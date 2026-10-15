"""A crude oxDNA engine (one oxDNA subprocess per path segment).

The engine runs the ``oxDNA`` binary on a user template input
(``<input_path>/input``) and overrides the keys it relies on from the
command line (``oxDNA input key=value ...``).

State layout passed to the order parameter:
    pos: (N, 9) = [x, a1, a3]  (centre of mass, base and normal vectors)
    vel: (N, 6) = [v, L]       (linear and body-frame angular momentum)
    box: (3,)   = box sides of the frame

Velocity reversal negates v and L. With ``thermostat = langevin`` it also
applies one Langevin update to the reversed state: oxDNA applies the
thermostat after each step, and this restores the update that otherwise
goes missing where a backward and a forward segment meet.
The MD barostat (``use_barostat``) is passed through untouched; its effect
on microscopic reversibility has not been verified.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

from infretis.classes.engines.enginebase import EngineBase

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

# input keys holding file names that must be absolute (oxDNA runs in exe_dir)
FILE_KEYS = ("topology", "seq_dep_file", "external_forces_file")
SUPPORTED_THERMOSTATS = ("no", "langevin")


def read_input_keys(filename: Union[str, Path]) -> Dict[str, str]:
    """Read top-level ``key = value`` pairs of an oxDNA input file."""
    keys = {}
    depth = 0
    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            line = line.split("#")[0].strip()
            depth += line.count("{") - line.count("}")
            match = re.match(r"^([\w.\-]+)\s*=\s*([^{]*)$", line)
            if depth == 0 and match and "}" not in line:
                keys[match.group(1)] = match.group(2).strip()
    return keys


def read_oxdna_conf(
    filename: Union[str, Path], idx: int, n_part: int
) -> Tuple[List[str], np.ndarray]:
    """Read frame ``idx`` of a .dat file: header lines, (N, 15) array."""
    block = n_part + 3
    with open(filename, encoding="utf-8") as fin:
        for _ in range(idx * block):
            if not fin.readline():
                raise ValueError(f"Frame {idx} not found in {filename}")
        lines = [fin.readline() for _ in range(block)]
    if not lines[-1]:
        raise ValueError(f"Frame {idx} not found in {filename}")
    head = [line.rstrip("\n") for line in lines[:3]]
    data = np.array([[float(x) for x in line.split()] for line in lines[3:]])
    if data.shape != (n_part, 15):
        raise ValueError(
            f"Frame {idx} in {filename} has shape {data.shape}, expected "
            f"({n_part}, 15). Are momenta included?"
        )
    return head, data


def write_oxdna_conf(
    filename: Union[str, Path], head: List[str], data: np.ndarray
) -> None:
    """Write one oxDNA configuration."""
    with open(filename, "w", encoding="utf-8") as fout:
        fout.write("\n".join(head) + "\n")
        for row in data:
            fout.write(" ".join(f"{x:.15g}" for x in row) + "\n")


def box_from_header(head: List[str]) -> np.ndarray:
    """Return the box sides from the ``b = Lx Ly Lz`` header line."""
    return np.array([float(x) for x in head[1].split("=")[1].split()])


class FrameReader:
    """Read complete frames from a trajectory that oxDNA is still writing."""

    def __init__(self, filename: str, n_part: int):
        """Set up the reader at the start of the file."""
        self.filename = filename
        self.n_part = n_part
        self.offset = 0

    def read_new(self) -> List[Tuple[int, List[str], np.ndarray]]:
        """Return the frames completed since the last call."""
        frames = []
        if not os.path.isfile(self.filename):
            return frames
        with open(self.filename, encoding="utf-8") as fin:
            fin.seek(self.offset)
            while True:
                lines = [fin.readline() for _ in range(self.n_part + 3)]
                # every line must be complete: readline() returns '' at EOF,
                # and later lines may already be there if oxDNA appends
                # in between two calls.
                if not all(line.endswith("\n") for line in lines):
                    break
                head = [line.rstrip("\n") for line in lines[:3]]
                data = np.array(
                    [[float(x) for x in line.split()] for line in lines[3:]]
                )
                frames.append((int(head[0].split()[2]), head, data))
                self.offset = fin.tell()
        return frames


class OxDNAEngine(EngineBase):
    """Run oxDNA as an external MD engine."""

    def __init__(
        self,
        timestep: float,
        subcycles: int,
        temperature: float,
        input_path: Union[str, Path],
        oxdna: str = "oxDNA",
        exe_path: Union[str, Path, None] = None,
        sleep: float = 0.1,
    ):
        """Set up the engine.

        Args:
            timestep: The oxDNA time step (overrides ``dt`` in the input).
            subcycles: Number of oxDNA steps between stored frames.
            temperature: Temperature in Kelvin (overrides ``T``).
            input_path: Directory with the template ``input`` file and
                the files it refers to (topology, ...).
            oxdna: Command to run oxDNA.
            exe_path: Directory that relative paths are taken from
                (default: the current working directory).
            sleep: Seconds between reads of the running trajectory.
        """
        super().__init__("oxDNA external engine", timestep, subcycles)
        self.name = "oxdna"
        self.ext = "dat"
        self.oxdna = shlex.split(oxdna)
        self.sleep = sleep
        exe_path = Path.cwd() if exe_path is None else Path(exe_path)
        self.input_path = (exe_path / input_path).resolve()
        self.input_files = {"input": self.input_path / "input"}
        if not self.input_files["input"].is_file():
            raise ValueError(f"No oxDNA input in {self.input_files['input']}")
        self.keys = read_input_keys(self.input_files["input"])

        # absolute paths for the files the template refers to
        self.file_overrides = {}
        for key in FILE_KEYS:
            if key in self.keys:
                path = Path(self.keys[key])
                if not path.is_absolute():
                    path = self.input_path / path
                if not path.is_file():
                    raise ValueError(f"oxDNA {key} file {path} not found")
                self.file_overrides[key] = str(path)
        if "topology" not in self.file_overrides:
            raise ValueError("The oxDNA input must define a topology")
        with open(self.file_overrides["topology"], encoding="utf-8") as fin:
            self.n_part = int(fin.readline().split()[0])

        # temperature: oxDNA reduced units, kT = T / 3000 K
        self.temperature = temperature
        self.kt = temperature / 3000.0
        self._beta = 1.0 / self.kt

        # thermostat: only those whose reversal we have checked
        self.thermostat = self.keys.get("thermostat", "no").lower()
        if self.thermostat not in SUPPORTED_THERMOSTATS:
            raise ValueError(
                f"thermostat = {self.thermostat} is not supported; "
                f"use one of {SUPPORTED_THERMOSTATS}"
            )
        if self.thermostat == "langevin":
            # same parameters as oxDNA's LangevinThermostat::init()
            if "gamma_trans" in self.keys:
                self.gamma = float(self.keys["gamma_trans"])
            elif "diff_coeff" in self.keys:
                self.gamma = self.kt / float(self.keys["diff_coeff"])
            else:
                raise ValueError("langevin needs gamma_trans or diff_coeff")
            self.gamma_rot = self.gamma / 3.0
        if self.keys.get("use_barostat", "false").lower() in ("1", "true"):
            logger.warning(
                "oxDNA barostat enabled: reversibility not verified."
            )

    def set_mdrun(self, md_items: Dict[str, Any]) -> None:
        """Set the execution directory for the worker."""
        self.exe_dir = md_items["exe_dir"]

    def _read_configuration(
        self, filename: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        """Read a single-frame configuration file."""
        head, data = read_oxdna_conf(filename, 0, self.n_part)
        return data[:, :9], data[:, 9:15], box_from_header(head), None

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Copy frame ``idx`` of a trajectory to a new file."""
        head, data = read_oxdna_conf(traj_file, idx, self.n_part)
        write_oxdna_conf(out_file, head, data)

    def _langevin_update(self, vel: np.ndarray) -> np.ndarray:
        """One oxDNA Langevin step on v and L (Euler-Maruyama)."""
        dt = self.timestep
        noise = self.rgen.normal(size=vel.shape)
        vel[:, :3] = (1 - self.gamma * dt) * vel[:, :3] + np.sqrt(
            2 * self.gamma * self.kt * dt
        ) * noise[:, :3]
        vel[:, 3:] = (1 - self.gamma_rot * dt) * vel[:, 3:] + np.sqrt(
            2 * self.gamma_rot * self.kt * dt
        ) * noise[:, 3:]
        return vel

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Negate v and L (plus one thermostat update, see module doc)."""
        head, data = read_oxdna_conf(filename, 0, self.n_part)
        data[:, 9:15] *= -1.0
        if self.thermostat == "langevin":
            data[:, 9:15] = self._langevin_update(data[:, 9:15])
        write_oxdna_conf(outfile, head, data)

    def modify_velocities(
        self, system: System, vel_settings: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Draw new v and L from Maxwell-Boltzmann (mass = inertia = 1)."""
        head, data = read_oxdna_conf(self.dump_frame(system), 0, self.n_part)
        kin_old = 0.5 * float(np.sum(data[:, 9:15] ** 2))
        data[:, 9:15] = self.rgen.normal(
            0.0, np.sqrt(self.kt), size=(self.n_part, 6)
        )
        if vel_settings.get("zero_momentum", False):
            data[:, 9:12] -= data[:, 9:12].mean(axis=0)
        kin_new = 0.5 * float(np.sum(data[:, 9:15] ** 2))
        conf_out = os.path.join(self.exe_dir, f"genvel.{self.ext}")
        write_oxdna_conf(conf_out, head, data)
        system.config = (conf_out, 0)
        system.ekin = kin_new
        dek = float("inf") if kin_old == 0.0 else kin_new - kin_old
        return dek, kin_new

    def _run_command(self, conf: str, name: str, nsteps: int) -> List[str]:
        """Return the oxDNA command line with the keys the engine needs."""
        overrides = {
            **self.file_overrides,
            "sim_type": "MD",
            "T": f"{self.temperature}K",
            "dt": f"{self.timestep}",
            "steps": f"{nsteps}",
            "seed": f"{self.rgen.integers(1, 2**31 - 1)}",
            "conf_file": conf,
            "trajectory_file": f"{name}_raw.dat",
            "lastconf_file": f"{name}_last.dat",
            "energy_file": f"{name}_energy.dat",
            "print_conf_interval": f"{self.subcycles}",
            "print_energy_every": f"{self.subcycles}",
            "time_scale": "linear",
            "restart_step_counter": "1",
            "refresh_vel": "0",
            "fix_diffusion": "0",
            "back_in_box": "0",
            "trajectory_print_momenta": "1",
            "no_stdout_energy": "1",
        }
        if "max_io" not in self.keys:
            # oxDNA aborts above max_io MB/s (default 1) after 30 s; frames
            # every `subcycles` steps easily exceed that for larger systems
            overrides["max_io"] = "1000"
        return [
            *self.oxdna,
            str(self.input_files["input"]),
            *[f"{key}={val}" for key, val in overrides.items()],
        ]

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ensemble: Dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """Run oxDNA from ``system.config`` and fill ``path``.

        Frame 0 is the start configuration itself (oxDNA does not write
        step 0). After the run it is prepended to the trajectory, so that
        phase point k is frame k of ``<name>.dat`` (step k * subcycles).
        """
        left, _, right = ensemble["interfaces"]
        initial_conf = system.config[0]
        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        raw_file = os.path.join(self.exe_dir, f"{name}_raw.dat")
        energy_file = os.path.join(self.exe_dir, f"{name}_energy.dat")
        msg_file.write(f"# Trajectory file is: {traj_file}")

        def add_frame(step_nr, data, box):
            order = self.calculate_order(
                system, xyz=data[:, :9], vel=data[:, 9:15], box=box
            )
            msg_file.write(f'{step_nr} {" ".join([str(j) for j in order])}')
            snapshot = {
                "order": order,
                "config": (traj_file, step_nr),
                "vel_rev": reverse,
            }
            phase_point = self.snapshot_to_system(system, snapshot)
            return self.add_to_path(path, phase_point, left, right)

        head0, data0 = read_oxdna_conf(initial_conf, 0, self.n_part)
        status, success, stop = add_frame(0, data0, box_from_header(head0))

        nsteps = (path.maxlen - 1) * self.subcycles
        cmd = self._run_command(initial_conf, name, nsteps)
        out_name = os.path.join(self.exe_dir, f"{name}.stdout")
        err_name = os.path.join(self.exe_dir, f"{name}.stderr")
        killed = False
        exe = None
        try:
            if not stop:
                with open(out_name, "wb") as fout, open(
                    err_name, "wb"
                ) as ferr:
                    exe = subprocess.Popen(
                        cmd,
                        stdout=fout,
                        stderr=ferr,
                        cwd=self.exe_dir,
                        preexec_fn=os.setsid,
                    )
                    reader = FrameReader(raw_file, self.n_part)
                    while not stop:
                        finished = exe.poll() is not None
                        for step, head, data in reader.read_new():
                            step_nr = step // self.subcycles
                            status, success, stop = add_frame(
                                step_nr, data, box_from_header(head)
                            )
                            if stop:
                                break
                        if finished:
                            break
                        sleep(self.sleep)
                    if stop and exe.poll() is None:
                        logger.debug("Terminating oxDNA execution")
                        os.killpg(os.getpgid(exe.pid), signal.SIGTERM)
                        killed = True
                    return_code = exe.wait(timeout=360)
                if return_code != 0 and not killed:
                    msg = (
                        f"oxDNA failed with return code {return_code}.\n"
                        f"Command: {' '.join(cmd)}\nSee {err_name}"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                if not stop:
                    status, success = "oxDNA ended before path end", False
        finally:
            if exe is not None and exe.poll() is None:
                os.killpg(os.getpgid(exe.pid), signal.SIGKILL)
                exe.wait()
            # prepend frame 0 so that (traj_file, k) is frame k of the file
            head0[0] = "t = 0"
            write_oxdna_conf(traj_file, head0, data0)
            if os.path.isfile(raw_file):
                complete = FrameReader(raw_file, self.n_part)
                complete.read_new()
                with open(raw_file, encoding="utf-8") as fin, open(
                    traj_file, "a", encoding="utf-8"
                ) as fout:
                    fout.write(fin.read(complete.offset))

        # energies: oxDNA prints per-particle values at steps 0, s, 2s, ...
        energies = {}
        if os.path.isfile(energy_file):
            for line in open(energy_file, encoding="utf-8"):
                cols = line.split()
                step = int(round(float(cols[0]) / self.timestep))
                energies[step] = [float(x) * self.n_part for x in cols[1:4]]
        ekin, vpot, etot, temp = [], [], [], []
        for k in range(path.length):
            upot, ukin, utot = energies.get(k * self.subcycles, [None] * 3)
            vpot.append(upot)
            ekin.append(ukin)
            etot.append(utot)
            # 6 degrees of freedom per nucleotide: kT = 2 K / (6 N)
            temp.append(
                None if ukin is None else ukin / (3 * self.n_part) * 3000.0
            )
        path.update_energies(ekin, vpot, etot, temp)

        last_file = os.path.join(self.exe_dir, f"{name}_last.dat")
        for fname in (raw_file, energy_file, out_name, err_name, last_file):
            self._removefile(fname)
        msg_file.write("# Propagation done.")
        return success, status
