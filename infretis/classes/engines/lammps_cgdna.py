"""A LAMMPS engine for the oxDNA/oxRNA models of the CG-DNA package.

Each nucleotide is a rigid ellipsoid with an orientation (quaternion) and an
angular momentum, which ``read_dump`` cannot load. This engine therefore
writes a complete data file (from the ``lammps.data`` template) for every
path segment and runs ``lammps.input`` with ``read_data ${initconf}``.

Frames are stored as .lammpstrj with the columns
    id type x y z ix iy iz vx vy vz c_q[1] c_q[2] c_q[3] c_q[4]
    angmomx angmomy angmomz
(wrapped positions + image flags, quaternion w,i,j,k, space-frame angular
momentum). LAMMPS writes the dump at step 0, so frame k is step k*subcycles.

State passed to the order parameter (same layout as the oxDNA engine):
    pos: (N, 9) = [unwrapped x, a1, a3], a1/a3 = body x/z axes
    vel: (N, 6) = [v, angular momentum (space frame)]
    box: (3,)   = box lengths

Supported integrators (checked in the input):
    fix nve/dot            NVE, exactly time reversible
    fix nve/dotc/langevin  Langevin (symmetric BAOAB splitting)
Velocity reversal negates v and the angular momentum. Thermostats and
barostats that cannot be reversed from the stored state (fix langevin,
Nose-Hoover nvt/npt/nph, temp/*, press/*, ...) are rejected.
Only ``units lj`` is supported; ``temperature`` is in LAMMPS lj units.
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
from infretis.classes.engines.lammps import read_energies, write_for_run

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

DUMP_COLUMNS = (
    "id type x y z ix iy iz vx vy vz c_q[1] c_q[2] c_q[3] c_q[4] "
    "angmomx angmomy angmomz"
)
MANDATORY = [
    "units lj",
    "variable subcycles index infretis_subcycles",
    "variable timestep index infretis_timestep",
    "variable nsteps index infretis_nsteps",
    "variable initconf index infretis_initconf",
    "variable name index infretis_name",
    "variable temperature index infretis_temperature",
    "variable seed index infretis_seed",
    "read_data ${initconf}",
    "compute q all property/atom quatw quati quatj quatk",
    "dump 1 all custom ${subcycles} ${name}.lammpstrj " + DUMP_COLUMNS,
    "dump_modify 1 sort id format float %.15g",
    "thermo ${subcycles}",
    "thermo_style custom step ke pe etotal temp",
    "thermo_modify norm no",
    "timestep ${timestep}",
    "run ${nsteps}",
]
INTEGRATORS = ("nve/dot", "nve/dotc/langevin")
FORBIDDEN_PREFIXES = (
    "nve", "nvt", "npt", "nph", "langevin", "temp/", "press/",
    "brownian", "gld", "gle", "rigid",
)


def check_cgdna_input(lmp_inp: Union[str, Path]) -> str:
    """Check the input template; return the integrator fix style."""
    with open(lmp_inp, encoding="utf-8") as fin:
        lines = [" ".join(line.split("#")[0].split()) for line in fin]
    missing = [cmd for cmd in MANDATORY if not any(cmd in ln for ln in lines)]
    if missing:
        raise ValueError(f"Missing commands in {lmp_inp}:\n{missing}")
    integrators = []
    for line in lines:
        tok = line.split()
        if len(tok) >= 4 and tok[0] == "fix":
            style = tok[3]
            if style in INTEGRATORS:
                integrators.append(style)
                if style == "nve/dotc/langevin" and not (
                    "${temperature}" in line and "${seed}" in line
                ):
                    raise ValueError(
                        "fix nve/dotc/langevin must use ${temperature} "
                        "and ${seed}"
                    )
            elif style.startswith(FORBIDDEN_PREFIXES):
                raise ValueError(
                    f"fix {style} is not supported (not time reversible "
                    f"from the stored state); use one of {INTEGRATORS}"
                )
        if (
            len(tok) >= 4
            and tok[0] == "pair_coeff"
            and re.search(r"ox[dr]na\d?/(stk|dh)", line)
            and "${temperature}" not in line
        ):
            raise ValueError(
                "oxDNA stacking and Debye-Huckel pair_coeff lines must use "
                f"${{temperature}}: '{line}'"
            )
        if re.search(r"ox[dr]na\d?/dh", line) and tok[0] == "pair_coeff":
            if "half_charged_ends yes" not in line:
                logger.warning(
                    "oxdna dh without 'half_charged_ends yes' differs "
                    "from the oxDNA package default."
                )
    if len(integrators) != 1:
        raise ValueError(
            f"Need exactly one integrator fix out of {INTEGRATORS}, "
            f"found {integrators}"
        )
    return integrators[0]


def read_data_template(filename: Union[str, Path]) -> Dict[str, Any]:
    """Split a LAMMPS data file into header lines and sections."""
    with open(filename, encoding="utf-8") as fin:
        lines = fin.read().splitlines()
    names = ("Masses", "Atoms", "Velocities", "Bonds", "Ellipsoids")
    header, sections, order, current = [], {}, [], None
    for line in lines:
        key = line.split("#")[0].strip()
        if key in names or key.split()[:1] == ["Atoms"]:
            current = key.split()[0]
            sections[current] = []
            order.append((current, line))
            continue
        if current is None:
            header.append(line)
        elif line.strip():
            sections[current].append(line.split())
    for name in ("Atoms", "Bonds", "Ellipsoids"):
        if name not in sections:
            raise ValueError(f"Section {name} missing in {filename}")
    for name in ("Atoms", "Ellipsoids"):
        sections[name].sort(key=lambda row: int(row[0]))
    return {"header": header, "sections": sections, "order": order}


def quat_to_axes(q: np.ndarray) -> np.ndarray:
    """Quaternions (N, 4: w, i, j, k) -> rotation matrices (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def read_cgdna_frame(
    filename: Union[str, Path], idx: int, n_atoms: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Read frame ``idx``: box bounds (3, 2) and atom data (N, 18)."""
    block = n_atoms + 9
    with open(filename, encoding="utf-8") as fin:
        for _ in range(idx * block):
            if not fin.readline():
                raise ValueError(f"Frame {idx} not in {filename}")
        lines = [fin.readline() for _ in range(block)]
    if not lines[-1]:
        raise ValueError(f"Frame {idx} not in {filename}")
    return parse_frame(lines, n_atoms)


def parse_frame(lines: List[str], n_atoms: int):
    """Parse one dump block (9 header lines + N atom lines)."""
    if lines[8].split()[2:] != DUMP_COLUMNS.split():
        raise ValueError(f"Unexpected dump columns: {lines[8].strip()}")
    bounds = np.array(
        [[float(v) for v in ln.split()[:2]] for ln in lines[5:8]]
    )
    data = np.array([[float(v) for v in ln.split()] for ln in lines[9:]])
    if data.shape != (n_atoms, 18):
        raise ValueError(f"Frame has shape {data.shape}")
    return bounds, data[np.argsort(data[:, 0])]


def write_cgdna_frame(
    filename: Union[str, Path], bounds: np.ndarray, data: np.ndarray,
    step: int = 0, append: bool = False,
) -> None:
    """Write one frame in the dump format used by this engine."""
    with open(filename, "a" if append else "w", encoding="utf-8") as fout:
        fout.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n")
        fout.write(f"{len(data)}\nITEM: BOX BOUNDS pp pp pp\n")
        for lo, hi in bounds:
            fout.write(f"{lo:.15g} {hi:.15g}\n")
        fout.write(f"ITEM: ATOMS {DUMP_COLUMNS}\n")
        for row in data:
            ints = " ".join(str(int(v)) for v in row[:2])
            xyz = " ".join(f"{v:.15g}" for v in row[2:5])
            img = " ".join(str(int(v)) for v in row[5:8])
            rest = " ".join(f"{v:.15g}" for v in row[8:])
            fout.write(f"{ints} {xyz} {img} {rest}\n")


class CGDNAFrameReader:
    """Read complete frames from a dump that LAMMPS is still writing."""

    def __init__(self, filename: str, n_atoms: int):
        """Start reading at the beginning of the file."""
        self.filename = filename
        self.n_atoms = n_atoms
        self.offset = 0

    def read_new(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return frames completed since the last call."""
        frames = []
        if not os.path.isfile(self.filename):
            return frames
        with open(self.filename, encoding="utf-8") as fin:
            fin.seek(self.offset)
            while True:
                lines = [fin.readline() for _ in range(self.n_atoms + 9)]
                # every line must be complete (readline returns '' at EOF,
                # and later lines may appear between two calls)
                if not all(line.endswith("\n") for line in lines):
                    break
                frames.append(parse_frame(lines, self.n_atoms))
                self.offset = fin.tell()
        return frames


class LAMMPSCGDNAEngine(EngineBase):
    """Run oxDNA-model simulations with LAMMPS (CG-DNA package)."""

    def __init__(
        self,
        lmp: str,
        input_path: Union[str, Path],
        timestep: float,
        subcycles: int,
        temperature: float,
        exe_path: Union[str, Path, None] = None,
        sleep: float = 0.1,
    ):
        """Set up the engine.

        Args:
            lmp: Command to run LAMMPS (built with CG-DNA).
            input_path: Directory with ``lammps.input`` and ``lammps.data``.
            timestep: Time step (lj units).
            subcycles: Steps between stored frames.
            temperature: Temperature in LAMMPS lj units (kT).
            exe_path: Directory relative paths refer to (default: cwd).
            sleep: Seconds between reads of the running dump.
        """
        super().__init__("LAMMPS CG-DNA engine", timestep, subcycles)
        self.name = "lammps_cgdna"
        self.ext = "lammpstrj"
        self.lmp = shlex.split(lmp)
        self.sleep = sleep
        exe_path = Path.cwd() if exe_path is None else Path(exe_path)
        self.input_path = (exe_path / input_path).resolve()
        self.input_files = {
            "input": self.input_path / "lammps.input",
            "data": self.input_path / "lammps.data",
        }
        self.integrator = check_cgdna_input(self.input_files["input"])
        self.template = read_data_template(self.input_files["data"])
        atoms = self.template["sections"]["Atoms"]
        ellip = self.template["sections"]["Ellipsoids"]
        self.n_atoms = len(atoms)
        # mass = density * ellipsoid volume; body inertia as in LAMMPS
        shape = np.array([[float(v) for v in row[1:4]] for row in ellip])
        density = np.array([float(row[7]) for row in atoms])
        self.mass = density * np.pi / 6.0 * np.prod(shape, axis=1)
        semi2 = (shape / 2.0) ** 2
        self.inertia = 0.2 * self.mass[:, None] * np.stack(
            [semi2[:, 1] + semi2[:, 2], semi2[:, 0] + semi2[:, 2],
             semi2[:, 0] + semi2[:, 1]], axis=1)
        self._check_mass_override()
        self.temperature = temperature
        self.kt = temperature
        self._beta = 1.0 / temperature

    def _check_mass_override(self) -> None:
        """Masses must come from the data file (density), not `set`."""
        with open(self.input_files["input"], encoding="utf-8") as fin:
            for line in fin:
                match = re.match(r"^\s*set\s+\S+\s+\S+\s+mass\s+(\S+)", line)
                if match and not np.allclose(
                    self.mass, float(match.group(1)), rtol=1e-6
                ):
                    raise ValueError(
                        f"'{line.strip()}' differs from the data-file "
                        f"masses ({self.mass.min():.6g}..."
                        f"{self.mass.max():.6g}); set masses via density."
                    )

    # ---------------------------------------------------------------- I/O
    def _frame_state(self, bounds, data):
        """(pos (N, 9), vel (N, 6), box (3,)) from a frame."""
        box = bounds[:, 1] - bounds[:, 0]
        xyz = data[:, 2:5] + data[:, 5:8] * box
        rot = quat_to_axes(data[:, 11:15])
        pos = np.hstack([xyz, rot[:, :, 0], rot[:, :, 2]])
        vel = np.hstack([data[:, 8:11], data[:, 15:18]])
        return pos, vel, box

    def _read_configuration(
        self, filename: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        """Read a single-frame configuration file."""
        pos, vel, box = self._frame_state(
            *read_cgdna_frame(filename, 0, self.n_atoms)
        )
        return pos, vel, box, None

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Copy frame ``idx`` of a trajectory to a new file."""
        bounds, data = read_cgdna_frame(traj_file, idx, self.n_atoms)
        write_cgdna_frame(out_file, bounds, data)

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Negate v and the angular momentum."""
        bounds, data = read_cgdna_frame(filename, 0, self.n_atoms)
        data[:, 8:11] *= -1.0
        data[:, 15:18] *= -1.0
        write_cgdna_frame(outfile, bounds, data)

    def _write_data(self, outfile: str, bounds, data) -> None:
        """Write a data file: template topology + this frame's state."""
        tmpl = self.template
        sec = tmpl["sections"]
        with open(outfile, "w", encoding="utf-8") as fout:
            for line in tmpl["header"]:
                tok = line.split()
                for k, ax in enumerate("xyz"):
                    if tok[2:4] == [f"{ax}lo", f"{ax}hi"]:
                        line = (f"{bounds[k, 0]:.15g} {bounds[k, 1]:.15g} "
                                f"{ax}lo {ax}hi")
                fout.write(line + "\n")
            for name, title in tmpl["order"]:
                fout.write(f"{title}\n\n")
                if name == "Atoms":
                    for row, d in zip(sec["Atoms"], data):
                        x = " ".join(f"{v:.15g}" for v in d[2:5])
                        img = " ".join(str(int(v)) for v in d[5:8])
                        fout.write(f"{row[0]} {row[1]} {x} "
                                   f"{' '.join(row[5:8])} {img}\n")
                elif name == "Velocities":
                    for d in data:
                        vals = " ".join(
                            f"{v:.15g}" for v in (*d[8:11], *d[15:18]))
                        fout.write(f"{int(d[0])} {vals}\n")
                elif name == "Ellipsoids":
                    for row, d in zip(sec["Ellipsoids"], data):
                        q = " ".join(f"{v:.15g}" for v in d[11:15])
                        fout.write(f"{row[0]} {' '.join(row[1:4])} {q}\n")
                else:
                    for row in sec[name]:
                        fout.write(" ".join(row) + "\n")
                fout.write("\n")
            if "Velocities" not in sec:
                fout.write("Velocities\n\n")
                for d in data:
                    vals = " ".join(f"{v:.15g}" for v in (*d[8:11], *d[15:18]))
                    fout.write(f"{int(d[0])} {vals}\n")

    # --------------------------------------------------------- dynamics
    def modify_velocities(
        self, system: System, vel_settings: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Draw v and angular momenta from Maxwell-Boltzmann."""
        bounds, data = read_cgdna_frame(
            self.dump_frame(system), 0, self.n_atoms)
        ekin_old = self._ekin(data)
        sig_v = np.sqrt(self.kt / self.mass)[:, None]
        data[:, 8:11] = self.rgen.normal(size=(self.n_atoms, 3)) * sig_v
        if vel_settings.get("zero_momentum", False):
            momentum = (self.mass[:, None] * data[:, 8:11]).sum(0)
            vcom = momentum / self.mass.sum()
            data[:, 8:11] -= vcom
        # body-frame angular momentum per principal axis, then to space frame
        l_body = self.rgen.normal(size=(self.n_atoms, 3)) * np.sqrt(
            self.kt * self.inertia)
        rot = quat_to_axes(data[:, 11:15])
        data[:, 15:18] = np.einsum("nij,nj->ni", rot, l_body)
        ekin_new = self._ekin(data)
        conf_out = os.path.join(self.exe_dir, f"genvel.{self.ext}")
        write_cgdna_frame(conf_out, bounds, data)
        system.config = (conf_out, 0)
        system.ekin = ekin_new
        dek = float("inf") if ekin_old == 0.0 else ekin_new - ekin_old
        return dek, ekin_new

    def _ekin(self, data: np.ndarray) -> float:
        """Translational + rotational kinetic energy."""
        rot = quat_to_axes(data[:, 11:15])
        l_body = np.einsum("nji,nj->ni", rot, data[:, 15:18])
        return float(0.5 * np.sum(self.mass[:, None] * data[:, 8:11] ** 2)
                     + 0.5 * np.sum(l_body ** 2 / self.inertia))

    def set_mdrun(self, md_items: Dict[str, Any]) -> None:
        """Set the execution directory for the worker."""
        self.exe_dir = md_items["exe_dir"]

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ensemble: Dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """Run LAMMPS from ``system.config`` and fill ``path``."""
        left, _, right = ensemble["interfaces"]
        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        data_file = os.path.join(self.exe_dir, f"{name}.data")
        run_input = os.path.join(self.exe_dir, f"{name}.input")
        log_file = os.path.join(self.exe_dir, f"{name}.log")
        out_name = os.path.join(self.exe_dir, f"{name}.stdout")
        msg_file.write(f"# Trajectory file is: {traj_file}")

        bounds, data = read_cgdna_frame(system.config[0], 0, self.n_atoms)
        self._write_data(data_file, bounds, data)
        write_for_run(
            self.input_files["input"],
            run_input,
            {
                "infretis_subcycles": self.subcycles,
                "infretis_timestep": self.timestep,
                "infretis_nsteps": (path.maxlen - 1) * self.subcycles,
                "infretis_initconf": data_file,
                "infretis_name": name,
                "infretis_temperature": self.temperature,
                # LAMMPS' Marsaglia generator accepts seeds up to 9e8
                "infretis_seed": int(self.rgen.integers(1, 900_000_000)),
            },
        )
        cmd = [*self.lmp, "-in", run_input, "-log", log_file,
               "-screen", "none"]
        status, success, stop, killed = "Running", False, False, False
        with open(out_name, "wb") as fout:
            exe = subprocess.Popen(
                cmd, stdout=fout, stderr=subprocess.STDOUT,
                cwd=self.exe_dir, preexec_fn=os.setsid,
            )
            try:
                reader = CGDNAFrameReader(traj_file, self.n_atoms)
                step_nr = 0
                ekin = []
                while not stop:
                    finished = exe.poll() is not None
                    for frame_bounds, frame in reader.read_new():
                        pos, vel, box = self._frame_state(frame_bounds, frame)
                        ekin.append(self._ekin(frame))
                        order = self.calculate_order(
                            system, xyz=pos, vel=vel, box=box)
                        msg_file.write(
                            f'{step_nr} {" ".join(str(j) for j in order)}')
                        snapshot = {"order": order,
                                    "config": (traj_file, step_nr),
                                    "vel_rev": reverse}
                        status, success, stop = self.add_to_path(
                            path, self.snapshot_to_system(system, snapshot),
                            left, right)
                        step_nr += 1
                        if stop:
                            break
                    if finished:
                        break
                    sleep(self.sleep)
                if stop and exe.poll() is None:
                    os.killpg(os.getpgid(exe.pid), signal.SIGTERM)
                    killed = True
                return_code = exe.wait(timeout=360)
            finally:
                if exe.poll() is None:
                    os.killpg(os.getpgid(exe.pid), signal.SIGKILL)
                    exe.wait()
        if return_code != 0 and not killed:
            msg = (f"LAMMPS failed with return code {return_code}.\n"
                   f"Command: {' '.join(cmd)}\nSee {out_name}")
            logger.error(msg)
            raise RuntimeError(msg)
        if not stop:
            status, success = "LAMMPS ended before path end", False

        # potential energy from the log; kinetic energy (translational +
        # rotational, which LAMMPS' ke/etotal leave out) from the frames
        energy = read_energies(log_file) if os.path.isfile(log_file) else {}
        end = path.length
        vpot = list(energy.get("PotEng", [])[:end])
        vpot += [None] * (end - len(vpot))
        ekin = ekin[:end]
        etot = [None if u is None else u + k for u, k in zip(vpot, ekin)]
        temp = [k / (3.0 * self.n_atoms) for k in ekin]  # 6 dof per site
        path.update_energies(ekin, vpot, etot, temp)
        for fname in (data_file, run_input, log_file, out_name):
            self._removefile(fname)
        msg_file.write("# Propagation done.")
        return success, status
