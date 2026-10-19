"""Test the input check of the LAMMPS CG-DNA engine."""

import pytest

from infretis.classes.engines.lammps_cgdna import MANDATORY, check_cgdna_input

LANGEVIN = (
    "fix 1 all nve/dotc/langevin ${temperature} ${temperature} 24.0 "
    "${seed} angmom 10"
)


def write_input(path, lines):
    """Write a template with the mandatory lines plus `lines`."""
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(MANDATORY + lines) + "\n")


def test_check_cgdna_input(tmp_path):
    """A complete template passes and returns the integrator."""
    inp = tmp_path / "lammps.input"
    write_input(inp, ["thermo_modify norm no flush yes", LANGEVIN])
    assert check_cgdna_input(inp) == "nve/dotc/langevin"


def test_check_cgdna_input_no_flush(tmp_path):
    """Without 'flush yes' the potential energies would be lost."""
    inp = tmp_path / "lammps.input"
    write_input(inp, [LANGEVIN])
    with pytest.raises(ValueError, match="flush yes"):
        check_cgdna_input(inp)


def test_check_cgdna_input_bad_thermostat(tmp_path):
    """A thermostat that cannot be reversed is rejected."""
    inp = tmp_path / "lammps.input"
    write_input(inp, ["thermo_modify norm no flush yes",
                      "fix 1 all langevin 0.1 0.1 24.0 1234"])
    with pytest.raises(ValueError, match="not supported"):
        check_cgdna_input(inp)
