"""Unit tests for `setup.expand_ensemble_move_policy`.

Pure-config tests. No REPEX, no MD, no I/O.

Ergonomic policy semantics under test:
  * "000" = [0-] and "001" = [0+] are protected default-sh ensembles. A
    normal user never mentions them; the ergonomic layer never assigns
    wf/mwf to them.
  * `default_move` applies only to the ordinary plus ensembles "002"
    onward.
  * `mwf_ensembles` and `mwf_subcycle_small` are sparse exceptions for
    ordinary plus ensembles; a selector resolving to "000"/"001" is an
    error.
  * `minus_move` is not required; only absent or an explicit "sh" is
    accepted, anything else is an error.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from infretis.setup import (
    TOMLConfigError,
    expand_ensemble_move_policy,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #


def _base_config(n_ens: int = 11) -> Dict[str, Any]:
    """Minimal config skeleton with only the fields the expander reads."""
    return {
        "simulation": {
            "interfaces": [round(0.1 * i, 4) for i in range(n_ens)],
            "tis_set": {},
        }
    }


def _with_policy(cfg: Dict[str, Any], **policy: Any) -> Dict[str, Any]:
    cfg["simulation"]["ensemble_move_policy"] = policy
    return cfg


def _with_infinit(cfg: Dict[str, Any], **infinit: Any) -> Dict[str, Any]:
    """Mark `cfg` as an `inft infinit` config (top-level `[infinit]` table)."""
    cfg["infinit"] = infinit
    return cfg


def _set_n_ens(cfg: Dict[str, Any], n_ens: int) -> Dict[str, Any]:
    """Resize `simulation.interfaces` to model an infinit interface update."""
    cfg["simulation"]["interfaces"] = [round(0.1 * i, 4) for i in range(n_ens)]
    return cfg


# --------------------------------------------------------------------------- #
# 1-2. no-ops                                                                 #
# --------------------------------------------------------------------------- #


def test_no_policy_section_is_noop():
    cfg = _base_config(5)
    snap = copy.deepcopy(cfg)
    expand_ensemble_move_policy(cfg)
    assert cfg == snap


def test_disabled_policy_is_noop_and_stripped():
    cfg = _with_policy(_base_config(5), enabled=False, default_move="mwf")
    expand_ensemble_move_policy(cfg)
    assert "ensemble_move_policy" not in cfg["simulation"]
    assert "shooting_moves" not in cfg["simulation"]
    assert cfg["simulation"]["tis_set"] == {}


# --------------------------------------------------------------------------- #
# 3-9. expansion + protected "000"/"001"                                      #
# --------------------------------------------------------------------------- #


def test_default_mwf_preserves_zero_minus_and_zero_plus_as_sh():
    """default_move='mwf' fills only "002" onward; "000"/"001" stay 'sh'."""
    cfg = _with_policy(_base_config(6), enabled=True, default_move="mwf")
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "mwf", "mwf", "mwf", "mwf",
    ]


def test_default_wf_preserves_zero_minus_and_zero_plus_as_sh():
    """default_move='wf' fills only "002" onward; "000"/"001" stay 'sh'."""
    cfg = _with_policy(_base_config(6), enabled=True, default_move="wf")
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "wf", "wf", "wf", "wf",
    ]


def test_default_wf_with_mwf_range_expands_shooting_moves():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "mwf", "mwf", "mwf", "wf", "wf", "wf",
    ]


def test_no_minus_move_required():
    """No minus_move key: expansion succeeds, "000"/"001" are 'sh'."""
    cfg = _with_policy(_base_config(4), enabled=True, default_move="wf")
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == ["sh", "sh", "wf", "wf"]


def test_minus_move_sh_is_accepted_but_not_required():
    """An explicit minus_move='sh' yields the same expansion as omitting it."""
    without = _with_policy(_base_config(6), enabled=True, default_move="wf")
    with_sh = _with_policy(
        _base_config(6), enabled=True, default_move="wf", minus_move="sh"
    )
    expand_ensemble_move_policy(without)
    expand_ensemble_move_policy(with_sh)
    assert (
        without["simulation"]["shooting_moves"]
        == with_sh["simulation"]["shooting_moves"]
        == ["sh", "sh", "wf", "wf", "wf", "wf"]
    )


def test_default_mwf_subcycle_small_becomes_canonical_scalar():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=7,
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small"] == 7


def test_sparse_subcycle_table_expands_per_ensemble():
    cfg = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "002": 2, "003": 2, "004": 2, "009": 8,
    }
    # ensembles 005, 006 covered by mwf_ensembles but not in subcycle map
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small"] == 4


def test_missing_ensembles_use_defaults():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=3,
        mwf_ensembles=["002"],
    )
    expand_ensemble_move_policy(cfg)
    moves = cfg["simulation"]["shooting_moves"]
    assert moves == ["sh", "sh", "mwf", "wf", "wf"]
    tis_set = cfg["simulation"]["tis_set"]
    assert tis_set["mwf_subcycle_small"] == 3
    assert "mwf_subcycle_small_by_ensemble" not in tis_set


# --------------------------------------------------------------------------- #
# protected-ensemble selectors                                                #
# --------------------------------------------------------------------------- #


def test_mwf_ensembles_cannot_target_000():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["000"],
    )
    with pytest.raises(TOMLConfigError, match="protected"):
        expand_ensemble_move_policy(cfg)


def test_mwf_ensembles_cannot_target_001():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["001"],
    )
    with pytest.raises(TOMLConfigError, match="protected"):
        expand_ensemble_move_policy(cfg)


def test_mwf_ensembles_open_left_range_hits_protected_and_raises():
    """':002' resolves to {000,001,002}; touching 000/001 is an error."""
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=[":002"],
    )
    with pytest.raises(TOMLConfigError, match="protected"):
        expand_ensemble_move_policy(cfg)


def test_mwf_ensembles_open_range_from_002():
    cfg = _with_policy(
        _base_config(6),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "mwf", "mwf", "mwf", "mwf",
    ]


def test_subcycle_override_cannot_target_000_or_001():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="mwf",
        default_mwf_subcycle_small=4,
        mwf_subcycle_small={"001": 2},
    )
    with pytest.raises(TOMLConfigError, match="protected"):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# minus_move handling                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_minus", ["wf", "mwf"])
def test_minus_move_non_sh_raises(bad_minus):
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        minus_move=bad_minus,
    )
    with pytest.raises(TOMLConfigError, match="minus_move"):
        expand_ensemble_move_policy(cfg)


def test_unknown_minus_move_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        minus_move="xx",
    )
    with pytest.raises(TOMLConfigError, match="minus_move"):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# validation errors                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "selector",
    ["abc", "5:2", "010", "-1", "", " ", "1:abc", 3, ["003"]],
)
def test_invalid_selectors_raise(selector):
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=[selector],
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.5, "4", None])
def test_invalid_subcycle_values_raise(bad_value):
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002"],
        mwf_subcycle_small={"002": bad_value},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_invalid_default_subcycle_value_raises():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=0,
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_unknown_default_move_raises():
    cfg = _with_policy(_base_config(4), enabled=True, default_move="xx")
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_subcycle_override_for_non_mwf_ensemble_still_raises():
    """Dead subcycle config (target ensemble is not 'mwf') is strict."""
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        # ensemble "002" is NOT in mwf_ensembles -> resolved as "wf"
        mwf_subcycle_small={"002": 4},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_overlapping_mwf_ensembles_selectors_raise():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004", "003"],
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


def test_overlapping_subcycle_selectors_raise():
    cfg = _with_policy(
        _base_config(8),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002:004"],
        mwf_subcycle_small={"002:003": 2, "003": 4},
    )
    with pytest.raises(TOMLConfigError):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# conflict policy                                                             #
# --------------------------------------------------------------------------- #


def test_conflict_with_existing_shooting_moves_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
    )
    cfg["simulation"]["shooting_moves"] = ["sh", "sh", "wf", "wf"]
    with pytest.raises(TOMLConfigError, match="shooting_moves"):
        expand_ensemble_move_policy(cfg)


def test_conflict_with_existing_scalar_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
    )
    cfg["simulation"]["tis_set"]["mwf_subcycle_small"] = 9
    with pytest.raises(TOMLConfigError, match="mwf_subcycle_small"):
        expand_ensemble_move_policy(cfg)


def test_conflict_with_existing_by_ensemble_table_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["002"],
    )
    cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] = {"002": 3}
    with pytest.raises(TOMLConfigError, match="by_ensemble"):
        expand_ensemble_move_policy(cfg)


def test_unknown_conflict_policy_raises():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        conflict_policy="overwrite",
    )
    with pytest.raises(TOMLConfigError, match="conflict_policy"):
        expand_ensemble_move_policy(cfg)


# --------------------------------------------------------------------------- #
# idempotence + canonical equivalence                                         #
# --------------------------------------------------------------------------- #


def test_policy_section_removed_after_successful_expansion():
    cfg = _with_policy(
        _base_config(4),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
    )
    expand_ensemble_move_policy(cfg)
    assert "ensemble_move_policy" not in cfg["simulation"]


def test_expansion_is_idempotent():
    cfg = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    expand_ensemble_move_policy(cfg)
    snap = copy.deepcopy(cfg)
    expand_ensemble_move_policy(cfg)  # no-op the second time
    assert cfg == snap


def test_ergonomic_equals_handwritten_canonical_after_expansion():
    ergonomic = _with_policy(
        _base_config(11),
        enabled=True,
        default_move="wf",
        minus_move="sh",  # accepted but not required
        default_mwf_subcycle_small=4,
        mwf_ensembles=["002:006", "009"],
        mwf_subcycle_small={"002:004": 2, "009": 8},
    )
    handwritten = _base_config(11)
    handwritten["simulation"]["shooting_moves"] = [
        "sh", "sh", "mwf", "mwf", "mwf", "mwf", "mwf", "wf", "wf", "mwf", "wf",
    ]
    handwritten["simulation"]["tis_set"]["mwf_subcycle_small"] = 4
    handwritten["simulation"]["tis_set"][
        "mwf_subcycle_small_by_ensemble"
    ] = {"002": 2, "003": 2, "004": 2, "009": 8}
    expand_ensemble_move_policy(ergonomic)
    assert ergonomic == handwritten


# --------------------------------------------------------------------------- #
# selector edge cases                                                         #
# --------------------------------------------------------------------------- #


def test_open_right_range_works():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["003:"],
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "wf", "mwf", "mwf",
    ]


def test_unpadded_selector_normalizes():
    cfg = _with_policy(
        _base_config(5),
        enabled=True,
        default_move="wf",
        mwf_ensembles=["3"],  # equivalent to "003"
        mwf_subcycle_small={"3": 6},
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "wf", "mwf", "wf",
    ]
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "003": 6,
    }


# --------------------------------------------------------------------------- #
# infinit mode: policy is the persistent, authoritative source of truth       #
# --------------------------------------------------------------------------- #


def test_infinit_policy_overwrites_auto_shooting_moves():
    """`inft infinit` auto-materializes shooting_moves; policy overwrites it."""
    cfg = _with_infinit(
        _with_policy(_base_config(8), enabled=True, default_move="mwf"),
        cstep=0,
    )
    # an earlier infinit iteration left an auto-generated (wf) field behind
    cfg["simulation"]["shooting_moves"] = ["sh", "wf"] + ["wf"] * 6
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "mwf", "mwf", "mwf", "mwf", "mwf", "mwf",
    ]


def test_infinit_policy_is_not_removed_after_expansion():
    """In infinit mode the policy section survives so it owns later cycles."""
    cfg = _with_infinit(
        _with_policy(_base_config(6), enabled=True, default_move="mwf"),
        cstep=0,
    )
    expand_ensemble_move_policy(cfg)
    policy = cfg["simulation"].get("ensemble_move_policy")
    assert policy is not None
    assert policy["enabled"] is True


def test_non_infinit_policy_conflict_still_errors():
    """Without [infinit], a pre-existing canonical field is still a conflict."""
    cfg = _with_policy(_base_config(6), enabled=True, default_move="mwf")
    cfg["simulation"]["shooting_moves"] = ["sh", "sh", "wf", "wf", "wf", "wf"]
    with pytest.raises(TOMLConfigError, match="shooting_moves"):
        expand_ensemble_move_policy(cfg)


def test_infinit_policy_persists_across_two_expansions():
    """Re-running setup must not let moves drift away from default_move."""
    cfg = _with_infinit(
        _with_policy(_base_config(8), enabled=True, default_move="mwf"),
        cstep=0,
    )
    expected = ["sh", "sh"] + ["mwf"] * 6
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == expected
    # second setup cycle on the same (still policy-carrying) config
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == expected
    assert cfg["simulation"]["ensemble_move_policy"]["enabled"] is True


def test_infinit_policy_regenerates_after_interface_count_change():
    """When infinit adds interfaces, moves regenerate for the new n_ens."""
    cfg = _with_infinit(
        _with_policy(_base_config(4), enabled=True, default_move="mwf"),
        cstep=0,
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == ["sh", "sh", "mwf", "mwf"]
    # infinit estimates new interface positions -> more ensembles
    _set_n_ens(cfg, 8)
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == [
        "sh", "sh", "mwf", "mwf", "mwf", "mwf", "mwf", "mwf",
    ]


def test_infinit_policy_subcycle_scalar_overwrites_previous_generated_value():
    """A stale canonical scalar from a restart TOML is overwritten by policy."""
    cfg = _with_infinit(
        _with_policy(
            _base_config(6),
            enabled=True,
            default_move="mwf",
            default_mwf_subcycle_small=1,
        ),
        cstep=2,
    )
    # previous restart wrote a different scalar
    cfg["simulation"]["tis_set"]["mwf_subcycle_small"] = 4
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small"] == 1


def test_infinit_policy_subcycle_table_regenerated():
    """The per-ensemble subcycle table is regenerated from policy each cycle."""
    cfg = _with_infinit(
        _with_policy(
            _base_config(8),
            enabled=True,
            default_move="mwf",
            default_mwf_subcycle_small=4,
            mwf_subcycle_small={"002:003": 2},
        ),
        cstep=2,
    )
    # stale table from a previous iteration that no longer matches policy
    cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] = {
        "005": 9,
    }
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "002": 2, "003": 2,
    }
    # idempotent: a second cycle yields the same table
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["tis_set"]["mwf_subcycle_small_by_ensemble"] == {
        "002": 2, "003": 2,
    }


def test_infinit_reload_corrects_wf_drift_from_update_toml():
    """End-to-end drift guard for the infinit update/write/reload cycle.

    The infinit interface-update step (``update_toml_interfaces``) rewrites
    ``simulation.shooting_moves`` as ``["sh", "sh", "wf", ...]``. Because the
    policy section is kept in the TOML, the next setup cycle re-expands it and
    must restore the policy's ``default_move`` ("mwf") from "002" onward.
    """
    cfg = _with_infinit(
        _with_policy(_base_config(8), enabled=True, default_move="mwf"),
        cstep=3,
    )
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == ["sh", "sh"] + ["mwf"] * 6

    # model update_toml_interfaces() hard-coding wf, then writing the TOML
    n_ens = len(cfg["simulation"]["interfaces"])
    cfg["simulation"]["shooting_moves"] = ["sh", "sh"] + ["wf"] * (n_ens - 2)

    # next `infretisrun` reloads that TOML and re-runs setup -> policy wins
    expand_ensemble_move_policy(cfg)
    assert cfg["simulation"]["shooting_moves"] == ["sh", "sh"] + ["mwf"] * 6
