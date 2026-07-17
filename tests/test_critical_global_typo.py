"""Regression test for CRITICAL #4: typo'd `global` declaration in
`pygad.gadget.config.read_config`.

The `global` statement (pygad/gadget/config.py:83) declared `std_nameto_HDF5`
(missing underscore) instead of the real module-level mapping
`std_name_to_HDF5` (config.py:59, exported via `__all__` and consumed by
`gadget/handler.py` and `snapshot/snapshot.py`). The declaration therefore did
not cover the name the function is supposed to maintain, leaving
`read_config` without a valid `global` binding for the module-level mapping.

The first test pins the defect directly: the `global` statement must name
`std_name_to_HDF5` (and must not name the typo). The second test exercises
`read_config` with a custom config and asserts the module-level mapping
actually reflects the update.
"""

import ast
import inspect

import pygad.gadget.config as config

# module-level state mutated by read_config (snapshot/restore for suite safety)
_STATE_NAMES = [
    "families",
    "block_order",
    "elements",
    "general",
    "default_gadget_units",
    "block_infos",
    "block_units",
    "std_name_to_HDF5",
    "HDF5_to_std_name",
]

MINIMAL_CFG = """\
[general]
kernel:                 cubic
vol_def_x:              1
UVB:                    HM01
IMF:                    Kroupa
unclear_blocks:         warning
block order:            POS, TSTB

[families]
gas:                    0
stars:                  4
dm:                     1,2,3
bh:                     5
baryons:                0,4,5

[base units]
LENGTH      =   ckpc / h_0
VELOCITY    =   a**(1/2) km / s
MASS        =   1e10 Msol / h_0

[hdf5 names]
POS  = Coordinates
TSTB = TestBlockDataset
"""


def _snapshot_state():
    snap = {}
    for name in _STATE_NAMES:
        obj = getattr(config, name)
        if isinstance(obj, dict):
            snap[name] = {k: (list(v) if isinstance(v, list) else v)
                          for k, v in obj.items()}
        else:
            snap[name] = list(obj)
    return snap


def _restore_state(snap):
    for name, saved in snap.items():
        obj = getattr(config, name)
        obj.clear()
        if isinstance(obj, dict):
            obj.update(saved)
        else:
            obj.extend(saved)


def test_read_config_declares_std_name_to_HDF5_as_global():
    """read_config's `global` statement must name `std_name_to_HDF5`.

    Pre-fix it declares the typo `std_nameto_HDF5` (missing underscore),
    which is a no-op for the intended module-level mapping.
    """
    tree = ast.parse(inspect.getsource(config.read_config))
    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            declared.update(node.names)
    assert "std_nameto_HDF5" not in declared, "typo global still present"
    assert "std_name_to_HDF5" in declared, (
        "read_config does not declare the module-level mapping "
        "`std_name_to_HDF5` as global"
    )


def test_read_config_updates_module_level_name_mappings(tmp_path):
    """read_config must update the module-level `std_name_to_HDF5` mapping
    (and the reverse `HDF5_to_std_name`) from the config's [hdf5 names]."""
    cfg_file = tmp_path / "gadget_test.cfg"
    cfg_file.write_text(MINIMAL_CFG)

    snap = _snapshot_state()
    try:
        config.read_config([str(cfg_file)])

        # the module-level mapping reflects exactly this config's entries
        assert set(config.std_name_to_HDF5) == {"POS ", "TSTB"}
        assert config.std_name_to_HDF5["POS "] == "Coordinates"
        assert config.std_name_to_HDF5["TSTB"] == "TestBlockDataset"

        # reverse mapping is rebuilt from the updated forward mapping
        assert config.HDF5_to_std_name["Coordinates"] == "POS "
        assert config.HDF5_to_std_name["TestBlockDataset"] == "TSTB"
    finally:
        _restore_state(snap)

    # state was fully restored
    assert config.std_name_to_HDF5 == snap["std_name_to_HDF5"]
    assert config.HDF5_to_std_name == snap["HDF5_to_std_name"]
