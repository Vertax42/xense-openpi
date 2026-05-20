"""Equivalence test: every YAML in configs/_examples/ must load to a TrainConfig
that is dataclass-equal to the original in _CONFIGS.

If this test fails after editing a YAML, either:
  a) update _CONFIGS to match (and we'll soon delete it anyway), or
  b) update the YAML to fix the drift.

If this test fails after editing config.py, regenerate the YAMLs:
    python scripts/migrate_configs_to_yaml.py --overwrite
"""

from __future__ import annotations

import pathlib

import pytest

import openpi.training.config as _config
import openpi.training.yaml_loader as _yaml_loader


_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "_examples"


def _yaml_files() -> list[pathlib.Path]:
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(_EXAMPLES_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_path", _yaml_files(), ids=lambda p: p.stem)
def test_yaml_matches_configs_dict(yaml_path: pathlib.Path):
    name = yaml_path.stem
    assert name in _config._CONFIGS_DICT, (  # noqa: SLF001
        f"{name}.yaml exists in configs/_examples/ but no _CONFIGS entry has that name. "
        "Either delete the stale YAML or add a matching _CONFIGS entry."
    )
    expected = _config._CONFIGS_DICT[name]  # noqa: SLF001
    loaded = _yaml_loader.load(yaml_path)
    assert loaded == expected, (
        f"YAML config '{name}' does not match the corresponding _CONFIGS entry. "
        "Run `python scripts/migrate_configs_to_yaml.py --overwrite` to regenerate."
    )


def test_examples_dir_is_present():
    """At least one example YAML must exist (sanity check the migration ran)."""
    files = _yaml_files()
    assert files, f"No YAML files found in {_EXAMPLES_DIR}. Run scripts/migrate_configs_to_yaml.py first."
