from pathlib import Path

import pytest

from agentic_colorlime.config_io import apply_overrides, load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_profile_preserves_reported_settings():
    config, profile = load_profile(ROOT / "configs" / "benchmark.yaml")
    assert profile["name"] == "benchmark"
    assert config.lime_num_samples == 1000
    assert config.colorlime_k == 256
    assert config.critical_area_fraction == 0.20


def test_overrides_do_not_mutate_the_loaded_profile():
    config, _ = load_profile(ROOT / "configs" / "agent-demo.yaml")
    updated = apply_overrides(config, colorlime_k=512, lime_num_samples=None)
    assert config.colorlime_k == 128
    assert updated.colorlime_k == 512
    assert updated.lime_num_samples == config.lime_num_samples


def test_unknown_experiment_setting_is_rejected(tmp_path):
    profile = tmp_path / "invalid.yaml"
    profile.write_text("experiment:\n  invented_parameter: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invented_parameter"):
        load_profile(profile)
