"""Feature-contract tests: window parsing + feature spec loading (PLAN.md §6)."""

from __future__ import annotations

import pytest

from streamguard.config import (
    REPO_ROOT,
    load_feature_spec,
    parse_window,
    velocity_window_seconds,
)


class TestWindowParsing:
    def test_known_windows(self):
        assert parse_window("5m") == 300
        assert parse_window("1h") == 3600
        assert parse_window("24h") == 86400
        assert parse_window("7d") == 604800

    def test_rejects_invalid(self):
        for bad in ("24", "m", "2w", "-1h", "1.5h"):
            with pytest.raises(ValueError):
                parse_window(bad)


class TestFeatureSpec:
    def test_repo_spec_loads(self):
        spec = load_feature_spec()  # the real configs/features.yaml
        assert spec["key"] == "card_key"
        assert spec["velocity"]["metrics"][0] == "txn_count"

    def test_velocity_windows_match_plan(self):
        spec = load_feature_spec()
        windows = velocity_window_seconds(spec)
        assert set(windows) == {"5m", "1h", "24h", "7d"}
        assert windows["24h"] == 86_400

    def test_missing_key_rejected(self, tmp_path):
        broken = tmp_path / "features.yaml"
        broken.write_text("velocity:\n  windows: [1h]\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_feature_spec(broken)

    def test_configs_dir_exists(self):
        assert (REPO_ROOT / "configs" / "pipeline.yaml").exists()
