"""Unit tests for schema validation and dataset inspection."""

import json
from pathlib import Path
import pytest
import yaml

from courtiq.data.inspector import inspect_dataset


def test_schema_config_structure():
    config_path = Path("configs/schema_config.yaml")
    assert config_path.exists(), "schema_config.yaml must exist in configs/"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "data_sources" in cfg
    assert "keys" in cfg
    assert "target" in cfg
    assert "filtering_rules" in cfg
    assert "features" in cfg

    # Leakage blacklist must contain critical post-shot fields
    blacklist = cfg["features"]["leakage_blacklist"]
    for required_blacklisted in ["outcome", "shotQuality", "fouled", "assisted", "blocked"]:
        assert required_blacklisted in blacklist


def test_inspect_dataset_synthetic(tmp_path):
    # Create mock match directory
    match_dir = tmp_path / "matches" / "999999"
    match_dir.mkdir(parents=True)

    # Dynamic events
    mock_events = {
        "shots": [
            {
                "id": "shot-999-1",
                "gameId": 999999,
                "shooterId": 101,
                "closestDefId": 202,
                "distance": 22.0,
                "outcome": True,
                "fouled": False,
                "contestLevel": "open",
                "three": True,
            },
            {
                "id": "shot-999-2",
                "gameId": 999999,
                "shooterId": 101,
                "closestDefId": 202,
                "distance": 5.0,
                "outcome": False,
                "fouled": True,
                "contestLevel": "plus",
                "three": False,
            },
        ]
    }
    with open(match_dir / "999999_dynamic_events.json", "w", encoding="utf-8") as f:
        json.dump(mock_events, f)

    # Game metadata
    mock_meta = {
        "home_team": {
            "name": "Team A",
            "players": [{"id": 101, "name": "Player A"}],
        },
        "away_team": {
            "name": "Team B",
            "players": [{"id": 202, "name": "Player B"}],
        },
    }
    with open(match_dir / "999999_game_data.json", "w", encoding="utf-8") as f:
        json.dump(mock_meta, f)

    audit = inspect_dataset(tmp_path)

    assert audit["status"] == "success"
    assert audit["row_counts"]["total_shots"] == 2
    assert audit["row_counts"]["total_games"] == 1
    assert audit["key_uniqueness"]["is_unique"] is True
    assert audit["join_coverage"]["shooter_id_join_rate"] == 1.0
    assert audit["join_coverage"]["closest_def_id_join_rate"] == 1.0
    assert audit["distributions"]["fouled_misses_count"] == 1
