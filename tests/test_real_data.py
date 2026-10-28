"""Tests for Frente D -- converting real ns-O-RAN runs into offline datasets.

They build a *fake* run folder (``database.db`` + ``EsActions.txt``) that mimics
the schema produced by :class:`nsoran.datalake.SQLiteDatabaseAPI`, so no ns-3
binary is required.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from environments.es_env import CELL_IDS, STATE_DIM
from nsoran.datalake import SQLiteDatabaseAPI
from nsoran.drl.real_data import (
    dataset_from_ns3_run,
    dataset_from_ns3_runs,
    transitions_from_ns3_run,
)

CONFIG = {"simTime": [10], "ues": [3], "indicationPeriodicity": [0.1]}

# Four indications, one per 100 ms period.
TIMESTAMPS = [1_700_000_000_100, 1_700_000_000_200, 1_700_000_000_300, 1_700_000_000_400]


def _make_run(directory, *, off_at_second_call: bool = True):
    directory.mkdir(parents=True, exist_ok=True)
    # Create the KPI tables exactly as the datalake does (they stay empty: the
    # LTE run has no NR KPM columns, so the environment falls back to zeros).
    SQLiteDatabaseAPI(str(directory), 3)

    connection = sqlite3.connect(directory / "database.db")
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS bsState "
        "(timestamp INTEGER, ueImsiComplete INTEGER, cellId INTEGER, state INTEGER)"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS grafana "
        "(timestamp INTEGER, ueImsiComplete INTEGER, time_grafana INTEGER, step INTEGER,"
        " throughput REAL, en_cons REAL, rlf REAL, on_cost REAL, reward REAL)"
    )

    for index, timestamp in enumerate(TIMESTAMPS):
        off = off_at_second_call and index == 1
        for cell in CELL_IDS:
            state = 0 if off else 1
            cursor.execute(
                "INSERT INTO bsState VALUES (?, ?, ?, ?)",
                (timestamp, None, cell, state),
            )
        cursor.execute(
            "INSERT INTO grafana VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, None, timestamp, index + 1, 0.0, 0.0, 0.0, 0.0, -0.1 * index),
        )
    connection.commit()
    connection.close()

    # Control log: all cells ON except the second indication (all OFF).
    lines = ["timestamp,cellId,hoAllowed"]
    for index, timestamp in enumerate(TIMESTAMPS):
        allowed = 0 if (off_at_second_call and index == 1) else 1
        for cell in CELL_IDS:
            lines.append(f"{timestamp},{cell},{allowed}")
    (directory / "EsActions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def test_transitions_from_ns3_run_shapes(tmp_path):
    run_dir = _make_run(tmp_path / "run")
    transitions = transitions_from_ns3_run(run_dir, scenario_configuration=CONFIG)

    n = len(TIMESTAMPS)
    assert transitions["observations"].shape == (n, STATE_DIM)
    assert transitions["next_observations"].shape == (n, STATE_DIM)
    assert transitions["actions"].shape == (n,)
    assert transitions["reward_components"].shape == (n, 5)
    assert transitions["timestamps"].tolist() == TIMESTAMPS
    # Last transition is terminal; the others are not.
    assert transitions["terminals"].tolist() == [0] * (n - 1) + [1]


def test_action_decoding_round_trip(tmp_path):
    run_dir = _make_run(tmp_path / "run")
    transitions = transitions_from_ns3_run(run_dir, scenario_configuration=CONFIG)
    # All ON = 127 (bits 0..6 set); all OFF = 0.
    assert transitions["actions"].tolist() == [127, 0, 127, 127]


def test_dataset_from_ns3_run_contract(tmp_path):
    run_dir = _make_run(tmp_path / "run")
    dataset = dataset_from_ns3_run(run_dir, scenario_configuration=CONFIG)  # validates

    assert len(dataset) == len(TIMESTAMPS)
    assert dataset.num_episodes == 1
    assert dataset.reward_components is not None
    assert dataset.metadata["source"] == "ns3"
    assert dataset.metadata["num_transitions"] == len(TIMESTAMPS)
    # The OFF indication must be visible in the observation (cell_state / bs_on).
    assert dataset.observations[1].sum() < dataset.observations[0].sum()


def test_dataset_from_ns3_runs_concatenates_episodes(tmp_path):
    first = _make_run(tmp_path / "runA")
    second = _make_run(tmp_path / "runB")

    dataset = dataset_from_ns3_runs([first, second], scenario_configuration=CONFIG)

    assert len(dataset) == 2 * len(TIMESTAMPS)
    assert dataset.num_episodes == 2
    assert sorted(set(dataset.episode_ids.tolist())) == [0, 1]
    assert len(dataset.metadata["run_dirs"]) == 2


def test_missing_database_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        dataset_from_ns3_run(empty, scenario_configuration=CONFIG)
