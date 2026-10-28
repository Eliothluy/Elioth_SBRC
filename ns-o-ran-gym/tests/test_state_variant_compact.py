"""Unit tests for the E5 compact state variant (Frente E5).

Covers the ``state_variant`` flag of ``EnergySavingEnv`` and the ``compact``
layout registered in ``nsoran.drl.config``.  The audit document is
``docs/plano_tecnico/kpm_audit.md`` (§3, "Tabela de cortes"): ``compact``
must remove EXACTLY the 15 columns

* ``mac_pdu_64qam_cell_{2..8}``  (indices 9, 21, 33, 45, 57, 69, 81)
* ``prb_percentage_cell_{2..8}`` (indices 6, 18, 30, 42, 54, 66, 78)
* ``bs_on``                       (index 84)

and the default must remain the paper's 85-dim state.  These tests never
start ns-3 (fake datalake, ``setup_sim=False``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from environments.es_env import (
    CELL_IDS,
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_FEATURES_PER_CELL,
    EnergySavingEnv,
    build_observation,
    cell_kpis_from_dataframe,
)
from nsoran.drl.config import (
    CELL_KPM_NAMES,
    E5_COMPACT_DROPPED_GLOBAL,
    E5_COMPACT_DROPPED_KPMS,
    E5_COMPACT_KEEP_INDICES,
    E5_DROPPED_KPMS,
    STATE_DIM_COMPACT,
    STATE_DIM_COMPACT70,
    STATE_FEATURE_NAMES as DRL_STATE_FEATURE_NAMES,
    STATE_VARIANTS,
    select_state_features,
    state_feature_names_for_dim,
    state_feature_names_for_variant,
    state_variant_indices,
)

TIMESTAMP = 1_700_000_000_100

#: The 15 indices of the audit's "Tabela de cortes" (kpm_audit.md §3).
AUDIT_DROPPED_INDICES = (
    # prb_percentage (offset 6)
    6, 18, 30, 42, 54, 66, 78,
    # mac_pdu_64qam (offset 9)
    9, 21, 33, 45, 57, 69, 81,
    # global bs_on
    84,
)

#: The 10 per-cell KPMs the audit keeps, in observation order.
AUDIT_KEPT_KPMS_PER_CELL = (
    "throughput_dl",
    "energy",
    "ee_ratio",
    "rlf_count",
    "rlf_percentage",
    "prb_used",
    "tb_64qam",
    "tb_64qam_ratio",
    "es_on_cost",
    "cell_state",
)


# ---------------------------------------------------------------------------
# config-level: shape and columns of the registered variants
# ---------------------------------------------------------------------------


def test_compact_removes_exactly_the_audit_columns():
    assert STATE_DIM_COMPACT70 == 70
    assert len(E5_COMPACT_KEEP_INDICES) == 70
    assert set(E5_COMPACT_DROPPED_KPMS) == {"mac_pdu_64qam", "prb_percentage"}
    assert E5_COMPACT_DROPPED_GLOBAL == ("bs_on",)
    assert sorted(set(range(STATE_DIM)) - set(E5_COMPACT_KEEP_INDICES)) == sorted(
        AUDIT_DROPPED_INDICES
    )


def test_compact_columns_are_the_audit_kept_kpms_in_cell_major_order():
    expected = tuple(
        f"{kpi}_cell_{cell}" for cell in CELL_IDS for kpi in AUDIT_KEPT_KPMS_PER_CELL
    )
    assert state_feature_names_for_variant("compact") == expected
    assert len(expected) == 70

    names = set(expected)
    assert not any(name.startswith("mac_pdu_64qam") for name in names)
    assert not any(name.startswith("prb_percentage") for name in names)
    assert "bs_on" not in names
    # Every kept column is a real feature of the 85-dim paper state.
    assert names <= set(DRL_STATE_FEATURE_NAMES)


def test_paper85_is_identity_and_default_stays_85():
    assert STATE_DIM == 85
    # The module-level default of the selector is still the full paper state.
    assert state_variant_indices() == tuple(range(85))
    assert state_variant_indices("paper85") == tuple(range(85))
    assert state_variant_indices("full") == tuple(range(85))
    assert state_feature_names_for_variant("paper85") == DRL_STATE_FEATURE_NAMES
    assert STATE_VARIANTS["paper85"]["obs_dim"] == 85
    assert STATE_VARIANTS["paper85"]["dropped_kpms"] == ()
    # Both modules describe the same 12 per-cell KPMs in the same order
    # (their name suffixes differ by convention: es_env uses ``_{cell}``,
    # nsoran.drl.config uses ``_cell_{cell}``).
    assert tuple(STATE_FEATURES_PER_CELL) == CELL_KPM_NAMES


def test_compact78_is_kept_for_compatibility():
    assert STATE_DIM_COMPACT == 78
    assert STATE_VARIANTS["compact78"]["obs_dim"] == 78
    assert STATE_VARIANTS["compact78"]["dropped_kpms"] == E5_DROPPED_KPMS
    assert len(state_feature_names_for_variant("compact78")) == 78


def test_state_feature_names_for_dim_knows_all_widths():
    assert state_feature_names_for_dim(85) == DRL_STATE_FEATURE_NAMES
    assert state_feature_names_for_dim(70) == state_feature_names_for_variant("compact")
    assert state_feature_names_for_dim(78) == state_feature_names_for_variant("compact78")
    with pytest.raises(ValueError):
        state_feature_names_for_dim(71)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="state variant"):
        state_variant_indices("inexistente")


def test_select_state_features_matches_compact_indices():
    rng = np.random.default_rng(7)
    obs = rng.random((3, 85), dtype=np.float32)
    selected = select_state_features(obs, "compact")
    assert selected.shape == (3, 70)
    np.testing.assert_array_equal(selected, obs[..., list(E5_COMPACT_KEEP_INDICES)])
    single = select_state_features(obs[0], "compact")
    assert single.shape == (70,)
    np.testing.assert_array_equal(single, obs[0][list(E5_COMPACT_KEEP_INDICES)])


# ---------------------------------------------------------------------------
# env-level: fake datalake (no ns-3), default vs compact
# ---------------------------------------------------------------------------


def make_ue_dataframe() -> pd.DataFrame:
    """Two UE rows per cell with per-cell distinct counters."""
    rows = []
    for index, cell in enumerate(CELL_IDS):
        throughput = 10_000.0 * (index + 1)
        transport_blocks = 100.0 + index
        prb_used = 10.0 + index
        qpsk, q16, q64 = 10.0, 5.0, 20.0
        for ue_index in range(2):
            rows.append(
                {
                    "ueImsiComplete": f"ue-{cell}-{ue_index}",
                    "nrCellId": cell,
                    "QosFlow.PdcpPduVolumeDL_Filter": throughput,
                    "TB.TotNbrDl.1": transport_blocks,
                    "RRU.PrbUsedDl": prb_used,
                    "TB.TotNbrDlInitial.Qpsk": qpsk,
                    "TB.TotNbrDlInitial.16Qam": q16,
                    "TB.TotNbrDlInitial.64Qam": q64,
                    "L3 serving SINR": -10.0 if (cell == 3 and ue_index == 0) else 10.0,
                }
            )
    return pd.DataFrame(rows)


class FakeDatalake:
    """Minimal stand-in for SQLiteDatabaseAPI (same contract as the unit tests)."""

    def __init__(self, rows, bs_rows):
        self.rows = rows
        self.bs_rows = bs_rows
        self.inserted = []

    def read_kpms(self, timestamp, kpms):
        if int(timestamp) != TIMESTAMP:
            return None
        columns = [
            "ueImsiComplete",
            "nrCellId",
            "QosFlow.PdcpPduVolumeDL_Filter",
            "TB.TotNbrDl.1",
            "L3 serving SINR",
            "RRU.PrbUsedDl",
            "TB.TotNbrDlInitial.64Qam",
            "TB.TotNbrDlInitial.Qpsk",
            "TB.TotNbrDlInitial.16Qam",
        ]
        return [tuple(row[column] for column in columns) for row in self.rows]

    def read_table(self, table_name):
        assert table_name == "bsState"
        return self.bs_rows

    def insert_data(self, table_name, data):
        self.inserted.append((table_name, data))


def make_env(state_variant: str = "paper85") -> EnergySavingEnv:
    configuration = {
        "simTime": [10],
        "ues": [9],
        "indicationPeriodicity": [0.1],
        "RngRun": [555],
    }
    return EnergySavingEnv(
        ns3_path=".",
        scenario_configuration=configuration,
        output_folder="/tmp/opencode/es-compact-tests",
        optimized=False,
        setup_sim=False,
        state_variant=state_variant,
    )


def _fake_datalake(env: EnergySavingEnv, cell_off: int | None = None) -> None:
    env.datalake = FakeDatalake(
        make_ue_dataframe().to_dict("records"),
        [
            (TIMESTAMP, None, cell, 0 if cell == cell_off else 1)
            for cell in CELL_IDS
        ],
    )
    env.last_timestamp = TIMESTAMP


def test_default_env_keeps_the_paper85_contract():
    env = make_env()
    try:
        assert env.state_variant == "paper85"
        assert env.obs_dim == 85
        assert env.observation_space.shape == (85,)
        assert env.observation_space.dtype == np.float32
        assert env.columns_state == list(STATE_FEATURE_NAMES)
        assert env.state_keep_indices == tuple(range(85))
    finally:
        env.close()


def test_compact_env_shape_and_columns():
    env = make_env(state_variant="compact")
    try:
        assert env.state_variant == "compact"
        assert env.obs_dim == 70
        assert env.observation_space.shape == (70,)
        assert env.observation_space.dtype == np.float32
        # env columns keep es_env's ``_{cell}`` convention, selected by the
        # audit's keep indices.
        assert env.columns_state == [
            STATE_FEATURE_NAMES[index] for index in E5_COMPACT_KEEP_INDICES
        ]
        assert env.columns_state == [
            f"{kpi}_{cell}" for cell in CELL_IDS for kpi in AUDIT_KEPT_KPMS_PER_CELL
        ]
        assert env.state_keep_indices == E5_COMPACT_KEEP_INDICES
        # Action space and reward contract are variant-independent.
        assert env.action_space.n == 128
    finally:
        env.close()


def test_compact_env_rejects_unknown_variant():
    with pytest.raises(ValueError, match="state variant"):
        make_env(state_variant="compact85")


def test_compact_observation_is_the_exact_selection_of_the_full_one():
    env_full = make_env()
    env_compact = make_env(state_variant="compact")
    try:
        _fake_datalake(env_full, cell_off=4)
        _fake_datalake(env_compact, cell_off=4)

        obs_full = env_full._get_obs()
        obs_compact = env_compact._get_obs()

        assert obs_full.shape == (85,)
        assert obs_compact.shape == (70,)
        assert obs_compact.dtype == np.float32
        assert np.all((obs_compact >= 0.0) & (obs_compact <= 1.0))
        # Same normalisation path, then the audit's column selection.
        np.testing.assert_array_equal(obs_compact, obs_full[list(E5_COMPACT_KEEP_INDICES)])
    finally:
        env_full.close()
        env_compact.close()


def test_compact_keeps_full_raw_observation_and_reward():
    env_full = make_env()
    env_compact = make_env(state_variant="compact")
    try:
        _fake_datalake(env_full, cell_off=4)
        _fake_datalake(env_compact, cell_off=4)

        env_full._get_obs()
        env_compact._get_obs()

        # The raw KPM audit trail stays 85-dim in both variants...
        assert len(env_full._last_raw["raw_observation"]) == 85
        assert len(env_compact._last_raw["raw_observation"]) == 85
        # ...and the reward (Eq. 2 on raw totals) is unchanged by the cut.
        assert env_full._compute_reward() == pytest.approx(env_compact._compute_reward())
        assert env_compact._last_raw["bs_on"] == 6
        assert env_compact.last_reward_components["normalized"].keys() == {
            "throughput", "energy", "rlf", "es_on_cost", "bs_on"
        }
    finally:
        env_full.close()
        env_compact.close()


def test_compact_reset_returns_compact_shape():
    env = make_env(state_variant="compact")
    try:
        env.sim_result = {}
        env.start_sim = lambda: setattr(env, "sim_result", {})
        env._wait_data_availability = lambda *args, **kwargs: True
        env._fill_datalake = lambda: None
        env._get_obs = lambda: np.zeros(env.obs_dim, dtype=np.float32)
        env._get_info = lambda: {}

        observation, _ = env.reset(seed=1234)
        assert observation.shape == (70,)
        assert env._last_obs.shape == (70,)
        assert env.scenario_configuration["RngRun"] == 1234
    finally:
        env.close()


# ---------------------------------------------------------------------------
# identities of the audit's redundant groups, on the fabricated frame
# ---------------------------------------------------------------------------


def test_group_a_mac_pdu_is_exact_copy_of_tb_64qam():
    cell_df = cell_kpis_from_dataframe(make_ue_dataframe(), timestamp=TIMESTAMP)
    np.testing.assert_array_equal(
        cell_df["mac_pdu_64qam"].to_numpy(), cell_df["tb_64qam"].to_numpy()
    )


def test_group_c_bs_on_is_sum_of_cell_states():
    states = {cell: (0 if cell == 4 else 1) for cell in CELL_IDS}
    cell_df = cell_kpis_from_dataframe(make_ue_dataframe(), timestamp=TIMESTAMP)
    cell_df["cell_state"] = [float(states[cell]) for cell in CELL_IDS]
    raw = build_observation(cell_df, states)

    assert raw[84] == 6.0  # global bs_on
    cell_states_in_obs = np.array([raw[11 + 12 * i] for i in range(7)])
    assert raw[84] == pytest.approx(cell_states_in_obs.sum())
