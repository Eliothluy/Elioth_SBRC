"""Unit tests for the paper energy-saving environment.

These tests never start ns-3.  They exercise the pure helpers exported by
``environments.es_env`` plus a fake datalake for the observation/reward glue.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environments.es_env import (
    CELL_IDS,
    DEFAULT_POWER_MODEL,
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_FEATURES_PER_CELL,
    EnergySavingEnv,
    MinMaxNormalizer,
    QuantileNormalizer,
    RewardNormalizer,
    action_to_cell_bits,
    build_observation,
    cell_bits_to_action,
    cell_kpis_from_dataframe,
    compute_reward,
    default_state_ranges,
    max_cell_power_w,
    normalize_action,
)

TIMESTAMP = 1_700_000_000_100


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------


def test_action_zero_is_all_off():
    assert action_to_cell_bits(0) == [(cell, 0) for cell in CELL_IDS]


def test_action_127_is_all_on():
    assert action_to_cell_bits(127) == [(cell, 1) for cell in CELL_IDS]


def test_action_bit_order_lsb_is_cell_2():
    assert action_to_cell_bits(1) == [(2, 1)] + [(cell, 0) for cell in CELL_IDS[1:]]
    assert action_to_cell_bits(2) == [(2, 0), (3, 1)] + [(cell, 0) for cell in CELL_IDS[2:]]
    # Bit 6 (value 64) is cell 8.
    assert action_to_cell_bits(64) == [(cell, 0) for cell in CELL_IDS[:-1]] + [(8, 1)]


def test_action_roundtrip_and_validation():
    bits = [1, 0, 1, 1, 0, 0, 1]
    assert cell_bits_to_action(bits) == 0b1001101
    assert [bit for _, bit in action_to_cell_bits(cell_bits_to_action(bits))] == bits

    with pytest.raises(ValueError):
        action_to_cell_bits(-1)
    with pytest.raises(ValueError):
        action_to_cell_bits(128)
    with pytest.raises(ValueError):
        cell_bits_to_action([1, 0])


def test_normalize_action_accepts_sequence_for_example_compat():
    assert normalize_action([1, 0, 1, 0, 0, 1, 1]) == [
        (2, 1), (3, 0), (4, 1), (5, 0), (6, 0), (7, 1), (8, 1)
    ]


# ---------------------------------------------------------------------------
# Fabricated datalake frame / state vector
# ---------------------------------------------------------------------------


def make_ue_dataframe() -> pd.DataFrame:
    """Two UE rows per cell; cell-level counters are repeated on each row."""
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


def test_cell_kpis_values():
    df = make_ue_dataframe()
    cell_df = cell_kpis_from_dataframe(df, timestamp=TIMESTAMP)

    assert list(cell_df.index) == list(CELL_IDS)
    # Cell 2 is the first cell: throughput 10k, 100 TB, 10 PRB.
    assert cell_df.loc[2, "throughput_dl"] == pytest.approx(10_000.0)
    assert cell_df.loc[2, "energy"] == pytest.approx(100.0)
    assert cell_df.loc[2, "ee_ratio"] == pytest.approx(100.0)
    assert cell_df.loc[2, "prb_percentage"] == pytest.approx(10.0 / 139.0 * 100.0)
    assert cell_df.loc[2, "tb_64qam_ratio"] == pytest.approx(20.0 / 35.0)
    assert cell_df.loc[2, "cell_state"] == pytest.approx(1.0)

    # Cell 3 has one UE with SINR < -5 dB.
    assert cell_df.loc[3, "rlf_count"] == pytest.approx(1.0)
    assert cell_df.loc[3, "rlf_percentage"] == pytest.approx(50.0)
    assert cell_df.loc[2, "rlf_count"] == pytest.approx(0.0)


def test_empty_dataframe_is_safe():
    cell_df = cell_kpis_from_dataframe(None)
    assert list(cell_df.index) == list(CELL_IDS)
    assert np.all(cell_df[list(STATE_FEATURES_PER_CELL)].to_numpy() == 0.0)


def test_observation_length_and_order():
    cell_df = cell_kpis_from_dataframe(make_ue_dataframe())
    cell_df["es_on_cost"] = 0.5
    states = {cell: 1 for cell in CELL_IDS}
    states[3] = 0

    observation = build_observation(cell_df, states)

    assert observation.shape == (STATE_DIM,)
    assert observation.dtype == np.float32
    assert len(STATE_FEATURE_NAMES) == STATE_DIM

    # Cell-major layout: 12 values per cell, then bs_on.
    assert STATE_FEATURE_NAMES[0] == "throughput_dl_2"
    assert STATE_FEATURE_NAMES[11] == "cell_state_2"
    assert STATE_FEATURE_NAMES[12] == "throughput_dl_3"
    assert STATE_FEATURE_NAMES[-1] == "bs_on"

    def value(name: str) -> float:
        return float(observation[STATE_FEATURE_NAMES.index(name)])

    assert value("throughput_dl_2") == pytest.approx(10_000.0)
    assert value("cell_state_2") == pytest.approx(1.0)
    assert value("cell_state_3") == pytest.approx(0.0)
    assert value("rlf_count_3") == pytest.approx(1.0)
    assert value("rlf_percentage_3") == pytest.approx(50.0)
    assert value("es_on_cost_8") == pytest.approx(0.5)
    assert value("bs_on") == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["uniform", "normal"])
def test_quantile_normalizer_fit_transform_inverse(method):
    references = np.linspace(-5.0, 5.0, 101)
    normalizer = QuantileNormalizer(method=method).fit(references)

    low = normalizer.transform(-5.0)
    high = normalizer.transform(5.0)
    middle = normalizer.transform(0.0)

    assert 0.0 <= low < 0.02
    assert 0.98 < high <= 1.0
    assert 0.4 < middle < 0.6

    transformed = normalizer.transform([-5.0, 0.0, 5.0])
    inverse = normalizer.inverse_transform(transformed)
    assert np.allclose(inverse, [-5.0, 0.0, 5.0], atol=0.2)

    assert np.all(np.diff(normalizer.transform(np.linspace(-5, 5, 50))) >= -1e-12)


def test_quantile_normalizer_duplicate_mass_is_monotone():
    """Regression: ties must use the mid-rank of the *sorted* sample.

    The EARTH energy model puts every sleeping cell at exactly 75 W, so the
    reference is dominated by a point mass.  Averaging the position-based
    ranks of the original order (the old implementation) put 75 W at 0.55,
    above 130 W, and made the reward non-monotone in the energy.
    """

    values = np.concatenate(
        [
            np.full(50, 75.0),
            np.linspace(130.0, 143.0, 100),
            np.full(10, 224.0),
        ]
    )
    normalizer = QuantileNormalizer(method="uniform").fit(values)
    transformed = normalizer.transform([75.0, 130.0, 140.0, 224.0])

    assert transformed[0] < transformed[1] < transformed[2] < transformed[3]
    # The 50 ties at 75 W occupy the first 50 sorted ranks out of 160.
    assert transformed[0] == pytest.approx(25.0 / 160.0, abs=1e-6)

    grid = np.linspace(75.0, 224.0, 200)
    assert np.all(np.diff(normalizer.transform(grid)) >= -1e-12)
    assert np.all(np.diff(normalizer.inverse_transform(
        normalizer.transform(grid)
    )) >= -1e-6)


def test_quantile_normalizer_calibration_roundtrip(tmp_path):
    normalizer = QuantileNormalizer(method="normal").fit(
        np.concatenate([np.zeros(20), np.linspace(0.0, 1.0, 20)])
    )
    path = tmp_path / "component.npz"
    normalizer.save(str(path))

    loaded = QuantileNormalizer.load(str(path))
    assert loaded.method == "normal"
    values = np.linspace(-0.1, 1.1, 30)
    assert np.allclose(loaded.transform(values), normalizer.transform(values))


def test_reward_normalizer_uses_calibration_file(tmp_path):
    calibration = tmp_path / "reward_calibration.npz"
    np.savez(
        calibration,
        throughput=np.array([0.0, 5.0, 10.0]),
        energy=np.array([0.0, 50.0, 100.0]),
        rlf=np.array([0.0, 5.0, 10.0]),
        es_on_cost=np.array([0.0, 0.5, 1.0]),
        bs_on=np.array([0.0, 3.5, 7.0]),
    )
    normalizer = RewardNormalizer("uniform", str(calibration))

    normalized = normalizer.transform(
        {"throughput": 10.0, "energy": 100.0, "rlf": 10.0, "es_on_cost": 1.0, "bs_on": 7.0}
    )
    assert all(0.8 < value <= 1.0 for value in normalized.values())

    inverse = normalizer.inverse_transform(normalized)
    assert inverse["throughput"] == pytest.approx(10.0, abs=1e-6)
    assert inverse["bs_on"] == pytest.approx(7.0, abs=1e-6)


def test_state_min_max_normalization():
    normalizer = MinMaxNormalizer(
        default_state_ranges(), feature_names=STATE_FEATURE_NAMES
    )
    zeros = np.zeros(STATE_DIM)
    assert np.allclose(normalizer.transform(zeros), 0.0)

    maxima = np.array([default_state_ranges()[name][1] for name in STATE_FEATURE_NAMES])
    assert np.allclose(normalizer.transform(maxima), 1.0)

    over_max = maxima * 10.0
    assert np.allclose(normalizer.transform(over_max), 1.0)

    # fit() learns the bounds from data and inverse_transform round-trips.
    data = np.vstack([zeros, maxima])
    fitted = MinMaxNormalizer(
        default_state_ranges(), feature_names=STATE_FEATURE_NAMES
    ).fit(data)
    transformed = fitted.transform(maxima)
    assert np.allclose(fitted.inverse_transform(transformed), maxima)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class _IdentityNormalizer:
    def transform(self, raw):
        return {key: float(value) for key, value in raw.items()}


def test_reward_formula_with_weights():
    raw = {
        "throughput": 1.0,
        "energy": 1.0,
        "rlf": 1.0,
        "es_on_cost": 1.0,
        "bs_on": 1.0,
    }
    weights = {"w1": 0.4, "w2": 0.4, "w3": 0.1, "w4": 0.1}
    reward, normalized, returned_weights = compute_reward(
        raw, weights, _IdentityNormalizer()
    )

    expected = 0.4 - 0.4 - 0.1 - 0.1 - 0.4
    assert reward == pytest.approx(expected)
    assert normalized == pytest.approx({key: 1.0 for key in raw})
    assert returned_weights == weights


def test_reward_uses_quantile_normalization():
    normalizer = RewardNormalizer("uniform")
    raw = {
        "throughput": 0.0,
        "energy": 0.0,
        "rlf": 0.0,
        "es_on_cost": 0.0,
        "bs_on": 0.0,
    }
    weights = {"w1": 0.51, "w2": 0.19, "w3": 0.2, "w4": 0.1}
    reward, normalized, _ = compute_reward(raw, weights, normalizer)

    expected = (
        weights["w1"] * normalized["throughput"]
        - weights["w2"] * normalized["energy"]
        - weights["w3"] * normalized["rlf"]
        - weights["w4"] * normalized["es_on_cost"]
        - weights["w2"] * normalized["bs_on"]
    )
    assert reward == pytest.approx(expected)
    assert all(0.0 <= value <= 1.0 for value in normalized.values())


# ---------------------------------------------------------------------------
# Environment glue (no ns-3)
# ---------------------------------------------------------------------------


def make_env(**kwargs) -> EnergySavingEnv:
    configuration = {
        "simTime": [10],
        "ues": [9],
        "indicationPeriodicity": [0.1],
        "RngRun": [555],
    }
    return EnergySavingEnv(
        ns3_path=".",
        scenario_configuration=configuration,
        output_folder="/tmp/opencode/es-env-tests",
        optimized=False,
        setup_sim=False,
        **kwargs,
    )


def test_env_spaces_and_action_conversion():
    env = make_env()
    try:
        assert env.action_space.n == 128
        assert env.observation_space.shape == (STATE_DIM,)
        assert env.observation_space.dtype == np.float32

        assert env._compute_action(0) == [[cell, 0] for cell in CELL_IDS]
        assert env._compute_action(127) == [[cell, 1] for cell in CELL_IDS]
        assert env._compute_action([1, 0, 1, 0, 1, 0, 1]) == [
            [2, 1], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0], [8, 1]
        ]
    finally:
        env.close()


class FakeDatalake:
    """Minimal stand-in for SQLiteDatabaseAPI."""

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


def test_env_get_obs_and_reward_with_fake_datalake():
    df = make_ue_dataframe()
    env = make_env()
    env.datalake = FakeDatalake(df.to_dict("records"), [
        (TIMESTAMP, None, cell, 0 if cell == 4 else 1) for cell in CELL_IDS
    ])
    env.last_timestamp = TIMESTAMP

    observation = env._get_obs()
    assert observation.shape == (STATE_DIM,)
    assert observation.dtype == np.float32
    assert np.all((observation >= 0.0) & (observation <= 1.0))

    assert env._last_raw["timestamps"]["kpm"] == TIMESTAMP
    assert env._last_raw["timestamps"]["bsstate_timestamp_unix"] == TIMESTAMP
    assert env._last_raw["bs_on"] == 6
    assert env._last_raw["cell_states"]["4"] == 0

    reward = env._compute_reward()
    assert isinstance(reward, float)
    assert set(env.last_reward_components["normalized"]) == {
        "throughput", "energy", "rlf", "es_on_cost", "bs_on"
    }
    assert env.datalake.inserted[0][0] == "grafana"
    env.close()


def test_reset_maps_seed_to_rng_run():
    env = make_env()
    try:
        env.sim_result = {}
        env.start_sim = lambda: setattr(env, "sim_result", {})
        env._wait_data_availability = lambda *args, **kwargs: True
        env._fill_datalake = lambda: None
        env._get_obs = lambda: np.zeros(STATE_DIM, dtype=np.float32)
        env._get_info = lambda: {}

        observation, info = env.reset(seed=1234)
        assert observation.shape == (STATE_DIM,)
        assert info == {}
        assert env.scenario_configuration["RngRun"] == 1234

        env.reset()
        assert env.scenario_configuration["RngRun"] == 1234  # unchanged without seed
    finally:
        env.close()


def test_paper_configuration_file():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "environments"
        / "scenario_configurations"
        / "paper_es_use_case.json"
    )
    with open(path, "r", encoding="utf-8") as file:
        configuration = json.load(file)

    expected = {
        "configuration": [3],
        "trafficModel": [4],
        "ues": [9],
        "simTime": [10],
        "indicationPeriodicity": [0.1],
        "heuristicType": [-1],
        "controlFileName": ["es_actions_for_ns3.csv"],
        "useSemaphores": [1],
        "handoverMode": ["DynamicTtt"],
        "outageThreshold": [-5],
        "numberOfRaPreambles": [40],
        "RngRun": [555],
    }
    for key, value in expected.items():
        assert configuration[key] == value


# ---------------------------------------------------------------------------
# D1: EARTH watts energy model
# ---------------------------------------------------------------------------


def test_watts_energy_model_uses_earth_parameters():
    df = make_ue_dataframe()
    states = {cell: 1 for cell in CELL_IDS}
    states[3] = 0
    cell_df = cell_kpis_from_dataframe(
        df,
        timestamp=TIMESTAMP,
        energy_model="watts",
        cell_states=states,
    )
    # Cell 2 uses 10 PRBs of 139: P = 130 + 4.7*20*(10/139).
    expected_on = 130.0 + 4.7 * 20.0 * (10.0 / 139.0)
    assert cell_df.loc[2, "energy"] == pytest.approx(expected_on)
    # Cell 3 is OFF: it still draws the EARTH sleep power.
    assert cell_df.loc[3, "energy"] == pytest.approx(75.0)
    # The energy-efficiency KPI is throughput / watts.
    assert cell_df.loc[2, "ee_ratio"] == pytest.approx(10_000.0 / expected_on)


def test_watts_energy_empty_frame_keeps_power_state():
    cell_df = cell_kpis_from_dataframe(
        None, energy_model="watts", cell_states={2: 1, 3: 0}
    )
    assert cell_df.loc[2, "energy"] == pytest.approx(130.0)
    assert cell_df.loc[3, "energy"] == pytest.approx(75.0)


def test_watts_energy_clips_prb_load_and_uses_max_power():
    assert max_cell_power_w() == pytest.approx(130.0 + 4.7 * 20.0)
    assert DEFAULT_POWER_MODEL["p_sleep_w"] == pytest.approx(75.0)
    ranges = default_state_ranges(energy_max=max_cell_power_w())
    assert ranges["energy_2"] == (0.0, pytest.approx(224.0))


def test_power_model_validation():
    with pytest.raises(ValueError):
        make_env(energy_model="not-a-model")
    with pytest.raises(ValueError):
        make_env(power_model={"p_sleep_w": 999.0})
    with pytest.raises(ValueError):
        make_env(es_cost_mode="not-a-mode")
    with pytest.raises(ValueError):
        make_env(per_cell_aggregation="not-a-mode")


# ---------------------------------------------------------------------------
# D2: dense vs impulse activation cost
# ---------------------------------------------------------------------------


def test_impulse_es_cost_only_on_off_to_on_transition():
    env = make_env(es_cost_mode="impulse")
    try:
        empty = pd.DataFrame(index=list(CELL_IDS))
        on = {cell: 1 for cell in CELL_IDS}
        off = dict(on)
        off[4] = 0

        env._update_es_on_cost(empty.copy(), on, 1000)
        frame = env._update_es_on_cost(empty.copy(), off, 1100)
        assert frame.loc[4, "es_on_cost"] == pytest.approx(0.0)
        frame = env._update_es_on_cost(empty.copy(), off, 2100)
        assert frame.loc[4, "es_on_cost"] == pytest.approx(0.0)
        # Cell 4 was OFF from t=1100 to t=3100: 2000 ms -> 0.9**20.
        frame = env._update_es_on_cost(empty.copy(), on, 3100)
        assert frame.loc[4, "es_on_cost"] == pytest.approx(0.9 ** 20)
        assert frame.loc[2, "es_on_cost"] == pytest.approx(0.0)
        frame = env._update_es_on_cost(empty.copy(), on, 3200)
        assert frame.loc[4, "es_on_cost"] == pytest.approx(0.0)
    finally:
        env.close()


def test_dense_es_cost_charges_every_on_step():
    env = make_env(es_cost_mode="dense")
    try:
        empty = pd.DataFrame(index=list(CELL_IDS))
        on = {cell: 1 for cell in CELL_IDS}
        first = env._update_es_on_cost(empty.copy(), on, 1000)
        second = env._update_es_on_cost(empty.copy(), on, 1100)
        third = env._update_es_on_cost(empty.copy(), on, 1200)
        assert first.loc[2, "es_on_cost"] == pytest.approx(0.9)
        # The ON clock starts at the first observation, so the second step is
        # still 100 ms and the third is 200 ms.
        assert second.loc[2, "es_on_cost"] == pytest.approx(0.9)
        assert third.loc[2, "es_on_cost"] == pytest.approx(0.9 ** 2)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# D3: per-cell normalisation
# ---------------------------------------------------------------------------


def test_per_cell_reward_averages_normalised_cells():
    normalizer = RewardNormalizer(method="uniform", per_cell=True)
    for cell in CELL_IDS:
        normalizer.cell_normalizers[("throughput", cell)] = QuantileNormalizer(
            "uniform"
        ).fit(np.linspace(0.0, 1.0, 101))
    per_cell = {
        "throughput": [(cell, 0.5) for cell in CELL_IDS],
        "energy": [(cell, 100.0) for cell in CELL_IDS],
        "rlf": [(cell, 0.0) for cell in CELL_IDS],
        "es_on_cost": [(cell, 0.5) for cell in CELL_IDS],
    }
    raw = {
        "throughput": 3.5,  # 7 * 0.5: global normalisation would give 0.35
        "energy": 700.0,
        "rlf": 0.0,
        "es_on_cost": 3.5,
        "bs_on": 7.0,
    }
    weights = {"w1": 0.51, "w2": 0.19, "w3": 0.20, "w4": 0.10}
    _, normalized, _ = compute_reward(
        raw, weights, normalizer, per_cell=per_cell, aggregation="mean"
    )
    assert normalized["throughput"] == pytest.approx(0.5)
    assert normalized["es_on_cost"] == pytest.approx(0.5)
    # bs_on stays global (fallback range 0..7); the mid-rank grid maps the
    # endpoint to 100.5/101, not exactly 1.
    assert normalized["bs_on"] == pytest.approx(100.5 / 101.0)


def test_per_cell_fallback_keeps_unit_scale():
    normalizer = RewardNormalizer(method="uniform", per_cell=True)
    # The fallback per-cell throughput range is 1e7 / 7; the top mid-rank
    # percentile is 100.5/101.
    assert normalizer.transform_cell("throughput", 2, 1.0e7 / 7.0) == pytest.approx(
        100.5 / 101.0
    )
    assert normalizer.transform_cell("throughput", 2, 0.0) == pytest.approx(0.5 / 101.0)


# ---------------------------------------------------------------------------
# E1 / CMDP: modos de restrição no ambiente (cmdp_design.md §6.1)
# ---------------------------------------------------------------------------


def _fake_env_with_data(**kwargs) -> EnergySavingEnv:
    """Env com datalake falso e observação já preenchida (sem ns-3)."""
    df = make_ue_dataframe()
    env = make_env(**kwargs)
    env.datalake = FakeDatalake(
        df.to_dict("records"),
        [(TIMESTAMP, None, cell, 0 if cell == 4 else 1) for cell in CELL_IDS],
    )
    env.last_timestamp = TIMESTAMP
    env._get_obs()
    return env


_PPO2_WEIGHTS = {"w1": 0.40, "w2": 0.40, "w3": 0.10, "w4": 0.10}


def test_env_cmdp_default_is_disabled_and_eq2_unchanged():
    env = _fake_env_with_data()
    try:
        assert env.cmdp_config.constraint_mode == "none"
        assert not env.cmdp_config.enabled
        assert len(env._throughput_window) == 0
        info = env._build_info_dict()
        assert info["env_config"]["cmdp"]["constraint_mode"] == "none"

        reward = env._compute_reward()
        # Nenhuma chave nova em reward_components no modo default.
        assert set(env.last_reward_components) == {
            "raw", "normalized", "weights", "reward"
        }
        assert env.last_reward_components["reward"] == pytest.approx(reward)
    finally:
        env.close()


def test_env_rejects_barrier_without_tau_sla():
    from nsoran.drl.config import CmdpConfig

    with pytest.raises(ValueError):
        make_env(constraint_mode="barrier")
    # "gate" é apelido de "barrier": mesma validação.
    with pytest.raises(ValueError):
        make_env(constraint_mode="gate")
    with pytest.raises(ValueError):
        make_env(constraint_mode="lagrangian")  # exige τ e/ou ε
    assert CmdpConfig(constraint_mode="barrier", tau_sla=1.0) is not None


def test_env_barrier_mode_gates_energy_channels():
    from nsoran.drl.config import compute_gated_reward, get_agent_config

    base = _fake_env_with_data(**_PPO2_WEIGHTS)
    raw_throughput = float(base._last_raw["totals"]["throughput"])
    base.close()

    # tau abaixo da janela => porta aberta (== Eq. 2); tau acima => fechada.
    cases = [
        (raw_throughput - 1.0, True),
        (raw_throughput + 1.0, False),
    ]
    for tau_sla, expected_open in cases:
        env = _fake_env_with_data(
            constraint_mode="barrier", tau_sla=tau_sla, gate_window=10,
            **_PPO2_WEIGHTS,
        )
        try:
            reward = env._compute_reward()
            components = env.last_reward_components
            assert env._throughput_window[-1] == pytest.approx(raw_throughput)
            assert components["throughput_window_mean"] == pytest.approx(
                raw_throughput
            )
            assert components["gate_open"] is expected_open
            expected_gate = 1.0 if expected_open else 0.0
            assert components["gate_value"] == pytest.approx(expected_gate)
            normalized = components["normalized"]
            expected = compute_gated_reward(
                normalized["throughput"],
                normalized["energy"],
                normalized["rlf"],
                normalized["es_on_cost"],
                normalized["bs_on"],
                get_agent_config("PPO-2"),
                expected_gate,
            )
            assert reward == pytest.approx(expected)
            if expected_open:
                # Porta aberta == Eq. (2) exata (mesmos pesos do env).
                weights = components["weights"]
                eq2 = (
                    weights["w1"] * normalized["throughput"]
                    - weights["w2"] * normalized["energy"]
                    - weights["w3"] * normalized["rlf"]
                    - weights["w4"] * normalized["es_on_cost"]
                    - weights["w2"] * normalized["bs_on"]
                )
                assert reward == pytest.approx(eq2)
        finally:
            env.close()


def test_env_barrier_gate_uses_window_mean_and_resets():
    env = _fake_env_with_data(
        constraint_mode="barrier",
        tau_sla=1.0e18,  # nenhum throughput real alcança: porta fechada
        gate_window=2,
        **_PPO2_WEIGHTS,
    )
    try:
        raw_throughput = float(env._last_raw["totals"]["throughput"])
        env._throughput_window.append(0.0)  # indicação anterior sem tráfego
        env._compute_reward()
        components = env.last_reward_components
        # A janela (2 indicações) média o tráfego atual com o zero anterior.
        assert components["throughput_window_mean"] == pytest.approx(
            raw_throughput / 2.0
        )
        assert components["gate_open"] is False
        assert components["gate_value"] == pytest.approx(0.0)

        # reset() esvazia a janela do episódio (design §6.1).
        env.sim_result = {}
        env.start_sim = lambda: setattr(env, "sim_result", {})
        env._wait_data_availability = lambda *args, **kwargs: True
        env._fill_datalake = lambda: None
        env._get_obs = lambda: np.zeros(STATE_DIM, dtype=np.float32)
        env._get_info = lambda: {}
        env.reset(seed=99)
        assert len(env._throughput_window) == 0
    finally:
        env.close()


def test_env_barrier_soft_gate_matches_config():
    from nsoran.drl.config import CmdpConfig

    env = _fake_env_with_data(
        constraint_mode="barrier",
        tau_sla=53_680.0,
        gate_temperature=3_834.0,  # ≈ 0.05·μ_AO (design §8.2-B1)
        **_PPO2_WEIGHTS,
    )
    try:
        raw_throughput = float(env._last_raw["totals"]["throughput"])
        env._compute_reward()
        expected = CmdpConfig(
            constraint_mode="barrier", tau_sla=53_680.0, gate_temperature=3_834.0
        ).gate_factor(raw_throughput)
        components = env.last_reward_components
        assert components["gate_value"] == pytest.approx(expected)
        assert components["gate_open"] is (expected >= 0.5)
    finally:
        env.close()


def test_env_lagrangian_mode_records_cost_without_changing_reward():
    default_env = _fake_env_with_data(**_PPO2_WEIGHTS)
    default_reward = default_env._compute_reward()
    default_keys = set(default_env.last_reward_components)
    default_env.close()
    assert default_keys == {"raw", "normalized", "weights", "reward"}

    env = _fake_env_with_data(
        constraint_mode="lagrangian", tau_sla=1.0e9, epsilon=0.0,
        **_PPO2_WEIGHTS,
    )
    try:
        reward = env._compute_reward()
        components = env.last_reward_components
        # A restrição vive no treino (design §6.1-a): a recompensa coletada
        # no env não muda; só o custo hinge é registrado para diagnóstico.
        assert reward == pytest.approx(default_reward)
        assert set(components) == default_keys | {"qos_cost"}
        raw = components["raw"]
        expected_cost = (
            (1.0e9 - raw["throughput"]) / 1.0e9  # S_rho = max(tau, 1) = 1e9
            + raw["rlf"] / 1.0  # S_varsigma = max(epsilon, 1) = 1
        )
        assert components["qos_cost"] == pytest.approx(expected_cost)
        # O info carrega a configuração do CMDP para rastreabilidade.
        info = env._build_info_dict()
        assert info["env_config"]["cmdp"]["constraint_mode"] == "lagrangian"
        assert info["env_config"]["cmdp"]["tau_sla"] == pytest.approx(1.0e9)
        assert info["env_config"]["cmdp"]["epsilon"] == pytest.approx(0.0)
    finally:
        env.close()
