"""Shape, contract and smoke tests for Frente C (DRL package).

Tests are written so that they degrade gracefully:

* configuration / lazy-import tests need no optional dependency at all;
* dataset and replay-environment tests require numpy + gymnasium;
* network and CQL tests require torch.

Run with::

    pytest tests/test_drl_shapes.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Dependency-free tests
# ---------------------------------------------------------------------------


def test_lazy_import_works_and_reports_dependencies():
    import nsoran.drl as drl

    assert callable(drl.dependency_status)
    status = drl.dependency_status()
    for name in drl.OPTIONAL_DEPENDENCIES:
        assert name in status
        assert isinstance(status[name], bool)
    # Accessing config symbols must work with the standard library alone.
    assert drl.STATE_DIM == 85
    assert len(drl.PAPER_AGENTS) == 6


def test_unknown_attribute_raises_attribute_error():
    import nsoran.drl as drl

    with pytest.raises(AttributeError):
        _ = drl.does_not_exist


def test_table_i_configs_match_the_paper():
    from nsoran.drl.config import PAPER_AGENTS, compute_reward, get_agent_config

    expected = {
        "PPO-1": (0.51, 0.19, 0.20, 0.10, "normal"),
        "PPO-2": (0.40, 0.40, 0.10, 0.10, "uniform"),
        "PPO-3": (0.20, 0.40, 0.20, 0.20, "uniform"),
        "PPO-4": (0.40, 0.32, 0.18, 0.10, "uniform"),
        "PPO-5": (0.45, 0.20, 0.25, 0.10, "normal"),
        "DQN": (0.40, 0.40, 0.10, 0.10, "uniform"),
    }
    assert set(PAPER_AGENTS) == set(expected)
    for name, (w1, w2, w3, w4, normalization) in expected.items():
        config = get_agent_config(name)
        assert (config.w1, config.w2, config.w3, config.w4) == (w1, w2, w3, w4)
        assert config.normalization == normalization
        assert abs(sum(config.weights.values()) - 1.0) < 1e-9
    assert get_agent_config("dqn").algo == "rem_dqn"
    assert get_agent_config("PPO-1").algo == "ppo"

    # Eq. (2) with unit components.
    config = get_agent_config("PPO-1")
    value = compute_reward(1.0, 1.0, 1.0, 1.0, 1.0, config)
    assert value == pytest.approx(0.51 - 0.19 - 0.20 - 0.10 - 0.19)


def test_entropy_coefficients_are_inside_the_paper_range():
    from nsoran.drl.config import PAPER_AGENTS

    for name, config in PAPER_AGENTS.items():
        if config.ent_coef is not None:
            assert 0.001 <= config.ent_coef <= 0.003, name


def test_action_encoding_matches_the_contract():
    from nsoran.drl.config import (
        action_to_bitstring,
        action_to_cells,
        cells_to_action,
    )

    assert action_to_cells(127) == {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    assert set(action_to_cells(0).values()) == {0}
    assert action_to_cells(1)[2] == 1
    assert action_to_cells(64)[8] == 1
    assert cells_to_action(action_to_cells(42)) == 42
    assert action_to_bitstring(0) == "0000000"
    assert action_to_bitstring(127) == "1111111"


# ---------------------------------------------------------------------------
# Dataset / environment tests (numpy + gymnasium)
# ---------------------------------------------------------------------------


def _import_offline():
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from nsoran.drl import offline

    return offline


def test_synthetic_dataset_roundtrip_and_validation(tmp_path):
    offline = _import_offline()

    dataset = offline.synthetic_dataset(
        num_transitions=250, episode_length=100, seed=3
    )
    dataset.validate()
    assert dataset.observations.shape == (250, 85)
    assert dataset.next_observations.shape == (250, 85)
    assert dataset.actions.shape == (250,)
    assert dataset.rewards.shape == (250,)
    assert dataset.terminals.shape == (250,)
    assert dataset.episode_ids.shape == (250,)
    assert dataset.timestamps.shape == (250,)
    assert dataset.num_episodes == 3

    path = offline.save_offline_dataset(tmp_path / "dataset.npz", dataset)
    assert path.exists()
    reloaded = offline.load_offline_dataset(path)
    assert reloaded.observations.shape == (250, 85)
    assert (reloaded.actions == dataset.actions).all()
    assert (reloaded.episode_ids == dataset.episode_ids).all()
    assert reloaded.reward_components is not None
    assert reloaded.reward_components.shape == (250, 5)


def test_dataset_validation_rejects_wrong_shapes(tmp_path):
    np = pytest.importorskip("numpy")
    offline = _import_offline()

    bad = offline.OfflineDataset(
        observations=np.zeros((4, 10), dtype=np.float32),
        actions=np.zeros(4, dtype=np.int64),
        rewards=np.zeros(4, dtype=np.float32),
        next_observations=np.zeros((4, 10), dtype=np.float32),
        terminals=np.zeros(4, dtype=np.int64),
        episode_ids=np.zeros(4, dtype=np.int64),
        timestamps=np.zeros(4, dtype=np.int64),
    )
    with pytest.raises(offline.DatasetValidationError):
        bad.validate(obs_dim=85)


def test_dataset_validation_rejects_out_of_range_actions():
    pytest.importorskip("numpy")
    offline = _import_offline()

    dataset = offline.synthetic_dataset(num_transitions=10, seed=0)
    dataset.actions[0] = 999
    with pytest.raises(offline.DatasetValidationError):
        dataset.validate()


def test_split_dataset_is_episode_disjoint():
    offline = _import_offline()

    dataset = offline.synthetic_dataset(
        num_transitions=300, episode_length=100, seed=7
    )
    train, validation = offline.split_dataset(dataset, val_fraction=0.34, seed=1)
    assert len(train) > 0 and len(validation) > 0
    train_episodes = set(train.episode_ids.tolist())
    validation_episodes = set(validation.episode_ids.tolist())
    assert not (train_episodes & validation_episodes)


def test_apply_reward_config_recomputes_rewards():
    offline = _import_offline()
    from nsoran.drl.config import compute_reward, get_agent_config

    dataset = offline.synthetic_dataset(
        num_transitions=40, episode_length=10, seed=5
    )
    agent = get_agent_config("DQN")
    rescored = dataset.with_rewards_from_components(agent)
    expected = compute_reward(
        dataset.reward_components[:, 0],
        dataset.reward_components[:, 1],
        dataset.reward_components[:, 2],
        dataset.reward_components[:, 3],
        dataset.reward_components[:, 4],
        agent,
    )
    assert rescored.rewards == pytest.approx(expected, rel=1e-6)
    assert rescored.metadata["reward_agent"] == "DQN"


def test_offline_replay_env_spaces_and_episode_cycle():
    np = pytest.importorskip("numpy")
    offline = _import_offline()

    dataset = offline.synthetic_dataset(
        num_transitions=10, episode_length=5, seed=11
    )
    env = offline.OfflineReplayEnv(dataset, mode="deterministic", seed=0)
    assert isinstance(env.observation_space, __import__("gymnasium").spaces.Box)
    assert env.observation_space.shape == (85,)
    assert int(env.action_space.n) == 128

    observation, info = env.reset()
    assert observation.shape == (85,)
    assert observation.dtype == np.float32
    assert info["episode_length"] == 5

    first_episode = info["episode_id"]
    terminated = truncated = False
    steps = 0
    last_info = info
    while not (terminated or truncated):
        observation, reward, terminated, truncated, last_info = env.step(0)
        steps += 1
        assert np.isfinite(reward)
        assert observation.shape == (85,)
        assert "action_mismatch" in last_info
        if steps > 10:  # safety
            pytest.fail("episode did not terminate")
    assert steps == 5
    # synthetic_dataset marks the last transition of each episode terminal.
    assert terminated and not truncated

    _, next_info = env.reset()
    assert next_info["episode_id"] != first_episode
    env.close()


def test_offline_replay_env_truncates_unterminated_episode():
    offline = _import_offline()

    dataset = offline.synthetic_dataset(
        num_transitions=6, episode_length=3, seed=4
    )
    dataset.terminals[:] = 0
    env = offline.OfflineReplayEnv(dataset, mode="deterministic", seed=0)
    env.reset()
    env.step(0)
    env.step(0)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated
    assert info["truncated_at_dataset_boundary"]
    env.close()


def test_offline_replay_env_shuffled_mode_runs():
    offline = _import_offline()

    dataset = offline.synthetic_dataset(
        num_transitions=200, episode_length=100, seed=2
    )
    env = offline.OfflineReplayEnv(dataset, mode="shuffled", seed=42)
    observation, info = env.reset()
    assert observation.shape == (85,)
    assert info["mode"] == "shuffled"
    env.close()


def test_extract_step_metrics_proxy_and_info_override():
    np = pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from nsoran.drl.evaluate import extract_step_metrics

    observation = np.zeros(85, dtype=np.float32)
    observation[0] = 0.5  # cell 2 throughput
    observation[12] = 0.25  # cell 3 throughput
    observation[84] = 0.7  # global bs_on
    metrics = extract_step_metrics(observation, {})
    assert metrics["throughput"] == pytest.approx(0.75)
    assert metrics["energy"] == 0.0
    assert metrics["bs_on"] == pytest.approx(0.7)
    assert metrics["sources"]["throughput"] == "observation_proxy"

    overridden = extract_step_metrics(
        observation,
        {"throughput": 3.0, "rlf": 2.0, "activation_cost": 0.1, "bs_on": 6},
        reward=1.0,
    )
    assert overridden["throughput"] == pytest.approx(3.0)
    assert overridden["rlf"] == pytest.approx(2.0)
    assert overridden["sources"]["throughput"] == "info"
    assert overridden["reward"] == pytest.approx(1.0)


def test_evaluation_result_serialization(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from nsoran.drl.evaluate import (
        EpisodeMetrics,
        EvaluationResult,
        save_result,
        summarize_result,
    )

    result = EvaluationResult(
        algo="ppo",
        model_path="model.zip",
        environment="test",
        episodes=[
            EpisodeMetrics(
                episode=0,
                steps=5,
                throughput=1.0,
                energy=2.0,
                rlf=0.0,
                activation_cost=0.1,
                bs_on_mean=6.0,
                reward=0.5,
                action_switches=2,
            )
        ],
        step_metrics={
            "throughput": [0.2, 0.2],
            "energy": [0.4, 0.4],
            "rlf": [0.0, 0.0],
            "activation_cost": [0.05, 0.05],
            "bs_on": [6.0, 6.0],
            "reward": [0.2, 0.3],
            "action": [1, 2],
        },
    )
    summary = summarize_result(result)
    assert summary["episodes"] == 1
    assert summary["metrics"]["throughput"]["mean"] == pytest.approx(1.0)

    payload = save_result(result, tmp_path, make_plots=False)
    assert (tmp_path / "evaluation_summary.json").exists()
    assert (tmp_path / "step_metrics.npz").exists()
    assert payload["algo"] == "ppo"


# ---------------------------------------------------------------------------
# Torch tests (network shapes, CQL smoke, checkpointing)
# ---------------------------------------------------------------------------


def _import_rem_dqn():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from nsoran.drl import rem_dqn

    return rem_dqn


def test_rem_network_output_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from nsoran.drl.networks import REMQNetwork

    network = REMQNetwork(
        obs_dim=85,
        n_actions=128,
        hidden_dims=(64, 64, 32),
        k_heads=3,
    )
    observations = torch.zeros(4, 85)
    q_values = network(observations)
    assert q_values.shape == (4, 3, 128)
    assert network.mean_q(observations).shape == (4, 128)
    assert network.num_parameters > 0


def test_rem_network_rejects_wrong_observation_size():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from nsoran.drl.networks import REMQNetwork

    network = REMQNetwork(obs_dim=85, hidden_dims=(16, 16, 8))
    with pytest.raises(ValueError):
        network(torch.zeros(2, 10))


def test_cql_loss_smoke_and_update():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    rem_dqn = _import_rem_dqn()

    config = rem_dqn.REMDQNConfig(
        hidden_dims=(32, 32, 16),
        conv_channels=(8, 8),
        k_heads=3,
        cql_alpha=0.5,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    agent = rem_dqn.REMDQNAgent(config)
    batch = {
        "observations": np.random.rand(8, 85).astype(np.float32),
        "actions": np.random.randint(0, 128, size=8),
        "rewards": np.random.randn(8).astype(np.float32),
        "next_observations": np.random.rand(8, 85).astype(np.float32),
        "terminals": np.zeros(8, dtype=np.float32),
    }
    losses = agent.compute_losses(batch)
    assert set(losses) >= {"td_loss", "cql_loss", "total_loss"}
    for name, value in losses.items():
        assert np.isfinite(float(value.detach()))
    assert float(losses["cql_loss"].detach()) >= 0.0
    expected_total = float(losses["td_loss"].detach()) + config.cql_alpha * float(
        losses["cql_loss"].detach()
    )
    assert float(losses["total_loss"].detach()) == pytest.approx(
        expected_total, rel=1e-5
    )

    metrics = agent.update(batch)
    assert metrics["global_step"] == 1.0
    assert 0.0 <= metrics["epsilon"] <= 1.0


def test_rem_dqn_act_and_epsilon_schedule():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    rem_dqn = _import_rem_dqn()

    config = rem_dqn.REMDQNConfig(
        hidden_dims=(16, 16, 8),
        conv_channels=(4, 4),
        k_heads=3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        device="cpu",
        seed=0,
    )
    agent = rem_dqn.REMDQNAgent(config)
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(100) == pytest.approx(0.05)

    single = agent.act(np.zeros(85, dtype=np.float32), deterministic=True)
    assert isinstance(single, int)
    assert 0 <= single < 128

    batch_actions = agent.act(
        np.zeros((5, 85), dtype=np.float32), deterministic=True
    )
    assert batch_actions.shape == (5,)
    assert ((batch_actions >= 0) & (batch_actions < 128)).all()


def test_rem_dqn_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    rem_dqn = _import_rem_dqn()

    config = rem_dqn.REMDQNConfig(
        hidden_dims=(16, 16, 8),
        conv_channels=(4, 4),
        k_heads=3,
        device="cpu",
        seed=0,
    )
    agent = rem_dqn.REMDQNAgent(config)
    observation = np.zeros(85, dtype=np.float32)
    expected = agent.act(observation, deterministic=True)
    path = agent.save(tmp_path / "agent.pt")
    loaded = rem_dqn.REMDQNAgent.load(path)
    assert loaded.act(observation, deterministic=True) == expected
    assert loaded.config.k_heads == 3
    assert loaded.config.cql_alpha == config.cql_alpha


def test_rem_replay_buffer_shapes():
    np = pytest.importorskip("numpy")
    rem_dqn = _import_rem_dqn()

    buffer = rem_dqn.OfflineReplayBuffer(
        observations=np.zeros((20, 85), dtype=np.float32),
        actions=np.zeros(20, dtype=np.int64),
        rewards=np.zeros(20, dtype=np.float32),
        next_observations=np.zeros((20, 85), dtype=np.float32),
        terminals=np.zeros(20, dtype=np.float32),
        seed=0,
    )
    batch = buffer.sample(8)
    assert batch["observations"].shape == (8, 85)
    assert batch["actions"].shape == (8,)
    assert batch["terminals"].shape == (8,)


# ---------------------------------------------------------------------------
# IQL tests (D4)
# ---------------------------------------------------------------------------


def _import_iql():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from nsoran.drl import iql

    return iql


def test_iql_losses_and_update():
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    iql = _import_iql()

    config = iql.IQLConfig(
        hidden_dims=(32, 32),
        batch_size=8,
        device="cpu",
        seed=0,
    )
    agent = iql.IQLAgent(config)
    batch = {
        "observations": np.random.rand(8, 85).astype(np.float32),
        "actions": np.random.randint(0, 128, size=8),
        "rewards": np.random.randn(8).astype(np.float32),
        "next_observations": np.random.rand(8, 85).astype(np.float32),
        "terminals": np.zeros(8, dtype=np.float32),
    }
    losses = agent.compute_losses(batch)
    assert set(losses) >= {
        "q_loss", "value_loss", "policy_loss", "total_loss"
    }
    for name, value in losses.items():
        assert np.isfinite(float(value.detach()))
    expected_total = sum(
        float(losses[name].detach())
        for name in ("q_loss", "value_loss", "policy_loss")
    )
    assert float(losses["total_loss"].detach()) == pytest.approx(
        expected_total, rel=1e-5
    )

    q_before = agent.q_values(batch["observations"]).detach().clone()
    metrics = agent.update(batch)
    assert metrics["global_step"] == 1.0
    q_after = agent.q_values(batch["observations"])
    assert not np.allclose(
        q_before.numpy(), q_after.detach().numpy(), atol=1e-7
    )
    validation = agent.validation_loss(batch)
    assert np.isfinite(validation["total_loss"])


def test_iql_predict_and_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    iql = _import_iql()

    config = iql.IQLConfig(hidden_dims=(16, 16), device="cpu", seed=0)
    agent = iql.IQLAgent(config)
    observation = np.zeros(85, dtype=np.float32)

    action, state = agent.predict(observation)
    assert state is None
    assert isinstance(action, int)
    assert 0 <= action < 128
    assert agent.act(observation, deterministic=True) == action

    batched, _ = agent.predict(
        np.zeros((5, 85), dtype=np.float32), deterministic=True
    )
    assert batched.shape == (5,)

    path = agent.save(tmp_path / "iql.pt")
    loaded = iql.IQLAgent.load(path)
    assert loaded.act(observation) == action
    assert loaded.config.tau == config.tau
    assert loaded.config.hidden_dims == (16, 16)


def test_iql_training_smoke(tmp_path):
    pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    offline = _import_offline()
    iql = _import_iql()

    dataset = offline.synthetic_dataset(
        num_transitions=48, episode_length=12, seed=5
    )
    summary = iql.train_iql(
        dataset,
        iql_config=iql.IQLConfig(
            hidden_dims=(16, 16), batch_size=8, device="cpu", seed=0
        ),
        output_dir=tmp_path,
        total_timesteps=8,
        eval_freq=4,
        min_timesteps=1,
        patience=2,
        val_fraction=0.25,
        seed=0,
        verbose=0,
    )
    assert summary["algo"] == "iql"
    assert summary["completed_steps"] == 8
    assert (tmp_path / "final_model.pt").is_file()
    assert (tmp_path / "training_summary.json").is_file()
    assert summary["train_transitions"] + summary["validation_transitions"] == 48


# ---------------------------------------------------------------------------
# E1 / CMDP: os três modos de restrição (docs/plano_tecnico/cmdp_design.md)
# ---------------------------------------------------------------------------


def test_constraint_mode_none_keeps_eq2_untouched():
    """Modo default: CmdpConfig desligado e Eq. (2) bit a bit intacta."""
    from nsoran.drl.config import (
        CONSTRAINT_MODES,
        CmdpConfig,
        compute_reward,
        get_agent_config,
    )

    cfg = get_agent_config("PPO-2")
    args = (0.6, 0.5, 0.4, 0.3, 0.2)
    baseline = compute_reward(*args, cfg)
    # Eq. (2) com os pesos de PPO-2 (0.40/0.40/0.10/0.10).
    assert baseline == pytest.approx(
        0.40 * 0.6 - 0.40 * 0.5 - 0.10 * 0.4 - 0.10 * 0.3 - 0.40 * 0.2
    )
    # Os novos kwargs opcionais, nos valores default, não mudam nada.
    assert compute_reward(
        *args, cfg, dual_lambda=0.0, qos_cost=0.7, gate_open=1.0
    ) == pytest.approx(baseline)

    default = CmdpConfig()
    assert CONSTRAINT_MODES == ("none", "lagrangian", "barrier")
    assert default.constraint_mode == "none"
    assert not default.enabled
    assert default.qos_cost(1.0, 1.0) == 0.0  # sem limiares -> custo zero


def test_constraint_mode_lagrangian_cost_penalty_and_dual():
    """Modo lagrangiano: custo hinge (Eq. 3.2), r̃ = r − λc (Eq. 4.1) e
    ascensão dual com projeção (Eq. 4.2)."""
    from nsoran.drl.config import CmdpConfig, compute_reward, get_agent_config

    cmdp = CmdpConfig(
        constraint_mode="lagrangian",
        tau_sla=53_680.0,
        epsilon=14.0,
        throughput_scale=76_686.0,
        rlf_scale=18.0,
        tolerance=0.05,
        lambda_lr=0.01,
        lambda_max=10.0,
    )
    # Hinge de throughput e de RLF, isolados e somados; folga => custo zero.
    assert cmdp.qos_cost(53_680.0, 14.0) == pytest.approx(0.0)
    assert cmdp.qos_cost(53_680.0 - 7_668.6, 14.0) == pytest.approx(0.1)
    assert cmdp.qos_cost(53_680.0, 18.0) == pytest.approx(4.0 / 18.0)
    assert cmdp.qos_cost(60_000.0, 10.0) == pytest.approx(0.0)
    assert cmdp.qos_cost(53_680.0 - 7_668.6, 18.0) == pytest.approx(
        0.1 + 4.0 / 18.0
    )

    # Recompensa lagrangiana: Eq. (2) − λ·c (DQN: w = 0.40/0.40/0.10/0.10).
    cfg = get_agent_config("DQN")
    args = (0.6, 0.5, 0.4, 0.3, 0.2)
    base = compute_reward(*args, cfg)
    penalised = compute_reward(*args, cfg, dual_lambda=2.0, qos_cost=0.25)
    assert penalised == pytest.approx(base - 2.0 * 0.25)

    # Ascensão dual (Eq. 4.2).  A deriva 0.01*(0.0757−0.05) = 2.57e-4 é a do
    # design §8.1 para o dataset misto real.
    assert cmdp.update_dual(0.0757) == pytest.approx(2.57e-4)
    assert cmdp.update_dual(1.0e6) == pytest.approx(10.0)  # teto lambda_max
    assert cmdp.update_dual(-1.0e6) == pytest.approx(0.0)  # piso 0.0

    # Arrays também funcionam (re-scoring offline de datasets reais).
    np = pytest.importorskip("numpy")
    costs = cmdp.qos_cost(
        np.array([53_680.0 - 7_668.6, 60_000.0]),
        np.array([14.0, 10.0]),
    )
    assert costs[0] == pytest.approx(0.1)
    assert costs[1] == pytest.approx(0.0)
    array_args = tuple(np.asarray(value) for value in args)
    gated_rewards = compute_reward(
        *array_args,
        cfg,
        dual_lambda=2.0,
        qos_cost=costs,
    )
    assert gated_rewards[0] == pytest.approx(base - 2.0 * 0.1)
    assert gated_rewards[1] == pytest.approx(base)


def test_constraint_mode_barrier_gates_both_energy_channels():
    """Modo barreira: Eq. (5.1′) porta os DOIS canais de economia (γ̂ e B̂)."""
    import math

    from nsoran.drl.config import (
        CmdpConfig,
        compute_gated_reward,
        compute_reward,
        get_agent_config,
        normalize_constraint_mode,
    )

    assert normalize_constraint_mode("gate") == "barrier"
    assert normalize_constraint_mode("BARRIER") == "barrier"

    cfg = get_agent_config("PPO-2")  # w2 = 0.40 pesa γ̂ e B̂
    cmdp = CmdpConfig(constraint_mode="barrier", tau_sla=53_680.0, gate_window=10)
    args = (0.6, 0.5, 0.4, 0.3, 0.2)

    # Porta aberta (média da janela >= τ): Eq. (5.1′) == Eq. (2).
    assert cmdp.gate_factor(53_680.0) == 1.0
    assert compute_gated_reward(*args, cfg, 1.0) == pytest.approx(
        compute_reward(*args, cfg)
    )

    # Porta fechada: w1*ρ̂ − w3*ς̂ − w4*δ̂ (0.4*0.6 − 0.1*0.4 − 0.1*0.3 = 0.17).
    assert cmdp.gate_factor(53_679.9) == 0.0
    gated = compute_gated_reward(*args, cfg, 0.0)
    assert gated == pytest.approx(0.17)
    assert gated == pytest.approx(compute_reward(*args, cfg, gate_open=0.0))

    # Porta suave (variante padrão de (b), §8.2-B1): g(τ) = 0.5 e monótona;
    # T = 0.05·μ_AO ≈ 3834 bytes para μ_AO = 76.686 (dados reais).
    smooth = CmdpConfig(
        constraint_mode="barrier", tau_sla=53_680.0, gate_temperature=3_834.0
    )
    assert smooth.gate_factor(53_680.0) == pytest.approx(0.5)
    assert smooth.gate_factor(53_680.0 + 3_834.0) == pytest.approx(
        1.0 / (1.0 + math.exp(-1.0))
    )
    values = [smooth.gate_factor(tp) for tp in (0.0, 30_000.0, 53_680.0, 80_000.0)]
    assert values == sorted(values)
    assert all(0.0 < value < 1.0 for value in values)

    # A porta também pode ser combinada com a penalidade dual (camadas
    # independentes em compute_reward).
    both = compute_reward(*args, cfg, dual_lambda=1.0, qos_cost=0.2, gate_open=0.0)
    assert both == pytest.approx(0.17 - 0.2)


def test_cmdp_config_validation():
    from nsoran.drl.config import CmdpConfig

    with pytest.raises(ValueError):
        CmdpConfig(constraint_mode="barrier")  # porta exige tau_sla
    with pytest.raises(ValueError):
        CmdpConfig(constraint_mode="lagrangian")  # exige τ e/ou ε
    with pytest.raises(ValueError):
        CmdpConfig(constraint_mode="ponderado")  # modo desconhecido
    with pytest.raises(ValueError):
        CmdpConfig(dual_lambda=11.0, lambda_max=10.0)  # λ fora da projeção
    with pytest.raises(ValueError):
        CmdpConfig(tau_sla=-1.0, constraint_mode="lagrangian")
    with pytest.raises(ValueError):
        CmdpConfig(tau_sla=1.0, epsilon=0.0, throughput_scale=0.0)
    with pytest.raises(ValueError):
        CmdpConfig(tau_sla=1.0, gate_window=0)
    with pytest.raises(ValueError):
        CmdpConfig(tau_sla=1.0, gate_temperature=0.0)

    # from_mapping ignora chaves desconhecidas e normaliza o apelido.
    cfg = CmdpConfig.from_mapping(
        {"constraint_mode": "gate", "tau_sla": 53_680.0, "descendente": True}
    )
    assert cfg.constraint_mode == "barrier"
    assert cfg.tau_sla == pytest.approx(53_680.0)
    roundtrip = CmdpConfig.from_mapping(cfg.as_dict())
    assert roundtrip.as_dict() == cfg.as_dict()


def test_train_cli_cmdp_flags_default_off():
    """As flags do E1/CMDP existem no CLI de treino, mas o default é 'none'."""
    from nsoran.drl import train

    parser = train.build_parser()
    default = parser.parse_args(
        ["--algo", "rem_dqn", "--agent", "DQN", "--dataset", "d.npz"]
    )
    assert default.cmdp_mode == "none"
    assert default.qos_throughput_min is None
    assert default.qos_rlf_max is None
    assert default.qos_lambda_max == pytest.approx(10.0)  # design §8.1-A4
    assert default.qos_tolerance == pytest.approx(0.05)  # design §2.1

    args = parser.parse_args(
        [
            "--algo", "rem_dqn", "--agent", "DQN", "--dataset", "d.npz",
            "--cmdp-mode", "lagrangian",
            "--qos-throughput-min", "53680",
            "--qos-rlf-max", "14",
            "--qos-throughput-scale", "76686",
            "--qos-rlf-scale", "18",
        ]
    )
    assert args.cmdp_mode == "lagrangian"
    assert args.qos_throughput_min == pytest.approx(53_680.0)
    assert args.qos_rlf_max == pytest.approx(14.0)
    assert args.qos_throughput_scale == pytest.approx(76_686.0)
    assert args.qos_rlf_scale == pytest.approx(18.0)


def test_train_cli_cmdp_mode_gating(tmp_path):
    """'barrier' e 'lagrangian' sem limiares falham antes de abrir o dataset."""
    from nsoran.drl import train

    def namespace(*extra):
        parser = train.build_parser()
        return parser.parse_args(
            [
                "--algo", "rem_dqn", "--agent", "DQN",
                "--dataset", "nao-existe.npz",
                "--offline-algo", "replay",
                "--output-dir", str(tmp_path / "saida"),
                *extra,
            ]
        )

    with pytest.raises(ValueError, match="barrier"):
        train._run(namespace("--cmdp-mode", "barrier"))
    with pytest.raises(ValueError, match="lagrangian"):
        train._run(namespace("--cmdp-mode", "lagrangian"))
    # Nada foi criado: a validação do CMDP precede o carregamento do dataset.
    assert not (tmp_path / "saida").exists()

    # SB3 PPO (ablação replay) ainda não recebe o lagrangiano (§8.1-A5).
    parser = train.build_parser()
    ppo_args = parser.parse_args(
        [
            "--algo", "ppo", "--agent", "PPO-2", "--dataset", "d.npz",
            "--offline-algo", "replay",
            "--cmdp-mode", "lagrangian", "--qos-throughput-min", "1",
        ]
    )
    with pytest.raises(ValueError, match="replay"):
        train._run(ppo_args)
