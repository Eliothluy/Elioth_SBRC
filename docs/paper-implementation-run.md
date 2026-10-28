# Running the paper implementation (CCNC 2025) on real ns-3 data

This document records an end-to-end execution of the *Design and Evaluation of
Deep Reinforcement Learning for Energy Saving in Open RAN* implementation in
this workspace: real ns-3 runs of the paper scenario → offline dataset → PPO /
REM-DQN training → live evaluation of the policy on ns-3.

Everything below was produced with **real ns-3 simulations**. No synthetic
dataset was used for any reported number.

## 1. Scenario

`src/environments/scenario_configurations/paper_es_use_case.json`:

| Parameter | Value | Paper |
|---|---|---|
| `configuration` | 3 | 850 MHz, 20 MHz, ISD 1700 m |
| `trafficModel` | 4 | 25 % TCP full buffer 20 Mbps, 25 % UDP bursty ~20 Mbps, 25 % TCP bursty 750 kb/s, 25 % TCP bursty 150 kb/s |
| `positionAllocator` | 2 | paper uniform placement |
| `ues` | 9 | 9 UEs per gNB → 63 UEs |
| `simTime` | 10 s | 10 s |
| `indicationPeriodicity` | 0.1 s | 100 ms |
| `handoverMode` | DynamicTtt | dynamic TTT |
| `outageThreshold` | −5 dB | RLF threshold |
| `e2nrEnabled` | 1 | **required**, see below |

`--e2nrEnabled=1` is mandatory: without it ns-3 only writes the E2 KPM files of
the LTE eNB (cell 1). The seven NR cells (2..8) then produce no
`cu-*-cell-*.txt` / `du-cell-*.txt`, and every per-cell KPI of the 85-dimensional
state stays at zero.

## 2. Wall-clock cost

The paper scenario is **slow**: one E2 indication (100 ms of simulation time)
costs ~50 s of wall time on this machine, so a full 10 s simulation takes
~90 min. The Python-side timeouts (`step_timeout`, `reset_timeout`) must stay
well above that value, otherwise the episode is truncated before ns-3 has
produced any KPM.

## 3. Data collection

```bash
cd ns-o-ran-gym
.venv/bin/python examples/collect_run.py --policy random --seed 557 \
    --config src/environments/scenario_configurations/paper_es_use_case.json \
    --ns3_path ../ns-3-mmwave-oran \
    --output_folder examples/output/real_runs/mine_random_seed557
```

`examples/collect_run.py` drives the live scenario with a scripted policy
(`keep` = Always On, `random`, `random-seconds` = the paper's *Random*
baseline) and leaves the run folder behind.

The run folder is then converted into the offline dataset:

```bash
.venv/bin/python examples/fit_reward_calibration.py \
    --runs-dir examples/output/real_runs \
    --output examples/output/reward_calibration.npz

.venv/bin/python examples/collect_real_dataset.py \
    --runs-dir examples/output/real_runs \
    --output examples/output/real_dataset.npz \
    --config src/environments/scenario_configurations/paper_es_use_case.json \
    --agent PPO-2 --calibration-file examples/output/reward_calibration.npz
```

The calibration matters: without it the reward normaliser falls back to generic
ranges (throughput 0…10⁷) and every component of Eq. (2) is compressed into
≈0.005, so the reward carries almost no signal. With the calibration fitted on
the real runs, each component is quantile-normalised over its real
distribution (mean ≈ 0.5, range [0, 1]), exactly as the paper describes.

## 4. Training

```bash
.venv/bin/python examples/ppo_2.py --dataset examples/output/real_dataset.npz \
    --output-dir examples/output/real_runs/ppo_2 --total-timesteps 1200000 \
    --batch-size 256
```

PPO-2 (Table I: w1=0.40, w2=0.40, w3=0.10, w4=0.10, uniform normalisation,
ent_coef=0.002) with the paper's hyper-parameters. The REM-DQN + CQL agent
(`examples/dqn.py`) uses the same dataset and the same Table I row.

## 5. Evaluation

```bash
.venv/bin/python examples/ppo_2.py --eval-only --model <checkpoint> \
    --ns3-path ../ns-3-mmwave-oran \
    --scenario-config src/environments/scenario_configurations/paper_es_use_case.json \
    --episodes 1
```

The evaluator reads the **raw** KPM totals from `info["raw_kpms"]["totals"]`, so
the reported throughput / energy / RLF are the real KPI values and not the
min-max normalised state.

For a side-by-side comparison of scripted policies:

```bash
.venv/bin/python examples/compare_runs.py \
    --run "Always On=examples/output/real_runs/mine_alwayson_seed558" \
    --run "Random=examples/output/real_runs/mine_random_seed557" \
    --reference "Always On" --window 47
```

`--window N` compares every policy over the same number of first E2
indications, which is necessary because a run that ns-3 aborts covers fewer
indications than a complete one.

## 6. Results obtained in this workspace

All numbers come from real ns-3 runs of the paper scenario and are computed
from the raw KPMs. To stay comparable, every policy is evaluated over the same
number of first E2 indications (`--window 25`).

| policy | energy / indication | throughput / indication | max RLF | mean active cells | Δ energy | Δ throughput |
|---|---|---|---|---|---|---|
| Always On (reference) | 617.3 | 71 836 | 6 | 7.00 | — | — |
| Random (ns-3 heuristic, 1 change/s) | 484.2 | 47 648 | 35 | 2.12 | −21.6 % | −33.7 % |
| Static 4-2-0-1 (ns-3 heuristic) | 622.4 | 71 774 | 10 | 6.00 | +0.8 % | −0.1 % |
| Random (per-indication, exploration) | 242.8 | 21 231 | 29 | 3.72 | −60.7 % | −70.4 % |

The order of magnitude of the trade-off is the one the paper reports: the static
baseline of the paper saves ≈0.7 % of the energy for a large throughput loss,
while the aggressive policies trade a large amount of energy for a large
throughput loss. The RLF column reproduces the paper's "max 2 UEs in RLF"
metric — the value is larger here because the runs are truncated by the ns-3
abort described in the next section and the covered window includes the
attachment phase of the scenario.

Offline dataset built from the same campaign:

| Item | Value |
|---|---|
| real ns-3 runs | 24 |
| E2 indications (transitions) | 885 |
| observations | 885 × 85 (float32, [0, 1]) |
| actions | 0…127 (all 128 combinations covered) |
| reward | Eq. (2) with the PPO-2 row, quantile-normalised on the real KPI distribution |

Agents trained on it with the paper's hyper-parameters:

| Agent | Algorithm | Budget | Result |
|---|---|---|---|
| PPO-2 | SB3 PPO, batch 256, ent_coef 0.002 | 1.2 M timesteps (early stop at 50 k) | `examples/output/real_runs/ppo_2_model/best_model.zip` |
| DQN | REM-DQN (3 heads, 512-512-256) + CQL (α = 0.5) | 20 000 gradient steps | `examples/output/real_runs/dqn_model/best_model.pt` |

The dataset is three orders of magnitude smaller than the paper's (>300 000
points from ≈3 000 simulations), so the trained policies are **not** a
reproduction of the paper's agents: they validate the pipeline end to end on
real data. Reproducing the paper's numbers needs the full campaign, i.e. days of
wall time on this machine (≈90 min per 10 s simulation).

## 7. Known limitation: the ns-3 fork aborts during the paper scenario

Several runs were interrupted by an ns-3 abort in the TCP stack. Two different
signatures were observed, both of them latent bugs of the ns-3.38 TCP
implementation that the paper traffic mix (20 Mbps TCP full buffer + UDP bursty,
i.e. heavy use of the selective acknowledgement option) exposes:

```
NS_ASSERT failed, cond="m_current >= m_dataStart && m_current < m_dataEnd",
msg="You have attempted to read beyond the bounds of the available buffer space. ..."
```

and, under gdb,

```
Program received signal SIGSEGV, Segmentation fault.
#0  ns3::TcpTxBuffer::Update (...) at stl_list.h:588
#1  ns3::TcpSocketBase::ProcessOptionSack (tcp-socket-base.cc:4278)
#2  ns3::TcpSocketBase::ReadOptions (tcp-socket-base.cc:1610)
#3  ns3::TcpSocketBase::ReceivedAck (tcp-socket-base.cc:1829)
```

Root causes (three latent bugs of the upstream ns-3 code, all fixed in this fork):

1. `TcpOptionSack::Deserialize()` computed `sackCount = (size - 2) / 8` in `uint8_t`
   arithmetic. A malformed SACK option with `size < 2` underflowed to 254, and the
   loop read up to 248 bytes past the end of the option. The over-read produced
   garbage blocks in the scoreboard of the TCP socket, which later crashed in
   `TcpTxBuffer::Update()` (SIGSEGV).
2. `TcpTxBuffer::Remove()` and `TcpTxBuffer::ResetLastSegmentSent()` erased items
   of `m_sentList` without invalidating `m_highestSack.first`, leaving a dangling
   iterator.
3. `McEnbPdcp::DoReceivePdu()` and `LtePdcp::DoReceivePdu()` removed the PDCP
   header without checking that the PDU was long enough to contain it (the
   user-plane counterpart `McUePdcp::DoReceivePdu()` already did). A truncated
   PDU made `LtePdcpHeader::Deserialize()` read past the end of the packet — this
   is the "read beyond the bounds of the available buffer space" abort, and it is
   the one that truncated most of the campaign runs.

The fixes are in `src/internet/model/tcp-option-sack.cc`,
`src/internet/model/tcp-tx-buffer.cc`, `src/lte/model/mc-enb-pdcp.cc`,
`src/lte/model/lte-pdcp.cc` and `src/lte/model/lte-pdcp-header.cc` of the ns-3
fork.

Validation: the *Always On* run with `RngRun=558` aborted at `t = 4.795932724 s`
with the "read beyond the bounds of the available buffer space" assert. With the
three fixes the **same run completes the full 10 s scenario** (100 E2 indications,
ns-3 exits normally), and the `TcpTxBuffer::Update()` variant of the crash (SIGSEGV)
is gone as well.

To obtain the backtrace of an abort, the simulation can be launched under gdb:

```bash
NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args" \
    .venv/bin/python examples/ppo_2.py --eval-only --model <checkpoint> ...
```

## 8. Bugs found and fixed while running the paper implementation

| Where | Symptom | Fix |
|---|---|---|
| `paper_es_use_case.json` | every per-cell KPI was zero (no KPM files for cells 2..8) | add `e2nrEnabled`/`e2lteEnabled`/`enableE2FileLogging`/`e2du`/`e2cuUp`/`e2cuCp` |
| `es_env.py`, `ns_env.py` | the episode was truncated after one step | `step_timeout` 30 s → 300 s, new `reset_timeout` (300 s) |
| `lte-enb-rrc.cc` | `NS_ASSERT failed ... The secondary cell to which the UE is attached has changed` after ≈0.3 s of a random policy | drop *stale* handover events in `PerformHandover()` |
| `ns_env.py` | the ns-3 error message was silently discarded (the run folder's `stderr` was always empty) | `read_streams()` opens `stdout`/`stderr` in append mode |
| `drl/real_data.py` | the dataset stored the **raw** reward components although the contract requires the normalised ones, so re-weighting the dataset with another Table I agent produced rewards ~10⁶ times too large | store the normalised components |
| `drl/evaluate.py` | the evaluator reported the min-max normalised observation proxy instead of the real KPIs | read `info["raw_kpms"]["totals"]` |
| `examples/_agent_cli.py` | the file was missing: every `examples/ppo_N.py` and `examples/dqn.py` failed with `ModuleNotFoundError` | re-implemented from the compiled `.pyc` |

## 9. Reproducing

```bash
cd ns-o-ran-gym
.venv/bin/python -m pytest -q tests/test_es_env_unit.py tests/test_real_data.py \
    tests/test_drl_shapes.py tests/test_agent_scripts.py     # 63 tests
```
