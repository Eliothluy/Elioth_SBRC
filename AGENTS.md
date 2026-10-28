# AGENTS.md

Research workspace implementing the CCNC 2025 paper *Design and Evaluation of Deep
Reinforcement Learning for Energy Saving in Open RAN*.

## Data policy — no synthetic data

**Do not use synthetic data and do not bypass the ns-3 simulation.** Every dataset,
training run, and reported metric must come from real ns-3 simulations launched
through the Gym environment: `examples/collect_run.py` (or `examples/energy_saving.py`)
→ run folder under `examples/output/<uuid>/` → `examples/collect_real_dataset.py
--run-dir ... --output ...` → `examples/ppo_N.py` / `examples/dqn.py --dataset ...
--evaluate --ns3-path ../ns-3-mmwave-oran`. The `--synthetic` script flags,
`nsoran.drl.offline.synthetic_dataset()`, and
`examples/paper_runs/dataset_paper_scale_300k.npz` are pipeline smoke tests only —
never a source of project results. Offline evaluation without `--ns3-path` merely
replays recorded transitions; it does not replace running ns-3. **A 10 s paper
simulation costs roughly 90 min of wall time on this machine** (one E2 indication of
100 ms of simulation time takes ~50 s of wall time, mostly UE attachment), so budget
for that instead of reaching for synthetic shortcuts. Unit tests with fake datalakes
(`setup_sim=False`) are fine: they test code, not results.

## Layout and repository boundaries

- **The root is not a git repository.** `ns-3-mmwave-oran/`, `ns-o-ran-gym/`, and
  `oran-e2sim/` are three independent git repos (branches `master`, `main`, `develop`).
  Run `git` inside the subproject you changed, not at the root.
- `ns-3-mmwave-oran/` — vendored ns-3.38 + mmWave/ns-O-RAN fork. Paper work lives in
  `scratch/scenario-three.cc`, `src/mmwave/helper/energy-heuristic.*`, and the LTE RRC
  handover guards. Local changes to `ns3`, `lte-enb-rrc.*`, and `sta-wifi-mac.h` are
  intentional; do not revert them.
- `ns-o-ran-gym/` — Python side. Two top-level packages under `src/`: `nsoran`
  (base env, datalake, DRL) and `environments` (Gym envs). **There is no
  `nsoran.environments` package**, despite what two legacy tests import.
- `oran-e2sim/` — upstream OSC E2 simulator fork. Not exercised by the Python tests;
  only touch it for E2-interface work.
- Paper sources at the root: the original PDF `Design and Evaluation of Deep
  Reinforcement\nLearning for Energy Saving in Open RAN.pdf` (arXiv:2410.14021,
  accepted at IEEE CCNC 2025 — **the filename contains a literal newline** between
  "Reinforcement" and "Learning", so escape it in shell commands), `Tradução.md`
  (full Portuguese translation of the paper), and `PAPER_IMPLEMENTATION.md`
  (Portuguese integration spec for this fork, organized as "Frentes" A–D:
  ns-3/ns-O-RAN, Gymnasium env, DRL, data/evaluation).
  `ns-o-ran-gym/examples/paper_runs/README.md` documents generated results and their
  caveats; it is not a source of truth.

## Paper sources and key numbers

Primary sources, in order of authority: the root PDF (arXiv:2410.14021, CCNC 2025),
`Tradução.md` (Portuguese translation; its section structure mirrors the paper), and
`PAPER_IMPLEMENTATION.md` (integration spec). Key numbers for quick reference:

- Problem: an energy-saving xApp turns the RF front-ends of NR cells ON/OFF (sleep
  mode 1, 71 µs wake-up; RF equipment ≈ 65% of BS power draw).
- State: 85 KPMs = 12 KPIs per cell × 7 NR cells + 1 global `bs_on` (number of active
  BSs). The 12 per-cell KPIs include throughput/energy ratio, RLF count and %
  (SINR < −5 dB), scheduled PRB count and %, 64-QAM transport blocks and MAC PDUs,
  PHY-layer bytes, and activation cost δ_i = 0.9^(0.01·T_Di) with T_Di in ms.
- Reward (Eq. 2): `w1·ρ − w2·γ − w3·ς − w4·δ − w2·B_sON`, Σ W_j = 1, each component
  quantile-normalized (uniform or normal).
- Table I agents (w1/w2/w3/w4 — normalization): PPO-1 0.51/0.19/0.20/0.10 — Normal;
  DQN 0.40/0.40/0.10/0.10 — Uniform; PPO-2 0.40/0.40/0.10/0.10 — Uniform;
  PPO-3 0.20/0.40/0.20/0.20 — Uniform; PPO-4 0.40/0.32/0.18/0.10 — Uniform;
  PPO-5 0.45/0.20/0.25/0.10 — Normal.
- Training is offline (as the O-RAN spec recommends): a dataset from ~3,000
  simulations / >300,000 data points. PPO via Stable-Baselines3 (batch 256,
  ent_coef 0.001–0.003, up to 1,200,000 timesteps, 100-step episodes, early stop
  after 4 evaluations without improvement, minimum 10,000 steps). DQN is REM-DQN
  with CQL (1-D CNN encoder + dense [512, 512, 256]).
- Scenario: NSA architecture, central LTE eNB with a co-located gNB plus 6 more gNBs
  1,700 m from the center (7 controlled NR cells, IDs 2–8; the translation text says
  "sete gNBs adicionais", a minor internal inconsistency — the paper's N=7 counts the
  NR cells). 3GPP UMi Street-Canyon, FR1 850 MHz, 20 MHz bandwidth, 9 UEs per
  BS = 63 UEs, random-walk mobility at 2–4 m/s, simTime 10 s, E2 indication
  periodicity 100 ms. Traffic mix: 25% TCP full-buffer (20 Mbps), 25% UDP bursty
  (~20 Mbps), 50% TCP bursty (half 750 kb/s, half 150 kb/s).
- Baselines: Always On; Random (uniform number of BSs off every 1 s); Static and
  Dynamic from [3] with tuples (N_On,P1, N_On,P2, N_Off) = (4,2,1), (3,2,2), (2,2,3),
  target SINR 13 dB, radius 2000 m, dynamic-TTT handover.
- Key results (500 evaluation points per agent/baseline): PPO-2 saves 24.2% energy vs
  Always On at −52.3% throughput; Static/Dynamic save +0.7%/+4.3% at −23.5%/−27.6%
  throughput; PPO-1 dominates the Static/Dynamic baselines (+1.2%/+4.8% energy,
  +7.2%/+3.1% throughput, max 2 UEs in RLF); DQN max 1 UE in RLF; PPO-4/PPO-5 save
  19.3%/13.6% at −46.2%/−34% throughput.

## Python commands

`python` is not on `PATH`. Run everything from `ns-o-ran-gym/` with the checked-in venv:

```bash
cd ns-o-ran-gym
.venv/bin/python -m pytest -q \
  tests/test_es_env_unit.py tests/test_real_data.py \
  tests/test_drl_shapes.py tests/test_agent_scripts.py   # 63 tests, ~2.5 s
.venv/bin/python -m pytest tests/test_drl_shapes.py -q -k rem   # single area
.venv/bin/python -m compileall src
```

- Full `pytest` currently fails collection: `tests/test_check_env.py` and
  `tests/test_time_ts.py` import the stale `nsoran.environments.ts_env` and hardcode
  `/workspace/...`. Use the four files above unless you are fixing those tests.
- `test_es_env_unit.py` uses fake datalakes (`setup_sim=False`); `test_real_data.py`
  builds fake run dirs. Neither needs ns-3. Live-env tests do.
- DRL smoke test (synthetic data, no ns-3; **pipeline check only, never a source of
  results** — see the Data policy above):

```bash
.venv/bin/python examples/ppo_1.py --synthetic 1000 --total-timesteps 256
.venv/bin/python examples/dqn.py   --synthetic 1000 --total-timesteps 256
```

- Real offline workflow: `examples/energy_saving.py` → a run folder under
  `examples/output/<uuid>/` → `examples/collect_real_dataset.py --run-dir ... --output ...`
  → `examples/ppo_N.py --dataset ... --evaluate --ns3-path ../ns-3-mmwave-oran` (or
  `examples/dqn.py`).
- Optional DRL deps (`torch`, `stable-baselines3`, `matplotlib`) are already installed
  in `.venv`; install extras with `pip install -e '.[drl]'` only if needed.

## ns-3 commands

```bash
cd ns-3-mmwave-oran
./ns3 build          # CMake wrapper, not waf
./ns3 run "scratch/scenario-three --configuration=3 --trafficModel=4 \
  --positionAllocator=2 --ues=9 --simTime=10 \
  --indicationPeriodicity=0.1 --heuristicType=3 --e2nrEnabled=1"
```

- Do **not** pass `--nBsNoUesAlloc` for the paper scenario: it excludes gNBs from the
  UE positioning, while the paper puts 9 UEs on each of the 7 gNBs (63 UEs).
  `paper_es_use_case.json` therefore leaves it out.

- The default profile is built (`build/`, not `build/optimized/`).
  `EnergySavingEnv(..., optimized=True)` expects `build/optimized/`; keep
  `optimized=False` for the default build.
- A 10 s paper simulation takes roughly 90 minutes of wall time (~50 s per 100 ms
  indication); use short runs for smoke checks.
- **`--e2nrEnabled=1` is mandatory for the paper scenario.** Without it ns-3 only
  writes the E2 KPM files of the LTE eNB (cell 1): the seven NR cells (2..8) produce
  no `cu-*-cell-*.txt` / `du-cell-*.txt` and every per-cell KPI of the 85-dimensional
  state stays at zero. `paper_es_use_case.json` now sets `e2nrEnabled`,
  `e2lteEnabled`, `enableE2FileLogging`, `e2du`, `e2cuUp` and `e2cuCp` explicitly.
- The ns-3 control handshake is: ns-3 posts `/sem_metrics_<uuid>` and then *blocks* on
  `/sem_control_<uuid>`; Python acquires the metrics semaphore, appends the action to
  `es_actions_for_ns3.csv` and releases the control semaphore; ns-3 then reads and
  truncates the file. A Python-side timeout that is shorter than the time ns-3 needs
  for one indication therefore truncates the episode — see the timeouts below.
- Running `./ns3 run` directly writes many trace files into the current working
  directory. Run it from a scratch/output directory if you want clean artifacts.

## Paper contract (shared by ns-3 and the Gym env)

- Controlled cells are `[2..8]` (cell 1 is the LTE eNB). Actions are `Discrete(128)`:
  bit `i` (LSB = cell 2) is 1 for ON, 0 for OFF.
- Observation is `Box(0, 1, shape=(85,))`: 12 KPIs × 7 cells + global `bs_on`.
- The external control file is `es_actions_for_ns3.csv` with header
  `timestamp,cellId,hoAllowed`; `hoAllowed=1` means ON. `bsState.txt` records the real
  RF state as `Timestamp UNIX Id State`.
- Paper scenario config: `src/environments/scenario_configurations/paper_es_use_case.json`.
  `examples/energy_saving.py` defaults to the legacy `es_use_case.json`; pass `--config`
  for paper runs.
- Table I agents and reward weights live in `src/nsoran/drl/config.py` (`PAPER_AGENTS`);
  `nsoran.drl` lazy-imports optional deps, so `--help` works without torch/SB3.
- Evaluation without `--ns3-path` replays the offline dataset and does not let actions
  affect transitions. Do not present those metrics as policy performance.
- `examples/paper_runs/dataset_paper_scale_300k.npz` is **synthetic**; its results
  validate pipeline/scale, not the paper. Real-run artifacts are in
  `examples/output/real_runs/`.

## Reference DRL implementation

### The example scripts (`ns-o-ran-gym/examples/`)

| Script | Purpose |
|---|---|
| `energy_saving.py` | canonical environment driver (keeps the current ON/OFF state) |
| `collect_run.py` | drives the live paper scenario with a scripted policy (`keep`, `random`, `random-seconds`) and leaves the run folder behind — the entry point for real data collection |
| `collect_real_dataset.py` | converts one or more ns-3 run folders into the offline `.npz` dataset (`--run-dir` / `--runs-dir`, `--agent` for the Table I reward weights) |
| `compare_runs.py` | aggregates real runs into the paper's metrics (throughput, energy, RLF, activation cost) and the energy-saving / throughput delta with respect to a reference run (`--window N` compares over the same number of indications) |
| `fit_reward_calibration.py` | fits the reward-component calibration (the quantiles of Eq. 2) on the raw KPIs of real runs, so the reward is not compressed into the generic fallback ranges |
| `_agent_cli.py` | shared CLI used by `ppo_1.py`..`ppo_5.py` and `dqn.py` (train offline → evaluate live) |
| `ppo_1.py`..`ppo_5.py`, `dqn.py` | one thin wrapper per Table I agent |

### `examples/energy_saving.py` — the canonical example

`examples/energy_saving.py` is the reference driver for the DRL side. It is an
environment driver, not training: after `reset()`, each step re-applies the current
ON/OFF state (`env.bs_states_list()`) — replace that call with a policy's predict to
train/evaluate. CLI flags and defaults:

| Flag | Default | Note |
|---|---|---|
| `--config` | `/mnt/legitbet/elioth_sbrc/ns-o-ran-gym/src/environments/scenario_configurations/es_use_case.json` | hardcoded at `energy_saving.py:10` |
| `--output_folder` | `output` | run dirs land in `output/<uuid>/` relative to cwd |
| `--ns3_path` | `/mnt/legitbet/elioth_sbrc/ns-3-mmwave-oran/` | hardcoded at `energy_saving.py:14` |
| `--num_steps` | `1000` | step-loop bound |
| `--optimized` | off (`store_true`) | expects `build/optimized/` |

It constructs `EnergySavingEnv(ns3_path, scenario_configuration, output_folder,
optimized)` and nothing else; every other knob comes from the constructor defaults.

### `src/environments/es_env.py` — where the parameters live

`EnergySavingEnv(ns3_path, scenario_configuration, output_folder, optimized,
do_heuristic=True, *, reward_weights=None, w1..w4=None, normalization="uniform",
calibration_file=None, state_calibration_file=None, p_tx=1.0, prb_total=139.0,
rlf_sinr_threshold=-5.0, es_decay=0.9, es_time_factor=0.01,
du_aggregation="max", step_timeout=300.0, reset_timeout=300.0, setup_sim=True)`.

- Default reward is Table I PPO-1 (`w1..w4 = 0.51/0.19/0.20/0.10`); per-component
  quantile normalization via `normalization="uniform"|"normal"`, with optional
  `calibration_file` (RewardNormalizer) and `state_calibration_file`
  (MinMaxNormalizer).
- Per-cell energy proxy `TB.TotNbrDl.1 * p_tx`; PRB % divides by `prb_total` (139);
  activation cost = `es_decay ** (es_time_factor * T_on_ms)`.
- Hardcoded in the `super()` call: `scenario="scenario-three"`,
  `control_file="es_actions_for_ns3.csv"`, `log_file="EsActions.txt"`.
- `step_timeout` (300 s) bounds the semaphore wait in `step()`; on expiry the step
  returns `truncated=True` with `info["timeout"]`. `reset_timeout` (300 s) bounds the
  wait for the *first* indication in `reset()`. **Both must stay well above the ~50 s
  that one 100 ms indication costs in wall time**, otherwise the episode is truncated
  before ns-3 has produced any KPM (the old 30 s / 5 s defaults made every run fail
  after one step).
- `reset(seed=...)` overwrites the scenario's `RngRun`; `simTime`,
  `indicationPeriodicity`, and `ues` are read from the scenario JSON.

### Scenario configurations (`src/environments/scenario_configurations/`)

`es_use_case.json` is the scenario that runs on ns-3 by default (what
`energy_saving.py` uses without `--config`); `paper_es_use_case.json` is the paper
scenario (pass `--config` for paper runs). Every JSON key is forwarded to the ns-3
binary as `--key=value` (the env flattens `{k: [v]}` to `k=v`).

| Key | `es_use_case.json` (default) | `paper_es_use_case.json` (paper) |
|---|---|---|
| `configuration` | 0 | 3 |
| `trafficModel` | 3 | 4 |
| `positionAllocator` | — | 2 |
| `ues` | 3 | 9 |
| `simTime` | 10 | 10 |
| `indicationPeriodicity` | 0.1 | 0.1 |
| `heuristicType` | -1 | -1 |
| `RngRun` | 555 | 555 |
| `numberOfRaPreambles` | 40 | 40 |
| `handoverMode` | DynamicTtt | DynamicTtt |
| `outageThreshold` | -5.0 | -5 |
| `controlFileName` | es_actions_for_ns3.csv | es_actions_for_ns3.csv |
| `useSemaphores` | 1 | 1 |
| `e2nrEnabled` | — (absent) | 1 |
| `e2lteEnabled` | — (absent) | 1 |
| `enableE2FileLogging` | — (absent) | 1 |
| `e2du` / `e2cuUp` / `e2cuCp` | — (absent) | 1 / 1 / 1 |
| `bsOn`/`bsIdle`/`bsSleep`/`bsOff` | 5/0/0/2 | — (absent) |
| `reducedPmValues` | 0 | — (absent) |

Only `simTime`, `indicationPeriodicity`, and `ues` are consumed by the Python side;
`controlFileName`/`useSemaphores` are ns-3-only flags (the control filename is
hardcoded in Python as `es_actions_for_ns3.csv`).

### ns-3 control and datalake (`src/nsoran/`)

- `ns_env.py` — `NsOranEnv`, the base class that owns the ns-3 process. On first
  construction it runs `./ns3 configure` + `./ns3 build` inside `ns3_path`, resolves
  the `scenario-three` executable from the ns-3 build status, then launches it with
  `subprocess.Popen(cwd=output_folder/<uuid4>)` and `LD_LIBRARY_PATH` pointing at the
  build dir. Communication is **named POSIX semaphores + files only** (`posix_ipc`):
  ns-3 posts `/sem_metrics_<uuid>` when new KPMs are ready; Python posts
  `/sem_control_<uuid>` after writing an action. There are no sockets/E2TERM here —
  `oran-e2sim` is not on this path. `close()` terminates the process and unlinks
  both semaphores.
- `action_controller.py` — `ActionController.create_control_action()` appends the
  action rows to `es_actions_for_ns3.csv` (read by ns-3) and to the `EsActions.txt`
  log.
- `datalake.py` — `SQLiteDatabaseAPI` creates `database.db` (SQLite) inside each run
  folder and ingests the ns-3 traces `cu-cp-cell-*.txt`, `cu-up-cell-*.txt`, and
  `du-cell-*.txt` into tables `lte_cu_cp`, `gnb_cu_cp`, `lte_cu_up`, `gnb_cu_up`,
  and `du` (cellId 1 → `lte_*`; cellId comes from the filename).
  `EnergySavingEnv` adds `bsState` (from `bsState.txt`, cells 2–8) and `grafana`
  (per-step KPIs + reward). Observations are read with `read_kpms()`, an INNER JOIN
  across those tables on `(timestamp, ueImsiComplete)`.
- `drl/` — the DRL stack ("Frente C"): `config.py` (`PAPER_AGENTS`, `STATE_DIM=85`,
  `N_CELLS=7`, `N_ACTIONS=128`, `CELL_IDS=(2..8)`, `compute_reward`, action
  bit-encoding), `ppo.py` (SB3 PPO + `EarlyStopCallback`), `rem_dqn.py` +
  `networks.py` (REM-DQN with CQL and a 1-D CNN encoder), `offline.py` (`.npz`
  dataset contract + `OfflineReplayEnv`), `real_data.py` (ns-3 run folder → `.npz`,
  actions replayed from `EsActions.txt`, rewards from the `grafana` table),
  `evaluate.py` (live/offline evaluation, plots, CLI), and `train.py` (training CLI).

## Gotchas

- Hardcoded absolute paths remain in `examples/energy_saving.py`
  (`/mnt/legitbet/...`) and the two legacy tests (`/workspace/...`). Always pass explicit
  `--ns3_path`, `--config`, and `--output_folder` rather than relying on defaults.
- The paper's aggressive ON/OFF policies can turn every cell off. The `lte-enb-rrc.cc`
  guards that keep UEs on LTE in that case are required to avoid an `EpcX2` crash;
  preserve them.
- A third guard in `LteEnbRrc::PerformHandover()` (`src/lte/model/lte-enb-rrc.cc`) drops
  *stale* handover events: the assertion
  `handoverInfo.sourceCellId == m_lastMmWaveCell[imsi]` does not hold when the xApp
  flips a cell while a handover is pending, and the abort
  (`The secondary cell to which the UE is attached has changed`) killed the simulation
  after ~0.3 s of a random policy. Preserve that guard too.
- `NsOranEnv.read_streams()` opens `stdout`/`stderr` in **append** mode. It used to
  open them with `"w+"`, which truncated them on every poll and silently discarded the
  ns-3 error message exactly when it was needed to diagnose a crash.
- `nsoran.drl.evaluate` reads the raw KPM totals from `info["raw_kpms"]["totals"]`
  (the key is `raw_kpms`, not `raw_kpis`). Without it the evaluator silently reported
  the min-max *normalised* observation proxy instead of the real throughput/energy.
- The offline dataset stores the **normalised** reward components (the dataset
  contract says so); `nsoran.drl.real_data` used to store the raw ones, which made
  `--apply-reward-config` produce rewards ~10⁶ times too large. Use
  `examples/fit_reward_calibration.py` to fit the quantile normalisation of Eq. (2)
  on the real runs before converting the dataset, otherwise the reward components
  are compressed into ≈0.005 by the generic fallback ranges.
- **Fixed:** the ns-3 fork aborted in the LTE/TCP stack during the paper scenario
  (the paper traffic mix exercises the SACK option much harder than the default
  mix). Three latent bugs of ns-3.38:
  (1) `TcpOptionSack::Deserialize()` computed `sackCount = (size - 2) / 8` in
  `uint8_t` arithmetic (a malformed option with `size < 2` underflowed to 254 and
  read 248 bytes past the option, corrupting the TCP scoreboard → SIGSEGV in
  `TcpTxBuffer::Update()`);
  (2) `TcpTxBuffer::Remove()`/`ResetLastSegmentSent()` left `m_highestSack.first`
  dangling after erasing an item of `m_sentList`;
  (3) `McEnbPdcp::DoReceivePdu()`/`LtePdcp::DoReceivePdu()` removed the PDCP
  header without checking the size of the PDU (the UE side already did) → a
  truncated PDU aborted the simulation with "read beyond the bounds of the
  available buffer space".
  Fixed in the fork (`src/internet/model/tcp-option-sack.cc`,
  `src/internet/model/tcp-tx-buffer.cc`, `src/lte/model/mc-enb-pdcp.cc`,
  `src/lte/model/lte-pdcp.cc`, `src/lte/model/lte-pdcp-header.cc`).
  Reproduce an abort with
  `NS3_LAUNCH_PREFIX="gdb -batch -ex run -ex 'bt 60' --args"`, which launches
  ns-3 under gdb (supported by `NsOranEnv.start_sim`).
  Validation: the *Always On* run with `RngRun=558` aborted at `t = 4.795932724 s`;
  with the three fixes the same run completes the full 10 s scenario.
- Gym run outputs (SQLite `database.db`, KPI traces, `EsActions.txt`) are written under
  `output_folder/<uuid>/`; the environment launches `scenario-three` with that as cwd.
- No test/lint CI exists (only mirror and PyPI release workflows). For C++ formatting,
  use the repo's `.clang-format`; Python has no enforced linter config.
