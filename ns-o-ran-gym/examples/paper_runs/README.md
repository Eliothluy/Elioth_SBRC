# Paper baseline runs

Real ns-3 runs of the CCNC 2025 paper (*Design and Evaluation of Deep
Reinforcement Learning for Energy Saving in Open RAN*) reproduced with
`scenario-three`.

**Every number in this folder comes from a real ns-3 simulation.** Nothing
here is synthetic: the runs are launched by
`examples/energy_saving.py` through the Gymnasium environment
(`environments.es_env.EnergySavingEnv`), which drives the ns-3 process with
the POSIX semaphores and the `es_actions_for_ns3.csv` control file.

## How the campaign is launched

```bash
cd ns-o-ran-gym
.venv/bin/python scripts/run_paper_baselines.py \
    --jobs 4 --seeds 555 556 557 \
    --output-dir examples/output/real_runs
```

`scripts/run_paper_baselines.py` enumerates the (policy, tuple, seed)
combinations and runs each one as a separate process of
`examples/energy_saving.py`. The runs are fully isolated: the ns-3 process,
its semaphores (`/sem_metrics_<uuid>`) and its control file all live in the
run folder, so several simulations can run concurrently on the same machine.

ns-3 is configured and compiled **once** by the orchestrator: the CMake cache
of ns-3 cannot be regenerated concurrently, so the workers are started with
`--skip-build` (`NsOranEnv(skip_build=True)`).

Useful options:

| option | effect |
|---|---|
| `--jobs N` | number of concurrent simulations (4 by default) |
| `--min-free-mb N` | do not start a run while less than N MiB are available |
| `--only NAME...` | run only the listed jobs (the names come from `--dry-run`) |
| `--force` | run the jobs again even if the manifest already holds a successful run |
| `--reclassify` | re-scan the logs of the recorded runs and fix their status |
| `--dataset PATH` | convert the runs that completed into an offline `.npz` |

## Policies

| policy | ns-3 `heuristicType` | description |
|---|---|---|
| `always_on` | `-1` | no heuristic, every RF front-end stays powered on (the reference) |
| `random` | `3` | uniform number of cells turned off every second (`EnergyHeuristic::RandomSleepingPolicy`) |
| `static` | `1` | static sleeping of [3], parametrized by the `(bsOn, bsIdle, bsSleep, bsOff)` tuple |
| `dynamic` | `2` | dynamic sleeping of [3], same tuple |

The tuples are the three of the paper: `(4, 2, 0, 1)`, `(3, 2, 0, 2)` and
`(2, 2, 0, 3)`; all of them add up to the seven controlled cells.

The baselines let **ns-3** decide which cells are powered off: the Python
side only releases the control semaphore and never writes to
`es_actions_for_ns3.csv` (`--no-action`, `EnergySavingEnv.step_baseline()`).
Writing to the control file would override the heuristic and would evict the
users of the cells it turned off at *every* indication, which makes the
simulation crash (see "The control file must stay empty" below). The decisions
of the heuristic are recorded in `EsActions.txt` so that the run can still be
converted into an offline dataset.

### The control file must stay empty

ns-3 reads `es_actions_for_ns3.csv` at every indication and, as soon as it
finds at least one row, it calls `LteEnbRrc::EvictUsersFromSecondaryCell()`.
Re-applying the current cell state (what `examples/energy_saving.py` does by
default) therefore evicts the users of the cells that are off at *every*
indication, which is not what a baseline is supposed to do and which also
makes the simulation abort. The baselines must therefore run with
`--no-action`: ns-3 owns the cell states and the Python side only logs them.

## Layout

```
examples/output/real_runs/
├── manifest.json            # one entry per run: policy, tuple, seed, log, run folder
├── logs/<job>.log           # stdout of examples/energy_saving.py
└── <job>/<uuid>/           # the ns-3 run folder
    ├── database.db          # SQLite datalake (cu-*, du, bsState, grafana)
    ├── bsState.txt          # real RF state per cell: Timestamp UNIX Id State
    ├── EsActions.txt        # decisions of the policy (7 rows per indication)
    ├── es_actions_for_ns3.csv  # control file delivered to ns-3
    ├── cu-cp-cell-*.txt     # ns-O-RAN CU-CP KPM traces
    ├── cu-up-cell-*.txt     # ns-O-RAN CU-UP KPM traces
    └── du-cell-*.txt        # ns-O-RAN DU KPM traces
```

## Cost and caveats

* The paper scenario (`configuration=3`, `trafficModel=4`, 7 gNBs, 9 UEs per
  gNB = 63 UEs, 10 s of simulation time) costs **about 50-60 minutes of wall
  time per run** on the default (non-optimized) ns-3 build of this workspace:
  one E2 indication (100 ms of simulation time) takes roughly 30-50 s of CPU.
  The 24 runs of the campaign were executed 8 at a time.
* A run whose ns-3 process aborts is **not** a result: the orchestrator scans
  the log of every run (`scripts/run_paper_baselines.py --reclassify`) and
  marks it as `crashed`. Such runs are excluded from the summary table.
* The paper scenario aborts in the LTE/TCP stack unless ns-3 tolerates a
  truncated packet. Three latent bugs of ns-3.38 are fixed in the fork:
  `TcpOptionSack::Deserialize()` underflowed on a malformed option length,
  `TcpTxBuffer::Remove()` left `m_highestSack.first` dangling, and the PDCP
  removed the PDCP header of a truncated PDU without checking its size.
  In addition, `TcpHeader::Deserialize()` now discards the options of a
  segment that does not carry them (`src/internet/model/tcp-header.cc`) and a
  malformed MSS option is discarded instead of aborting
  (`src/internet/model/tcp-option-rfc793.cc`), so a corrupted segment can
  never abort the simulation. The reproduction is
  `examples/energy_saving.py --heuristic 3 --seed 555 --no-action`, which
  aborted at t = 1.457 s before the fix.
* A paper simulation grows to **~1.8 GB of resident memory** (the fork
  accumulates state over the simulated window), so the campaign runs four
  simulations at a time and the orchestrator does not start a new one while
  less than `--min-free-mb` (4 GB by default) are available: the kernel would
  otherwise OOM-kill a run, and a killed run is not a result.
* `--optimized` expects a build in `build/optimized/`, which does not exist
  in this workspace; the default `build/` profile is used.
* The seeds are ns-3 `RngRun` values (555, 556 and 557). They change the
  UE positions, the mobility and the random sleeping decisions of ns-3.
* The static and dynamic heuristics use `sinrTh=73`, the ns-3 default, which
  the code documents as the 13 dB target of the paper.
* The KPIs of the campaign are *not* the paper's final numbers: the paper
  evaluates the DRL agents on top of them. These runs are the baselines the
  agents are compared against.

## Reading the results

```bash
# per-policy table (throughput / energy / RLF and the delta vs Always On)
.venv/bin/python scripts/summarize_baselines.py \
    --campaign-dir examples/output/real_runs \
    --csv examples/output/real_runs/baselines.csv \
    --markdown examples/output/real_runs/baselines.md

# offline dataset (one .npz with every run)
.venv/bin/python examples/collect_real_dataset.py \
    --runs-dir examples/output/real_runs/<job>/<uuid> ...
```
