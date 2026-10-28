#!/usr/bin/env python3
"""PPO-5 -- coverage-oriented agent of Table I (Frente C).

Table I configuration: w1=0.45 (throughput), w2=0.20 (energy),
w3=0.25 (RLF), w4=0.10 (activation cost), normalization "normal",
ent_coef=0.0015.

Usage::

    # train offline and evaluate on the live ns-3 paper scenario
    python examples/ppo_5.py --dataset data/offline.npz --evaluate \
        --ns3-path ../ns-3-mmwave-oran

    # short smoke training on a synthetic dataset
    python examples/ppo_5.py --synthetic 1000 --total-timesteps 256

    # evaluate an existing checkpoint only
    python examples/ppo_5.py --eval-only --model runs/PPO-5/best_model.zip \
        --ns3-path ../ns-3-mmwave-oran

Run ``python examples/ppo_5.py --help`` for all options.
"""

from _agent_cli import run_agent

AGENT = "PPO-5"

if __name__ == "__main__":
    raise SystemExit(run_agent(AGENT))
