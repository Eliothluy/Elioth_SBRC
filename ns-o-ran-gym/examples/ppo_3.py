#!/usr/bin/env python3
"""PPO-3 -- aggressive energy-saving agent of Table I (Frente C).

Table I configuration: w1=0.20 (throughput), w2=0.40 (energy),
w3=0.20 (RLF), w4=0.20 (activation cost), normalization "uniform",
ent_coef=0.003.

Usage::

    # train offline and evaluate on the live ns-3 paper scenario
    python examples/ppo_3.py --dataset data/offline.npz --evaluate \
        --ns3-path ../ns-3-mmwave-oran

    # short smoke training on a synthetic dataset
    python examples/ppo_3.py --synthetic 1000 --total-timesteps 256

    # evaluate an existing checkpoint only
    python examples/ppo_3.py --eval-only --model runs/PPO-3/best_model.zip \
        --ns3-path ../ns-3-mmwave-oran

Run ``python examples/ppo_3.py --help`` for all options.
"""

from _agent_cli import run_agent

AGENT = "PPO-3"

if __name__ == "__main__":
    raise SystemExit(run_agent(AGENT))
