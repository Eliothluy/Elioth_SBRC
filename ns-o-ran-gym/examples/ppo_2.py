#!/usr/bin/env python3
"""PPO-2 -- energy-saving-focused agent of Table I (Frente C).

Table I configuration: w1=0.40 (throughput), w2=0.40 (energy),
w3=0.10 (RLF), w4=0.10 (activation cost), normalization "uniform",
ent_coef=0.002.

Usage::

    # train offline and evaluate on the live ns-3 paper scenario
    python examples/ppo_2.py --dataset data/offline.npz --evaluate \
        --ns3-path ../ns-3-mmwave-oran

    # short smoke training on a synthetic dataset
    python examples/ppo_2.py --synthetic 1000 --total-timesteps 256

    # evaluate an existing checkpoint only
    python examples/ppo_2.py --eval-only --model runs/PPO-2/best_model.zip \
        --ns3-path ../ns-3-mmwave-oran

Run ``python examples/ppo_2.py --help`` for all options.
"""

from _agent_cli import run_agent

AGENT = "PPO-2"

if __name__ == "__main__":
    raise SystemExit(run_agent(AGENT))
