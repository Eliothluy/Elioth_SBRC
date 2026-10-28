#!/usr/bin/env python3
"""DQN -- REM-DQN + CQL offline agent of Table I (Frente C).

Table I configuration: w1=0.40 (throughput), w2=0.40 (energy),
w3=0.10 (RLF), w4=0.10 (activation cost), normalization "uniform".
The network is the paper's hybrid 1-D CNN + dense [512, 512, 256] with a
Random Ensemble Mixture head bank (3 heads) and a conservative Q-learning
loss (CQL alpha 0.5 by default).

Usage::

    # train offline and evaluate on the live ns-3 paper scenario
    python examples/dqn.py --dataset data/offline.npz --evaluate \
        --ns3-path ../ns-3-mmwave-oran

    # short smoke training on a synthetic dataset
    python examples/dqn.py --synthetic 1000 --total-timesteps 256

    # evaluate an existing checkpoint only
    python examples/dqn.py --eval-only --model runs/DQN/best_model.pt \
        --ns3-path ../ns-3-mmwave-oran

Run ``python examples/dqn.py --help`` for all options.
"""

from _agent_cli import run_agent

AGENT = "DQN"

if __name__ == "__main__":
    raise SystemExit(run_agent(AGENT))
