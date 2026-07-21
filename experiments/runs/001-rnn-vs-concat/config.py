"""Experiment config: BASE (full base config), GRID (dict of lists, swept as
a cross product), and optionally SEEDS (default (0, 1, 2)).

Run with: pixi run sweep experiments/runs/001-rnn-vs-concat
"""

from experiments.protocols import PROTOCOLS

BASE = {**PROTOCOLS['ref-v2'], 'model': 'rnn'}

# dim_hidden is listed first so the required H=256 cells run before the
# param-matched H=325 cells if the sweep is cut short.
# context_k None = unbounded recurrence (pre-existing rnn behavior);
# 8 = windowed recurrence matching mlp_concat k=8's context reach.
# H=325 puts the rnn at 152,100 params vs mlp_concat k=8's 152,193.
GRID = {
    'dim_hidden': [256, 325],
    'context_k': [None, 8],
}

SEEDS = (0, 1, 2)
