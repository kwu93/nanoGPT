"""Experiment config: BASE (full base config), GRID (dict of lists, swept as
a cross product), and optionally SEEDS (default (0, 1, 2)).

Run with: pixi run sweep experiments/runs/NNN-<slug>
"""

from experiments.protocols import PROTOCOLS

BASE = {**PROTOCOLS['ref-v2'], 'model': 'mlp_concat'}

GRID = {
    'context_k': [1, 2, 3, 4, 5, 8, 16, 32],
}

SEEDS = (0, 1, 2)
