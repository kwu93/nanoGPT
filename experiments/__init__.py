"""Experiment framework for the from-scratch models (solo.py lineage).

- model.py: architecture registry, select with config['model']
- train.py: one training run as a pure function of a config dict
- sweep.py: grid sweeps over configs, results appended to runs.jsonl
- analyze.ipynb (repo root): pandas/matplotlib analysis of runs.jsonl
"""
