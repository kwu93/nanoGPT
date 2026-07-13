"""Experiment framework for the from-scratch models (solo.py lineage).

- model.py: architecture registry, select with config['model']
- train.py: one training run as a pure function of a config dict
- sweep.py: grid sweeps over configs, results appended to runs.jsonl
- ngram.py: count-based baselines and floors (fixed/bag tables, Witten-Bell
  and Kneser-Ney backoff); run as a module for the NLL ladder
- analyze.ipynb (repo root): pandas/matplotlib analysis of runs.jsonl

Results and learnings are logged in JOURNAL.md (repo root).
"""
