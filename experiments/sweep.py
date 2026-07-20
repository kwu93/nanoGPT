"""Grid sweeps over train(config), one JSON line per run.

CLI: python -m experiments.sweep experiments/runs/NNN-<slug>
The experiment directory's config.py defines BASE, GRID, and optionally
SEEDS; results land in that directory (runs.jsonl, models/), so parallel
experiments in separate worktrees never collide.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .train import save_model, train


def git_info():
    """Current commit sha and dirty flag, recorded per run for provenance."""
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def sweep(base_config, grid, seeds=(0, 1, 2), out='runs.jsonl', models_dir='models', exp=None):
    """Train every combination in `grid` at every seed, appending one JSON line
    per run to `out`. Append + flush means a crashed sweep loses nothing and
    results accumulate across sweeps. Weights land in `models_dir`, keyed
    by config hash + seed so re-running a config overwrites its own file, and
    each row records its `model_path` for load_model(). Every row also carries
    provenance: experiment id, git sha (+ dirty flag), and a UTC timestamp."""
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    git_sha, git_dirty = git_info()
    print(f'{len(combos)} configs x {len(seeds)} seeds = {len(combos) * len(seeds)} runs -> {out}')
    with open(out, 'a') as f:
        for n, values in enumerate(combos, 1):
            cfg = {**base_config, **dict(zip(keys, values))}
            cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
            for seed in seeds:
                result = train(cfg, seed=seed)
                model_path = f"{models_dir}/{cfg['model']}-{cfg_hash}-s{seed}.pt"
                save_model(model_path, result['model'], result['chars'])
                row = {
                    'exp': exp,
                    'git_sha': git_sha,
                    'git_dirty': git_dirty,
                    'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'config': cfg,
                    'seed': seed,
                    'train_loss': result['train_loss'],
                    'val_loss': result['val_loss'],
                    'best_val_loss': result['best_val_loss'],
                    'best_iter': result['best_iter'],
                    'num_params': result['num_params'],
                    'model_path': model_path,
                    'curves': result['curves'],
                }
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(f"[{n}/{len(combos)}] {dict(zip(keys, values))} seed={seed} "
                      f"val_loss={result['val_loss']:.4f} params={result['num_params']}")


def load_experiment(exp_dir):
    """Import an experiment directory's config.py and return (BASE, GRID, SEEDS)."""
    path = exp_dir / 'config.py'
    spec = importlib.util.spec_from_file_location(f'exp_config_{exp_dir.name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BASE, module.GRID, getattr(module, 'SEEDS', (0, 1, 2))


def main():
    parser = argparse.ArgumentParser(description='Run the sweep defined by an experiment directory.')
    parser.add_argument('exp_dir', type=Path,
                        help='experiment directory containing config.py (e.g. experiments/runs/001-lambda-sweep)')
    args = parser.parse_args()
    base, grid, seeds = load_experiment(args.exp_dir)
    sweep(base, grid, seeds=seeds,
          out=args.exp_dir / 'runs.jsonl',
          models_dir=args.exp_dir / 'models',
          exp=args.exp_dir.name)


if __name__ == '__main__':
    main()
