import hashlib
import itertools
import json

from .train import BASE_CONFIG, save_model, train


def sweep(base_config, grid, seeds=(0, 1, 2), out='runs.jsonl', models_dir='models'):
    """Train every combination in `grid` at every seed, appending one JSON line
    per run to `out`. Append + flush means a crashed sweep loses nothing and
    results accumulate across sweeps. Final weights land in `models_dir`, keyed
    by config hash + seed so re-running a config overwrites its own file, and
    each row records its `model_path` for load_model()."""
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
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


if __name__ == '__main__':
    sweep(BASE_CONFIG, {
        'dim_embed': [16, 32, 64],
        'dim_hidden': [64, 128, 256],
    })
