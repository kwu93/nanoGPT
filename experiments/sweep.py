import itertools
import json

from .train import BASE_CONFIG, train


def sweep(base_config, grid, seeds=(0, 1, 2), out='runs.jsonl'):
    """Train every combination in `grid` at every seed, appending one JSON line
    per run to `out`. Append + flush means a crashed sweep loses nothing and
    results accumulate across sweeps."""
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f'{len(combos)} configs x {len(seeds)} seeds = {len(combos) * len(seeds)} runs -> {out}')
    with open(out, 'a') as f:
        for n, values in enumerate(combos, 1):
            cfg = {**base_config, **dict(zip(keys, values))}
            for seed in seeds:
                result = train(cfg, seed=seed)
                row = {
                    'config': cfg,
                    'seed': seed,
                    'train_loss': result['train_loss'],
                    'val_loss': result['val_loss'],
                    'num_params': result['num_params'],
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
