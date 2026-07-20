"""Named training protocols: the frozen base configs experiments build on.

An experiment's config.py starts from one of these and overrides only what
the question needs; specs say `base: ref-v2` instead of restating ten
hyperparameters, and protocol drift across entries becomes impossible.
Protocols are append-only: never edit one in place, add a new version and
leave the old one for reproducibility.
"""

PROTOCOLS = {
    # Page-1 standard setting (journal/page1.md scoreboard): ~128 tokens/step,
    # E32/H128, 5000 iters. Context ladders held tokens/step at ~128 by
    # trading seq_len against batch_size (1x128, 2x64, 3x42, 4x32, 8x16,
    # 16x8, 32x4). weight_decay 0.0 makes AdamW coincide with the plain Adam
    # this era actually ran on (identical update at wd=0).
    'std-v1': {
        'data_path': 'input.txt',
        'test_ratio': 0.2,
        'seq_len': 32,
        'batch_size': 4,
        'dim_embed': 32,
        'dim_hidden': 128,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'num_iters': 5000,
        'eval_every': 250,
        'eval_iters': 100,
        'device': 'cpu',
    },
    # Page-2 reference scale: 4x capacity, 2,048 tokens/step, weight decay
    # 0.1, checkpoint-at-val-min (built into train()). Scoreboard entries at
    # this scale quote best_val_loss, not final-iterate val.
    'ref-v2': {
        'data_path': 'input.txt',
        'test_ratio': 0.2,
        'seq_len': 32,
        'batch_size': 64,
        'dim_embed': 64,
        'dim_hidden': 256,
        'lr': 1e-3,
        'weight_decay': 0.1,
        'num_iters': 10000,
        'eval_every': 1000,
        'eval_iters': 100,
        'device': 'cpu',
    },
}
