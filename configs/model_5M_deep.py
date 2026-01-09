# GroundThink 5.5M - Deep (6 layers, dim=256)
# Good for: reasoning, narrative

MODEL_CONFIG = dict(
    dim=256,
    n_layers=6,
    n_heads=8,
    head_dim=32,
)

TRAIN_CONFIG = dict(
    seq_len=256,
    batch_size=16,
    num_steps=50000,
    lr_base=3e-4,
    lr_decay=1e-3,
    warmup=2000,
)

SAVE_NAME = "groundthink_5M_deep"
