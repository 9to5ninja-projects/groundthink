# GroundThink 8M - Wide (4 layers, dim=384)
# Good for: conversation, Q&A, fast recall

MODEL_CONFIG = dict(
    dim=384,
    n_layers=4,
    n_heads=8,
    head_dim=48,
)

TRAIN_CONFIG = dict(
    seq_len=256,
    batch_size=16,
    num_steps=10000,
    lr_base=1e-4,      # Lower LR for wider model
    lr_decay=5e-4,
    warmup=500,
)

SAVE_NAME = "groundthink_8M_wide"
