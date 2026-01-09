# GroundThink V3 - 8M Prototype Config
# Per V3_RESEARCH_NOTES.md Section 2.30
#
# Architecture: 12 layers × 256 dim (deep-narrow for reasoning)
# Attention anchor at layer 6 (1:11 ratio)
# Target: Break 7.0 loss wall, validate Identity Coalescence

MODEL_CONFIG = dict(
    n_layers=12,           # Deep for reasoning depth
    dim=256,               # Narrow but sufficient
    n_heads=8,
    head_dim=32,           # 256 / 8 = 32
    state_dim=16,          # Latent state size per head
    attn_positions=[6],    # Attention anchor at middle (1:11 ratio)
)

TRAIN_CONFIG = dict(
    seq_len=256,           # Start short, curriculum will grow
    batch_size=8,          # Stateful batching
    num_steps=10000,       # Quick validation run
    lr_base=6e-4,          # Per Section 2.30
    warmup=500,            # ~5% warmup
)

SAVE_NAME = "groundthink_8M_v3"

# Expected outcomes:
# - Loss should drop below 6.5 within 10k steps
# - State entropy should stay in 2.0-5.0 range
# - State norms should stay < 10
# - If successful, ready to scale to 30M
