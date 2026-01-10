"""Quick check of h0 values in trained checkpoint."""
import torch

ckpt = torch.load('groundthink_8M_v3_1k.pt', weights_only=False)
h0 = ckpt['model']['h0']

print(f'h0 shape: {h0.shape}')
print(f'h0 dtype: {h0.dtype}')
print(f'\nPer-layer h0 norms:')
for i in range(h0.shape[0]):
    layer_h0 = h0[i]  # [n_heads, head_dim, head_dim]
    norm = layer_h0.norm().item()
    numel = layer_h0.numel()
    rms = norm / (numel ** 0.5)
    print(f'  Layer {i:2d}: norm={norm:8.4f}, RMS={rms:.6f}, elements={numel}')

print(f'\nFresh h0 comparison:')
fresh_h0 = torch.randn_like(h0) * 0.01
print(f'Fresh h0 norm (Layer 0): {fresh_h0[0].norm().item():.4f}')
print(f'Trained h0 is {h0[0].norm().item() / fresh_h0[0].norm().item():.1f}x larger than fresh init')
