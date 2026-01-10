"""Trace state norm evolution through forward passes."""
import torch
from layers_v030 import GroundThinkV3

torch.set_grad_enabled(False)

model = GroundThinkV3(vocab_size=97).cuda()
model.eval()

# Initial state
states = model.get_initial_states(1, 'cuda')
print(f'INITIAL STATE:')
for i, s in enumerate(states):
    if s is not None:
        print(f'  Layer {i:2d}: norm={s.norm().item():.4f}')
    else:
        print(f'  Layer {i:2d}: None')

# After 1 token
x = torch.randint(0, 97, (1, 1)).cuda()
logits, states = model(x, states)
print(f'\nAFTER 1 TOKEN:')
for i, s in enumerate(states):
    if s is not None:
        print(f'  Layer {i:2d}: norm={s.norm().item():.4f}')
    else:
        print(f'  Layer {i:2d}: None')

# After 10 tokens
for _ in range(9):
    x = torch.randint(0, 97, (1, 1)).cuda()
    logits, states = model(x, states)
print(f'\nAFTER 10 TOKENS:')
for i, s in enumerate(states):
    if s is not None:
        print(f'  Layer {i:2d}: norm={s.norm().item():.4f}')
    else:
        print(f'  Layer {i:2d}: None')

# After 100 tokens
for _ in range(90):
    x = torch.randint(0, 97, (1, 1)).cuda()
    logits, states = model(x, states)
print(f'\nAFTER 100 TOKENS:')
for i, s in enumerate(states):
    if s is not None:
        print(f'  Layer {i:2d}: norm={s.norm().item():.4f}')
    else:
        print(f'  Layer {i:2d}: None')
