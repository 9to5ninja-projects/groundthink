"""
State Update Delta Verification
Per V3.5 Research: Check if FLA kernel is actually updating state.

If Delta = 0.0: Kernel not updating state (fatal)
If Delta very small (1e-6): Rigid state, In-Context LR too low
If Delta reasonable: State is updating, proceed to cosine similarity
"""
import torch
from layers_v030 import GroundThinkV3

torch.set_grad_enabled(False)

print("="*60)
print("STATE UPDATE DELTA CHECK (5 tokens)")
print("="*60)

model = GroundThinkV3(vocab_size=97).cuda()
model.eval()

# Use tokens within vocab range (0-96)
# Vocab 97 = ASCII 32-127 shifted to 0-95 + one special token
# "Hello" shifted: H=72-32=40, e=101-32=69, l=108-32=76, l=76, o=111-32=79
test_tokens = [40, 69, 76, 76, 79]  # "Hello" in tokenizer space

# Get initial state
states = model.get_initial_states(1, 'cuda')

print(f"\nInitial state[0] sum: {states[0].sum().item():.8f}")
print(f"Initial state[0] norm: {states[0].norm().item():.8f}")

print("\n--- Processing 5 tokens ---")
for i, tok_id in enumerate(test_tokens):
    prev_state_sum = states[0].sum().item()
    prev_state_norm = states[0].norm().item()
    
    # Process single token
    x = torch.tensor([[tok_id]]).cuda()
    logits, states = model(x, states)
    
    current_state_sum = states[0].sum().item()
    current_state_norm = states[0].norm().item()
    
    delta_sum = abs(current_state_sum - prev_state_sum)
    delta_norm = abs(current_state_norm - prev_state_norm)
    
    # Token display: shift back to ASCII for readability
    char_display = chr(tok_id + 32) if tok_id + 32 < 128 else '?'
    print(f"Token {i} ('{char_display}'): delta_sum={delta_sum:.8f}, delta_norm={delta_norm:.8f}")

print("\n--- Analysis ---")
print(f"Final state[0] sum: {states[0].sum().item():.8f}")
print(f"Final state[0] norm: {states[0].norm().item():.8f}")

# Check multiple layers
print("\n--- All Layers Final State ---")
for layer_idx, layer_state in enumerate(states):
    if layer_state is not None:
        print(f"Layer {layer_idx:2d}: sum={layer_state.sum().item():12.4f}, norm={layer_state.norm().item():.4f}")
    else:
        print(f"Layer {layer_idx:2d}: None (Attention)")
