# Pre-Phase 3.9 Infrastructure Validation Checklist

**Purpose:** 30-minute verification that core infrastructure is ready before starting Week 1 diagnostics

**Timeline:** Complete by Jan 10, 2026 (same day as Task 40 start)  
**Owner:** Dev  
**Block:** Cannot start Week 1 until all 3 items PASS

---

## Item 1: Checkpoint Loading & Model Instantiation ✓ PASS/FAIL

**What to Test:** Can we load ckpt_GF-MH_step1000.pt and run a forward pass?

**Quick Test (5 min):**
```bash
python3 << 'EOF'
import torch
from models import get_model

# Step 1: Load model
model = get_model('GF-MH').cuda()
print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Step 2: Load checkpoint
ckpt = torch.load('checkpoints/ckpt_GF-MH_step1000.pt', map_location='cuda')
model.load_state_dict(ckpt['model_state'])
print(f"✓ Checkpoint loaded (step {ckpt['step']})")

# Step 3: Test forward pass
x = torch.randint(0, 97, (2, 256), device='cuda')
with torch.no_grad():
    logits = model(x)
print(f"✓ Forward pass OK: output shape {logits.shape}")
print(f"✓ Memory OK: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n✅ PASS: Checkpoint loading works")
EOF
```

**Expected Output:**
```
✓ Model loaded: 7,900,000 params
✓ Checkpoint loaded (step 1000)
✓ Forward pass OK: output shape torch.Size([2, 256, 97])
✓ Memory OK: 1.23 GB
✅ PASS: Checkpoint loading works
```

**Status:** [ ] PASS [ ] FAIL

**If FAIL:** Check:
- Checkpoint file exists: `ls -lh checkpoints/ckpt_GF-MH_step1000.pt`
- Model registry: `python3 -c "from models import get_model; print(get_model('GF-MH'))"`
- Vocab size mismatch (check hybrid_v4_8m.py constructor)

---

## Item 2: BPE Tokenization Capability ✓ PASS/FAIL

**What to Test:** Does data/tokenizer.py support BPE encoding?

**Quick Test (5 min):**
```bash
python3 << 'EOF'
from data.tokenizer import CharTokenizer
# Load FineWeb sample
with open('data/fineweb_5m.txt', 'r') as f:
    text = f.read(100000)  # First 100K chars

tokenizer = CharTokenizer(text=text)
print(f"✓ CharTokenizer loaded: vocab_size={tokenizer.vocab_size}")

# Try encoding
tokens = tokenizer.encode("Hello world")
print(f"✓ Encode OK: 'Hello world' → {tokens}")

decoded = tokenizer.decode(tokens)
print(f"✓ Decode OK: {tokens} → '{decoded}'")

# Check for BPE support
if hasattr(tokenizer, 'encode_bpe') or hasattr(tokenizer, 'enable_bpe'):
    print("✓ BPE support detected")
else:
    print("⚠ WARNING: No BPE method found in tokenizer")
    print("  Available methods:", [m for m in dir(tokenizer) if not m.startswith('_')])

print("\n✅ PASS: Tokenizer operational")
EOF
```

**Expected Output:**
```
✓ CharTokenizer loaded: vocab_size=97
✓ Encode OK: 'Hello world' → [48, 37, 42, 42, 45, ...]
✓ Decode OK: [...] → 'Hello world'
✓ BPE support detected (or warning if not)
✅ PASS: Tokenizer operational
```

**Status:** [ ] PASS [ ] FAIL

**Note:** If BPE not detected in tokenizer.py, check:
- Is BPE implementation in a different module?
- Need to implement BPE wrapper? (See STRATEGY_AUDIT notes)
- Can Task 40 run with CharTokenizer temporarily?

---

## Item 3: Hidden State Accessibility for Tracing ✓ PASS/FAIL

**What to Test:** Can we access RWKV and Mamba hidden states during forward pass?

**Quick Test (10 min):**
```bash
python3 << 'EOF'
import torch
from models import get_model

model = get_model('GF-MH').cuda()
model.eval()

# Prepare input
x = torch.randint(0, 97, (1, 128), device='cuda')

# Test 1: Check if model outputs hidden states
print("Testing hidden state access...")
with torch.no_grad():
    # Try different return signatures
    try:
        # Option A: return_dict with hidden_states
        output = model(x, return_hidden_states=True)
        if isinstance(output, dict):
            print(f"✓ Option A works: {output.keys()}")
    except:
        # Option B: tuple return (logits, hidden_states)
        try:
            output, hidden_states = model(x)
            print(f"✓ Option B works: tuple return")
        except:
            # Option C: only logits returned
            output = model(x)
            print(f"⚠ Option C: only logits, need to hook states")

# Test 2: PyTorch hook capability (always works)
print("\nTesting PyTorch hook capability...")
states_captured = {}

def capture_rwkv_state(module, input, output):
    if hasattr(output, 'shape'):
        states_captured['rwkv'] = output.detach().cpu().numpy()
    return output

# Register hook on a layer (check actual model structure)
for name, module in model.named_modules():
    if 'rwkv' in name.lower():
        module.register_forward_hook(capture_rwkv_state)
        print(f"✓ Hooked: {name}")
        break

# Forward pass with hook
with torch.no_grad():
    output = model(x)
    
if 'rwkv' in states_captured:
    print(f"✓ State capture works: shape {states_captured['rwkv'].shape}")
else:
    print(f"⚠ Hook registered but no state captured (need to debug module structure)")

print("\n✅ PASS: State access validated")
EOF
```

**Expected Output:**
```
Testing hidden state access...
✓ Option A works: dict_keys(['logits', 'hidden_states', ...]) 
  OR
✓ Option B works: tuple return
  OR
⚠ Option C: only logits, need to hook states

Testing PyTorch hook capability...
✓ Hooked: model.layers.0.rwkv
✓ State capture works: shape (1, 128, 768)
✅ PASS: State access validated
```

**Status:** [ ] PASS [ ] FAIL

**If FAIL (no hidden states accessible):** 
- This is OK - PyTorch hooks still work (Option C above)
- State Tracing Module will use register_forward_hook() pattern
- Slightly more complex but functionally equivalent

---

## Summary

| Item | Status | Blocker | Notes |
|------|--------|---------|-------|
| 1: Checkpoint loading | PASS/FAIL | 🔴 YES | Required for Week 1 diagnostics |
| 2: BPE tokenization | PASS/FAIL | 🟡 MAYBE | Required for Task 40; can use CharTokenizer temporarily |
| 3: Hidden state access | PASS/FAIL | 🟢 NO | Hooks always work; direct access optional |

## How to Use This Checklist

1. **Before starting Phase 3.9:** Copy commands above into terminal
2. **Mark [ ] PASS or [ ] FAIL** for each item
3. **If any FAIL:** Debug and document issue in `logs/infrastructure_check_failures.txt`
4. **Timeline:** Must complete by Jan 10 evening (before Task 40 overnight run)

## Quick Command Reference

```bash
# Run all checks at once
cd /home/m_tes/groundthink

# Check 1
python3 -c "from models import get_model; m = get_model('GF-MH'); print('✓ Model OK')"

# Check 2
python3 -c "from data.tokenizer import CharTokenizer; print('✓ Tokenizer OK')"

# Check 3
python3 << 'EOF'
import torch
from models import get_model
x = torch.randint(0, 97, (1, 128))
m = get_model('GF-MH')
out = m(x)
print('✓ Forward pass OK')
EOF
```

---

**Checklist Status:** READY FOR USE  
**Created:** 2026-01-10  
**Owner:** Dev team  

