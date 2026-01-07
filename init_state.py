"""
RWKV-7 State Initialization - Tool Training
Teaches the model how to use WRITE_WORLDVIEW syntax through examples.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

MODEL_PATH = "models/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096"
STATE_DIR = "states"
INIT_STATE_FILE = "initialized_0.4B.state"

os.makedirs(STATE_DIR, exist_ok=True)

print("Loading model...")
model = RWKV(model=MODEL_PATH, strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
print("Model loaded.\n")

state = None
worldview_log = []

def inject(text):
    global state
    tokens = pipeline.encode(text)
    for i in range(0, len(tokens), 256):
        _, state = model.forward(tokens[i:i+256], state)
    print(f"[Injected {len(tokens)} tokens]")

def generate(prompt, max_tokens=200):
    global state
    tokens = pipeline.encode(prompt)
    all_tokens = []
    out_str = ""
    out_last = 0
    occurrence = {}
    
    for i in range(max_tokens):
        toks = tokens if i == 0 else [token]
        while len(toks) > 0:
            out, state = model.forward(toks[:256], state)
            toks = toks[256:]
        
        for n in occurrence:
            out[n] -= (0.25 + occurrence[n] * 0.25)
        
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.7, top_k=100)
        if token == 0:
            break
        
        all_tokens.append(token)
        for k in occurrence:
            occurrence[k] *= 0.996
        
        ttt = pipeline.decode([token])
        occurrence[token] = occurrence.get(token, 0) + (0 if ttt in " \t0123456789" else 1)
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            print(tmp, end="", flush=True)
            out_str += tmp
            out_last = i + 1
        
        if "User:" in out_str:
            out_str = out_str.split("User:")[0]
            break
    
    print()
    return out_str.strip()

def save_state():
    global state, worldview_log
    filepath = os.path.join(STATE_DIR, INIT_STATE_FILE)
    torch.save({"state": state, "worldview_log": worldview_log}, filepath)
    print(f"\n*** State saved to {filepath} ***")

# =============================================================================
# TOOL TRAINING
# =============================================================================

print("=" * 60)
print("Teaching WRITE_WORLDVIEW tool usage")
print("=" * 60 + "\n")

# Functional examples only
TOOL_EXAMPLES = (
    "System: You have a tool to save notes permanently.\n\n"
    "Syntax: WRITE_WORLDVIEW[your note here]\n\n"
    "Examples:\n\n"
    "User: Remember that my name is Alex.\n\n"
    "Assistant: I will remember that. WRITE_WORLDVIEW[User name is Alex]\n\n"
    "User: What is 2+2? Save the answer.\n\n"
    "Assistant: 2+2 equals 4. WRITE_WORLDVIEW[2+2=4]\n\n"
    "User: I prefer dark mode.\n\n"
    "Assistant: Noted. WRITE_WORLDVIEW[User prefers dark mode]\n\n"
)

inject(TOOL_EXAMPLES)

print("\n" + "=" * 60)
print("Testing tool usage")
print("=" * 60 + "\n")

print("User: My favorite color is blue. Please save this.\n")
print("Assistant: ", end="")
response = generate("User: My favorite color is blue. Please save this.\n\nAssistant:")

print("\n" + "=" * 60)
print("Saving state")
print("=" * 60)
save_state()
print("Done.")
