"""
RWKV-7 Worldview Service - Simplified
Uses pre-trained state from init_state.py
"""

import os
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

MODEL_PATH = "models/rwkv7-g1c-2.9b-20251231-ctx8192"
STATE_DIR = "states"
STATE_FILE = "initialized.state"

print("Loading model...")
model = RWKV(model=MODEL_PATH, strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
print("Model loaded.\n")

# Load pre-trained state
state = None
worldview_log = []

state_path = os.path.join(STATE_DIR, STATE_FILE)
if os.path.exists(state_path):
    data = torch.load(state_path, weights_only=False)
    state = data["state"]
    worldview_log = data.get("worldview_log", [])
    print(f"Loaded state from {state_path}")
    print(f"Worldview entries: {len(worldview_log)}")
else:
    print("No pre-trained state found. Run init_state.py first.")

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

def intercept_commands(response):
    global worldview_log
    
    # Match Action: calls the model naturally wants to make
    action_pattern = r'Action:\s*(\w+)\s*\(([^)]*)\)'
    matches = re.findall(action_pattern, response)
    
    for action, args in matches:
        # Memory-related actions -> save to worldview
        if action.lower() in ['remember_user_info', 'save_info', 'store_memory', 
                               'update_memory', 'note', 'remember', 'save']:
            # Extract what to remember from context around the action
            note = extract_memory_context(response, action)
            if note and note not in worldview_log:
                worldview_log.append(note)
                print(f"\n  [SAVED] {note}")
    
    return response

def extract_memory_context(response, action):
    """Extract what the model is trying to remember from the response context"""
    # Look for patterns like "remembered that X" or "user's X is Y"
    patterns = [
        r"remembered? that (.+?)(?:\.|$)",
        r"user'?s? (\w+ (?:is|are|prefers?) .+?)(?:\.|$)",
        r"favorite (\w+ is \w+)",
        r"name is (\w+)",
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: just note the action was called
    return f"User info recorded via {action}"

def save_state():
    filepath = os.path.join(STATE_DIR, STATE_FILE)
    torch.save({"state": state, "worldview_log": worldview_log}, filepath)
    print(f"State saved to {filepath}")

print("\n" + "-" * 50)
print("Commands: save, worldview, quit")
print("-" * 50 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            save_state()
            break
        if user_input.lower() == "save":
            save_state()
            continue
        if user_input.lower() == "worldview":
            print("\n=== Worldview ===")
            for i, entry in enumerate(worldview_log, 1):
                print(f"  {i}. {entry}")
            print("=================\n")
            continue
        
        print("Assistant: ", end="")
        response = generate(f"User: {user_input}\n\nAssistant:")
        intercept_commands(response)
        print()
        
    except KeyboardInterrupt:
        print("\nSaving...")
        save_state()
        break
