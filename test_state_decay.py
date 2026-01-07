"""
Test how state degrades over time with filler conversation
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

MODEL_PATH = "models/rwkv7-g1c-2.9b-20251231-ctx8192"

print("Loading model...")
model = RWKV(model=MODEL_PATH, strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
print("Model loaded.\n")

state = None

def inject(text):
    global state
    tokens = pipeline.encode(text)
    for i in range(0, len(tokens), 256):
        _, state = model.forward(tokens[i:i+256], state)
    return len(tokens)

def generate(prompt, max_tokens=100):
    global state
    tokens = pipeline.encode(prompt)
    all_tokens = []
    out_str = ""
    
    for i in range(max_tokens):
        toks = tokens if i == 0 else [token]
        while len(toks) > 0:
            out, state = model.forward(toks[:256], state)
            toks = toks[256:]
        
        token = pipeline.sample_logits(out, temperature=0.5, top_p=0.5)
        if token == 0 or "User:" in out_str:
            break
        all_tokens.append(token)
        out_str = pipeline.decode(all_tokens)
    
    return out_str.split("User:")[0].strip()

# Plant a fact
print("="*60)
print("PLANTING FACT")
print("="*60)
inject("User: My secret code word is BANANA. Remember this.\n\nAssistant: I will remember that your secret code word is BANANA.\n\n")
print("Planted: secret code word = BANANA\n")

# Test recall immediately
print("Immediate recall:")
print("Q: What is my secret code word?")
resp = generate("User: What is my secret code word?\n\nAssistant:")
print(f"A: {resp}\n")

# Now inject filler conversation
print("="*60)
print("INJECTING FILLER (simulating long conversation)")
print("="*60)

filler_topics = [
    ("weather", "The weather today is sunny with mild temperatures."),
    ("cooking", "A good recipe for pasta involves garlic, olive oil, and parmesan."),
    ("history", "The Roman Empire fell in 476 AD."),
    ("science", "Water boils at 100 degrees Celsius at sea level."),
    ("music", "Beethoven composed nine symphonies."),
    ("sports", "Soccer is the most popular sport worldwide."),
    ("travel", "Paris is known for the Eiffel Tower."),
    ("technology", "The first computer was called ENIAC."),
    ("nature", "The Amazon rainforest produces 20% of Earth's oxygen."),
    ("art", "The Mona Lisa was painted by Leonardo da Vinci."),
]

total_filler_tokens = 0
for topic, answer in filler_topics:
    filler = f"User: Tell me about {topic}.\n\nAssistant: {answer}\n\n"
    tokens = inject(filler)
    total_filler_tokens += tokens

print(f"Injected {total_filler_tokens} tokens of filler conversation\n")

# Test recall after filler
print("="*60)
print("TESTING RECALL AFTER FILLER")
print("="*60)
print("Q: What is my secret code word?")
resp = generate("User: What is my secret code word?\n\nAssistant:")
print(f"A: {resp}\n")

# Inject even more filler
print("="*60)
print("INJECTING MORE FILLER (2x)")
print("="*60)
for topic, answer in filler_topics * 2:
    filler = f"User: Tell me more about {topic}.\n\nAssistant: {answer} There's much more to learn about this topic.\n\n"
    tokens = inject(filler)
    total_filler_tokens += tokens

print(f"Total filler tokens now: {total_filler_tokens}\n")

print("Q: What is my secret code word?")
resp = generate("User: What is my secret code word?\n\nAssistant:")
print(f"A: {resp}\n")

# One more round
print("="*60)
print("INJECTING EVEN MORE FILLER (4x)")
print("="*60)
for topic, answer in filler_topics * 4:
    filler = f"User: Explain {topic} in detail.\n\nAssistant: {answer} This is a fascinating subject.\n\n"
    tokens = inject(filler)
    total_filler_tokens += tokens

print(f"Total filler tokens now: {total_filler_tokens}\n")

print("Q: What is my secret code word?")
resp = generate("User: What is my secret code word?\n\nAssistant:")
print(f"A: {resp}\n")

print("="*60)
print("CONCLUSION")
print("="*60)
print("If the model still remembers BANANA after all that filler,")
print("the state has good retention. If not, we've demonstrated decay.")