"""
Quick test script for RWKV-7 Goose model
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "RWKV/RWKV7-Goose-World2.9-0.4B-HF"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("RWKV-7 'Goose' Quick Test")
print("=" * 60)
print(f"\nDevice: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print("\nLoading model...")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if DEVICE == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

print("Model loaded!\n")

# Test prompt
prompt = "What is a large language model?"
print(f"Prompt: {prompt}\n")

# Create chat messages
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response...\n")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        top_p=0.3,
        repetition_penalty=1.2
    )

# Decode
generated_ids = [outputs[0][len(inputs.input_ids[0]):]]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("=" * 60)
print("Response:")
print("=" * 60)
print(response)
print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
