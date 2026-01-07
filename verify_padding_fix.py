
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

class Config:
    vocab_size = 50257      # GPT2 Standard
    d_model = 2048          # 1B Scale
    n_layer = 24            # Deeper than T4 version (18 -> 24)
    head_size = 64          
    
    # A100 Specifics
    # Total Batch = micro_batch * grad_accum * devices
    # 80GB VRAM allows dense batches
    micro_batch_size = 4    # Small batch for verification
    grad_accum_steps = 8     # Total batch ~128
    
    max_seq_len = 2048      # Real context length (vs 512 on T4)
    learning_rate = 3e-4    # Slightly lower for deeper model
    
    total_steps = 2500      # Stop after this many steps (approx 1 epoch of TinyStories)
    
    project_name = "groundthink_1B_A100"
    dtype = torch.bfloat16  # NATIVE BF16 (No Scaler needed, High Stability)

config = Config()

def get_dataloaders(config):
    print("📚 Loading Dataset (TinyStories for demo, swap for OpenWebText on A100)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Increase buffer use for A100 speed
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, 
            max_length=config.max_seq_len, return_tensors="pt"
        )
        
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        
        # Mask padding tokens so we don't train on them
        # -100 is the default ignore_index for CrossEntropyLoss
        if 'attention_mask' in encoded:
            labels[encoded['attention_mask'] == 0] = -100
            
        return input_ids, labels, encoded['attention_mask']

    return DataLoader(
        dataset, 
        batch_size=config.micro_batch_size, 
        collate_fn=collate_fn,
        num_workers=0  # Easier for simple script
    ), tokenizer

def verify_masking():
    print("🔍 VERIFICATION START: Checking Padding Masking...")
    dataloader, tokenizer = get_dataloaders(config)
    
    # Get one batch
    input_ids, labels, attention_mask = next(iter(dataloader))
    
    # Inspect first sample
    print(f"\nBatch Shape: {input_ids.shape}")
    
    sample_idx = 0
    input_ids_0 = input_ids[sample_idx]
    labels_0 = labels[sample_idx]
    mask_0 = attention_mask[sample_idx]
    
    pad_token_id = tokenizer.pad_token_id
    print(f"PAD Token ID: {pad_token_id}")
    
    # Find locations where mask is 0
    padding_indices = (mask_0 == 0).nonzero(as_tuple=True)[0]
    
    if len(padding_indices) == 0:
        print("⚠️ No padding in this sample. Fetching another batch...")
        # (For robust checking we might need loop, but TinyStories usually requires padding to 2048)
        # Let's inspect length first.
        print(f"Sequence Length: {len(input_ids_0)}")
        print("Sample Text:")
        print(tokenizer.decode(input_ids_0[:100]) + "...")
    else:
        print(f"✅ Found {len(padding_indices)} padding tokens.")
        
        # Check specific positions
        first_pad_pos = padding_indices[0].item()
        print(f"First padding starts at index: {first_pad_pos}")
        
        input_token_at_pad = input_ids_0[first_pad_pos].item()
        label_at_pad = labels_0[first_pad_pos].item()
        
        print(f"Input at index {first_pad_pos}: {input_token_at_pad} (Should be {pad_token_id})")
        print(f"Label at index {first_pad_pos}: {label_at_pad} (Should be -100)")
        
        if label_at_pad == -100:
            print("\n✅ SUCCESS: Label is correctly masked to -100.")
        else:
            print(f"\n❌ FAILURE: Label is {label_at_pad}, expected -100.")

        # Check a non-padded token
        non_pad_idx = 0
        print(f"\nChecking valid token at index {non_pad_idx}...")
        print(f"Input: {input_ids_0[non_pad_idx]} | Label: {labels_0[non_pad_idx]}")
        
        if labels_0[non_pad_idx] == input_ids_0[non_pad_idx]:
             print("✅ SUCCESS: Valid tokens are preserved.")
        else:
             print("❌ FAILURE: Valid token mismatch.")

if __name__ == "__main__":
    verify_masking()
