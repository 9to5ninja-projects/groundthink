"""
GroundThink V3 Data Pipeline

Implements StatefulDataset per V3_RESEARCH_NOTES.md Section 2.11:
- Parallel track splitting for stateful batching
- is_new_doc detection for state reset
- State-handoff support for identity persistence
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

# Import tokenizer infrastructure
from tokenizer_v030 import (
    GroundThinkTokenizer, CharTokenizer, BPETokenizer,
    get_tokenizer_for_scale, EOS_ID, SPECIAL_TOKENS
)


class StatefulDataset(Dataset):
    """
    Stateful Dataset for State-Handoff Training
    
    From V3_RESEARCH_NOTES.md Section 2.11:
    - Treats corpus as one giant continuous stream
    - Splits into N parallel tracks (one per batch element)
    - Each batch[i] at step T is continuation of batch[i] at step T-1
    - Detects new document boundaries for state reset
    
    Args:
        token_array: 1D tensor of all tokens
        batch_size: Number of parallel tracks
        seq_len: Sequence length per step
        doc_separator_id: Token ID that marks document boundaries (for state reset)
        val_ratio: Fraction of each track reserved for validation (default 0.1)
    """
    
    def __init__(
        self, 
        token_array: torch.Tensor, 
        batch_size: int, 
        seq_len: int,
        doc_separator_id: int | None = None,
        val_ratio: float = 0.1,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.doc_separator_id = doc_separator_id
        self.val_ratio = val_ratio
        
        # Truncate to be divisible by batch_size
        n_tokens = len(token_array)
        n_tokens = (n_tokens // batch_size) * batch_size
        tokens = token_array[:n_tokens]
        
        # Reshape into parallel tracks: [batch_size, tokens_per_track]
        self.tracks = tokens.view(batch_size, -1)
        self.tokens_per_track = self.tracks.size(1)
        
        # Split into train/val portions
        # Validation is the LAST val_ratio of each track (no overlap with training)
        self.train_tokens = int(self.tokens_per_track * (1 - val_ratio))
        self.val_start = self.train_tokens  # Validation starts where training ends
        
        # Total steps we can take (training only)
        self.num_steps = (self.train_tokens - 1) // seq_len
        
        # Validation steps available
        val_tokens = self.tokens_per_track - self.train_tokens
        self.val_steps = (val_tokens - 1) // seq_len
        self._val_step_idx = 0  # Current validation step for sequential access
        
        print(f"StatefulDataset: {batch_size} tracks × {self.tokens_per_track:,} tokens")
        print(f"  Train: {self.train_tokens:,} tokens, {self.num_steps:,} steps")
        print(f"  Val: {val_tokens:,} tokens, {self.val_steps:,} steps")
    
    def __len__(self) -> int:
        return self.num_steps
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training batch at step idx.
        
        Returns:
            x: Input tokens [batch_size, seq_len]
            y: Target tokens [batch_size, seq_len]
            is_new_doc: Boolean mask [batch_size] indicating state reset needed
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Ensure we don't read into validation portion
        if end_idx + 1 > self.train_tokens:
            end_idx = self.train_tokens - 1
            start_idx = end_idx - self.seq_len
        
        # x is input, y is target (shifted by 1)
        x = self.tracks[:, start_idx:end_idx]
        y = self.tracks[:, start_idx + 1:end_idx + 1]
        
        # Detect new document boundaries
        if self.doc_separator_id is not None:
            is_new_doc = (x == self.doc_separator_id).any(dim=1)
        else:
            # If no separator defined, never reset (pure stateful)
            is_new_doc = torch.zeros(self.batch_size, dtype=torch.bool)
        
        return x, y, is_new_doc
    
    def get_val_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Get next validation batch (sequential access).
        
        Returns:
            x: Input tokens [batch_size, seq_len]
            y: Target tokens [batch_size, seq_len]
            is_new_doc: Boolean mask [batch_size] indicating state reset needed
            
        Returns None when all validation batches exhausted.
        Call reset_val() to start over.
        """
        if self._val_step_idx >= self.val_steps:
            return None
        
        start_idx = self.val_start + (self._val_step_idx * self.seq_len)
        end_idx = start_idx + self.seq_len
        
        # x is input, y is target (shifted by 1)
        x = self.tracks[:, start_idx:end_idx]
        y = self.tracks[:, start_idx + 1:end_idx + 1]
        
        # Detect new document boundaries
        if self.doc_separator_id is not None:
            is_new_doc = (x == self.doc_separator_id).any(dim=1)
        else:
            is_new_doc = torch.zeros(self.batch_size, dtype=torch.bool)
        
        self._val_step_idx += 1
        return x, y, is_new_doc
    
    def reset_val(self):
        """Reset validation iterator to beginning."""
        self._val_step_idx = 0


class LineDataset(Dataset):
    """
    Simple line-based dataset (non-stateful, for comparison).
    Each line is an independent sample.
    """
    
    def __init__(self, filepath: str | Path, tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 50]
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(self.lines[idx])
        
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


def load_stateful_dataset(
    filepath: str | Path,
    batch_size: int,
    seq_len: int,
    tokenizer: GroundThinkTokenizer = None,
    scale: str = "8M",
    val_ratio: float = 0.1,
) -> tuple[StatefulDataset, GroundThinkTokenizer]:
    """
    Load a text file as a StatefulDataset.
    
    Args:
        filepath: Path to text file
        batch_size: Number of parallel tracks
        seq_len: Sequence length per step
        tokenizer: Optional pre-built tokenizer
        scale: Model scale for tokenizer selection ("8M", "30M", "125M")
        val_ratio: Fraction of each track reserved for validation (default 0.1)
        
    Returns:
        dataset: StatefulDataset
        tokenizer: GroundThinkTokenizer
    """
    filepath = Path(filepath)
    
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"  {len(text):,} characters")
    
    # Build tokenizer if not provided
    if tokenizer is None:
        tokenizer = get_tokenizer_for_scale(scale, str(filepath))
        print(f"  Built {scale} tokenizer: {tokenizer.vocab_size} vocab")
    
    # Tokenize
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"  {len(tokens):,} tokens")
    
    # Create dataset with EOS as doc separator
    dataset = StatefulDataset(tokens, batch_size, seq_len, doc_separator_id=EOS_ID, val_ratio=val_ratio)
    
    return dataset, tokenizer
