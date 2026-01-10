"""
GroundThink Tokenizer Infrastructure

Scalable tokenization from char-level (small models) to BPE (larger models).

Features:
- Scalable from small vocab (8M) to 24k BPE (125M)
- Special tokens as SINGLE tokens for state transitions
- Uses HuggingFace tokenizers library (Rust-based, fast)

Renamed from tokenizer_v030.py for standardization.
"""

from pathlib import Path

# Special tokens that MUST be single tokens per Section 2.31
# These ensure state transitions happen in one mathematical step
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endofturn|>",
]

# Token IDs (fixed across all vocab sizes for consistency)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6
ENDOFTURN_ID = 7

# Reserved range for special tokens (0-31)
RESERVED_SPECIAL = 32


class GroundThinkTokenizer:
    """
    Base tokenizer interface for GroundThink.
    
    All tokenizer implementations (char-level, BPE) must:
    1. Reserve IDs 0-31 for special tokens
    2. Treat SPECIAL_TOKENS as single tokens
    3. Provide encode/decode methods
    """
    
    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
    
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
    
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class CharTokenizer(GroundThinkTokenizer):
    """
    Character-level tokenizer for architecture validation.
    
    - Reserves IDs 0-31 for special tokens
    - Characters start at ID 32
    - Small vocab = minimal embedding tax
    """
    
    def __init__(self, text: str = None, vocab: dict = None):
        super().__init__()
        
        if vocab is not None:
            # Load from saved vocab
            self.char_to_id = vocab
        else:
            # Build from text
            self.char_to_id = {}
            
            # Reserve special tokens
            for i, token in enumerate(SPECIAL_TOKENS):
                self.char_to_id[token] = i
            
            # Build char vocab starting at RESERVED_SPECIAL
            if text:
                chars = sorted(set(text))
                for i, c in enumerate(chars):
                    if c not in self.char_to_id:
                        self.char_to_id[c] = RESERVED_SPECIAL + i
        
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
    
    @property
    def vocab_size(self) -> int:
        # Must return max_id + 1 to cover all token indices
        # (IDs are sparse: 0-7 for special, then 32+ for chars)
        return max(self.char_to_id.values()) + 1
    
    def encode(self, text: str) -> list[int]:
        # Fast path: if no special tokens in text, just map chars directly
        has_special = any(token in text for token in SPECIAL_TOKENS)
        
        if not has_special:
            # Fast vectorized encoding - no special token checks needed
            return [self.char_to_id.get(c, self.unk_id) for c in text]
        
        # Slow path: check for special tokens
        result = []
        i = 0
        while i < len(text):
            matched = False
            for token in SPECIAL_TOKENS:
                if text[i:].startswith(token):
                    result.append(self.char_to_id[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                result.append(self.char_to_id.get(text[i], self.unk_id))
                i += 1
        return result
    
    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


class BPETokenizer(GroundThinkTokenizer):
    """
    BPE tokenizer using HuggingFace tokenizers library.
    
    - Trains custom BPE on target corpus
    - Reserves IDs 0-31 for special tokens
    - Vocab size configurable (16k for 8M, 24k for 125M)
    """
    
    def __init__(self, tokenizer_path: str = None):
        super().__init__()
        self._tokenizer = None
        self._vocab_size = 0
        
        if tokenizer_path:
            self.load(tokenizer_path)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def train(self, files: list[str], vocab_size: int = 16000, min_frequency: int = 2):
        """
        Train BPE tokenizer on corpus files.
        
        Args:
            files: List of text file paths
            vocab_size: Target vocabulary size
            min_frequency: Minimum token frequency
        """
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        except ImportError:
            raise ImportError("Please install: pip install tokenizers")
        
        # Create BPE tokenizer
        self._tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()
        
        # Train with special tokens
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        
        self._tokenizer.train(files, trainer)
        self._vocab_size = self._tokenizer.get_vocab_size()
        
        print(f"Trained BPE tokenizer: {self._vocab_size} vocab")
    
    def save(self, path: str):
        """Save tokenizer to file."""
        if self._tokenizer:
            self._tokenizer.save(path)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(path)
        self._vocab_size = self._tokenizer.get_vocab_size()
    
    def encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self._tokenizer.encode(text).ids
    
    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self._tokenizer.decode(ids)


def get_tokenizer_for_scale(scale: str, corpus_path: str = None) -> GroundThinkTokenizer:
    """
    Get appropriate tokenizer for model scale.
    
    Per Section 2.9:
    - 8M prototype: char-level or 16k BPE
    - 30M test: 16k BPE  
    - 125M target: 24k BPE
    
    Args:
        scale: "8M", "30M", or "125M"
        corpus_path: Path to training corpus (for BPE training)
    """
    if scale == "8M":
        # Char-level for quick architecture validation
        if corpus_path:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return CharTokenizer(text)
        else:
            import string
            return CharTokenizer(string.printable)
    
    elif scale in ("30M", "125M"):
        vocab_sizes = {"30M": 16000, "125M": 24000}
        tok = BPETokenizer()
        if corpus_path:
            tok.train([corpus_path], vocab_size=vocab_sizes[scale])
        return tok
    
    else:
        raise ValueError(f"Unknown scale: {scale}. Use '8M', '30M', or '125M'")
