"""
Download training data for 5M model - no dependencies except requests
Mix: TinyStories (40%), OpenWebText sample (30%), BookCorpus sample (20%), Dialogue (10%)
"""

import os
import json
import random
import hashlib
from urllib.request import urlopen, Request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, dest, chunk_size=8192):
    """Download a file with progress"""
    print(f"Downloading {url}...")
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req) as response:
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {downloaded:,} / {total:,} bytes ({pct}%)", end='', flush=True)
        print()

def get_tinystories():
    """Download TinyStories - perfect for small models"""
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    dest = DATA_DIR / "tinystories.txt"
    if not dest.exists():
        download_file(url, dest)
    
    print("Processing TinyStories...")
    texts = []
    with open(dest, 'r', encoding='utf-8', errors='ignore') as f:
        story = []
        for line in f:
            if line.strip() == '<|endoftext|>':
                if story:
                    text = ' '.join(story).strip()
                    if 50 < len(text) < 2000:  # Filter by length
                        texts.append(text)
                    story = []
            else:
                story.append(line.strip())
    print(f"  Got {len(texts):,} stories")
    return texts

def get_gutenberg_books():
    """Get public domain books from Project Gutenberg - expanded list"""
    books = [
        # Classic novels
        ("Pride and Prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
        ("Alice in Wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
        ("Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
        ("Moby Dick", "https://www.gutenberg.org/files/2701/2701-0.txt"),
        ("Great Expectations", "https://www.gutenberg.org/files/1400/1400-0.txt"),
        ("Tale of Two Cities", "https://www.gutenberg.org/files/98/98-0.txt"),
        ("Frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
        ("Dracula", "https://www.gutenberg.org/files/345/345-0.txt"),
        ("Jane Eyre", "https://www.gutenberg.org/files/1260/1260-0.txt"),
        ("Wuthering Heights", "https://www.gutenberg.org/files/768/768-0.txt"),
        # More classics
        ("War and Peace", "https://www.gutenberg.org/files/2600/2600-0.txt"),
        ("Crime and Punishment", "https://www.gutenberg.org/files/2554/2554-0.txt"),
        ("The Count of Monte Cristo", "https://www.gutenberg.org/files/1184/1184-0.txt"),
        ("Les Miserables", "https://www.gutenberg.org/files/135/135-0.txt"),
        ("Anna Karenina", "https://www.gutenberg.org/files/1399/1399-0.txt"),
        ("Don Quixote", "https://www.gutenberg.org/files/996/996-0.txt"),
        ("The Brothers Karamazov", "https://www.gutenberg.org/files/28054/28054-0.txt"),
        ("Ulysses", "https://www.gutenberg.org/files/4300/4300-0.txt"),
        ("The Odyssey", "https://www.gutenberg.org/files/1727/1727-0.txt"),
        ("The Iliad", "https://www.gutenberg.org/files/6130/6130-0.txt"),
        # Adventure
        ("Treasure Island", "https://www.gutenberg.org/files/120/120-0.txt"),
        ("Robinson Crusoe", "https://www.gutenberg.org/files/521/521-0.txt"),
        ("The Call of the Wild", "https://www.gutenberg.org/files/215/215-0.txt"),
        ("White Fang", "https://www.gutenberg.org/files/910/910-0.txt"),
        ("The Jungle Book", "https://www.gutenberg.org/files/236/236-0.txt"),
        ("Around the World in 80 Days", "https://www.gutenberg.org/files/103/103-0.txt"),
        ("20000 Leagues Under the Sea", "https://www.gutenberg.org/files/164/164-0.txt"),
        ("The Time Machine", "https://www.gutenberg.org/files/35/35-0.txt"),
        ("The War of the Worlds", "https://www.gutenberg.org/files/36/36-0.txt"),
        ("The Invisible Man", "https://www.gutenberg.org/files/5230/5230-0.txt"),
        # American Literature
        ("The Scarlet Letter", "https://www.gutenberg.org/files/25344/25344-0.txt"),
        ("The Adventures of Tom Sawyer", "https://www.gutenberg.org/files/74/74-0.txt"),
        ("Huckleberry Finn", "https://www.gutenberg.org/files/76/76-0.txt"),
        ("The Great Gatsby", "https://www.gutenberg.org/files/64317/64317-0.txt"),
        ("Little Women", "https://www.gutenberg.org/files/514/514-0.txt"),
        # More variety
        ("The Picture of Dorian Gray", "https://www.gutenberg.org/files/174/174-0.txt"),
        ("The Strange Case of Dr Jekyll and Mr Hyde", "https://www.gutenberg.org/files/43/43-0.txt"),
        ("A Christmas Carol", "https://www.gutenberg.org/files/46/46-0.txt"),
        ("Oliver Twist", "https://www.gutenberg.org/files/730/730-0.txt"),
        ("David Copperfield", "https://www.gutenberg.org/files/766/766-0.txt"),
        ("Sense and Sensibility", "https://www.gutenberg.org/files/161/161-0.txt"),
        ("Emma", "https://www.gutenberg.org/files/158/158-0.txt"),
        ("Northanger Abbey", "https://www.gutenberg.org/files/121/121-0.txt"),
        ("Persuasion", "https://www.gutenberg.org/files/105/105-0.txt"),
        ("The Hound of the Baskervilles", "https://www.gutenberg.org/files/2852/2852-0.txt"),
    ]
    
    texts = []
    for name, url in books:
        dest = DATA_DIR / f"gutenberg_{name.replace(' ', '_').lower()}.txt"
        if not dest.exists():
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"  Failed to download {name}: {e}")
                continue
        
        print(f"Processing {name}...")
        with open(dest, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Skip header/footer (Gutenberg boilerplate)
        start = content.find("*** START OF")
        end = content.find("*** END OF")
        if start > 0 and end > start:
            content = content[start:end]
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            para = ' '.join(para.split())  # Normalize whitespace
            if 100 < len(para) < 2000:
                texts.append(para)
    
    print(f"  Got {len(texts):,} book paragraphs")
    return texts

def get_simple_wikipedia():
    """Get Simple Wikipedia dumps - educational, simple language"""
    # Using a smaller sample from HF
    url = "https://huggingface.co/datasets/wikipedia/resolve/main/20220301.simple/train-00000-of-00001.parquet"
    dest = DATA_DIR / "simple_wiki.parquet"
    
    texts = []
    if not dest.exists():
        print("Simple Wikipedia requires pyarrow - using fallback...")
        # Fallback: use some educational text from a simpler source
        fallback_texts = [
            "The sun is a star at the center of our solar system. It provides light and heat to Earth.",
            "Water is made of hydrogen and oxygen atoms. It is essential for all life on Earth.",
            "Plants use sunlight to make food through photosynthesis. They take in carbon dioxide and release oxygen.",
            "The moon orbits Earth once every 27 days. It has no atmosphere and no liquid water.",
            "Electricity flows through wires like water flows through pipes. It powers our homes and devices.",
        ]
        return fallback_texts * 100  # Repeat to have some volume
    
    return texts

def get_dialogue_samples():
    """Generate some conversational patterns"""
    # Since we can't easily get dialogue datasets, let's create templates
    # These teach the model conversational structure
    
    dialogues = [
        "What do you think about that? I think it's quite interesting, actually. The way things work together.",
        "Have you ever wondered why the sky is blue? It's because of how light scatters in the atmosphere.",
        "Tell me more about your ideas. Well, I've been thinking about how we learn from experience.",
        "That's a good point. I hadn't considered it from that angle before. Let me think about it.",
        "How does this work? It's actually simpler than it looks. You just need to understand the basics.",
        "I disagree with that view. Here's why I think differently about it.",
        "What happened next? Things took an unexpected turn. Nobody saw it coming.",
        "Can you explain that again? Of course. Let me try a different approach this time.",
        "Why do you believe that? Because the evidence points in that direction. Consider this example.",
        "That reminds me of something. There's a connection here that's worth exploring.",
    ]
    
    # Expand with variations
    expanded = []
    starters = ["So,", "Well,", "Actually,", "Honestly,", "Interestingly,", "You know,", "The thing is,"]
    middles = ["I was thinking", "it occurred to me", "I noticed", "I realized", "I've observed"]
    
    for d in dialogues:
        expanded.append(d)
        for s in starters[:3]:
            expanded.append(f"{s} {d}")
    
    print(f"  Got {len(expanded):,} dialogue samples")
    return expanded

def create_mixed_dataset(target_samples=50000):
    """Create a mixed dataset - NO DUPLICATES, use natural proportions"""
    
    print("\n=== Downloading and Processing Data ===\n")
    
    # Get all sources
    tinystories = get_tinystories()
    books = get_gutenberg_books() 
    
    # Use all unique data we have, no repeating
    print(f"\n=== Available Data ===")
    print(f"  TinyStories: {len(tinystories):,}")
    print(f"  Books: {len(books):,}")
    
    # Calculate natural proportions based on what we have
    total_available = len(tinystories) + len(books)
    
    # Target proportions: use what we have, up to target
    # Stories are abundant, books are precious - use all books
    n_books = min(len(books), target_samples // 2)  # Up to 50% books
    n_stories = min(len(tinystories), target_samples - n_books)  # Rest from stories
    
    print(f"\n=== Mixing Dataset (no duplicates) ===")
    print(f"  TinyStories: {n_stories:,}")
    print(f"  Books: {n_books:,}")
    print(f"  Total: {n_stories + n_books:,}")
    
    # Sample without replacement
    random.seed(42)
    
    stories_sample = random.sample(tinystories, n_stories)
    books_sample = random.sample(books, n_books) if n_books <= len(books) else books
    
    # Combine and shuffle
    all_texts = stories_sample + books_sample
    random.shuffle(all_texts)
    
    # Save as simple text file (one sample per line)
    output_path = DATA_DIR / "mixed_training_data.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in all_texts:
            # Clean up
            text = ' '.join(text.split())
            f.write(text + '\n')
    
    print(f"\n=== Dataset Created ===")
    print(f"Total samples: {len(all_texts):,}")
    print(f"Saved to: {output_path}")
    
    # Stats
    total_chars = sum(len(t) for t in all_texts)
    avg_len = total_chars // len(all_texts)
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {avg_len} chars")
    print(f"Estimated tokens: ~{total_chars // 4:,}")
    
    return output_path

if __name__ == "__main__":
    create_mixed_dataset(target_samples=200000)
