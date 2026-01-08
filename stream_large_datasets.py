"""
Stream large datasets via direct parquet downloads (no rate limits).
Target: 100M+ tokens from Cosmopedia + FineWeb-Edu + UltraChat
"""
import argparse
from pathlib import Path
import requests
import pandas as pd
from io import BytesIO


def count_tokens_approx(text: str) -> int:
    return len(text) // 4


def stream_cosmopedia(target_tokens: int, output_path: Path, min_length: int = 200):
    """Stream Cosmopedia via parquet API"""
    print("\n=== Streaming Cosmopedia ===")
    
    api_url = "https://huggingface.co/api/datasets/HuggingFaceTB/cosmopedia/parquet"
    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        print(f"Failed to get file list: {response.status_code}")
        return 0
    
    parquet_map = response.json()
    priority_splits = ['stories', 'wikihow', 'web_samples_v1', 'auto_math_text']
    
    total_tokens = 0
    samples_written = 0
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for split in priority_splits:
            if total_tokens >= target_tokens:
                break
            if split not in parquet_map:
                continue
            files = parquet_map[split].get('train', [])
            print(f"\n  {split}: {len(files)} files")
            
            for file_url in files[:5]:
                if total_tokens >= target_tokens:
                    break
                try:
                    print(f"    Downloading {file_url.split('/')[-1]}...", end=" ", flush=True)
                    resp = requests.get(file_url, timeout=120)
                    if resp.status_code != 200:
                        print("skip")
                        continue
                    df = pd.read_parquet(BytesIO(resp.content))
                    file_tokens = 0
                    file_samples = 0
                    for _, row in df.iterrows():
                        if total_tokens >= target_tokens:
                            break
                        text = str(row.get('text', ''))
                        if len(text) >= min_length:
                            text = text.strip().replace('\n', '<nl>')
                            tokens = count_tokens_approx(text)
                            f.write(text + '\n')
                            total_tokens += tokens
                            file_tokens += tokens
                            file_samples += 1
                            samples_written += 1
                    print(f"{file_samples:,} samples, {file_tokens:,} tokens")
                except Exception as e:
                    print(f"error: {e}")
    
    print(f"\nCosmopedia total: {samples_written:,} samples, {total_tokens:,} tokens")
    return total_tokens


def stream_fineweb_edu(target_tokens: int, output_path: Path, min_length: int = 200):
    """Stream FineWeb-Edu via parquet API"""
    print("\n=== Streaming FineWeb-Edu ===")
    
    api_url = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet"
    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        print(f"Failed to get file list: {response.status_code}")
        return 0
    
    parquet_map = response.json()
    files = parquet_map.get('sample-10BT', {}).get('train', [])
    if not files:
        print("No sample-10BT files found")
        return 0
    print(f"  Found {len(files)} files")
    
    total_tokens = 0
    samples_written = 0
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for file_url in files[:10]:
            if total_tokens >= target_tokens:
                break
            try:
                print(f"    Downloading {file_url.split('/')[-1]}...", end=" ", flush=True)
                resp = requests.get(file_url, timeout=120)
                if resp.status_code != 200:
                    print("skip")
                    continue
                df = pd.read_parquet(BytesIO(resp.content))
                file_tokens = 0
                file_samples = 0
                for _, row in df.iterrows():
                    if total_tokens >= target_tokens:
                        break
                    text = str(row.get('text', ''))
                    if len(text) >= min_length:
                        text = text.strip().replace('\n', '<nl>')
                        tokens = count_tokens_approx(text)
                        f.write(text + '\n')
                        total_tokens += tokens
                        file_tokens += tokens
                        file_samples += 1
                        samples_written += 1
                print(f"{file_samples:,} samples, {file_tokens:,} tokens")
            except Exception as e:
                print(f"error: {e}")
    
    print(f"\nFineWeb-Edu total: {samples_written:,} samples, {total_tokens:,} tokens")
    return total_tokens


def stream_ultrachat(target_tokens: int, output_path: Path):
    """Stream UltraChat via parquet API"""
    print("\n=== Streaming UltraChat ===")
    
    api_url = "https://huggingface.co/api/datasets/stingning/ultrachat/parquet"
    response = requests.get(api_url, timeout=30)
    if response.status_code != 200:
        print(f"Failed to get file list: {response.status_code}")
        return 0
    
    parquet_map = response.json()
    files = parquet_map.get('default', {}).get('train', [])
    if not files:
        print("No files found")
        return 0
    print(f"  Found {len(files)} files")
    
    total_tokens = 0
    samples_written = 0
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for file_url in files[:10]:
            if total_tokens >= target_tokens:
                break
            try:
                print(f"    Downloading {file_url.split('/')[-1]}...", end=" ", flush=True)
                resp = requests.get(file_url, timeout=120)
                if resp.status_code != 200:
                    print("skip")
                    continue
                df = pd.read_parquet(BytesIO(resp.content))
                file_tokens = 0
                file_samples = 0
                for _, row in df.iterrows():
                    if total_tokens >= target_tokens:
                        break
                    turns = row.get('data', None)
                    if turns is not None:
                        # Handle numpy array or list
                        if hasattr(turns, 'tolist'):
                            turns = turns.tolist()
                        if isinstance(turns, list) and len(turns) > 0:
                            text = ' '.join(str(t) for t in turns).strip().replace('\n', '<nl>')
                            tokens = count_tokens_approx(text)
                            f.write(text + '\n')
                            total_tokens += tokens
                            file_tokens += tokens
                            file_samples += 1
                            samples_written += 1
                print(f"{file_samples:,} samples, {file_tokens:,} tokens")
            except Exception as e:
                print(f"error: {e}")
    
    print(f"\nUltraChat total: {samples_written:,} samples, {total_tokens:,} tokens")
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-tokens', type=int, default=100_000_000)
    parser.add_argument('--output', type=str, default='data/100M_mix.txt')
    parser.add_argument('--sources', type=str, default='cosmopedia,fineweb,ultrachat')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('')
    
    sources = args.sources.split(',')
    tokens_per_source = args.target_tokens // len(sources)
    
    print(f"Target: {args.target_tokens:,} tokens")
    print(f"Sources: {sources}")
    print(f"Per source: ~{tokens_per_source:,} tokens")
    
    total_tokens = 0
    if 'cosmopedia' in sources:
        total_tokens += stream_cosmopedia(tokens_per_source, output_path)
    if 'fineweb' in sources:
        total_tokens += stream_fineweb_edu(tokens_per_source, output_path)
    if 'ultrachat' in sources:
        total_tokens += stream_ultrachat(tokens_per_source, output_path)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"DONE! Total: {total_tokens:,} tokens, {file_size:.1f} MB")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
