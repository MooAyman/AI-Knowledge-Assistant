import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_TEXT_PATH

def naive_chunking(input_path, output_path, chunk_size=500, overlap=50):
    print(f"Reading text from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print(f"Total text length: {len(text)} characters.")
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        chunks.append({
            "chunk_id": chunk_id,
            "content": chunk_content,
            "metadata": {
                "source": os.path.basename(input_path),
                "chunk_size": len(chunk_content)
            }
        })
        
        chunk_id += 1
        start += (chunk_size - overlap)
        if start >= len(text) or len(chunk_content) < chunk_size:
            break
            
    print(f"Created {len(chunks)} chunks.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Saved chunks to {output_path}")

if __name__ == "__main__":
    naive_chunking(PROCESSED_TEXT_PATH, "data/processed/naive_chunks.json")
