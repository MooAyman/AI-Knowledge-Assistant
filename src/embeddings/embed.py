import os
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_embedding(texts, api_key, model="intfloat/multilingual-e5-small"):
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # E5 models expect "passage: " prefix for document text chunks
    prefixed_texts = [f"passage: {text}" for text in texts]
    
    payload = {
        "inputs": prefixed_texts,
        "options": {"wait_for_model": True}
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"HF Inference API error: {response.status_code} - {response.text}")
        
    return response.json()

def embed_chunks(input_path, output_path):
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY environment variable not found.")
        
    print(f"Reading chunks from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    print("Generating embeddings via HF Inference API...")
    batch_size = 16
    embedded_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_texts = [chunk["content"] for chunk in batch]
        
        print(f"Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        for attempt in range(5):
            try:
                embeddings = get_embedding(batch_texts, api_key)
                if isinstance(embeddings, list) and len(embeddings) == len(batch):
                    break
                else:
                    raise Exception(f"Unexpected response format: {embeddings}")
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == 4:
                    raise e
                time.sleep(5)
                
        for chunk, embedding in zip(batch, embeddings):
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            })
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Saved embedded chunks to {output_path}")

if __name__ == "__main__":
    input_file = "data/processed/naive_chunks.json"
    output_file = "data/processed/naive_embedded_chunks.json"
    embed_chunks(input_file, output_file)
