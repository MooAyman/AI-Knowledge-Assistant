import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def store_in_qdrant(input_path, db_path, collection_name="naive_chunks"):
    print(f"Reading embedded chunks from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        embedded_chunks = json.load(f)
        
    if not embedded_chunks:
        print("No chunks to process.")
        return
        
    print(f"Loaded {len(embedded_chunks)} embedded chunks.")
    
    # Determine the vector dimension dynamically from the first chunk
    first_embedding = embedded_chunks[0]["embedding"]
    vector_dim = len(first_embedding)
    print(f"Detected vector dimension: {vector_dim}")
    
    # Initialize persistent local Qdrant client
    os.makedirs(db_path, exist_ok=True)
    client = QdrantClient(path=db_path)
    
    # Recreate collection if we want to update it, or ensure it exists
    # To make it safe to run again without duplicates and start clean:
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Recreating it to ensure clean write...")
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection_name}'.")
    
    # Prepare points
    points = []
    for chunk in embedded_chunks:
        # payload structure containing original details
        payload = {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "source": chunk["metadata"].get("source", "unknown"),
            "chunk_size": chunk["metadata"].get("chunk_size", 0)
        }
        
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload=payload
            )
        )
        
    # Upload points
    print(f"Uploading {len(points)} points to Qdrant...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Verification
    collection_info = client.get_collection(collection_name)
    points_count = collection_info.points_count
    print(f"Verification: Collection contains {points_count} points.")
    if points_count == len(embedded_chunks):
        print("Success! All chunks successfully stored.")
    else:
        print(f"Warning: Expected {len(embedded_chunks)} points, but found {points_count}.")

if __name__ == "__main__":
    input_file = "data/processed/naive_embedded_chunks.json"
    database_path = "data/qdrant"
    store_in_qdrant(input_file, database_path)
