import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from src.embeddings.embed import get_embedding

# Load environment variables
load_dotenv()

def retrieve_similar_chunks(query, db_path="data/qdrant", collection_name="naive_chunks", top_k=5):
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY environment variable not found.")

    # E5 models expect "query: " prefix for query text
    prefixed_query = f"query: {query}"
    
    # Get embedding for the query
    safe_query = query.encode('ascii', errors='replace').decode('ascii')
    print(f"Embedding query: '{safe_query}'...")
    embeddings = get_embedding([prefixed_query], api_key)
    query_vector = embeddings[0]
    
    # Initialize Qdrant Client
    client = QdrantClient(path=db_path)
    
    # Search for top_k results
    print(f"Retrieving top {top_k} results from collection '{collection_name}'...")
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points
    
    retrieved_docs = []
    for hit in results:
        retrieved_docs.append({
            "score": hit.score,
            "chunk_id": hit.id,
            "content": hit.payload.get("content"),
            "source": hit.payload.get("source")
        })
        
    return retrieved_docs

if __name__ == "__main__":
    test_query = "What are the official working hours?"
    try:
        docs = retrieve_similar_chunks(test_query)
        print("\n--- Retrieval Results ---")
        for i, doc in enumerate(docs):
            print(f"\nResult {i+1} (Score: {doc['score']:.4f}, ID: {doc['chunk_id']}):")
            print(f"Source: {doc['source']}")
            snippet = doc['content'][:150]
            # Replace characters that can't be encoded in console default
            print(f"Content snippet: {snippet.encode('ascii', errors='replace').decode('ascii')}...")
    except Exception as e:
        print(f"Error during retrieval: {e}")
