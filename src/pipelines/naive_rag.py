from src.generation.generate import generate_answer

class NaiveRAGPipeline:
    def __init__(self):
        pass
        
    def query(self, user_query):
        """
        Accepts a user query, retrieves top chunks, and generates an answer.
        Returns a dict containing the generated 'answer' and the retrieved 'contexts'.
        """
        return generate_answer(user_query)

if __name__ == "__main__":
    # Initialize pipeline
    rag = NaiveRAGPipeline()
    
    # Run test queries
    test_queries = [
        "What are the official working hours?",
        "ما هي سياسة الحضور؟",
    ]
    
    for q in test_queries:
        safe_q = q.encode('ascii', errors='replace').decode('ascii')
        print(f"\n========================================\nQuery: {safe_q}")
        res = rag.query(q)
        if res:
            # Safely encode/decode to avoid Windows console errors
            ans = res["answer"].encode('ascii', errors='replace').decode('ascii')
            print(f"Answer:\n{ans}")
            print(f"Retrieved {len(res['contexts'])} context chunks.")
        else:
            print("Failed to run RAG query.")
