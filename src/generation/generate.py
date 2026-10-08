import os
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.retrieve import retrieve_similar_chunks

# Load environment variables
load_dotenv()

def generate_answer(query):
    # Retrieve top 5 relevant chunks
    try:
        retrieved_docs = retrieve_similar_chunks(query)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return None
        
    # Build prompt context
    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        context_text += f"\n--- Context Source: {doc['source']} ---\n{doc['content']}\n"
        
    system_instruction = (
        "You are a helpful assistant for answering questions from an organization's documents. "
        "Answer the user's question using ONLY the provided contexts below. "
        "Do not use external knowledge. If the answer is not contained within the contexts, reply with: "
        "'Information not found in the guide.' "
        "Ensure you answer in the same language as the user's question (either English or Arabic)."
    )
    
    # Initialize OpenAI Client
    # It automatically reads OPENAI_API_KEY from environment variables
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Contexts:\n{context_text}\n\nQuestion: {query}"}
        ],
        temperature=0.0
    )
    
    return {
        "answer": response.choices[0].message.content,
        "contexts": retrieved_docs
    }

if __name__ == "__main__":
    # Small end-to-end tests
    test_queries = [
        "What are the official working hours?",
        "ما هي سياسة الحضور؟",
    ]
    
    for q in test_queries:
        print(f"\n========================================\nQuery: {q}")
        res = generate_answer(q)
        if res:
            # Safely encode/decode to avoid Windows console errors
            ans = res["answer"].encode('ascii', errors='replace').decode('ascii')
            print(f"Answer:\n{ans}")
        else:
            print("Failed to generate answer.")
