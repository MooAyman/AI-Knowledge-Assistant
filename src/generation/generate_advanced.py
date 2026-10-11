from dotenv import load_dotenv
from langfuse import observe
from openai import OpenAI

load_dotenv()


@observe(name="generation", as_type="generation")
def generate_answer_from_contexts(query: str, contexts: list[dict]) -> dict:
    context_text = ""
    for doc in contexts:
        metadata = doc.get("metadata") or {}
        source = doc.get("source") or metadata.get("source", "unknown")
        section_title = doc.get("section_title") or metadata.get("section_title")

        header = f"--- Context Source: {source} ---"
        if section_title:
            header += f"\nSection Title: {section_title}"

        context_text += f"\n{header}\n{doc['content']}\n"

    system_instruction = (
        "You are a helpful assistant for answering questions from an organization's documents. "
        "Answer the user's question using ONLY the provided contexts below. "
        "Do not use external knowledge. If the answer is not contained within the contexts, reply with: "
        "'Information not found in the guide.' "
        "Ensure you answer in the same language as the user's question (either English or Arabic)."
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Contexts:\n{context_text}\n\nQuestion: {query}",
            },
        ],
        temperature=0.0,
    )

    return {
        "answer": response.choices[0].message.content,
        "contexts": contexts,
    }
