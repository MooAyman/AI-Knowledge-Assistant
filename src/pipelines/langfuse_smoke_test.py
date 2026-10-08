from dotenv import load_dotenv
from langfuse import get_client

from src.pipelines.advanced_rag import ADVANCED_PIPELINE_TAG, AdvancedRAGPipeline


def _safe_print(text: str) -> None:
    print(text.encode("ascii", errors="replace").decode("ascii"))


def run_langfuse_smoke_test() -> None:
    load_dotenv()

    query = "What are the official working hours and working days?"
    pipeline = AdvancedRAGPipeline(candidate_k=20, top_k=5)

    _safe_print("Running Advanced RAG Langfuse smoke test...")
    _safe_print(f"Query: {query}")
    _safe_print(f"Expected trace tag: {ADVANCED_PIPELINE_TAG}")

    result = pipeline.query(query)
    get_client().flush()

    _safe_print("Langfuse flush completed.")
    _safe_print(f"Top context chunk_id: {result['contexts'][0]['chunk_id']}")
    _safe_print(f"Answer preview: {result['answer'][:120]}")


if __name__ == "__main__":
    run_langfuse_smoke_test()
