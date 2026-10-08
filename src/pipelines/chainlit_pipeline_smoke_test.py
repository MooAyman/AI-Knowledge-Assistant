from dotenv import load_dotenv

from app import PIPELINE_ADVANCED, PIPELINE_NAIVE, create_pipeline

load_dotenv()


def _safe_print(text: str) -> None:
    print(text.encode("ascii", errors="replace").decode("ascii"))


def run_chainlit_pipeline_smoke_test() -> None:
    query = "What are the official working hours?"

    for pipeline_name in (PIPELINE_NAIVE, PIPELINE_ADVANCED):
        _safe_print(f"\n{'=' * 60}")
        _safe_print(f"Pipeline: {pipeline_name}")
        _safe_print(f"Query: {query}")

        pipeline = create_pipeline(pipeline_name)
        result = pipeline.query(query)

        if not result or "answer" not in result:
            raise RuntimeError(f"{pipeline_name} returned no answer.")

        _safe_print(f"Top chunk_id: {result['contexts'][0]['chunk_id']}")
        _safe_print(f"Answer: {result['answer']}")

    _safe_print("\nChainlit pipeline smoke test passed.")


if __name__ == "__main__":
    run_chainlit_pipeline_smoke_test()
