from typing import Any

from src.pipelines.advanced_rag import AdvancedRAGPipeline, _safe_print
from src.retrieval.hybrid_retrieve import hybrid_retrieve
from src.text.arabic import prepare_arabic_query


def _top_ids(results: list[dict[str, Any]], limit: int = 5) -> list[int]:
    return [item["chunk_id"] for item in results[:limit]]


def _print_top_results(label: str, results: list[dict[str, Any]]) -> None:
    _safe_print(f"\n{label}:")
    for rank, item in enumerate(results[:5], start=1):
        _safe_print(
            f"  rank={rank} chunk_id={item['chunk_id']} "
            f"combined={item['combined_score']:.4f} "
            f"section={item.get('section_title', '')}"
        )


def run_arabic_retrieval_smoke_test() -> None:
    test_queries = [
        "ما هي ساعات العمل الرسمية؟",
        "ما هي سياسة الحضور؟",
        "كيف يتم تسجيل الحضور؟",
    ]

    pipeline = AdvancedRAGPipeline(candidate_k=20, top_k=5)

    for query in test_queries:
        retrieval_query, arabic_meta = prepare_arabic_query(query)
        raw_hybrid = hybrid_retrieve(query=query, top_k=5)
        norm_hybrid = hybrid_retrieve(query=retrieval_query, top_k=5)
        advanced = pipeline.retrieve(query)

        _safe_print(f"\n{'=' * 70}")
        _safe_print(f"Query: {query}")
        _safe_print(f"Retrieval query: {retrieval_query}")
        _safe_print(f"Normalized: {arabic_meta['normalized']}")
        _safe_print(f"Original tokens: {arabic_meta['original_tokens']}")
        _safe_print(f"Retrieval tokens: {arabic_meta['retrieval_tokens']}")

        _print_top_results("Hybrid (raw query, no Arabic prep)", raw_hybrid)
        _print_top_results("Hybrid (normalized retrieval query)", norm_hybrid)
        _print_top_results(
            "Advanced pipeline (normalized hybrid + rerank)",
            advanced["contexts"],
        )

        raw_ids = _top_ids(raw_hybrid)
        norm_ids = _top_ids(norm_hybrid)
        advanced_ids = _top_ids(advanced["contexts"])
        _safe_print(
            f"\nTop-5 chunk_ids -> raw: {raw_ids} | normalized: {norm_ids} | advanced: {advanced_ids}"
        )

        if raw_ids != norm_ids:
            _safe_print(
                "Issue: raw vs normalized hybrid ranking differs "
                "(likely alef/hamza or letter-form mismatch)."
            )
        elif arabic_meta["original_tokens"] != arabic_meta["retrieval_tokens"]:
            _safe_print(
                "Note: tokenization changed after normalization, "
                "but hybrid top-5 stayed the same."
            )
        else:
            _safe_print("Tokenization and hybrid top-5 are stable for this query.")


if __name__ == "__main__":
    run_arabic_retrieval_smoke_test()
