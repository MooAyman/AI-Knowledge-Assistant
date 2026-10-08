from typing import Any

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes

from src.generation.generate_advanced import generate_answer_from_contexts
from src.retrieval.hybrid_retrieve import hybrid_retrieve
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, rerank_candidates
from src.text.arabic import prepare_arabic_query

load_dotenv()

ADVANCED_PIPELINE_TAG = "pipeline:advanced"


class AdvancedRAGPipeline:
    def __init__(
        self,
        candidate_k: int = 20,
        top_k: int = 5,
        alpha: float = 0.6,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.alpha = alpha
        self.reranker_model = reranker_model

    @observe(name="hybrid-retrieval")
    def _run_hybrid_retrieval(self, retrieval_query: str) -> list[dict[str, Any]]:
        return hybrid_retrieve(
            query=retrieval_query,
            top_k=self.candidate_k,
            alpha=self.alpha,
        )

    @observe(name="reranking")
    def _run_reranking(
        self,
        retrieval_query: str,
        hybrid_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return rerank_candidates(
            query=retrieval_query,
            candidates=hybrid_results,
            top_k=self.top_k,
            model_name=self.reranker_model,
        )

    @observe(name="retrieval")
    def retrieve(self, user_query: str) -> dict[str, Any]:
        retrieval_query, arabic_meta = prepare_arabic_query(user_query)
        hybrid_results = self._run_hybrid_retrieval(retrieval_query)
        reranked_results = self._run_reranking(retrieval_query, hybrid_results)
        return {
            "hybrid_candidates": hybrid_results,
            "contexts": reranked_results,
            "arabic_meta": arabic_meta,
        }

    @observe(name="advanced-rag-query")
    def query(self, user_query: str) -> dict[str, Any]:
        with propagate_attributes(
            tags=[ADVANCED_PIPELINE_TAG],
            metadata={"pipeline": "advanced"},
        ):
            retrieval = self.retrieve(user_query)
            generation = generate_answer_from_contexts(
                query=user_query,
                contexts=retrieval["contexts"],
            )
            return {
                **generation,
                "hybrid_candidates": retrieval["hybrid_candidates"],
            }


def _safe_print(text: str) -> None:
    print(text.encode("ascii", errors="replace").decode("ascii"))


def _print_ranking_comparison(
    query: str,
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> None:
    _safe_print(f"\n{'=' * 60}\nQuery: {query}")
    _safe_print("\nBefore reranking (hybrid top 5):")
    for rank, item in enumerate(hybrid_results[:5], start=1):
        _safe_print(
            f"  rank={rank} chunk_id={item['chunk_id']} "
            f"combined={item['combined_score']:.4f} "
            f"section={item.get('section_title', '')}"
        )

    _safe_print("\nAfter reranking (BGE top 5):")
    for item in reranked_results:
        _safe_print(
            f"  rank={item['post_rerank_rank']} (was #{item['pre_rerank_rank']}) "
            f"chunk_id={item['chunk_id']} rerank={item['rerank_score']:.4f} "
            f"combined={item['combined_score']:.4f} section={item.get('section_title', '')}"
        )


def run_rerank_test() -> None:
    test_queries = [
        "What are the official working hours and working days?",
        "What is the attendance policy?",
        "ما هي سياسة الحضور؟",
    ]

    pipeline = AdvancedRAGPipeline(candidate_k=20, top_k=5)

    for query in test_queries:
        retrieval = pipeline.retrieve(query)
        _print_ranking_comparison(
            query=query,
            hybrid_results=retrieval["hybrid_candidates"],
            reranked_results=retrieval["contexts"],
        )

        result = generate_answer_from_contexts(query, retrieval["contexts"])
        _safe_print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    run_rerank_test()
