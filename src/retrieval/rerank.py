from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    model = get_reranker(model_name)
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = model.predict(pairs)

    reranked: list[dict[str, Any]] = []
    for rank, (candidate, score) in enumerate(
        zip(candidates, scores), start=1
    ):
        item = dict(candidate)
        item["pre_rerank_rank"] = rank
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["post_rerank_rank"] = rank

    return reranked[:top_k]
