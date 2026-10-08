import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings.embed import get_embedding


load_dotenv()


WORD_RE = re.compile(r"[A-Za-z0-9_\u0600-\u06FF]+")


@dataclass
class BM25Index:
    tokenized_docs: list[list[str]]
    doc_len: list[int]
    avg_doc_len: float
    term_doc_freq: dict[str, int]
    total_docs: int


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def load_better_chunks(path: str = "data/processed/better_chunks.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_better_embeddings(
    chunks_path: str = "data/processed/better_chunks.json",
    embedded_path: str = "data/processed/better_embedded_chunks.json",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    if os.path.exists(embedded_path):
        with open(embedded_path, "r", encoding="utf-8") as f:
            return json.load(f)

    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY environment variable not found.")

    chunks = load_better_chunks(chunks_path)
    embedded_chunks: list[dict[str, Any]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        embeddings = get_embedding(texts, api_key=api_key)

        for chunk, emb in zip(batch, embeddings):
            embedded_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "embedding": emb,
                }
            )

    os.makedirs(os.path.dirname(embedded_path), exist_ok=True)
    with open(embedded_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    return embedded_chunks


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Index:
    tokenized_docs: list[list[str]] = []
    doc_len: list[int] = []
    term_doc_freq: dict[str, int] = {}

    for chunk in chunks:
        tokens = tokenize(chunk["content"])
        tokenized_docs.append(tokens)
        doc_len.append(len(tokens))

        seen = set(tokens)
        for token in seen:
            term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

    total_docs = len(chunks)
    avg_doc_len = sum(doc_len) / total_docs if total_docs else 0.0

    return BM25Index(
        tokenized_docs=tokenized_docs,
        doc_len=doc_len,
        avg_doc_len=avg_doc_len,
        term_doc_freq=term_doc_freq,
        total_docs=total_docs,
    )


def bm25_scores(
    query: str,
    index: BM25Index,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    query_tokens = tokenize(query)
    scores = [0.0 for _ in range(index.total_docs)]

    for i, doc_tokens in enumerate(index.tokenized_docs):
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in query_tokens:
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue

            df = index.term_doc_freq.get(term, 0)
            idf = math.log(((index.total_docs - df + 0.5) / (df + 0.5)) + 1.0)

            numerator = term_tf * (k1 + 1.0)
            denominator = term_tf + k1 * (1.0 - b + b * (index.doc_len[i] / max(index.avg_doc_len, 1e-9)))
            score += idf * (numerator / denominator)

        scores[i] = score

    return scores


def get_query_embedding(query: str, model: str = "intfloat/multilingual-e5-small") -> list[float]:
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY environment variable not found.")

    api_url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": [f"query: {query}"],
        "options": {"wait_for_model": True},
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"HF Inference API error: {resp.status_code} - {resp.text}")

    data = resp.json()
    if not isinstance(data, list) or not data:
        raise Exception("Unexpected embedding response for query.")

    return data[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    alpha: float = 0.6,
    chunks_path: str = "data/processed/better_chunks.json",
    embedded_path: str = "data/processed/better_embedded_chunks.json",
) -> list[dict[str, Any]]:
    chunks = load_better_chunks(chunks_path)
    embedded_chunks = ensure_better_embeddings(chunks_path=chunks_path, embedded_path=embedded_path)

    if len(chunks) != len(embedded_chunks):
        raise ValueError("Mismatch between better chunks and embedded chunks count.")

    bm25_index = build_bm25_index(chunks)
    bm25_raw = bm25_scores(query, bm25_index)

    query_embedding = get_query_embedding(query)
    semantic_raw = [cosine_similarity(query_embedding, item["embedding"]) for item in embedded_chunks]

    bm25_norm = normalize_scores(bm25_raw)
    semantic_norm = normalize_scores(semantic_raw)

    fused = []
    for i, chunk in enumerate(chunks):
        combined_score = alpha * semantic_norm[i] + (1.0 - alpha) * bm25_norm[i]
        fused.append(
            {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "source": chunk["metadata"].get("source"),
                "section_title": chunk["metadata"].get("section_title"),
                "bm25_score": bm25_norm[i],
                "semantic_score": semantic_norm[i],
                "combined_score": combined_score,
            }
        )

    fused.sort(key=lambda x: x["combined_score"], reverse=True)
    return fused[:top_k]


def run_basic_hybrid_test() -> None:
    test_query = "What are the official working hours and working days?"
    results = hybrid_retrieve(query=test_query, top_k=5)

    print("Hybrid retrieval smoke test")
    print(f"Query: {test_query}")
    print(f"Top results: {len(results)}")
    for i, item in enumerate(results, start=1):
        print(
            f"{i}. chunk_id={item['chunk_id']} combined={item['combined_score']:.4f} "
            f"semantic={item['semantic_score']:.4f} bm25={item['bm25_score']:.4f}"
        )
        print(f"   section={item['section_title']}")


if __name__ == "__main__":
    run_basic_hybrid_test()
