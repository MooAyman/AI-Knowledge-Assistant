from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.jsonl"
RESULTS_PATH = ROOT / "evaluation" / "advanced_results.jsonl"
SUMMARY_PATH = ROOT / "evaluation" / "advanced_summary.json"


def ensure_ragas_importable() -> None:
    try:
        import ragas  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "langchain_community.chat_models.vertexai":
            raise

    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module
    import ragas  # noqa: F401


ensure_ragas_importable()

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from src.pipelines.advanced_rag import AdvancedRAGPipeline


def build_ragas_metrics() -> list[Any]:
    openai_client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=openai_client)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    m_faithfulness = deepcopy(faithfulness)
    m_answer_relevancy = deepcopy(answer_relevancy)
    m_context_recall = deepcopy(context_recall)

    m_faithfulness.llm = llm
    m_answer_relevancy.llm = llm
    m_answer_relevancy.embeddings = embeddings
    m_context_recall.llm = llm

    return [m_faithfulness, m_answer_relevancy, m_context_recall]


def load_benchmark() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with BENCHMARK_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_single_turn_record(item: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    retrieved_contexts = [
        safe_text(doc.get("content", ""))
        for doc in generated.get("contexts", [])
        if isinstance(doc, dict)
    ]

    return {
        "user_input": safe_text(item.get("question", "")),
        "response": safe_text(generated.get("answer", "")),
        "retrieved_contexts": retrieved_contexts,
        "reference": safe_text(item.get("gold_answer", "")),
    }


def aggregate_scores(result: Any) -> dict[str, float]:
    if hasattr(result, "_repr_dict"):
        return {
            key: float(value)
            for key, value in result._repr_dict.items()
            if value is not None
        }
    if hasattr(result, "scores"):
        score_rows = result.scores
        if score_rows:
            return {
                key: float(value)
                for key, value in score_rows[0].items()
                if isinstance(value, (int, float)) and value is not None
            }
    return {}


def run_evaluation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_dotenv(ROOT / ".env", override=False)
    rows = load_benchmark()
    pipeline = AdvancedRAGPipeline()

    result_rows: list[dict[str, Any]] = []
    answerable_dataset: list[dict[str, Any]] = []

    for item in rows:
        question = safe_text(item.get("question", ""))
        unanswerable = bool(item.get("unanswerable"))
        gold_answer = safe_text(item.get("gold_answer", ""))

        try:
            generated = pipeline.query(question)
        except Exception as exc:
            result_rows.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": None,
                    "retrieved_contexts": [],
                    "evaluated": False,
                    "status": "error",
                    "error": str(exc),
                    "metric_scores": {},
                }
            )
            continue

        generated_answer = safe_text(
            generated.get("answer", "") if isinstance(generated, dict) else ""
        )
        retrieved_contexts = []
        if isinstance(generated, dict):
            for doc in generated.get("contexts", []) or []:
                if isinstance(doc, dict):
                    retrieved_contexts.append(
                        {
                            "chunk_id": doc.get("chunk_id"),
                            "content": safe_text(doc.get("content", "")),
                            "source": safe_text(doc.get("source", "")),
                            "score": doc.get("score"),
                        }
                    )

        record = {
            "id": item.get("id"),
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "evaluated": False,
            "status": "ok",
            "error": None,
            "metric_scores": {},
        }

        if unanswerable:
            record["status"] = "skipped_unanswerable"
            record["evaluated"] = False
            record["metric_scores"] = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
            }
            result_rows.append(record)
            continue

        answerable_dataset.append(build_single_turn_record(item, generated or {}))
        record["evaluated"] = True
        result_rows.append(record)

    summary: dict[str, Any] = {
        "pipeline": "advanced",
        "total_questions": len(rows),
        "answerable_questions": len(answerable_dataset),
        "unanswerable_questions": sum(1 for item in rows if item.get("unanswerable")),
        "failed_questions": sum(1 for item in result_rows if item.get("status") == "error"),
        "metrics": {},
        "failed_question_ids": [],
    }

    if answerable_dataset:
        dataset = EvaluationDataset.from_list(answerable_dataset)
        result = evaluate(
            dataset,
            metrics=build_ragas_metrics(),
            show_progress=False,
        )
        summary["metrics"] = aggregate_scores(result)

        for item in result_rows:
            if item.get("evaluated"):
                item["metric_scores"] = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_recall": None,
                }
                matches = [
                    row for row in answerable_dataset if row["user_input"] == item["question"]
                ]
                if not matches:
                    continue
                idx = answerable_dataset.index(matches[0])
                if hasattr(result, "scores"):
                    score_row = result.scores[idx] if idx < len(result.scores) else {}
                    item["metric_scores"] = {
                        key: value
                        for key, value in score_row.items()
                        if isinstance(value, (int, float))
                    }

        summary["metrics"] = {
            "faithfulness": float(summary["metrics"].get("faithfulness", 0.0) or 0.0),
            "answer_relevancy": float(summary["metrics"].get("answer_relevancy", 0.0) or 0.0),
            "context_recall": float(summary["metrics"].get("context_recall", 0.0) or 0.0),
        }

    summary["failed_question_ids"] = [
        item["id"] for item in result_rows if item.get("status") == "error"
    ]

    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        for item in result_rows:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return result_rows, summary


def main() -> None:
    result_rows, summary = run_evaluation()
    print("Advanced RAG RAGAS evaluation complete")
    print(f"total_questions={summary['total_questions']}")
    print(f"answerable={summary['answerable_questions']}")
    print(f"unanswerable={summary['unanswerable_questions']}")
    print(f"failed={summary['failed_questions']}")
    for name in ["faithfulness", "answer_relevancy", "context_recall"]:
        value = summary.get("metrics", {}).get(name)
        print(f"{name}={value if value is not None else 'n/a'}")
    if summary.get("failed_question_ids"):
        print(f"failed_question_ids={summary['failed_question_ids']}")
    print(f"results={RESULTS_PATH}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
