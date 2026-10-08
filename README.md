# Ask AMAN HR

Bilingual (English / Arabic) Retrieval-Augmented Generation assistant for HR and onboarding questions.

Policy answers usually live in a long handbook, so people hunt through pages for working hours, attendance, tracks, and evaluation rules. This project turns a local handbook into a grounded Q&A assistant: retrieve the relevant passages first, then answer only from those passages, in the same language as the question.

This is Week 2 of an applied AI engineering track. It compares a naive RAG baseline with an advanced pipeline, and measures both with RAGAS.

The original knowledge base used during development is **not published**. Clone the repo, add your own `.docx` handbook, and rebuild the index locally.

## What it does

- Answers HR / onboarding questions from a local knowledge base
- Replies in the same language as the question (English or Arabic)
- Refuses to guess when the answer is not in the retrieved context
- Lets you switch between **Naive RAG** and **Advanced RAG** in a Chainlit UI

## Architecture

```
DOCX handbook
  → text extraction
  → chunking
  → embeddings
  → retrieval
  → LLM answer (grounded on retrieved chunks)
```

### Naive RAG

1. Extract text from a DOCX handbook
2. Split into overlapping character windows (500 chars, 50 overlap)
3. Embed with `intfloat/multilingual-e5-small` via Hugging Face
4. Store vectors in local Qdrant
5. Retrieve top-5 by cosine similarity
6. Generate with `gpt-4o-mini`, using only retrieved context

### Advanced RAG

1. Structure-aware chunking: headings, EN/AR paragraph pairs, and list blocks stay together
2. Hybrid retrieval: BM25 + dense embeddings (`alpha = 0.6`)
3. Arabic query normalization before retrieval (alef variants, taa marbuta, diacritics)
4. Cross-encoder reranking with `BAAI/bge-reranker-v2-m3` (20 candidates → top 5)
5. Same grounded generation prompt
6. Langfuse tracing on retrieval, rerank, and generation

## Evaluation (RAGAS)

The published summaries are aggregate scores only. The original question set and retrieved passages are private.

Benchmark used during development: **40** questions (20 English, 20 Arabic), **36** answerable, **4** deliberately unanswerable.

| Pipeline     | Faithfulness | Answer relevancy | Context precision | Context recall |
| ------------ | -----------: | ---------------: | ----------------: | -------------: |
| Naive RAG    |         0.92 |             0.67 |              0.66 |           0.76 |
| Advanced RAG |         0.84 |             0.70 |                 — |         **0.97** |

Advanced retrieval improved context recall a lot. Faithfulness did not automatically follow. That gap is the main engineering takeaway: better retrieval is not the same as a better answer.

To run evaluation on your own documents, copy `evaluation/benchmark.example.jsonl` to `evaluation/benchmark.jsonl` and edit the questions.

## Tech stack

- **UI:** Chainlit
- **Generation:** OpenAI `gpt-4o-mini`
- **Embeddings:** Hugging Face Inference API, `intfloat/multilingual-e5-small`
- **Vector store (naive):** Qdrant (local)
- **Hybrid retrieval:** BM25 + dense cosine fusion
- **Reranker:** `BAAI/bge-reranker-v2-m3`
- **Observability:** Langfuse
- **Evaluation:** RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`)

## Project structure

```
app.py                          Chainlit entry point
src/config.py                   Local path configuration
src/ingestion/                  DOCX extract + naive/better chunking
src/embeddings/                 Hugging Face embeddings
src/retrieval/                  Qdrant, hybrid BM25+dense, rerank
src/generation/                 Grounded LLM prompts
src/pipelines/                  Naive + advanced orchestration
src/text/arabic.py              Arabic normalization
evaluation/                     RAGAS runners, example benchmark, score summaries
data/raw/                       Your source DOCX (local only)
data/processed/                 Generated text, chunks, embeddings (local only)
data/qdrant/                    Local vector DB (local only)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env`. Never commit that file.

### Build the knowledge base

Put your handbook at `data/raw/hr_guide.docx` (or set `RAW_DOC_PATH`), then:

```bash
python src/ingestion/extract.py
python src/ingestion/chunk.py
python src/embeddings/embed.py
python src/retrieval/store.py
python src/ingestion/chunk_better.py
```

Better embeddings for the advanced pipeline are created on first hybrid retrieval.

### Run the app

```bash
chainlit run app.py
```

In the UI settings panel, switch between Naive RAG and Advanced RAG.

### Evaluate

```bash
copy evaluation\benchmark.example.jsonl evaluation\benchmark.jsonl
python evaluation/run_ragas.py
python evaluation/run_ragas_advanced.py
```

## Privacy

This repository publishes the RAG implementation, not the original handbook.

Not included:

- Source HR / onboarding documents
- Extracted text, chunks, embeddings, and the Qdrant store
- Full evaluation questions and per-question retrieved contexts
- API keys

Answers from this assistant are not official HR advice.

## License

Educational / portfolio project.
