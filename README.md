# AI Knowledge Assistant

A reusable English/Arabic Retrieval-Augmented Generation system for questions over a local document collection.

Organizations often keep answers in long handbooks, policies, manuals, and FAQs. This project turns those files into a grounded Q&A assistant: retrieve the relevant passages first, then answer only from those passages, in the same language as the question.

It compares a **Naive RAG** baseline with an **Advanced RAG** pipeline and measures both with RAGAS.

Bring your own documents. The knowledge base used during development is not included.

## What it is for

Use it on internal material such as:

- HR documents and company policies
- Manuals, guides, and procedures
- FAQs and internal knowledge bases
- Bilingual (Arabic/English) organizational documents
- Other internal business documentation

## What it does

- Answers questions from a local knowledge base you provide
- Replies in the same language as the question (English or Arabic)
- Refuses to guess when the answer is not in the retrieved context
- Lets you switch between **Naive RAG** and **Advanced RAG** in a Chainlit UI

## Architecture

```
Your DOCX document
  → text extraction
  → chunking
  → embeddings
  → retrieval
  → LLM answer (grounded on retrieved chunks)
```

### Naive RAG

1. Extract text from a DOCX file
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

Published files include aggregate scores only. The original question set and retrieved passages are not in this repository.

Benchmark used during development: **40** questions (20 English, 20 Arabic), **36** answerable, **4** deliberately unanswerable, on a private bilingual organizational handbook.

| Pipeline     | Faithfulness | Answer relevancy | Context precision | Context recall |
| ------------ | -----------: | ---------------: | ----------------: | -------------: |
| Naive RAG    |         0.92 |             0.67 |              0.66 |           0.76 |
| Advanced RAG |         0.84 |             0.70 |                 — |         **0.97** |

Advanced retrieval improved context recall substantially. Faithfulness did not automatically follow. The takeaway: better retrieval is not the same as a better answer.

To evaluate on your own documents, copy `evaluation/benchmark.example.jsonl` to `evaluation/benchmark.jsonl` and replace the questions with ones that match your corpus.

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

## Use your own documents

1. Copy `.env.example` to `.env` and add your API keys. Never commit `.env`.
2. Place a DOCX file at `data/raw/knowledge_base.docx`, or set `RAW_DOC_PATH` in `.env`.
3. Build the index (commands below).
4. Run the Chainlit app and ask questions in English or Arabic.
5. Optionally copy `evaluation/benchmark.example.jsonl` to `evaluation/benchmark.jsonl` and run RAGAS against your own questions.

Supported source format for ingestion is **DOCX**. Multiple organizations can reuse the same code; only the local files and `.env` paths change.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # macOS/Linux: cp .env.example .env
```

Fill in `.env`. Never commit that file.

### Build the knowledge base

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
# macOS/Linux: cp evaluation/benchmark.example.jsonl evaluation/benchmark.jsonl
python evaluation/run_ragas.py
python evaluation/run_ragas_advanced.py
```

## Privacy

This repository publishes the RAG implementation, not any organization's source documents.

Not included:

- Source documents
- Extracted text, chunks, embeddings, and the Qdrant store
- Full evaluation questions and per-question retrieved contexts
- API keys

Answers are generated from retrieved passages only and are not official organizational advice.

## License

Educational / portfolio project. Adapt it to your own documents under your own policies and access controls.
