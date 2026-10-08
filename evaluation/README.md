# Evaluation

RAGAS runners and metric summaries live here.

Private files are kept locally and ignored by git:

- `benchmark.jsonl` — full evaluation questions for the original knowledge base
- `*_results.jsonl` — per-question outputs, including retrieved document text
- assignment PDFs

To run evaluation on your own corpus, copy `benchmark.example.jsonl` to `benchmark.jsonl` and replace the questions with ones that match your documents.

```bash
copy evaluation\benchmark.example.jsonl evaluation\benchmark.jsonl
python evaluation/run_ragas.py
python evaluation/run_ragas_advanced.py
```

`naive_summary.json` and `advanced_summary.json` contain aggregate RAGAS scores only. They do not include source-document text.
