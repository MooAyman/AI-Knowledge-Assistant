# Evaluation

RAGAS runners and metric summaries live here.

Local files that should not be published:

- `benchmark.jsonl` — evaluation questions for a specific knowledge base
- `*_results.jsonl` — per-question outputs, including retrieved document text

To evaluate on your own corpus, copy `benchmark.example.jsonl` to `benchmark.jsonl` and replace the questions with ones that match your documents.

```bash
copy evaluation\benchmark.example.jsonl evaluation\benchmark.jsonl
python evaluation/run_ragas.py
python evaluation/run_ragas_advanced.py
```

`naive_summary.json` and `advanced_summary.json` contain aggregate RAGAS scores only. They do not include source-document text.
