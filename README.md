# LLM + DuckDB Numeric Analysis (Offline, Ollama)

This project lets you analyze **numeric CSV datasets** with a **local LLM (Ollama)** that writes **SELECT-only SQL**, runs it on **DuckDB**, and explains the results. No cloud, no API keys.

## 1) Prereqs

- macOS with **Ollama** running (menu bar icon). Example models:
  ```bash
  ollama pull llama3:8b-instruct-q4_K_M
  ```
- Python 3.10+

## 2) Setup

```bash
# from repo root (this folder)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Quick test with the sample dataset

```bash
python textsql_numeric_ollama.py --csv data/sample_metrics.csv --q "Weekly average value per product; last 6 weeks; include WoW change."
```

## 4) Use your own data

Put your CSV in `data/` and run:

```bash
python textsql_numeric_ollama.py --csv data/your_file.csv --q "Top 10 products by average monthly revenue and their growth vs previous month."
```

Optional flags:
- `--table` name for DuckDB (default: `data`)
- `--limit` rows to display (default: 50)
- `--model` Ollama model tag (default: `llama3:8b-instruct-q4_K_M`)
- `--log-db` path to a DuckDB file that will store prompts/SQL/results for later eval or fine-tuning
- `--max-repairs` how many times to retry with the repair prompt if SQL is unsafe or fails (default: 1)

## 5) Offline evaluation

You can run a small eval suite of questions using the new eval runner. Cases are newline-delimited JSON, e.g. `data/eval_cases.example.jsonl`.

```bash
python eval_runner.py --cases data/eval_cases.example.jsonl --log-db logs.duckdb
```

## 6) Notes

- The script **only allows SELECT** queries for safety.
- If the model outputs invalid SQL, you'll see the SQL and an error; re-run with a simpler question or clarify columns.
- For charts, copy the SQL and plot in a notebook, or extend the script.
