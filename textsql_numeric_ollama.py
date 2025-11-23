import argparse
import sys

from app.runner import format_result, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--table", default="data", help="DuckDB table name")
    ap.add_argument("--q", required=True, help="Natural-language question")
    ap.add_argument("--limit", type=int, default=50, help="Row limit for printing")
    ap.add_argument("--model", default="llama3:8b-instruct-q4_K_M", help="Ollama model name")
    ap.add_argument("--log-db", default=None, help="Optional DuckDB file path to log runs for eval/finetune")
    ap.add_argument("--max-repairs", type=int, default=2, help="Retries with repair prompt when SQL is unsafe or fails")
    args = ap.parse_args()

    result = run_pipeline(
        csv_path=args.csv,
        table_name=args.table,
        question=args.q,
        model=args.model,
        limit=args.limit,
        log_db=args.log_db,
        max_repairs=args.max_repairs,
    )

    print(format_result(result, args.limit))
    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
