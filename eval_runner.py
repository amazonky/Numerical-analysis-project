import argparse

from app.evaluation import run_eval


def main():
    ap = argparse.ArgumentParser(description="Run offline eval cases for the text-to-SQL pipeline.")
    ap.add_argument("--cases", required=True, help="Path to JSONL file with eval cases")
    ap.add_argument("--model", default="llama3:8b-instruct-q4_K_M", help="Ollama model name")
    ap.add_argument("--log-db", default=None, help="Optional DuckDB file path to log eval runs")
    ap.add_argument("--limit", type=int, default=50, help="Row limit for executing queries")
    ap.add_argument("--max-repairs", type=int, default=1, help="Retries with repair prompt")
    args = ap.parse_args()

    results = run_eval(
        cases_path=args.cases,
        model=args.model,
        log_db=args.log_db,
        limit=args.limit,
        max_repairs=args.max_repairs,
    )

    total = len(results)
    passed = sum(1 for r in results if r.success)
    print(f"Eval results: {passed}/{total} passed")
    for idx, r in enumerate(results, 1):
        status = "PASS" if r.success else "FAIL"
        msg = f"[{status}] {idx}. Q: {r.case.question} | SQL: {r.sql}"
        if r.error:
            msg += f" | Error: {r.error}"
        print(msg)


if __name__ == "__main__":
    main()
