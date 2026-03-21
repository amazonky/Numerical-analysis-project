import argparse
import json

from app.evaluation import run_benchmark, save_benchmark_report


def main():
    ap = argparse.ArgumentParser(description="Run research-grade offline benchmark for the text-to-SQL pipeline.")
    ap.add_argument("--cases", required=True, help="Path to JSONL file with benchmark cases")
    ap.add_argument("--model", default="llama3:8b-instruct-q4_K_M", help="Ollama model name")
    ap.add_argument("--log-db", default=None, help="Optional DuckDB file path to log runs")
    ap.add_argument("--limit", type=int, default=50, help="Row limit for executing queries")
    ap.add_argument("--max-repairs", type=int, default=1, help="Retries with repair prompt")
    ap.add_argument("--split", default=None, help="Optional split filter (e.g. dev/test)")
    ap.add_argument("--repetitions", type=int, default=1, help="Number of full benchmark repetitions")
    ap.add_argument("--strict-cases", action="store_true", help="Fail fast on case validation errors")
    ap.add_argument("--output-dir", default="benchmark_reports", help="Directory to write JSON/CSV reports")
    args = ap.parse_args()

    report = run_benchmark(
        cases_path=args.cases,
        model=args.model,
        log_db=args.log_db,
        limit=args.limit,
        max_repairs=args.max_repairs,
        split=args.split,
        repetitions=args.repetitions,
        strict_cases=args.strict_cases,
    )
    paths = save_benchmark_report(report, args.output_dir)

    print(f"Benchmark run: {report.run_id}")
    if report.case_validation.errors:
        print("Case validation errors:")
        for e in report.case_validation.errors:
            print(f"- {e}")
    if report.case_validation.warnings:
        print("Case validation warnings:")
        for w in report.case_validation.warnings:
            print(f"- {w}")

    summary = report.summary
    print(
        f"Case evals: {summary.total_case_evals} "
        f"(unique cases={summary.unique_cases}, repetitions={summary.repetitions})"
    )
    print(f"Passed: {summary.passed_case_evals}")
    print(f"% successful execution: {summary.successful_execution_rate:.3f}")
    print(f"Failed execution count: {summary.failed_execution_count}")
    print(f"Syntax error rate: {summary.syntax_error_rate:.3f}")
    if summary.correction_success_rate is not None:
        print(f"Correction success rate: {summary.correction_success_rate:.3f}")
    else:
        print("Correction success rate: N/A (no repair attempts)")
    if summary.success_rate_ci95 is not None:
        lo, hi = summary.success_rate_ci95
        print(f"% successful execution 95% CI: [{lo:.3f}, {hi:.3f}]")

    if summary.execution_accuracy is not None:
        print(f"Execution accuracy: {summary.execution_accuracy:.3f}")
    if summary.execution_accuracy_ci95 is not None:
        lo, hi = summary.execution_accuracy_ci95
        print(f"Execution accuracy 95% CI: [{lo:.3f}, {hi:.3f}]")

    if summary.sql_exact_match_rate is not None:
        print(f"SQL exact match rate: {summary.sql_exact_match_rate:.3f}")
    if summary.sql_exact_match_ci95 is not None:
        lo, hi = summary.sql_exact_match_ci95
        print(f"SQL exact match 95% CI: [{lo:.3f}, {hi:.3f}]")

    if summary.avg_duration_ms is not None:
        print(f"Avg duration (ms): {summary.avg_duration_ms:.2f}")
    if summary.std_duration_ms is not None:
        print(f"Std duration (ms): {summary.std_duration_ms:.2f}")

    print("Component F1 macro:")
    print(json.dumps(summary.component_f1_macro, indent=2))

    print("Failure breakdown:")
    print(json.dumps(summary.failure_breakdown, indent=2))

    print(f"Saved JSON report: {paths['json']}")
    print(f"Saved CSV report: {paths['csv']}")


if __name__ == "__main__":
    main()
