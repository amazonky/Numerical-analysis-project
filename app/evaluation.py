import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from .benchmarking import compare_results, normalize_sql_for_match, result_hash, score_sql_components
from .prompts import EXPLAIN_PROMPT, GENERATOR_PROMPT, REPAIR_PROMPT
from .runner import run_pipeline


@dataclass
class EvalCase:
    case_id: str
    csv: str
    question: str
    table: str = "data"
    split: str = "unspecified"
    tags: List[str] = field(default_factory=list)
    expect_sql_contains: Optional[List[str]] = None
    expect_min_rows: Optional[int] = None
    gold_sql: Optional[str] = None
    gold_result_csv: Optional[str] = None
    gold_result_hash: Optional[str] = None
    order_sensitive: bool = False


@dataclass
class EvalResult:
    case_id: str
    split: str
    question: str
    repetition: int
    success: bool
    safe: bool
    error: Optional[str]
    sql: str
    row_count: Optional[int]
    duration_ms: Optional[float]
    failure_category: str
    sql_exact_match: Optional[bool]
    execution_match: Optional[bool]
    sql_component_f1: Dict[str, float]
    result_hash_pred: Optional[str]
    result_hash_gold: Optional[str]
    details: Dict[str, Any]


@dataclass
class CaseValidation:
    valid: bool
    warnings: List[str]
    errors: List[str]


@dataclass
class BenchmarkSummary:
    total_case_evals: int
    unique_cases: int
    repetitions: int
    passed_case_evals: int
    successful_execution_rate: float
    failed_execution_count: int
    success_rate_ci95: Optional[Tuple[float, float]]
    execution_accuracy: Optional[float]
    execution_accuracy_ci95: Optional[Tuple[float, float]]
    syntax_error_rate: float
    correction_success_rate: Optional[float]
    sql_exact_match_rate: Optional[float]
    sql_exact_match_ci95: Optional[Tuple[float, float]]
    avg_duration_ms: Optional[float]
    std_duration_ms: Optional[float]
    component_f1_macro: Dict[str, float]
    failure_breakdown: Dict[str, int]


@dataclass
class BenchmarkRun:
    run_id: str
    created_at_utc: str
    metadata: Dict[str, Any]
    case_validation: CaseValidation
    summary: BenchmarkSummary
    results: List[EvalResult]


def _prompt_hashes() -> Dict[str, str]:
    import hashlib

    def h(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    return {
        # Backward-compatible key name kept for existing reports/consumers.
        "sql_prompt_sha256": h(GENERATOR_PROMPT.template),
        "repair_prompt_sha256": h(REPAIR_PROMPT.template),
        "explain_prompt_sha256": h(EXPLAIN_PROMPT.template),
    }


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _execute_gold_sql(case: EvalCase) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE {case.table} AS SELECT * FROM read_csv_auto('{case.csv}')")
    return con.execute(case.gold_sql or "").fetchdf()


def _load_gold_df(case: EvalCase) -> Optional[pd.DataFrame]:
    if case.gold_sql:
        return _execute_gold_sql(case)
    if case.gold_result_csv:
        return pd.read_csv(case.gold_result_csv)
    return None


def _failure_category(
    *,
    safe: bool,
    error: Optional[str],
    sql_exact_match: Optional[bool],
    execution_match: Optional[bool],
    contains_ok: Optional[bool],
    min_rows_ok: Optional[bool],
) -> str:
    if error:
        low = error.lower()
        if "parse" in low or "sqlglot" in low:
            return "parse_error"
        if "safety" in low or "only select" in low:
            return "safety_rejection"
        return "execution_error"
    if not safe:
        return "unsafe"
    if sql_exact_match is False:
        return "sql_exact_mismatch"
    if execution_match is False:
        return "execution_mismatch"
    if contains_ok is False:
        return "expectation_sql_contains_mismatch"
    if min_rows_ok is False:
        return "expectation_min_rows_mismatch"
    return "ok"


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def load_cases(path: str) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=obj.get("case_id", f"case_{idx:04d}"),
                    csv=obj["csv"],
                    question=obj["question"],
                    table=obj.get("table", "data"),
                    split=obj.get("split", "unspecified"),
                    tags=obj.get("tags", []),
                    expect_sql_contains=obj.get("expect_sql_contains"),
                    expect_min_rows=obj.get("expect_min_rows"),
                    gold_sql=obj.get("gold_sql"),
                    gold_result_csv=obj.get("gold_result_csv"),
                    gold_result_hash=obj.get("gold_result_hash"),
                    order_sensitive=bool(obj.get("order_sensitive", False)),
                )
            )
    return cases


def validate_cases(cases: List[EvalCase]) -> CaseValidation:
    warnings: List[str] = []
    errors: List[str] = []

    seen: set[str] = set()
    for c in cases:
        if c.case_id in seen:
            errors.append(f"duplicate case_id: {c.case_id}")
        seen.add(c.case_id)

        if not Path(c.csv).exists():
            errors.append(f"case {c.case_id}: csv path missing -> {c.csv}")
        if c.gold_result_csv and not Path(c.gold_result_csv).exists():
            errors.append(f"case {c.case_id}: gold_result_csv missing -> {c.gold_result_csv}")
        if not (c.gold_sql or c.gold_result_csv or c.gold_result_hash):
            warnings.append(f"case {c.case_id}: no gold target (SQL/result/hash), accuracy metrics limited")
        if not c.question.strip():
            errors.append(f"case {c.case_id}: empty question")

    return CaseValidation(valid=(len(errors) == 0), warnings=warnings, errors=errors)


def run_benchmark(
    *,
    cases_path: str,
    model: str,
    log_db: Optional[str],
    limit: int = 50,
    max_repairs: int = 1,
    split: Optional[str] = None,
    repetitions: int = 1,
    strict_cases: bool = False,
) -> BenchmarkRun:
    cases = load_cases(cases_path)
    if split:
        cases = [c for c in cases if c.split == split]

    case_validation = validate_cases(cases)
    if strict_cases and not case_validation.valid:
        raise ValueError("Invalid benchmark cases: " + "; ".join(case_validation.errors))

    started = datetime.now(timezone.utc)
    run_id = started.strftime("bench_%Y%m%dT%H%M%SZ")

    results: List[EvalResult] = []
    for rep in range(1, max(1, repetitions) + 1):
        for case in cases:
            res = run_pipeline(
                csv_path=case.csv,
                table_name=case.table,
                question=case.question,
                model=model,
                limit=limit,
                log_db=log_db,
                max_repairs=max_repairs,
            )

            sql_exact_match: Optional[bool] = None
            component_f1: Dict[str, float] = {}
            if case.gold_sql:
                sql_exact_match = normalize_sql_for_match(res.sql) == normalize_sql_for_match(case.gold_sql)
                comp = score_sql_components(res.sql, case.gold_sql)
                component_f1 = {k: round(v.f1, 6) for k, v in comp.items()}

            gold_df = _load_gold_df(case)
            execution_match: Optional[bool] = None
            result_hash_gold: Optional[str] = None
            result_hash_pred = result_hash(res.df, order_sensitive=case.order_sensitive)
            execution_detail = None
            if gold_df is not None:
                cmp = compare_results(res.df, gold_df, order_sensitive=case.order_sensitive)
                execution_match = cmp.match
                result_hash_gold = cmp.result_hash_gold
                execution_detail = cmp.detail

            if case.gold_result_hash is not None and result_hash_pred is not None:
                hash_match = result_hash_pred == case.gold_result_hash
                execution_match = hash_match if execution_match is None else (execution_match and hash_match)
                if hash_match is False:
                    execution_detail = "predicted result hash differs from gold_result_hash"

            contains_ok: Optional[bool] = None
            if case.expect_sql_contains:
                contains_ok = all(substr.lower() in res.sql.lower() for substr in case.expect_sql_contains)

            min_rows_ok: Optional[bool] = None
            if case.expect_min_rows is not None:
                min_rows_ok = res.df is not None and len(res.df) >= case.expect_min_rows

            success = res.error is None and res.safe
            if contains_ok is False:
                success = False
            if min_rows_ok is False:
                success = False
            if sql_exact_match is False:
                success = False
            if execution_match is False:
                success = False

            failure = _failure_category(
                safe=res.safe,
                error=res.error,
                sql_exact_match=sql_exact_match,
                execution_match=execution_match,
                contains_ok=contains_ok,
                min_rows_ok=min_rows_ok,
            )

            results.append(
                EvalResult(
                    case_id=case.case_id,
                    split=case.split,
                    question=case.question,
                    repetition=rep,
                    success=success,
                    safe=res.safe,
                    error=res.error,
                    sql=res.sql,
                    row_count=len(res.df) if res.df is not None else None,
                    duration_ms=res.duration_ms,
                    failure_category=failure,
                    sql_exact_match=sql_exact_match,
                    execution_match=execution_match,
                    sql_component_f1=component_f1,
                    result_hash_pred=result_hash_pred,
                    result_hash_gold=result_hash_gold,
                    details={
                        "contains_ok": contains_ok,
                        "min_rows_ok": min_rows_ok,
                        "execution_detail": execution_detail,
                        "repair_attempts": res.repair_attempts,
                    },
                )
            )

    total = len(results)
    passed = sum(1 for r in results if r.success)
    exec_cases = [r for r in results if r.execution_match is not None]
    em_cases = [r for r in results if r.sql_exact_match is not None]
    durations = [r.duration_ms for r in results if r.duration_ms is not None]

    def _mean(v: List[float]) -> Optional[float]:
        if not v:
            return None
        return sum(v) / len(v)

    component_keys = sorted({k for r in results for k in r.sql_component_f1.keys()})
    component_macro = {}
    for key in component_keys:
        vals = [r.sql_component_f1[key] for r in results if key in r.sql_component_f1]
        component_macro[key] = round(_mean(vals) or 0.0, 6)

    failure_breakdown: Dict[str, int] = {}
    for r in results:
        failure_breakdown[r.failure_category] = failure_breakdown.get(r.failure_category, 0) + 1

    success_ci = _wilson_ci(passed, total)
    exec_pass = sum(1 for r in exec_cases if r.execution_match)
    exact_pass = sum(1 for r in em_cases if r.sql_exact_match)
    failed = total - passed
    syntax_errors = failure_breakdown.get("parse_error", 0)
    repair_attempted = [r for r in results if int(r.details.get("repair_attempts") or 0) > 0]
    repair_successes = sum(1 for r in repair_attempted if r.success)

    summary = BenchmarkSummary(
        total_case_evals=total,
        unique_cases=len(cases),
        repetitions=max(1, repetitions),
        passed_case_evals=passed,
        successful_execution_rate=(passed / total) if total else 0.0,
        failed_execution_count=failed,
        success_rate_ci95=success_ci,
        execution_accuracy=(exec_pass / len(exec_cases)) if exec_cases else None,
        execution_accuracy_ci95=_wilson_ci(exec_pass, len(exec_cases)) if exec_cases else None,
        syntax_error_rate=(syntax_errors / total) if total else 0.0,
        correction_success_rate=(repair_successes / len(repair_attempted)) if repair_attempted else None,
        sql_exact_match_rate=(exact_pass / len(em_cases)) if em_cases else None,
        sql_exact_match_ci95=_wilson_ci(exact_pass, len(em_cases)) if em_cases else None,
        avg_duration_ms=_mean(durations),
        std_duration_ms=_std(durations),
        component_f1_macro=component_macro,
        failure_breakdown=failure_breakdown,
    )

    metadata = {
        "model": model,
        "cases_path": cases_path,
        "split": split,
        "limit": limit,
        "max_repairs": max_repairs,
        "log_db": log_db,
        "repetitions": max(1, repetitions),
        "strict_cases": strict_cases,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "prompt_hashes": _prompt_hashes(),
        "created_at_utc": started.isoformat(),
    }

    return BenchmarkRun(
        run_id=run_id,
        created_at_utc=started.isoformat(),
        metadata=metadata,
        case_validation=case_validation,
        summary=summary,
        results=results,
    )


def run_eval(
    *,
    cases_path: str,
    model: str,
    log_db: Optional[str],
    limit: int = 50,
    max_repairs: int = 1,
) -> List[EvalResult]:
    """
    Backward-compatible entry point expected by existing scripts.
    """
    report = run_benchmark(
        cases_path=cases_path,
        model=model,
        log_db=log_db,
        limit=limit,
        max_repairs=max_repairs,
    )
    return report.results


def benchmark_to_dict(report: BenchmarkRun) -> Dict[str, Any]:
    return {
        "run_id": report.run_id,
        "created_at_utc": report.created_at_utc,
        "metadata": report.metadata,
        "case_validation": asdict(report.case_validation),
        "summary": asdict(report.summary),
        "results": [asdict(r) for r in report.results],
    }


def save_benchmark_report(report: BenchmarkRun, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{report.run_id}.json"
    csv_path = out / f"{report.run_id}.csv"

    with open(json_path, "w") as f:
        json.dump(benchmark_to_dict(report), f, indent=2)

    rows = []
    for r in report.results:
        row = asdict(r)
        row["sql_component_f1"] = json.dumps(r.sql_component_f1)
        row["details"] = json.dumps(r.details)
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }
