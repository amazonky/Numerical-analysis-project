import json
from dataclasses import dataclass
from typing import List, Optional

from .runner import run_pipeline


@dataclass
class EvalCase:
    csv: str
    question: str
    table: str = "data"
    expect_sql_contains: Optional[List[str]] = None
    expect_min_rows: Optional[int] = None


@dataclass
class EvalResult:
    case: EvalCase
    success: bool
    error: Optional[str]
    sql: str
    row_count: Optional[int]


def load_cases(path: str) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    csv=obj["csv"],
                    question=obj["question"],
                    table=obj.get("table", "data"),
                    expect_sql_contains=obj.get("expect_sql_contains"),
                    expect_min_rows=obj.get("expect_min_rows"),
                )
            )
    return cases


def run_eval(
    *,
    cases_path: str,
    model: str,
    log_db: Optional[str],
    limit: int = 50,
    max_repairs: int = 1,
) -> List[EvalResult]:
    cases = load_cases(cases_path)
    results: List[EvalResult] = []

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

        success = res.error is None and res.safe
        if success and case.expect_sql_contains:
            success = all(substr.lower() in res.sql.lower() for substr in case.expect_sql_contains)
        if success and case.expect_min_rows is not None and res.df is not None:
            success = len(res.df) >= case.expect_min_rows

        results.append(
            EvalResult(
                case=case,
                success=success,
                error=res.error,
                sql=res.sql,
                row_count=len(res.df) if res.df is not None else None,
            )
        )
    return results

