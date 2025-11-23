import time
from dataclasses import dataclass
from typing import Optional

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from tabulate import tabulate

from .logging_utils import log_run
from .prompts import SQL_PROMPT, EXPLAIN_PROMPT
from .repair import repair_sql
from .safety import normalize_sql, is_safe
from .schema_utils import summarize_schema


def _invoke(llm: OllamaLLM, prompt: str) -> str:
    return llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)


@dataclass
class RunResult:
    sql: str
    df: Optional[pd.DataFrame]
    explanation: Optional[str]
    safe: bool
    error: Optional[str]
    repair_attempts: int
    schema_txt: str
    stats_txt: str
    duration_ms: float


def run_pipeline(
    *,
    csv_path: str,
    table_name: str,
    question: str,
    model: str,
    limit: int,
    log_db: Optional[str],
    max_repairs: int = 2,
) -> RunResult:
    start = time.time()
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")

    schema_txt, stats_txt = summarize_schema(con, table_name)
    llm = OllamaLLM(model=model)

    raw_sql = _invoke(
        llm,
        SQL_PROMPT.format(
            table_name=table_name,
            schema=schema_txt,
            stats=stats_txt or "(no numeric preview available)",
            question=question,
        ),
    )
    sql = normalize_sql(raw_sql, table_name)

    repair_attempts = 0
    df: Optional[pd.DataFrame] = None
    error: Optional[str] = None

    while True:
        safe = is_safe(sql)
        if not safe:
            error = "Generated SQL failed safety checks"
        else:
            try:
                df = con.execute(sql).fetchdf()
                error = None
            except Exception as exc:
                error = str(exc)

        if error and repair_attempts < max_repairs:
            repair_attempts += 1
            sql = repair_sql(
                llm,
                table_name=table_name,
                schema=schema_txt,
                question=question,
                previous_sql=sql,
                error=error,
            )
            continue
        break

    duration_ms = (time.time() - start) * 1000
    preview_txt = df.head(min(limit, 10)).to_markdown(index=False) if df is not None else None
    explanation = (
        _invoke(
            llm,
            EXPLAIN_PROMPT.format(
                question=question,
                sql=sql,
                preview=preview_txt or "(no rows)",
            ),
        ).strip()
        if df is not None
        else None
    )

    # Persist log for eval/finetune
    log_run(
        log_db,
        csv_path=csv_path,
        table_name=table_name,
        question=question,
        model=model,
        sql=sql,
        safe=error is None and safe is True,
        execution_error=error,
        preview=preview_txt,
        row_count=len(df) if df is not None else None,
        schema_summary=schema_txt,
        stats_preview=stats_txt,
        repair_attempts=repair_attempts,
        duration_ms=duration_ms,
    )

    return RunResult(
        sql=sql,
        df=df,
        explanation=explanation,
        safe=error is None and safe is True,
        error=error,
        repair_attempts=repair_attempts,
        schema_txt=schema_txt,
        stats_txt=stats_txt,
        duration_ms=duration_ms,
    )


def format_result(result: RunResult, limit: int) -> str:
    """
    Render the result to a human-friendly string for CLI output.
    """
    lines = []
    lines.append("\n---- SQL ----")
    lines.append(result.sql)

    if result.error:
        lines.append("\n---- ERROR ----")
        lines.append(str(result.error))
    else:
        lines.append(f"\n---- RESULT (up to {limit} rows) ----")
        lines.append(
            tabulate(result.df.head(limit), headers="keys", tablefmt="github", showindex=False)  # type: ignore[arg-type]
        )
        if result.explanation:
            lines.append("\n---- EXPLANATION ----")
            lines.append(result.explanation)
    return "\n".join(lines)
