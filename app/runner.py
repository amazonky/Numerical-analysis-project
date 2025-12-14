import time
from dataclasses import dataclass
from typing import Optional

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from tabulate import tabulate

from .graph_runner import build_graph
from .logging_utils import log_run
from .schema_utils import get_date_bounds, get_date_columns, summarize_schema


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
    date_columns = [c.lower() for c in get_date_columns(con, table_name)]
    date_bounds = get_date_bounds(con, table_name, date_columns)
    llm = OllamaLLM(model=model)

    # Build LangGraph pipeline
    graph = build_graph()
    initial_state = {
        "csv_path": csv_path,
        "table_name": table_name,
        "question": question,
        "model": model,
        "limit": limit,
        "log_db": log_db,
        "max_repairs": max_repairs,
        "llm": llm,
        "con": con,
        "schema_txt": schema_txt,
        "stats_txt": stats_txt or "(no numeric preview available)",
        "date_columns": date_columns,
        "date_bounds": date_bounds,
    }

    state = graph.invoke(initial_state)

    sql = state.get("sql", "")
    df = state.get("df")
    error = state.get("error")
    explanation = state.get("explanation")
    repair_attempts = state.get("repair_attempts", 0)
    safe_flag = error is None and state.get("safe", False)

    duration_ms = (time.time() - start) * 1000
    preview_txt = df.head(min(limit, 10)).to_markdown(index=False) if df is not None else None

    # Persist log for eval/finetune
    log_run(
        log_db,
        csv_path=csv_path,
        table_name=table_name,
        question=question,
        model=model,
        sql=sql,
        safe=safe_flag,
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
        safe=safe_flag,
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
        if result.df is not None and not result.df.empty:
            lines.append(
                tabulate(result.df.head(limit), headers="keys", tablefmt="github", showindex=False)  # type: ignore[arg-type]
            )
        else:
            lines.append("(no rows)")
        if result.explanation:
            lines.append("\n---- EXPLANATION ----")
            lines.append(result.explanation)
    return "\n".join(lines)
