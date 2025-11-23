from typing import Optional, TypedDict

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph

from .logging_utils import log_run
from .prompts import EXPLAIN_PROMPT, SQL_PROMPT
from .repair import repair_sql
from .safety import is_safe, normalize_sql, validate_with_sqlglot, SqlValidationError
from .schema_utils import summarize_schema


class State(TypedDict, total=False):
    csv_path: str
    table_name: str
    question: str
    model: str
    limit: int
    log_db: Optional[str]
    max_repairs: int
    llm: OllamaLLM
    con: duckdb.DuckDBPyConnection
    schema_txt: str
    stats_txt: str
    sql: str
    safe: bool
    df: Optional[pd.DataFrame]
    error: Optional[str]
    repair_attempts: int
    explanation: Optional[str]
    duration_ms: float


def _call_llm(llm: OllamaLLM, prompt: str) -> str:
    return llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)


def build_graph():
    g = StateGraph(State)

    def generate(state: State):
        llm = state["llm"]
        raw_sql = _call_llm(
            llm,
            SQL_PROMPT.format(
                table_name=state["table_name"],
                schema=state["schema_txt"],
                stats=state["stats_txt"] or "(no numeric preview available)",
                question=state["question"],
            )
        )
        sql = normalize_sql(raw_sql, state["table_name"])
        return {"sql": sql, "repair_attempts": 0}

    def validate_and_execute(state: State):
        sql = state["sql"]
        table = state["table_name"]
        error = None
        safe = False
        df = None
        try:
            sql = validate_with_sqlglot(sql, table=table)
            safe = is_safe(sql)
            if not safe:
                error = "Generated SQL failed safety checks"
        except SqlValidationError as exc:
            error = str(exc)

        if not error and safe:
            try:
                df = state["con"].execute(sql).fetchdf()
            except Exception as exc:
                error = str(exc)

        return {"sql": sql, "safe": safe, "df": df, "error": error}

    def needs_repair(state: State):
        return state.get("error") is not None and state.get("repair_attempts", 0) < state.get("max_repairs", 0)

    def repair(state: State):
        attempts = state.get("repair_attempts", 0) + 1
        sql = repair_sql(
            state["llm"],
            table_name=state["table_name"],
            schema=state["schema_txt"],
            question=state["question"],
            previous_sql=state["sql"],
            error=state.get("error") or "unknown error",
        )
        return {"sql": sql, "error": None, "df": None, "safe": False, "repair_attempts": attempts}

    def explain(state: State):
        if state.get("df") is None:
            return {}
        preview_txt = state["df"].head(min(state["limit"], 10)).to_markdown(index=False)
        explanation = _call_llm(
            state["llm"],
            EXPLAIN_PROMPT.format(
                question=state["question"],
                sql=state["sql"],
                preview=preview_txt or "(no rows)",
            )
        ).strip()
        return {"explanation": explanation}

    g.add_node("generate", generate)
    g.add_node("validate_and_execute", validate_and_execute)
    g.add_node("repair", repair)
    g.add_node("explain", explain)

    g.set_entry_point("generate")
    g.add_edge("generate", "validate_and_execute")
    g.add_conditional_edges(
        "validate_and_execute",
        needs_repair,
        {
            True: "repair",
            False: "explain",
        },
    )
    g.add_edge("repair", "validate_and_execute")
    g.add_edge("explain", END)

    return g.compile()
