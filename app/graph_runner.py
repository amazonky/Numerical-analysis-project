from typing import Optional, TypedDict

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph
from sqlglot import exp

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
    date_columns: list[str]
    date_bounds: dict[str, tuple[Optional[str], Optional[str]]]
    sql: str
    safe: bool
    df: Optional[pd.DataFrame]
    error: Optional[str]
    repair_attempts: int
    explanation: Optional[str]
    date_range_message: Optional[str]
    duration_ms: float


def _call_llm(llm: OllamaLLM, prompt: str) -> str:
    return llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)


def build_graph():
    g = StateGraph(State)

    def _extract_date_predicates(sql: str, date_cols: set[str]):
        """
        Pull out predicates that reference date/time columns so we can test coverage.
        Returns (predicate_sql, referenced_cols) or (None, set()).
        """
        try:
            parsed_expr = exp.parse_one(sql, read="duckdb")
        except Exception:
            return None, set()

        where = parsed_expr.args.get("where")
        if where is None:
            return None, set()

        date_preds = []
        referenced: set[str] = set()

        def collect(expr: exp.Expression):
            if isinstance(expr, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between)):
                cols = {c.name.lower() for c in expr.find_all(exp.Column)}
                matched = cols & date_cols
                if matched:
                    date_preds.append(expr)
                    referenced.update(matched)
                    return
            for child in expr.args.values():
                if isinstance(child, exp.Expression):
                    collect(child)
                elif isinstance(child, list):
                    for ch in child:
                        if isinstance(ch, exp.Expression):
                            collect(ch)

        collect(where)
        if not date_preds:
            return None, set()

        combined = date_preds[0]
        for p in date_preds[1:]:
            combined = exp.and_(combined, p)

        return combined.sql(dialect="duckdb"), referenced

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
        date_range_message = None
        date_filter_sql = None
        date_cols_in_query: set[str] = set()
        try:
            sql = validate_with_sqlglot(sql, table=table)
            safe = is_safe(sql)
            if not safe:
                error = "Generated SQL failed safety checks"
        except SqlValidationError as exc:
            error = str(exc)

        if not error:
            date_filter_sql, date_cols_in_query = _extract_date_predicates(sql, set(c.lower() for c in state.get("date_columns", [])))

        if not error and safe:
            con = state["con"]
            # If the date filter yields zero rows against the table, short-circuit with a friendly message.
            if date_filter_sql:
                try:
                    date_only_count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {date_filter_sql}").fetchone()[0]
                    if date_only_count == 0:
                        bounds_parts = []
                        for col in date_cols_in_query:
                            min_max = state.get("date_bounds", {}).get(col)
                            if min_max and all(v is not None for v in min_max):
                                bounds_parts.append(f"{col}: {min_max[0]} to {min_max[1]}")
                        bounds_txt = "; ".join(bounds_parts) if bounds_parts else None
                        msg = "Date range is not provided in the dataset."
                        if bounds_txt:
                            msg = f"{msg} Available date coverage -> {bounds_txt}"
                        date_range_message = msg
                        return {"sql": sql, "safe": safe, "df": pd.DataFrame(), "error": None, "date_range_message": date_range_message}
                except Exception:
                    # If the diagnostic check fails, fall back to normal execution.
                    pass
            try:
                df = con.execute(sql).fetchdf()
            except Exception as exc:
                error = str(exc)

        # Fallback: if query ran but returned no rows and there was a date filter, emit the coverage message.
        if not error and date_filter_sql and df is not None and df.empty and not date_range_message:
            bounds_parts = []
            for col in date_cols_in_query:
                min_max = state.get("date_bounds", {}).get(col)
                if min_max and all(v is not None for v in min_max):
                    bounds_parts.append(f"{col}: {min_max[0]} to {min_max[1]}")
            if bounds_parts:
                date_range_message = f"Date range is not provided in the dataset. Available date coverage -> {'; '.join(bounds_parts)}"

        return {
            "sql": sql,
            "safe": safe,
            "df": df,
            "error": error,
            "date_range_message": date_range_message,
        }

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
        if state.get("date_range_message"):
            return {"explanation": state["date_range_message"]}
        if state.get("df") is None:
            return {}
        if state["df"].empty:
            return {"explanation": "Query returned no rows."}
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
