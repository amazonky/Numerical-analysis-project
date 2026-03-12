import json
import re
from typing import Optional, TypedDict

import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph
from sqlglot import exp

from .prompts import DECOMPOSER_PROMPT, EXPLAIN_PROMPT, GENERATOR_PROMPT, SELECTOR_PROMPT
from .repair import repair_sql
from .safety import SqlValidationError, is_safe, normalize_sql, validate_with_sqlglot


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
    selected_columns: list[str]
    selected_schema_txt: str
    plan_steps: list[str]
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


def _extract_json_obj(raw: str) -> dict:
    text = raw.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def build_graph():
    g = StateGraph(State)

    def _extract_date_predicates(sql: str, date_cols: set[str]):
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

    def selector(state: State):
        llm = state["llm"]
        raw = _call_llm(
            llm,
            SELECTOR_PROMPT.format(
                question=state["question"],
                schema=state["schema_txt"],
            ),
        )
        obj = _extract_json_obj(raw)
        selected_columns = obj.get("selected_columns") or []
        selected_schema = obj.get("selected_schema") or state["schema_txt"]

        if not isinstance(selected_columns, list):
            selected_columns = []
        selected_columns = [str(c).strip() for c in selected_columns if str(c).strip()]

        if not isinstance(selected_schema, str) or not selected_schema.strip():
            selected_schema = state["schema_txt"]

        return {
            "selected_columns": selected_columns,
            "selected_schema_txt": selected_schema,
            "repair_attempts": 0,
        }

    def decomposer(state: State):
        llm = state["llm"]
        raw = _call_llm(
            llm,
            DECOMPOSER_PROMPT.format(
                question=state["question"],
                selected_schema=state.get("selected_schema_txt") or state["schema_txt"],
            ),
        )
        obj = _extract_json_obj(raw)
        plan_steps = obj.get("plan_steps") or []
        if not isinstance(plan_steps, list) or not plan_steps:
            plan_steps = [
                "Interpret question intent",
                "Map required fields to schema",
                "Compose SQL with proper filters and aggregations",
                "Validate SQL semantics",
            ]
        plan_steps = [str(s).strip() for s in plan_steps if str(s).strip()]
        return {"plan_steps": plan_steps}

    def generator(state: State):
        llm = state["llm"]
        plan_text = "\n".join(f"- {s}" for s in state.get("plan_steps", []))
        raw_sql = _call_llm(
            llm,
            GENERATOR_PROMPT.format(
                table_name=state["table_name"],
                selected_schema=state.get("selected_schema_txt") or state["schema_txt"],
                stats=state["stats_txt"] or "(no numeric preview available)",
                plan=plan_text or "- Create a valid query",
                question=state["question"],
            ),
        )
        sql = normalize_sql(raw_sql, state["table_name"])
        return {"sql": sql}

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
                    pass
            try:
                df = con.execute(sql).fetchdf()
            except Exception as exc:
                error = str(exc)

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

    def needs_refine(state: State):
        return state.get("error") is not None and state.get("repair_attempts", 0) < state.get("max_repairs", 0)

    def refiner(state: State):
        attempts = state.get("repair_attempts", 0) + 1
        plan_text = "\n".join(f"- {s}" for s in state.get("plan_steps", []))
        sql = repair_sql(
            state["llm"],
            table_name=state["table_name"],
            schema=state.get("selected_schema_txt") or state["schema_txt"],
            question=state["question"],
            plan=plan_text,
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
            ),
        ).strip()
        return {"explanation": explanation}

    g.add_node("selector", selector)
    g.add_node("decomposer", decomposer)
    g.add_node("generator", generator)
    g.add_node("validate_and_execute", validate_and_execute)
    g.add_node("refiner", refiner)
    g.add_node("explain", explain)

    g.set_entry_point("selector")
    g.add_edge("selector", "decomposer")
    g.add_edge("decomposer", "generator")
    g.add_edge("generator", "validate_and_execute")
    g.add_conditional_edges(
        "validate_and_execute",
        needs_refine,
        {
            True: "refiner",
            False: "explain",
        },
    )
    g.add_edge("refiner", "validate_and_execute")
    g.add_edge("explain", END)

    return g.compile()
