import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import sqlglot
from sqlglot import exp


@dataclass
class ComponentScore:
    precision: float
    recall: float
    f1: float


@dataclass
class ExecutionComparison:
    match: bool
    row_count_pred: Optional[int]
    row_count_gold: Optional[int]
    result_hash_pred: Optional[str]
    result_hash_gold: Optional[str]
    detail: str


def normalize_sql_for_match(sql: str) -> str:
    s = sql.strip().rstrip(";")
    try:
        parsed = sqlglot.parse_one(s, read="duckdb")
        return parsed.sql(dialect="duckdb", pretty=False).lower().strip()
    except Exception:
        return " ".join(s.lower().split())


def _safe_sql_list(exprs: Iterable[exp.Expression]) -> List[str]:
    out = []
    for e in exprs:
        try:
            out.append(e.sql(dialect="duckdb", pretty=False).lower())
        except Exception:
            out.append(str(e).lower())
    return out


def extract_sql_components(sql: str) -> Dict[str, List[str]]:
    components = {
        "select": [],
        "where": [],
        "group_by": [],
        "order_by": [],
        "joins": [],
        "tables": [],
    }
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return components

    if isinstance(parsed, exp.Select):
        components["select"] = _safe_sql_list(parsed.expressions)

    where = parsed.args.get("where")
    if where is not None:
        components["where"] = [where.this.sql(dialect="duckdb", pretty=False).lower()]

    group = parsed.args.get("group")
    if group is not None:
        components["group_by"] = _safe_sql_list(group.expressions)

    order = parsed.args.get("order")
    if order is not None:
        components["order_by"] = _safe_sql_list(order.expressions)

    components["joins"] = _safe_sql_list(parsed.find_all(exp.Join))
    components["tables"] = sorted({t.name.lower() for t in parsed.find_all(exp.Table)})

    return components


def _f1(pred: Sequence[str], gold: Sequence[str]) -> ComponentScore:
    p = set(pred)
    g = set(gold)
    if not p and not g:
        return ComponentScore(1.0, 1.0, 1.0)
    if not p or not g:
        return ComponentScore(0.0, 0.0, 0.0)

    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    if precision + recall == 0:
        return ComponentScore(0.0, 0.0, 0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return ComponentScore(precision, recall, f1)


def score_sql_components(pred_sql: str, gold_sql: str) -> Dict[str, ComponentScore]:
    pred = extract_sql_components(pred_sql)
    gold = extract_sql_components(gold_sql)
    keys = ["select", "where", "group_by", "order_by", "joins", "tables"]
    return {k: _f1(pred.get(k, []), gold.get(k, [])) for k in keys}


def _normalize_value(v: Any, float_precision: int) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return round(v, float_precision)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _canonical_rows(
    df: pd.DataFrame,
    *,
    order_sensitive: bool,
    float_precision: int,
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cols = [str(c) for c in df.columns]
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append(tuple(_normalize_value(v, float_precision) for v in row))

    if not order_sensitive:
        rows = sorted(rows, key=lambda x: repr(x))
    return cols, rows


def result_hash(
    df: Optional[pd.DataFrame],
    *,
    order_sensitive: bool = False,
    float_precision: int = 6,
) -> Optional[str]:
    if df is None:
        return None
    cols, rows = _canonical_rows(df, order_sensitive=order_sensitive, float_precision=float_precision)
    payload = repr((cols, rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_results(
    pred_df: Optional[pd.DataFrame],
    gold_df: Optional[pd.DataFrame],
    *,
    order_sensitive: bool = False,
    float_precision: int = 6,
) -> ExecutionComparison:
    if pred_df is None:
        return ExecutionComparison(
            match=False,
            row_count_pred=None,
            row_count_gold=len(gold_df) if gold_df is not None else None,
            result_hash_pred=None,
            result_hash_gold=result_hash(gold_df, order_sensitive=order_sensitive, float_precision=float_precision),
            detail="predicted query did not produce a dataframe",
        )
    if gold_df is None:
        return ExecutionComparison(
            match=False,
            row_count_pred=len(pred_df),
            row_count_gold=None,
            result_hash_pred=result_hash(pred_df, order_sensitive=order_sensitive, float_precision=float_precision),
            result_hash_gold=None,
            detail="gold dataframe missing",
        )

    pred_cols, pred_rows = _canonical_rows(pred_df, order_sensitive=order_sensitive, float_precision=float_precision)
    gold_cols, gold_rows = _canonical_rows(gold_df, order_sensitive=order_sensitive, float_precision=float_precision)

    pred_hash = hashlib.sha256(repr((pred_cols, pred_rows)).encode("utf-8")).hexdigest()
    gold_hash = hashlib.sha256(repr((gold_cols, gold_rows)).encode("utf-8")).hexdigest()

    if pred_cols != gold_cols:
        return ExecutionComparison(
            match=False,
            row_count_pred=len(pred_df),
            row_count_gold=len(gold_df),
            result_hash_pred=pred_hash,
            result_hash_gold=gold_hash,
            detail=f"column mismatch: predicted={pred_cols}, gold={gold_cols}",
        )

    match = pred_rows == gold_rows
    detail = "match" if match else "row content mismatch"
    return ExecutionComparison(
        match=match,
        row_count_pred=len(pred_df),
        row_count_gold=len(gold_df),
        result_hash_pred=pred_hash,
        result_hash_gold=gold_hash,
        detail=detail,
    )
