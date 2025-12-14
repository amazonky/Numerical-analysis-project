import duckdb
from typing import Dict, Iterable, List, Optional, Tuple


def summarize_schema(con: duckdb.DuckDBPyConnection, table: str):
    """
    Return (schema_txt, stats_txt) strings for the given table.
    """
    cols = con.execute(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
    """).fetchdf()

    col_lines = [f"- {r['column_name']} {r['data_type']}" for _, r in cols.iterrows()]
    schema_txt = "\n".join(col_lines) if len(cols) else "(no columns found)"

    # quick numeric stats (first 5 numeric columns)
    num_cols = con.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table}'
          AND upper(data_type) IN ('DOUBLE', 'INTEGER', 'BIGINT', 'DECIMAL', 'REAL', 'HUGEINT', 'SMALLINT')
        ORDER BY ordinal_position
        LIMIT 5
    """).fetchdf()['column_name'].tolist()

    stats_txt = ""
    if num_cols:
        sel = ", ".join([f"avg({c}) AS avg_{c}, min({c}) AS min_{c}, max({c}) AS max_{c}" for c in num_cols])
        try:
            stats = con.execute(f"SELECT {sel} FROM {table}").fetchdf()
            stats_txt = stats.to_string(index=False)
        except Exception:
            stats_txt = "(stats unavailable)"
    return schema_txt, stats_txt


def get_date_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    """
    Return list of column names that are date or timestamp-like.
    """
    df = con.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table}'
          AND upper(data_type) IN ('DATE', 'TIMESTAMP', 'TIMESTAMP WITH TIME ZONE', 'TIME')
        ORDER BY ordinal_position
    """).fetchdf()
    return df["column_name"].tolist()


def get_date_bounds(
    con: duckdb.DuckDBPyConnection,
    table: str,
    date_columns: Iterable[str],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Return min/max (as strings) for each date-like column (column names expected normalized for the query).
    """
    bounds: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for col in date_columns:
        try:
            min_val, max_val = con.execute(
                f"SELECT CAST(min({col}) AS TEXT), CAST(max({col}) AS TEXT) FROM {table}"
            ).fetchone()
            bounds[col.lower()] = (min_val, max_val)
        except Exception:
            bounds[col.lower()] = (None, None)
    return bounds
