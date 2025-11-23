import duckdb


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

