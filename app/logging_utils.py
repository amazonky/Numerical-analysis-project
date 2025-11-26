import duckdb
from typing import Optional


def log_run(
    log_db_path: Optional[str],
    *,
    csv_path: str,
    table_name: str,
    question: str,
    model: str,
    sql: str,
    safe: bool,
    execution_error: Optional[str],
    preview: Optional[str],
    row_count: Optional[int],
    schema_summary: Optional[str],
    stats_preview: Optional[str],
    repair_attempts: int,
    duration_ms: Optional[float],
) -> None:
    """
    Persist run metadata to a DuckDB file for later analysis/evaluation.
    """
    if not log_db_path:
        return

    con = duckdb.connect(log_db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id BIGINT,
            ts TIMESTAMPTZ DEFAULT current_timestamp,
            csv_path TEXT,
            table_name TEXT,
            question TEXT,
            model TEXT,
            sql TEXT,
            safe BOOLEAN,
            execution_error TEXT,
            preview TEXT,
            row_count INTEGER,
            schema_summary TEXT,
            stats_preview TEXT,
            repair_attempts INTEGER,
            duration_ms DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO runs (
            csv_path, table_name, question, model, sql, safe, execution_error,
            preview, row_count, schema_summary, stats_preview, repair_attempts, duration_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            csv_path,
            table_name,
            question,
            model,
            sql,
            safe,
            execution_error,
            preview,
            row_count,
            schema_summary,
            stats_preview,
            repair_attempts,
            duration_ms,
        ],
    )

    # Also persist successful safe SQL into a compact table for reuse/fine-tuning
    if safe and not execution_error:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS safe_queries (
                id BIGINT,
                question TEXT,
                sql TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO safe_queries (question, sql) VALUES (?, ?)",
            [question, sql],
        )

    con.close()
