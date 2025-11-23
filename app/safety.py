import re

# Simple regex to ensure we only run SELECT statements
SAFE_SQL = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)


def normalize_sql(raw: str, table: str) -> str:
    s = raw.strip()
    m = re.search(r"(?is)\bselect\b[\s\S]*$", s)
    if m:
        s = m.group(0)
    # Remove code fences or 'sql' labels
    s = re.sub(r"^```\s*sql\s*|^```|```\s*$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Drop trailing semicolon if any
    s = s.rstrip().rstrip(";").strip()
    # Replace placeholder table names
    s = re.sub(r"\byour_table\b", table, s, flags=re.IGNORECASE)
    s = re.sub(r"\bmy_table\b", table, s, flags=re.IGNORECASE)
    # DuckDB interval syntax: INTERVAL 6 WEEK (no quotes)
    s = re.sub(r"INTERVAL\s*'\s*(\d+)\s*week\s*'", r"INTERVAL \1 WEEK", s, flags=re.IGNORECASE)
    # Prefer current_date over DATE 'now'
    s = re.sub(r"DATE\s*'\s*now\s*'", "current_date", s, flags=re.IGNORECASE)
    return s


def is_safe(sql: str) -> bool:
    if ";" in sql.strip().rstrip(";"):
        return False
    if not SAFE_SQL.match(sql.strip()):
        return False
    banned = ("insert", "update", "delete", "drop", "alter", "create", "attach", "pragma", "grant", "revoke", "copy")
    return not any(b in sql.lower() for b in banned)


def clean_sql(s: str) -> str:
    s = s.strip()
    # strip code fences
    s = re.sub(r"^```sql\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"^```|\s*```$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^sql\s*", "", s, flags=re.IGNORECASE)
    return s.strip().rstrip(";")

