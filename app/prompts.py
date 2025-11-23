import textwrap
from langchain.prompts import PromptTemplate

# Core SQL generation prompt
SQL_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You are a data analyst writing ONE safe DuckDB SQL query.

Rules:
- Use ONLY table name: {table_name}
- Output ONLY the SQL (no prose, no code fences, no explanations, no comments, no trailing semicolon)
- SELECT-only (no DDL/DML)
- If dates come from CSV, use: CAST(date AS DATE) AS d in a CTE, then group by date_trunc('week', d)
- When grouping, GROUP BY the derived fields (e.g., d, product) rather than re-calling date_trunc with extra arguments
- For "last N weeks", filter with date >= current_date - INTERVAL N WEEK
- Use current_date (NOT DATE 'now')
- For week-over-week, first aggregate in a subquery/CTE, THEN use LAG on the aggregated results
- If unsure, LIMIT 20

Return ONLY the SQL.
                                                          
Table schema:
{schema}

Sample numeric stats (for reference):
{stats}

User question:
{question}
"""))

# Explanation prompt for the model to describe the result
EXPLAIN_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You are a data analyst. Explain the SQL result in 5 concise bullet points.
- Highlight key trends, outliers, and comparisons.
- Keep it factual; avoid speculation.
- If the sample is small (LIMIT), mention that as a caveat.

Question:
{question}

SQL:
{sql}

Result preview (first rows):
{preview}
"""))

# Repair prompt for unsafe or failing SQL
REPAIR_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You must return one safe DuckDB SELECT query only.

Rules:
- SELECT-only (no DDL/DML)
- No semicolons
- No table names except: {table_name}
- Banned keywords: insert, update, delete, drop, alter, create, attach, pragma, grant, revoke, copy
- Keep current_date over DATE 'now'
- If dates come from CSV, use: CAST(date AS DATE) AS d in a CTE, then group by date_trunc('week', d)
- When grouping, GROUP BY the derived fields (e.g., d, product) rather than re-calling date_trunc with extra arguments
- For "last N weeks", filter with date >= current_date - INTERVAL N WEEK
- Prefer week aggregation before using LAG
- If unsure, LIMIT 20

Schema:
{schema}

Original question:
{question}

Previous SQL:
{previous_sql}

Problem to fix (safety or execution error):
{error}

Return corrected SQL only, no prose, no fences.
"""))
