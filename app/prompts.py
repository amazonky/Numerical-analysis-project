import textwrap
from langchain_core.prompts import PromptTemplate

# Selector agent prompt
SELECTOR_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You are the Selector agent in a multi-agent Text2SQL system.
Select the minimal schema subset needed for the question.

Return strict JSON only:
{{
  "selected_columns": ["col1", "col2"],
  "selected_schema": "- col1 TYPE\\n- col2 TYPE"
}}

Rules:
- Only include columns that exist in the full schema.
- Keep the selected subset concise and sufficient.

Question:
{question}

Full schema:
{schema}
"""))

# Decomposer agent prompt
DECOMPOSER_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You are the Decomposer agent in a multi-agent Text2SQL system.
Create short planning steps for constructing SQL.

Return strict JSON only:
{{
  "plan_steps": ["step 1", "step 2", "step 3"]
}}

Question:
{question}

Selected schema:
{selected_schema}
"""))

# Generator agent prompt
GENERATOR_PROMPT = PromptTemplate.from_template(textwrap.dedent("""
You are the Generator agent writing ONE safe DuckDB SQL query.

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
                                                          
Selected schema:
{selected_schema}

Sample numeric stats (for reference):
{stats}

Plan:
{plan}

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

# Refiner prompt for unsafe or failing SQL
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

Plan:
{plan}

Previous SQL:
{previous_sql}

Problem to fix (safety or execution error):
{error}

Return corrected SQL only, no prose, no fences.
"""))
