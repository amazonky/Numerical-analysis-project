from typing import Optional
from langchain_ollama import OllamaLLM
from .prompts import REPAIR_PROMPT
from .safety import normalize_sql


def repair_sql(
    llm: OllamaLLM,
    *,
    table_name: str,
    schema: str,
    question: str,
    previous_sql: str,
    error: str,
) -> str:
    """
    Ask the LLM to repair an unsafe or failing SQL statement.
    """
    prompt = REPAIR_PROMPT.format(
        table_name=table_name,
        schema=schema,
        question=question,
        previous_sql=previous_sql,
        error=error,
    )
    raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    return normalize_sql(raw, table_name)

