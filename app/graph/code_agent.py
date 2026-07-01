"""Code-exec agent node (Bullet 2: sandboxed execution + self-correcting retry).

Flow: generate Python from the question + retrieved data -> run it in the
sandbox -> if it errors, feed the traceback back to the LLM and retry, up to
settings.code_agent_max_retries times. The final answer carries the computed
output, the code that produced it, any chart artifact, and the retry count.
"""
import re

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.graph.retrieval import docs_to_citations, format_context
from app.graph.state import GraphState
from app.llm.provider import get_llm
from app.sandbox import executor
from app.sandbox.executor import ARTIFACT_NAME

_CODE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write short, self-contained Python to answer the question using the "
     "figures in the context. Rules:\n"
     "- Use only the standard library, pandas, and matplotlib.\n"
     "- Read the numbers you need directly from the context (hard-code them).\n"
     f"- If a chart is requested, save it to '{ARTIFACT_NAME}' via plt.savefig.\n"
     "- print() the key textual result so it can be shown to the user.\n"
     "- Output ONLY a Python code block."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

_FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Your previous Python failed. Return a corrected, complete script that fixes "
     "the error. Output ONLY a Python code block."),
    ("human", "Failed code:\n{code}\n\nError output:\n{error}"),
])


def _extract_code(text: str) -> str:
    """Pull the Python out of a ```python ...``` block (or use it as-is)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def run_code_agent(state: GraphState) -> GraphState:
    llm = get_llm()
    context = format_context(state.get("context", []))

    code = _extract_code(
        (_CODE_PROMPT | llm).invoke(
            {"context": context, "question": state["question"]}
        ).content
    )
    result = executor.run(code)

    retries = 0
    while not result.ok and retries < settings.code_agent_max_retries:
        retries += 1
        code = _extract_code(
            (_FIX_PROMPT | llm).invoke(
                {"code": code, "error": result.stderr}
            ).content
        )
        result = executor.run(code)

    if result.ok:
        answer = result.stdout.strip() or "Done."
        if result.artifact_path:
            answer += f"\n\n📊 Chart generated: {result.artifact_path}"
    else:
        answer = (f"I couldn't produce a working result after {retries} "
                  f"self-correction attempts. Last error:\n{result.stderr[:300]}")

    return {
        "answer": answer,
        "generated_code": code,
        "artifact_path": result.artifact_path,
        "retries": retries,
        "citations": docs_to_citations(state.get("context", [])),
    }
