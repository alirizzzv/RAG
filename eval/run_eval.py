"""Evaluation harness for the Agentic Knowledge Retrieval System.

Measures three things:
  1. Router accuracy  — does the intent classifier pick the right handler?
  2. Citation recall  — for retrieval questions, does the correct source doc
                        appear in the top-k citations?
  3. Answer accuracy  — do the expected key phrases appear in the answer?
     (Code questions: did execution succeed and, where expected, produce an
     artifact?)

Run:
    python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

# Repo root on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.build import get_graph

QA_PATH = Path(__file__).parent / "qa_set.json"


def run_eval():
    questions = json.loads(QA_PATH.read_text())
    graph = get_graph()

    router_correct = 0
    citation_correct = 0
    answer_correct = 0
    citation_total = 0
    answer_total = 0

    rows = []
    for q in questions:
        result = graph.invoke({"question": q["question"]})

        # --- router ---
        got_intent = result.get("intent").value
        router_ok = got_intent == q["intent"]
        if router_ok:
            router_correct += 1

        # --- citation recall (retrieval questions only) ---
        cite_ok = None
        if q.get("expected_source"):
            citation_total += 1
            sources = [c.source for c in result.get("citations", [])]
            cite_ok = q["expected_source"] in sources
            if cite_ok:
                citation_correct += 1

        # --- answer accuracy ---
        ans_ok = None
        answer = result.get("answer", "").lower()

        if q.get("expect_artifact"):
            answer_total += 1
            ans_ok = bool(result.get("artifact_path"))
            if ans_ok:
                answer_correct += 1
        elif q.get("expected_answer_contains"):
            answer_total += 1
            ans_ok = all(kw.lower() in answer for kw in q["expected_answer_contains"])
            if ans_ok:
                answer_correct += 1

        rows.append({
            "id": q["id"],
            "router": "✓" if router_ok else "✗",
            "citation": "✓" if cite_ok else ("✗" if cite_ok is False else "—"),
            "answer": "✓" if ans_ok else ("✗" if ans_ok is False else "—"),
            "q": q["question"][:55],
        })

    n = len(questions)
    print(f"\n{'ID':>2}  {'Router':6}  {'Citation':8}  {'Answer':6}  Question")
    print("-" * 80)
    for r in rows:
        print(f"{r['id']:>2}  {r['router']:6}  {r['citation']:8}  {r['answer']:6}  {r['q']}")

    print("\n── Summary ─────────────────────────────────────")
    print(f"  Router accuracy  : {router_correct}/{n}  ({100*router_correct//n}%)")
    if citation_total:
        print(f"  Citation recall  : {citation_correct}/{citation_total}  ({100*citation_correct//citation_total}%)")
    if answer_total:
        print(f"  Answer accuracy  : {answer_correct}/{answer_total}  ({100*answer_correct//answer_total}%)")
    print()
    return router_correct, citation_correct, citation_total, answer_correct, answer_total


if __name__ == "__main__":
    run_eval()
