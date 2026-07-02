"""Evaluation harness for the Agentic Knowledge Retrieval System.

Runs the full graph over a labelled question set (`eval/qa_set.json`) and
reports five metrics. Retrieval-quality metrics are named the way the RAG
literature (e.g. RAGAS) uses them, so the numbers mean what they say:

  1. Router accuracy    — does the intent classifier pick the right handler?
  2. Retrieval hit-rate — for grounded questions, is the correct source doc in
                          the top-k retrieved chunks? (This is hit-rate / recall@k,
                          NOT precision — it only asks whether the right doc showed up.)
  3. Answer match       — do the expected key facts appear in the answer? A keyword
                          check, so it verifies presence of the fact, not full
                          semantic correctness. For chart questions it checks an
                          artifact was produced; for compute questions, the number.
  4. Faithfulness       — LLM-as-judge: is every claim in the answer supported by
                          the retrieved context? This is the anti-hallucination
                          metric. Judged by the same provider as the app, so treat
                          it as a self-consistency proxy, not an independent audit.
  5. Abstention rate    — for deliberately out-of-scope questions (facts NOT in any
                          document), does the system correctly decline instead of
                          fabricating an answer?

`expected_answer_contains` entries may be a string (must appear) or a list
(any-of must appear), both matched case-insensitively.

Run:
    python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

# Repo root on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.build import get_graph
from app.graph.retrieval import format_context
from app.llm.provider import get_llm

QA_PATH = Path(__file__).parent / "qa_set.json"

# Phrases that signal the model correctly declined to answer out-of-scope questions.
_ABSTAIN_MARKERS = (
    "don't have enough", "do not have enough", "not enough information",
    "no information", "not contain", "isn't in", "is not in", "cannot find",
    "can't find", "not available", "not mentioned", "not provided", "unable to",
)


def _keyword_hit(answer: str, expected) -> bool:
    """True if every expected entry is satisfied. A list entry is an any-of group."""
    a = answer.lower()
    for entry in expected:
        if isinstance(entry, list):
            if not any(opt.lower() in a for opt in entry):
                return False
        elif entry.lower() not in a:
            return False
    return True


def _judge_faithful(answer: str, context: str) -> bool:
    """LLM-as-judge: is every claim in the answer supported by the context?"""
    verdict = get_llm().invoke(
        "You are a strict grader. Reply with a single word, YES or NO.\n"
        "Is EVERY factual claim in the ANSWER directly supported by the CONTEXT?\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nSupported (YES/NO)?"
    ).content.strip().upper()
    return verdict.startswith("YES")


def _abstained(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in _ABSTAIN_MARKERS)


def run_eval():
    questions = json.loads(QA_PATH.read_text())
    graph = get_graph()

    router_ok = router_n = 0
    hit_ok = hit_n = 0
    ans_ok = ans_n = 0
    faith_ok = faith_n = 0
    abstain_ok = abstain_n = 0

    rows = []
    for q in questions:
        result = graph.invoke({"question": q["question"]})
        answer = result.get("answer", "")
        docs = result.get("context", [])

        # 1. router
        router_n += 1
        r_ok = result.get("intent").value == q["intent"]
        router_ok += r_ok

        # 5. abstention (out-of-scope) — checked before answer/faithfulness
        out_of_scope = q.get("out_of_scope")
        abst = None
        if out_of_scope:
            abstain_n += 1
            abst = _abstained(answer)
            abstain_ok += abst

        # 2. retrieval hit-rate
        hit = None
        if q.get("expected_source"):
            hit_n += 1
            hit = q["expected_source"] in [c.source for c in result.get("citations", [])]
            hit_ok += hit

        # 3. answer match
        am = None
        if q.get("expect_artifact"):
            ans_n += 1
            am = bool(result.get("artifact_path"))
            ans_ok += am
        elif q.get("expected_answer_contains"):
            ans_n += 1
            am = _keyword_hit(answer, q["expected_answer_contains"])
            ans_ok += am

        # 4. faithfulness (grounded retrieval answers only — skip out-of-scope & code)
        faith = None
        if q["intent"] == "retrieval" and not out_of_scope and docs:
            faith_n += 1
            faith = _judge_faithful(answer, format_context(docs))
            faith_ok += faith

        rows.append({
            "id": q["id"],
            "router": "✓" if r_ok else "✗",
            "hit": _mark(hit),
            "ans": _mark(am),
            "faith": _mark(faith),
            "abst": _mark(abst),
            "q": q["question"][:48],
        })

    _print_report(rows, router_ok, router_n, hit_ok, hit_n,
                   ans_ok, ans_n, faith_ok, faith_n, abstain_ok, abstain_n)


def _mark(v):
    return "✓" if v else ("✗" if v is False else "—")


def _pct(ok, n):
    return f"{ok}/{n}  ({100 * ok // n}%)" if n else "n/a"


def _print_report(rows, router_ok, router_n, hit_ok, hit_n,
                  ans_ok, ans_n, faith_ok, faith_n, abstain_ok, abstain_n):
    print(f"\n{'ID':>2}  {'Route':5}  {'Hit':3}  {'Ans':3}  {'Faith':5}  {'Abst':4}  Question")
    print("-" * 82)
    for r in rows:
        print(f"{r['id']:>2}  {r['router']:5}  {r['hit']:3}  {r['ans']:3}  "
              f"{r['faith']:5}  {r['abst']:4}  {r['q']}")
    print("\n── Summary ──────────────────────────────────────")
    print(f"  Router accuracy    : {_pct(router_ok, router_n)}")
    print(f"  Retrieval hit-rate : {_pct(hit_ok, hit_n)}")
    print(f"  Answer match       : {_pct(ans_ok, ans_n)}")
    print(f"  Faithfulness       : {_pct(faith_ok, faith_n)}")
    print(f"  Abstention rate    : {_pct(abstain_ok, abstain_n)}")
    print()


if __name__ == "__main__":
    run_eval()
