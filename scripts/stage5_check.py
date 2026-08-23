"""Stage 5 self-check for IR__Chat.

Run with:  python3 scripts/stage5_check.py

Runs the whole gold set through the whole system and prints the report that
belongs in the write-up: retrieval and extraction scored separately, a
per-question-type breakdown, and every failure listed by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_chat import load_bundled_corpus  # noqa: E402
from ir_chat.config import PROJECT_ROOT  # noqa: E402
from ir_chat.dialogue import DialogueManager  # noqa: E402
from ir_chat.evaluate import evaluate, load_pairs, token_f1  # noqa: E402

RULE = "-" * 72
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


corpus, _ = load_bundled_corpus()
manager = DialogueManager(corpus)
pairs = load_pairs()

# --- 1. the metrics themselves -----------------------------------------------
print(RULE)
print("1. METRIC SANITY")
print(RULE)

print(f"  F1('Stephen Robertson', 'Stephen Robertson and Karen Sparck Jones')"
      f" = {token_f1('Stephen Robertson', 'Stephen Robertson and Karen Sparck Jones'):.2f}")
print(f"  F1('Joseph Weizenbaum', 'Joseph Weizenbaum') = "
      f"{token_f1('Joseph Weizenbaum', 'Joseph Weizenbaum'):.2f}")
print(f"  F1('Alan Turing', 'Joseph Weizenbaum') = "
      f"{token_f1('Alan Turing', 'Joseph Weizenbaum'):.2f}")

check("identical answers score 1.0", token_f1("1985", "1985") == 1.0)
check("unrelated answers score 0.0", token_f1("Alan Turing", "Joseph Weizenbaum") == 0.0)
check("partial answers get partial credit",
      0 < token_f1("Stephen Robertson", "Stephen Robertson and Karen Sparck Jones") < 1)
check("articles are ignored", token_f1("the Journal of Documentation",
                                       "Journal of Documentation") == 1.0)

# --- 2. the gold set ---------------------------------------------------------
print(f"\n{RULE}")
print("2. GOLD SET")
print(RULE)

answerable = [p for p in pairs if p.get("a") is not None]
print(f"  {len(pairs)} questions, {len(answerable)} answerable, "
      f"{len(pairs) - len(answerable)} deliberately unanswerable")
check("gold set is loaded", len(pairs) >= 30, f"{len(pairs)} questions")
check("every gold document exists",
      all(corpus.get(p["doc"]) for p in answerable))
check("unanswerable questions have no gold document",
      all(p.get("doc") is None for p in pairs if p.get("a") is None))

# --- 3. the run --------------------------------------------------------------
print(f"\n{RULE}")
print("3. RESULTS")
print(RULE)

report = evaluate(manager, pairs)
summary = report.summary()

print("  retrieval")
print(f"    recall@5              {summary['recall@k']:.3f}")
print(f"    mean reciprocal rank  {summary['MRR']:.3f}")
print("\n  answers")
print(f"    exact match           {summary['exact_match']:.3f}")
print(f"    token F1              {summary['token_f1']:.3f}")
print(f"    gold answer present   {summary['answer_found']:.3f}")
print("\n  refusals")
print(f"    correctly declined    {summary['refusal_accuracy']:.3f}")
print(f"\n  overall correct         {summary['overall_correct']:.3f}")

print(f"\n  {'question type':<16}{'n':>4}{'correct':>10}{'token F1':>10}")
for name, stats in report.by_type().items():
    print(f"  {name:<16}{stats['n']:>4}{stats['correct']:>9.0%}{stats['f1']:>10.2f}")

print(f"\n  route used: " + ", ".join(f"{r} {c}" for r, c in report.by_route().items()))

if report.failures():
    print(f"\n  failures:")
    for failure in report.failures():
        print(f"    {failure.question}")
        print(f"      expected: {failure.gold}")
        print(f"      got:      {failure.predicted[:80]}")

check("recall@5 above 0.90", summary["recall@k"] >= 0.90, f"{summary['recall@k']:.3f}")
check("MRR above 0.85", summary["MRR"] >= 0.85, f"{summary['MRR']:.3f}")
check("token F1 above 0.80", summary["token_f1"] >= 0.80, f"{summary['token_f1']:.3f}")
check("refuses every unanswerable question", summary["refusal_accuracy"] == 1.0)
check("overall correctness above 0.90", summary["overall_correct"] >= 0.90,
      f"{summary['overall_correct']:.3f}")

# --- 4. deployment files -----------------------------------------------------
print(f"\n{RULE}")
print("4. READY TO DEPLOY")
print(RULE)

app = PROJECT_ROOT / "app.py"
theme = PROJECT_ROOT / ".streamlit" / "config.toml"
requirements = PROJECT_ROOT / "requirements.txt"

for path in (app, theme, requirements):
    print(f"  {'found  ' if path.exists() else 'MISSING'} {path.relative_to(PROJECT_ROOT)}")

check("app entry point exists", app.exists())
check("theme is configured", theme.exists())
check("requirements pin streamlit", "streamlit" in requirements.read_text())
check("requirements pin the spaCy model", "en_core_web_sm" in requirements.read_text())

# --- verdict -----------------------------------------------------------------
print(f"\n{RULE}")
if failures:
    print(f"STAGE 5: {len(failures)} CHECK(S) FAILED -> {failures}")
    sys.exit(1)
print("STAGE 5 OK - IR__Chat is complete and ready to deploy.")
print(RULE)
