"""
Systematic verification of evaluation/results.json — closes the gap between
"spot-checked a few answers" and "systematically checked all 25."

Two checks, with different reliability:

1. CITATION VALIDITY (reliable, exact match): confirms every citation's
   (doc_title, section) pair actually exists in the real corpus (docs/),
   not hallucinated. This is a hard pass/fail — a citation either matches a
   real document section or it doesn't.

2. GROUNDEDNESS HEURISTIC (best-effort screening, not exhaustive proof):
   extracts numeric claims from each answer and checks whether they appear
   in that question's retrieved text (RAG chunks or structured tool output
   values). Flags answers where a number appears that wasn't found in any
   retrieved content, as candidates for manual review. This is a heuristic,
   not a full groundedness score — it can produce false positives (e.g. a
   number restated from the user's own question) and won't catch
   non-numeric unsupported claims. Its purpose is narrowing 25 answers down
   to the handful most worth a human second look, not replacing judgment.

Usage:
    python evaluation/verify_results.py
"""

import json
import re
import glob
from pathlib import Path

from bs4 import BeautifulSoup

DOCS_DIR = Path(__file__).parent.parent / "docs"
RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "verification_report.json"


def _doc_title_from_id(doc_id: str) -> str:
    words = doc_id.replace("_", " ").split()
    return " ".join(w.upper() if w.lower() == "pto" else w.capitalize() for w in words)


def build_corpus_reference() -> dict:
    """
    Reads the actual docs/ folder and builds {doc_title: {section, ...}} —
    the ground truth used to validate citations against, rather than trusting
    the eval results' own claims about what's in the corpus.
    """
    ref = {}

    for path in glob.glob(str(DOCS_DIR / "*.md")):
        doc_id = Path(path).stem
        title = _doc_title_from_id(doc_id)
        sections = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    sections.add(line[3:].strip())
        ref[title] = sections

    for path in glob.glob(str(DOCS_DIR / "*.html")):
        doc_id = Path(path).stem
        title = _doc_title_from_id(doc_id)
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        sections = {h2.get_text().strip() for h2 in soup.find_all("h2")}
        ref[title] = sections

    return ref


def extract_numbers(text: str) -> set:
    """Pulls standalone numbers (e.g. '16', '5.5') out of a text blob."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def collect_retrieved_text(trace: list) -> str:
    """
    Gathers all text the agent actually had access to for this question:
    RAG chunk text, plus any numeric values from structured tool outputs
    (e.g. pto_days_remaining: 16) — both are legitimate grounding sources.
    """
    parts = []
    for step in trace or []:
        output = step.get("output")
        if not isinstance(output, dict):
            continue
        for r in output.get("results") or []:
            if isinstance(r, dict) and "text" in r:
                parts.append(r["text"])
        for r in output.get("retrieved_evidence") or []:
            if isinstance(r, dict) and "text" in r:
                parts.append(r["text"])
        for value in output.values():
            if isinstance(value, (int, float)):
                parts.append(str(value))
    return " ".join(parts)


def verify():
    corpus_ref = build_corpus_reference()
    results = json.load(open(RESULTS_PATH))

    report = []
    for r in results:
        citations = r.get("citations") or []
        citation_issues = []
        for c in citations:
            title, section = c.get("doc_title"), c.get("section")
            if title not in corpus_ref:
                citation_issues.append(f"Unknown document: '{title}'")
            elif section not in corpus_ref[title]:
                citation_issues.append(f"Unknown section '{section}' in '{title}'")

        answer_numbers = extract_numbers(r.get("answer"))
        retrieved_text = collect_retrieved_text(r.get("trace"))
        retrieved_numbers = extract_numbers(retrieved_text)
        ungrounded_numbers = sorted(answer_numbers - retrieved_numbers)

        report.append({
            "id": r["id"],
            "category": r["category"],
            "citations_checked": len(citations),
            "citation_issues": citation_issues,
            "citations_valid": len(citation_issues) == 0,
            "answer_numbers": sorted(answer_numbers),
            "ungrounded_numbers": ungrounded_numbers,
            "groundedness_flag": len(ungrounded_numbers) > 0,
        })

    total = len(report)
    citation_valid_count = sum(1 for r in report if r["citations_valid"])
    flagged = [r for r in report if r["groundedness_flag"]]

    print("=" * 70)
    print("CITATION VALIDITY (exact match against actual corpus docs/)")
    print("=" * 70)
    print(f"{citation_valid_count}/{total} results had 100% valid citations\n")
    for r in report:
        if not r["citations_valid"]:
            print(f"  {r['id']} ({r['category']}): {r['citation_issues']}")

    print()
    print("=" * 70)
    print("GROUNDEDNESS HEURISTIC (numeric claims not found in retrieved text)")
    print("=" * 70)
    print(f"{len(flagged)}/{total} results flagged for manual review\n")
    for r in flagged:
        print(f"  {r['id']} ({r['category']}): answer has {r['answer_numbers']}, "
              f"not found in retrieved text: {r['ungrounded_numbers']}")

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull per-question report saved to {REPORT_PATH}")


if __name__ == "__main__":
    verify()
