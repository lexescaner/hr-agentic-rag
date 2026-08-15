"""
Answer correctness check — exact match against gold answers, testing factual
ACCURACY, not just groundedness.

This is the automated implementation of the rubric's "exact/partial match
against gold answers" metric (Section 9), complementing
evaluation/verify_results.py's groundedness and citation-validity checks.
The two scripts test genuinely different failure modes:

  verify_results.py asks: "Does this number appear ANYWHERE in what was
  retrieved?" — catches pure fabrication (nothing retrieved supports the
  claim at all), checked against the agent's OWN retrieved evidence.

  This script asks: "Is this the ACTUAL CORRECT value for THIS SPECIFIC
  record?" — catches misattribution (a number that is real and grounded,
  but assigned to the wrong person or fact), checked against mock_data/*.json
  directly, completely independent of the agent's own trace.

Concretely, a misattribution failure this script catches but verify_results.py
would not: if the agent retrieves EMP001's real PTO data but the LLM writes
"EMP002 has 16 days remaining," that number IS grounded (it exists in what
was retrieved) but is factually wrong for the person asked about. Only a
check against the real EMP002 record catches this.

Usage:
    python evaluation/answer_correctness_check.py --url http://localhost:5000
"""

import argparse
import json
import re
from pathlib import Path

import requests

MOCK_DATA_DIR = Path(__file__).parent.parent / "mock_data"

CORRECTNESS_QUESTIONS = [
    {
        "id": "C1",
        "employee_id": "EMP001",
        "token": "token-emp001-demo",
        "question": "How many PTO days do I have left?",
        "ground_truth_field": "pto_days_remaining",
    },
    {
        "id": "C2",
        "employee_id": "EMP001",
        "token": "token-emp001-demo",
        "question": "How many total PTO days do I get per year, and how many have I used?",
        "ground_truth_field": None,  # checked specially below (two numbers)
    },
    {
        "id": "C3",
        "employee_id": "EMP002",
        "token": "token-emp002-demo",
        "question": "How many PTO days do I have left?",
        "ground_truth_field": "pto_days_remaining",
    },
]


def load_ground_truth_pto(employee_id: str) -> dict:
    """Reads the REAL record directly from mock_data — not from anything the
    agent returned. This is the independent gold-answer source."""
    with open(MOCK_DATA_DIR / "pto_balances.json") as f:
        records = json.load(f)
    for record in records:
        if record.get("employee_id") == employee_id:
            return record
    raise ValueError(f"No ground-truth record found for {employee_id}")


def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(int(n) for n in re.findall(r"\b\d+\b", text))


def check_correctness(question_spec: dict, answer: str) -> dict:
    ground_truth = load_ground_truth_pto(question_spec["employee_id"])
    answer_numbers = extract_numbers(answer)

    if question_spec["ground_truth_field"]:
        expected = ground_truth[question_spec["ground_truth_field"]]
        if expected in answer_numbers:
            return {
                "passed": True,
                "detail": f"Correct — answer contains the true value ({expected}) for {question_spec['employee_id']}.",
            }
        return {
            "passed": False,
            "detail": (
                f"MISMATCH — ground truth {question_spec['ground_truth_field']}={expected} "
                f"for {question_spec['employee_id']}, but answer's numbers were {sorted(answer_numbers)}."
            ),
        }

    # C2-style: multiple specific facts must ALL be present and correct
    expected_total = ground_truth["pto_days_total"]
    expected_used = ground_truth["pto_days_used"]
    missing = []
    if expected_total not in answer_numbers:
        missing.append(f"pto_days_total={expected_total}")
    if expected_used not in answer_numbers:
        missing.append(f"pto_days_used={expected_used}")

    if not missing:
        return {
            "passed": True,
            "detail": f"Correct — both total ({expected_total}) and used ({expected_used}) present and accurate.",
        }
    return {
        "passed": False,
        "detail": f"MISMATCH — missing/incorrect: {missing}. Answer's numbers were {sorted(answer_numbers)}.",
    }


def run_correctness_check(base_url: str, output_path: str):
    results = []

    for q in CORRECTNESS_QUESTIONS:
        print(f"Running {q['id']} ({q['employee_id']}): {q['question']}")

        resp = requests.post(
            f"{base_url}/chat",
            json={"question": q["question"]},
            headers={"Authorization": f"Bearer {q['token']}"},
            timeout=180,
        )
        data = resp.json()
        answer = data.get("answer", "")

        check = check_correctness(q, answer)
        print(f"  -> {'CORRECT' if check['passed'] else 'INCORRECT'}: {check['detail']}")

        results.append({
            "id": q["id"],
            "employee_id": q["employee_id"],
            "question": q["question"],
            "answer": answer,
            "ground_truth_check": check,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["ground_truth_check"]["passed"])
    print(f"\n{passed}/{len(results)} answers verified correct against independent ground truth (exact match).")
    print(f"Full evidence saved to {output_path}")
    return passed == len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="evaluation/answer_correctness_results.json")
    args = parser.parse_args()

    run_correctness_check(args.url.rstrip("/"), args.output)
