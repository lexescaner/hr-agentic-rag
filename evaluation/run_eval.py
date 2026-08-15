"""
Evaluation runner. Runs every question in eval_questions.py against the app
(local or deployed), captures the response + trace + latency, and writes
results to evaluation/results.json for manual review and metric reporting.

Usage:
    python evaluation/run_eval.py --url http://localhost:5000
    python evaluation/run_eval.py --url https://hr-agentic-rag-lmj7.onrender.com

This script does NOT auto-score correctness (that requires human judgment
against gold_answer_notes) — it captures raw data (response, trace, timing)
so you can score groundedness/citation accuracy/tool selection manually or
semi-automatically afterward, per the rubric's required metrics.

NOTE on rate limiting: OpenRouter's free tier enforces a per-minute limit
(20 requests/min observed) in addition to the daily cap. A delay is added
between questions to stay under that per-minute ceiling — without it, a
25-question run firing requests back-to-back reliably triggers 429s partway
through, producing misleading "errors" that reflect rate limiting rather
than actual agent behavior (confirmed in an earlier run of this script).
"""

import argparse
import json
import time
from pathlib import Path

import requests

from eval_questions import EVAL_SET

# Seconds to wait between each question. Each question may trigger multiple
# LLM calls (agent loop iterations + internal retries), so spacing questions
# out is more reliable than just respecting the per-request limit alone.
DELAY_BETWEEN_QUESTIONS = 8


def run_eval(base_url: str, output_path: str, delay: float = DELAY_BETWEEN_QUESTIONS):
    results = []

    for i, q in enumerate(EVAL_SET):
        print(f"Running {q['id']} ({q['category']}): {q['question'][:60]}...")

        start = time.time()
        try:
            headers = {}
            if q.get("auth_token"):
                headers["Authorization"] = f"Bearer {q['auth_token']}"

            resp = requests.post(
                f"{base_url}/chat",
                json={"question": q["question"]},
                headers=headers,
                timeout=180,  # generous, cold starts can be slow
            )
            elapsed = time.time() - start
            data = resp.json()
            status_code = resp.status_code
        except Exception as e:
            elapsed = time.time() - start
            data = {"error": str(e)}
            status_code = None

        actual_tools = [step["tool"] for step in data.get("trace", [])]

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "gold_answer_notes": q["gold_answer_notes"],
            "expected_tools": q["expected_tools"],
            "actual_tools": actual_tools,
            "tool_selection_match": set(actual_tools) == set(q["expected_tools"]),
            "answer": data.get("answer"),
            "citations": data.get("citations"),
            "trace": data.get("trace"),
            "escalation_occurred": any(step.get("escalation") for step in data.get("trace", [])),
            "authenticated": data.get("authenticated"),
            "http_status": status_code,
            "latency_seconds": round(elapsed, 2),
            "error": data.get("error"),
        })

        print(f"  -> {elapsed:.1f}s, tools called: {actual_tools}"
              + (f", ERROR: {data.get('error')[:80]}" if data.get("error") else ""))

        # Pause between questions (not after the last one) to stay under
        # OpenRouter's per-minute free-tier rate limit.
        if i < len(EVAL_SET) - 1:
            time.sleep(delay)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")

    latencies = [r["latency_seconds"] for r in results if r["http_status"] == 200]
    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        print(f"Latency p50: {p50:.1f}s, p95: {p95:.1f}s")

    tool_matches = sum(1 for r in results if r["tool_selection_match"])
    print(f"Tool selection matched expected: {tool_matches}/{len(results)}")

    errors = sum(1 for r in results if r["error"])
    print(f"Errors: {errors}/{len(results)}")
    if errors:
        rate_limit_errors = sum(1 for r in results if r["error"] and "429" in str(r["error"]))
        print(f"  (of which rate-limit 429s: {rate_limit_errors})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the app (local or deployed)")
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_QUESTIONS,
                         help="Seconds to wait between questions (default: 8)")
    args = parser.parse_args()

    run_eval(args.url.rstrip("/"), args.output, args.delay)
