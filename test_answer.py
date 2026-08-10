"""
Test the full answer pipeline, including all 3 required guardrails.
Run after setting OPENROUTER_API_KEY in .env:
    python test_answer.py
"""

from rag.answer import answer_question

print("=" * 70)
print("TEST 1: Normal grounded question (should cite pto_policy)")
print("=" * 70)
result = answer_question("How many PTO days do I get, and can I carry them over?")
print(result["answer"])
print("Citations:", [c["doc_title"] for c in result["citations"]])

print()
print("=" * 70)
print("TEST 2: Out-of-corpus question (guardrail 1 — should refuse/redirect)")
print("=" * 70)
result = answer_question("What is the company's stock option vesting schedule?")
print(result["answer"])

print()
print("=" * 70)
print("TEST 3: Multi-document question (should cite both remote work + security)")
print("=" * 70)
result = answer_question(
    "I want to work remotely from a country outside my normal location for 6 weeks. "
    "What do I need to consider?"
)
print(result["answer"])
print("Citations:", [c["doc_title"] for c in result["citations"]])
