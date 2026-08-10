"""
Quick manual test for the RAG pipeline. Run after building the index:
    python rag/policy_rag.py --build
    python test_rag.py

Confirms:
  1. A straightforward single-document question retrieves the right doc
  2. The required complex multi-document question retrieves chunks from
     BOTH remote_work_policy.md and data_security_policy.md
"""

from rag.policy_rag import search

print("=" * 70)
print("TEST 1: Straightforward question (should hit pto_policy)")
print("=" * 70)
results = search("How many PTO days do I get per year?", top_k=3)
for r in results:
    print(f"  [{r['doc_id']}] {r['section']} (distance={r['distance']:.4f})")

print()
print("=" * 70)
print("TEST 2: Complex multi-document question")
print("(should hit BOTH remote_work_policy AND data_security_policy)")
print("=" * 70)
results = search(
    "If I want to work remotely from a Tier 3 country for 6 weeks, "
    "what security and approval requirements apply?",
    top_k=5,
)
doc_ids_hit = set()
for r in results:
    print(f"  [{r['doc_id']}] {r['section']} (distance={r['distance']:.4f})")
    doc_ids_hit.add(r["doc_id"])

print()
if {"remote_work_policy", "data_security_policy"}.issubset(doc_ids_hit):
    print("✅ PASS — retrieval correctly pulled from both policy documents")
else:
    print(f"⚠️  Only hit: {doc_ids_hit} — may need top_k increased or chunking adjusted")
