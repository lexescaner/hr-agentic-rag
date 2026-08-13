# Deployment

## Live URL

**Application:** https://hr-agentic-rag-lmj7.onrender.com

**Health endpoint:** https://hr-agentic-rag-lmj7.onrender.com/health

**Chat endpoint (POST):** https://hr-agentic-rag-lmj7.onrender.com/chat

## Platform

Deployed to **Render** (free tier), single-service architecture — web app, agent orchestrator,
both MCP server subprocesses, and the Chroma vector index all run within one deployed service, per
the project brief's recommended free-tier architecture.

- **Build command:** `pip install -r requirements.txt && python rag/policy_rag.py --build`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Instance type:** Free (512 MB RAM, 0.1 CPU)
- **Repo:** github.com/lexescaner/hr-agentic-rag (auto-deploys on push to `main`)

## Cold-Start Behavior

Render's free tier spins the instance down after a period of inactivity. The dashboard explicitly
warns: *"Your free instance will spin down with inactivity, which can delay requests by 50 seconds
or more."*

In practice this means:
- The **first request** after a period of inactivity will be slow (Render has to boot a fresh
  container, which re-runs the full startup sequence: gunicorn starts, `app.py` is imported, the
  agent initializes and connects to both MCP server subprocesses).
- **Subsequent requests** are fast, since the instance stays warm as long as traffic continues.
- The `/health` endpoint reflects this directly — a cold-started instance will show `agent_ready:
  true` only once startup has actually completed, and will correctly report `agent_ready: false`
  (with a `startup_error` message) if the agent failed to initialize for any reason.

For grading/demo purposes: if the deployed URL is visited after being idle, expect the first
response to take up to a minute; this is expected free-tier behavior, not an application defect.

## Memory Constraint and Fix

The initial deployment attempt **failed** with `Ran out of memory (used over 512MB) while running
your code` — Render free tier's 512 MB RAM ceiling was exceeded by the original embedding approach
(`SentenceTransformerEmbeddingFunction`, which pulls in PyTorch — commonly 500 MB-1 GB+ once loaded
into memory), combined with gunicorn's main process plus two separate MCP server subprocesses each
running their own Python interpreter.

**Fix:** switched to Chroma's built-in `ONNXMiniLM_L6_V2` embedding function, which wraps the same
underlying `all-MiniLM-L6-v2` model via `onnxruntime` instead of `sentence-transformers`/PyTorch —
same embedding quality and dimensionality, substantially smaller runtime memory footprint.
`sentence-transformers` (and its transitive dependencies `torch`/`transformers`) was removed from
`requirements.txt` entirely, since nothing else in the stack required it.

Notably, this same `sentence-transformers`/PyTorch pattern was also used in a prior project
(PraxaNew), though that project was run locally rather than deployed to a memory-constrained host,
so it's untested there. Still, this suggests the underlying tradeoff isn't specific to this
codebase — `sentence-transformers` is a reasonable default for local development, but worth
reconsidering for any future free-tier deployment, based on the concrete memory failure observed
here.

After the fix, the deploy succeeded cleanly on the first retry, with the agent initializing
correctly and both `/health` and `/chat` verified working against the live public URL.

## Verification

Both endpoints tested directly against the live URL post-deployment:

```bash
curl https://hr-agentic-rag-lmj7.onrender.com/health
# {"agent_ready":true,"status":"ok"}

curl -X POST https://hr-agentic-rag-lmj7.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days does EMP001 have left?"}'
# Returns correct answer, tool-call trace, and citations (where applicable)
```

## Environment Variables (set in Render dashboard, not committed)

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | (secret — set directly in Render, never in git) |
| `LLM_MODEL` | `google/gemma-4-26b-a4b-it:free` |

See `README.md` for local setup instructions and `design-and-evaluation.md` for the full
architecture rationale.
