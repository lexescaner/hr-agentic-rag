# HR Agentic RAG Assistant

Agentic AI system for HR policy Q&A and multi-step HR workflows, built for the MSAIE
"AI Engineering Techniques and Architectures" project.

**Live app:** https://hr-agentic-rag-lmj7.onrender.com
**Health check:** https://hr-agentic-rag-lmj7.onrender.com/health

## Architecture

- **RAG layer** (`rag/`): policy corpus (`docs/`, 9 files) chunked, embedded via Chroma's
  built-in ONNX `all-MiniLM-L6-v2`, and indexed in Chroma. Pattern adapted from
  [PraxaNew](https://github.com/lexescaner/PraxaNew).
- **Agent + MCP layer** (`mcp/`, `agent.py`): single LangGraph ReAct agent that discovers and
  calls tools from two MCP servers, with a code-level guard preventing tool calls with
  fabricated employee IDs. Pattern adapted from
  [real-estate-agent](https://github.com/lexescaner/real-estate-agent).
  - `mcp/policy_mcp_server.py` — RAG search / citation tools
  - `mcp/hr_data_mcp_server.py` — mock employee/PTO/benefits/ticket tools
- **Web app** (`app.py`): Flask chat UI (`/`) + `/chat` and `/health` API endpoints.

Full architecture diagram and design rationale: see `design-and-evaluation.md`.

## Status

**Complete.** All required workflows verified end-to-end (locally and on the live deployment),
25-question evaluation completed with systematic verification, both required demo tasks working.

## Setup (local development)

1. `python3 -m venv env && source env/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (and optionally `LLM_MODEL`,
   default is `google/gemma-4-26b-a4b-it:free`)
4. Build the RAG index: `python rag/policy_rag.py --build`
5. Run locally: `python app.py` (or `gunicorn app:app --bind 0.0.0.0:5000` to test under the
   same server used in production)
6. Visit `http://localhost:5000/` for the chat UI, or `POST /chat` with `{"question": "..."}`

## Testing

```bash
pytest tests/test_smoke.py -v
```
Runs 6 smoke tests: app import/startup, route registration, `/health` graceful degradation,
live MCP tool discovery, and a direct MCP tool call.

## Deployment

Deployed to **Render** (free tier), single-service architecture. Build and start commands,
environment variables, cold-start behavior, and the memory-limit fix encountered along the way:
see `deployed.md`.

## Evaluation

25-question evaluation set covering straightforward, multi-document, tool-requiring, ambiguous,
and out-of-scope questions. Question set, runner, and systematic verification script (citation
validity + groundedness heuristic) are in `evaluation/`. Full results and analysis: see
`design-and-evaluation.md`.

## Repository Structure

| File / Folder | Contents |
|---|---|
| `app.py` | Flask web app — `/health`, `/chat`, chat UI |
| `agent.py` | LangGraph agent orchestrator, employee_id guard, trace logging |
| `rag/` | Chunking, embedding, retrieval, LLM answer generation |
| `mcp/` | Two MCP servers (policy search, HR data) |
| `docs/` | Policy corpus (9 files, markdown + HTML) |
| `mock_data/` | Synthetic employee/PTO/benefits/ticket data |
| `assets/` | Diagram source files (SVG), referenced by `design-and-evaluation.md` |
| `evaluation/` | Eval question set, runner, verification script, results |
| `tests/` | CI smoke tests (`test_smoke.py`) |
| `test_answer.py`, `test_rag.py` | Standalone manual test scripts for the RAG/answer pipeline |
| `.github/workflows/ci.yml` | CI/CD pipeline |
| `Procfile` | Render start command (`gunicorn app:app ...`) |
| `render.yaml` | Render build/deploy configuration |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required environment variables |
| `design-and-evaluation.md` | Architecture, design rationale, evaluation results |
| `ai-tooling.md` | How AI tools were used in this build |
| `deployed.md` | Deployment details, cold-start notes, memory-fix story |
| `architecture.drawio` | Editable source for the architecture diagram |
