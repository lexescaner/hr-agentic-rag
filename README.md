# HR Agentic RAG Assistant

Agentic AI system for HR policy Q&A and multi-step HR workflows, built for the MSAIE
"AI Engineering Techniques and Architectures" project.

## Architecture

- **RAG layer** (`rag/`): policy corpus (`docs/`) chunked, embedded, and indexed in Chroma.
  Pattern adapted from [PraxaNew](https://github.com/lexescaner/PraxaNew).
- **Agent + MCP layer** (`mcp/`, `agent.py`): single LangGraph agent that discovers and calls
  tools from two MCP servers. Pattern adapted from
  [real-estate-agent](https://github.com/lexescaner/real-estate-agent).
  - `mcp/policy_mcp_server.py` — RAG search / citation tools
  - `mcp/hr_data_mcp_server.py` — mock employee/PTO/benefits/ticket tools
- **Web app** (`app.py`): Flask chat UI + `/chat` and `/health` endpoints.

## Status
Day 1 scaffold — corpus, mock data, and MCP server stubs in place. RAG wiring and agent
orchestration land Day 2-3.

## Setup
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set API keys (OpenRouter, etc.)
4. Run locally: `python app.py`

## Deployment
Target: Render/Railway free tier, single-service deployment. See `deployed.md` (added Day 4).
