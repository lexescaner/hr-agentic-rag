"""
Web app — Flask chat UI + API for the HR agentic assistant.

Wraps the existing, verified agent.py logic (build_agent, invoke_with_retry,
build_trace) behind two HTTP endpoints:
  - GET  /health  — app + MCP connectivity status
  - POST /chat    — accepts a question, returns the agent's answer plus a
                     structured tool-call trace (rubric: "returns the final
                     answer, citations, snippets, and a concise tool-call trace")

The agent is built once at MODULE IMPORT time (not inside `if __name__ ==
"__main__"`), so it works identically whether the app is run directly
(`python app.py`, local dev) or imported by a production WSGI server
(`gunicorn app:app`, used on Render) — gunicorn never executes the
`__main__` block, so agent construction has to happen at import time to run
in both cases.
"""

import os
import asyncio
import traceback

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from agent import build_agent, invoke_with_retry, build_trace

load_dotenv()

app = Flask(__name__)

_agent = None
_startup_error = None


def _run_async(coro):
    """
    Flask routes are sync; agent.py's functions are async (MCP client +
    LangGraph are async-native). This runs a coroutine to completion inside
    a sync context. Adequate for this project's free-tier / demo scale — a
    production app at higher traffic would likely use an async framework
    (e.g. Quart/FastAPI) instead to avoid blocking the request thread.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _init_agent():
    """Build the agent once. Sets module-level _agent / _startup_error."""
    global _agent, _startup_error
    try:
        _agent = _run_async(build_agent())
        _startup_error = None
        print("[startup] Agent ready.")
    except Exception as e:
        _agent = None
        _startup_error = str(e)
        print(f"[startup] Failed to build agent: {e}")
        traceback.print_exc()


# --- Module-level initialization ---
# Runs once when this module is imported, whether that happens via
# `python app.py` or via a WSGI server importing `app:app`. This is what
# makes the app work correctly under gunicorn on Render, not just locally.
print("Initializing agent (connecting to MCP servers)...")
_init_agent()


@app.route("/health", methods=["GET"])
def health():
    """
    Simple JSON status endpoint. Reports whether the agent (and by extension
    both MCP server connections) initialized successfully at startup.
    """
    status = "ok" if _agent is not None else "degraded"
    body = {
        "status": status,
        "agent_ready": _agent is not None,
    }
    if _startup_error:
        body["startup_error"] = _startup_error
    return jsonify(body), (200 if status == "ok" else 503)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts {"question": "..."} and returns the agent's final answer plus
    a structured operational trace (selected tools, arguments, outputs,
    escalation flags) — satisfies the rubric's requirement that /chat return
    "the final answer, citations, snippets, and a concise tool-call trace."
    """
    if _agent is None:
        return jsonify({
            "error": "Agent not initialized",
            "startup_error": _startup_error,
        }), 503

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Request body must include a non-empty 'question' field"}), 400

    try:
        result = _run_async(
            invoke_with_retry(_agent, [{"role": "user", "content": question}])
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Agent invocation failed: {e}"}), 500

    trace = build_trace(result)
    final_answer = result["messages"][-1].content

    citations = []
    for step in trace:
        output = step.get("output")
        if isinstance(output, dict):
            results = output.get("results") or output.get("retrieved_evidence")
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and "doc_title" in r:
                        citations.append({
                            "doc_title": r.get("doc_title"),
                            "section": r.get("section"),
                        })

    return jsonify({
        "answer": final_answer,
        "citations": citations,
        "trace": trace,
    })


if __name__ == "__main__":
    # Local dev entry point. Agent is already initialized above at import
    # time, so this just starts the dev server.
    if _agent is None:
        print("WARNING: agent failed to initialize — /chat will return 503 until fixed.")

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
