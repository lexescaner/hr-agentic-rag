"""
Web app — Flask chat UI + API for the HR agentic assistant.

SECURITY UPDATE: adds three fixes based on adversarial testing
(evaluation/run_security_eval.py) documented in design-and-evaluation.md
Section 5.3:
  1. Bearer token authentication on /chat, resolved to an employee_id and
     passed to the agent's authorization guard (fixes 3/3 authorization
     bypass).
  2. A live output filter scanning the draft answer for internal
     prompt/guardrail markers before returning it (fixes 1/2 prompt
     disclosure).
  3. Basic per-IP rate limiting on /chat (previously untested/absent).

For local testing, valid demo tokens are in mock_data/auth_tokens.json —
send one as: Authorization: Bearer token-emp001-demo
"""

import os
import json
import asyncio
import traceback

from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from agent import build_agent, invoke_with_retry, build_trace

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)

_agent = None
_startup_error = None

AUTH_TOKENS_PATH = os.path.join(os.path.dirname(__file__), "mock_data", "auth_tokens.json")

SYSTEM_PROMPT_MARKERS = [
    "GUARDED_EMPLOYEE_TOOLS",
    "NEVER invent or guess an employee_id",
    "Rejected: employee_id",
    "Rejected: this action requires authentication",
    "confirmation_required",
    "RECURSION_LIMIT",
]


def _load_auth_tokens() -> dict:
    try:
        with open(AUTH_TOKENS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[startup] WARNING: {AUTH_TOKENS_PATH} not found — all requests will be unauthenticated.")
        return {}


_AUTH_TOKENS = _load_auth_tokens()


def _resolve_authenticated_employee_id():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[len("Bearer "):].strip()
    return _AUTH_TOKENS.get(token)


def _filter_prompt_disclosure(answer: str) -> str:
    if not answer:
        return answer
    lower = answer.lower()
    for marker in SYSTEM_PROMPT_MARKERS:
        if marker.lower() in lower:
            print(f"[security] Blocked response containing internal marker: '{marker}'")
            return (
                "I can't share my internal instructions or system configuration. "
                "I'm happy to help with HR policy questions or your own employee data instead."
            )
    return answer


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _init_agent():
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


print("Initializing agent (connecting to MCP servers)...")
_init_agent()


_CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HR Assistant</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  #question, #token { width: 100%; padding: 10px; font-size: 15px; box-sizing: border-box; margin-bottom: 8px; }
  button { margin-top: 8px; padding: 8px 16px; font-size: 15px; cursor: pointer; }
  #answer { white-space: pre-wrap; margin-top: 20px; padding: 12px; background: #f4f4f4; border-radius: 6px; }
  #trace { margin-top: 12px; font-size: 12px; color: #555; }
  .status { font-size: 13px; color: #888; }
  label { font-size: 12px; color: #666; }
</style>
</head>
<body>
  <h1>HR Assistant</h1>
  <p class="status">Ask about PTO, remote work, benefits, expenses, and other HR policies.</p>
  <label>Auth token (optional — required for employee-specific data)</label>
  <input type="text" id="token" placeholder="e.g. token-xxxx-demo (see mock_data/auth_tokens.json)" />
  <input type="text" id="question" placeholder="e.g. How many PTO days do I get?" />
  <button onclick="ask()">Ask</button>
  <div id="answer"></div>
  <div id="trace"></div>

<script>
async function ask() {
  const q = document.getElementById('question').value;
  const token = document.getElementById('token').value;
  const answerDiv = document.getElementById('answer');
  const traceDiv = document.getElementById('trace');
  if (!q.trim()) return;

  answerDiv.textContent = 'Thinking... (first request after inactivity can take up to a minute)';
  traceDiv.textContent = '';

  try {
    const headers = {'Content-Type': 'application/json'};
    if (token.trim()) headers['Authorization'] = 'Bearer ' + token.trim();

    const resp = await fetch('/chat', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({question: q})
    });
    const data = await resp.json();

    if (data.error) {
      answerDiv.textContent = 'Error: ' + data.error;
      return;
    }

    answerDiv.textContent = data.answer;

    if (data.trace && data.trace.length > 0) {
      const toolNames = data.trace.map(s => s.tool).join(', ');
      traceDiv.textContent = 'Tools called: ' + toolNames;
    }
  } catch (e) {
    answerDiv.textContent = 'Request failed: ' + e;
  }
}

document.getElementById('question').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') ask();
});
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def chat_ui():
    return Response(_CHAT_UI_HTML, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if _agent is not None else "degraded"
    body = {
        "status": status,
        "agent_ready": _agent is not None,
    }
    if _startup_error:
        body["startup_error"] = _startup_error
    return jsonify(body), (200 if status == "ok" else 503)


@app.route("/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat():
    if _agent is None:
        return jsonify({
            "error": "Agent not initialized",
            "startup_error": _startup_error,
        }), 503

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Request body must include a non-empty 'question' field"}), 400

    authenticated_employee_id = _resolve_authenticated_employee_id()

    # Tell the model its authenticated identity directly, rather than letting
    # it guess a placeholder (e.g. "current_user_id") and discover the real
    # ID only by having a tool call rejected and reading the error message.
    # That guess-then-correct pattern was safe (the guard still blocked the
    # wrong ID) but wasteful and fragile — it relied on the rejection text
    # happening to reveal the correct ID, which isn't guaranteed behavior.
    if authenticated_employee_id:
        message_content = (
            f"[Authenticated as employee_id: {authenticated_employee_id}. "
            f"Use this exact ID for any employee-specific tool call — do not guess "
            f"or use a placeholder.]\n\n{question}"
        )
    else:
        message_content = question

    try:
        result = _run_async(
            invoke_with_retry(
                _agent,
                [{"role": "user", "content": message_content}],
                authenticated_employee_id=authenticated_employee_id,
            )
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Agent invocation failed: {e}"}), 500

    trace = build_trace(result)
    final_answer = _filter_prompt_disclosure(result["messages"][-1].content)

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
        "authenticated": authenticated_employee_id is not None,
    })


if __name__ == "__main__":
    if _agent is None:
        print("WARNING: agent failed to initialize — /chat will return 503 until fixed.")

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
