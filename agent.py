"""
Agent orchestrator — single LangGraph agent that discovers and calls tools
from both MCP servers (policy_mcp_server.py, hr_data_mcp_server.py).
Pattern adapted from real-estate-agent's real_estate_agent.py.

STATUS: both required workflows verified end-to-end. Confirmation gate verified
(negative and positive paths). Operational trace logging implemented.

SECURITY UPDATE: adversarial testing (evaluation/run_security_eval.py) found a
real authorization bypass — the original employee_id guard checked only whether
an ID was MENTIONED in the user's message, not whether the requester actually
IS that employee (3/3 bypass attempts succeeded). This version replaces that
check with a real, if lightweight, authentication mechanism: a bearer token
(mock_data/auth_tokens.json) resolved to an authenticated employee_id, checked
against every employee-specific tool call. Also fixes a partial system-prompt
disclosure gap (1/2 in testing) via a live output filter in app.py.
"""

import os
import json
import asyncio
import contextvars
from dotenv import load_dotenv
from langchain.agents import create_agent as create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag.answer import get_model

load_dotenv()

MCP_SERVERS = {
    "policy": {
        "command": "python",
        "args": ["mcp/policy_mcp_server.py"],
        "transport": "stdio",
    },
    "hr_data": {
        "command": "python",
        "args": ["mcp/hr_data_mcp_server.py"],
        "transport": "stdio",
    },
}

AGENT_SYSTEM_PROMPT = """You are an HR assistant agent. You have access to tools for searching HR
policy documents and looking up employee/PTO/benefits data. Decide which tools are needed to answer
each question — some questions need only policy search, others need employee data lookups too.

IMPORTANT — employee-specific tools (lookup_employee_profile, check_pto_balance,
lookup_benefits_status, create_mock_hr_ticket, draft_hr_email) are restricted to the authenticated
requester's own employee_id. If the requester is authenticated, their real employee_id will be
stated explicitly at the start of their message (e.g. "[Authenticated as employee_id: EMP001]").
Use that exact ID for any employee-specific tool call — never guess, invent, or use a placeholder
like "current_user_id". If no such authenticated employee_id is stated in the message, do not call
any employee-specific tool at all — tell the user they need to authenticate first. If a tool call
is rejected due to authentication/authorization, relay that clearly to the user rather than
retrying with a different ID. If a question is about general policy and doesn't need
employee-specific data, answer using search_policy_documents / check_policy_compliance alone.

A policy question phrased personally (e.g. "how many PTO days do I get", "what floating holidays
do I get") is still a GENERAL policy question if it can be fully answered from policy content
alone (e.g. standard tiers by tenure) without looking up this specific employee's stored data.
Answer these via search_policy_documents — do not ask the user to authenticate just because the
phrasing uses "I" or "my". Only require authentication when the question genuinely needs data
specific to this employee's own record (their actual balance, their actual profile, etc.), not
merely because the question is phrased in the first person.

Always explain which tools you used and why in your final answer. Never take irreversible actions
(like creating a ticket) without the user's explicit confirmation first. If a tool response has
status "confirmation_required", stop and relay that request to the user — do not call the tool
again with confirmed=True unless the user has explicitly said yes in this conversation."""

RECURSION_LIMIT = 10

GUARDED_EMPLOYEE_TOOLS = {
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "create_mock_hr_ticket",
    "draft_hr_email",
}

# The authenticated requester's employee_id for the current request, resolved
# from a bearer token in app.py. None if the request was unauthenticated.
_authenticated_employee_id: contextvars.ContextVar = contextvars.ContextVar(
    "authenticated_employee_id", default=None
)


def _guard_employee_id_tool(tool):
    """
    Wraps a single tool so it refuses to execute unless the employee_id (or
    to_employee_id) argument matches the AUTHENTICATED requester's own ID —
    not merely an ID mentioned somewhere in the conversation. This is a real
    authorization check, replacing the earlier message-mention check that
    adversarial testing showed could be bypassed 3/3 times.

    Tuple-aware: newer langchain-mcp-adapters versions can configure a tool
    with response_format="content_and_artifact", expecting every return
    value (including rejections from this guard) to be a 2-tuple of
    (content, artifact) rather than a bare value. Detecting this at wrap
    time and shaping the rejection accordingly avoids a runtime error when
    that dependency version changes underneath this code.
    """
    original_coroutine = tool.coroutine
    response_format = getattr(tool, "response_format", "content")

    async def guarded_coroutine(*args, **kwargs):
        target_id = kwargs.get("employee_id") or kwargs.get("to_employee_id")
        authenticated_id = _authenticated_employee_id.get()

        def _reject(message: str):
            error_dict = {"error": message}
            if response_format == "content_and_artifact":
                return json.dumps(error_dict), error_dict
            return error_dict

        if not authenticated_id:
            return _reject(
                "Rejected: this action requires authentication. No valid session/token was "
                "provided with this request, so no employee-specific data can be accessed. "
                "Tell the user they need to authenticate before this request can proceed."
            )

        if target_id and target_id.upper() != authenticated_id.upper():
            return _reject(
                f"Rejected: employee_id '{target_id}' does not match the authenticated "
                f"requester. You may only access data for your own employee_id "
                f"('{authenticated_id}'). Do not retry with a different ID — tell the user "
                f"this request is not authorized."
            )

        if "employee_id" in kwargs:
            kwargs["employee_id"] = authenticated_id
        if "to_employee_id" in kwargs:
            kwargs["to_employee_id"] = authenticated_id

        return await original_coroutine(*args, **kwargs)

    tool.coroutine = guarded_coroutine
    return tool


def apply_employee_id_guard(tools: list) -> list:
    for tool in tools:
        if tool.name in GUARDED_EMPLOYEE_TOOLS:
            _guard_employee_id_tool(tool)
    return tools


async def build_agent():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    tools = apply_employee_id_guard(tools)

    model = get_model()
    agent = create_react_agent(model, tools, system_prompt=AGENT_SYSTEM_PROMPT)
    return agent


async def invoke_with_retry(
    agent,
    messages,
    max_attempts: int = 3,
    delay: float = 3.0,
    authenticated_employee_id: str = None,
):
    """
    Sets the authenticated requester's employee_id for this request (used by
    the guard above), then invokes the agent with the existing retry logic
    for OpenRouter's transient 504/429 errors.
    """
    _authenticated_employee_id.set(authenticated_employee_id)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except ValueError as e:
            last_error = e
            print(f"[retry] attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(delay)
    raise last_error


def _unwrap_tool_output(raw_content):
    if isinstance(raw_content, list) and raw_content:
        block = raw_content[0]
        if isinstance(block, dict) and "text" in block:
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, TypeError):
                return block["text"]
    if isinstance(raw_content, str):
        try:
            return json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            return raw_content
    return raw_content


def build_trace(result: dict) -> list[dict]:
    trace = []
    step_num = 0
    pending_calls = {}

    for msg in result["messages"]:
        msg_type = msg.__class__.__name__

        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                pending_calls[call["id"]] = {
                    "tool": call["name"],
                    "args": call["args"],
                }

        elif msg_type == "ToolMessage":
            call_id = getattr(msg, "tool_call_id", None)
            call_info = pending_calls.get(call_id, {"tool": "unknown", "args": {}})
            step_num += 1
            output = _unwrap_tool_output(msg.content)

            is_escalation = isinstance(output, dict) and (
                output.get("status") == "confirmation_required"
                or (isinstance(output.get("error"), str) and output["error"].startswith("Rejected:"))
            )

            trace.append({
                "step": step_num,
                "tool": call_info["tool"],
                "args": call_info["args"],
                "output": output,
                "escalation": is_escalation,
            })

    return trace


def print_trace(trace: list[dict], question: str, result: dict) -> None:
    print("\n" + "=" * 70)
    print("OPERATIONAL TRACE")
    print("=" * 70)

    print(f"\n[LOOP START] recursion_limit={RECURSION_LIMIT}")
    print(f"  user question: \"{question}\"")

    if not trace:
        print("\n(no tool calls — answered from policy corpus or general reasoning alone)")
    for step in trace:
        flag = "  [ESCALATION]" if step["escalation"] else ""
        print(f"\nStep {step['step']}: called `{step['tool']}`{flag}")
        print(f"  args:   {json.dumps(step['args'], indent=2)}")
        output_str = json.dumps(step["output"], indent=2) if isinstance(step["output"], (dict, list)) else str(step["output"])
        print(f"  output: {output_str}")

    final_msg = result["messages"][-1]
    ended_naturally = final_msg.__class__.__name__ == "AIMessage" and not getattr(final_msg, "tool_calls", None)
    any_escalation = any(step["escalation"] for step in trace)

    print(f"\n[LOOP END] {len(trace)} tool call(s) executed")
    if ended_naturally:
        reason = "model returned a final answer with no further tool_calls (natural completion)"
    else:
        reason = "loop ended without a plain-text final message (unexpected — check recursion_limit or agent errors)"
    print(f"  termination reason: {reason}")
    if any_escalation:
        print(f"  note: at least one step required confirmation or was blocked by a guardrail")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":

    async def main():
        agent = await build_agent()

        question = "How many PTO days does EMP001 have left?"

        result = await invoke_with_retry(
            agent,
            [{"role": "user", "content": question}],
            authenticated_employee_id=None,
        )

        trace = build_trace(result)
        print_trace(trace, question, result)

        print("FINAL ANSWER:\n")
        print(result["messages"][-1].content)

    asyncio.run(main())
