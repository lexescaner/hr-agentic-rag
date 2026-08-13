"""
Agent orchestrator — single LangGraph agent that discovers and calls tools
from both MCP servers (policy_mcp_server.py, hr_data_mcp_server.py).
Pattern adapted from real-estate-agent's real_estate_agent.py.

STATUS: both required workflows verified end-to-end. Confirmation gate
verified. Operational trace logging implemented.

Day 5: eval found the agent occasionally hallucinated an employee_id when
calling employee-specific tools on questions that never provided one (e.g.
"How many floating holidays do I get?" -> fabricated employee_id
"EMP12345"). A prompt-level fix (AGENT_SYSTEM_PROMPT instruction) reduced
but did not reliably eliminate this — tested 1/3 success. This version adds
a CODE-LEVEL guard: employee-specific tool calls are intercepted and
blocked unless the employee_id actually appears in the user's own message,
mirroring the hard-enforced pattern already used for create_mock_hr_ticket's
confirmed=True check. Unlike the prompt-only fix, this cannot be bypassed by
the model choosing not to follow an instruction.
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
lookup_benefits_status, create_mock_hr_ticket, draft_hr_email) all require a real employee_id.
NEVER invent or guess an employee_id. Only call these tools when the user has explicitly provided
their employee ID in the conversation. If a question is about general policy (e.g. "how many
floating holidays do employees get") and no employee ID has been given, answer using
search_policy_documents / check_policy_compliance alone — do not call an employee-specific tool
with a fabricated ID. If the question genuinely requires employee-specific data and no ID was
given, ask the user for their employee ID instead of guessing one.

Note: employee-specific tool calls are also validated in code — a call with an employee_id that
was not actually provided by the user will be rejected automatically. If you see a rejection, ask
the user for their real employee ID rather than retrying with a different guess.

Always explain which tools you used and why in your final answer. Never take irreversible actions
(like creating a ticket) without the user's explicit confirmation first. If a tool response has
status "confirmation_required", stop and relay that request to the user — do not call the tool
again with confirmed=True unless the user has explicitly said yes in this conversation."""

# Explicit ceiling on agent loop iterations — a documented design decision rather
# than relying on the framework's undocumented default.
RECURSION_LIMIT = 10

# Names of tools that accept an employee_id / to_employee_id argument and
# must not be allowed to run with a value the user never actually provided.
GUARDED_EMPLOYEE_TOOLS = {
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "create_mock_hr_ticket",
    "draft_hr_email",
}

# Holds the current request's raw user message so the tool guard (below) can
# check it during execution. Set fresh at the start of every invoke_with_retry
# call. Using a ContextVar (not a plain global) so this is safe under
# asyncio's concurrent execution model — each request's value stays isolated
# to that request's async context.
_current_user_message: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_message", default=""
)


def _guard_employee_id_tool(tool):
    """
    Wraps a single tool so it refuses to execute if the employee_id (or
    to_employee_id) argument the model chose does not actually appear in the
    user's own message for this request. This is a hard, code-enforced
    guardrail — unlike a system-prompt instruction, the model cannot bypass
    it by simply not following the request.

    On rejection, returns a structured error (not an exception) so the
    agent's ReAct loop sees it as a normal tool result and can course-correct
    (e.g. ask the user for their real ID) within the same turn, exactly like
    it already does for create_mock_hr_ticket's confirmation_required case.
    """
    original_coroutine = tool.coroutine

    async def guarded_coroutine(*args, **kwargs):
        employee_id = kwargs.get("employee_id") or kwargs.get("to_employee_id")
        if employee_id:
            user_message = _current_user_message.get()
            if employee_id.upper() not in user_message.upper():
                return {
                    "error": (
                        f"Rejected: employee_id '{employee_id}' was not provided by the user in "
                        f"this message. This tool call was blocked before running. Do not guess "
                        f"or invent an employee ID — ask the user to provide their real employee "
                        f"ID, or answer from policy documents alone if the question doesn't "
                        f"actually require employee-specific data."
                    )
                }
        return await original_coroutine(*args, **kwargs)

    tool.coroutine = guarded_coroutine
    return tool


def apply_employee_id_guard(tools: list) -> list:
    """Applies the employee_id guard to every tool named in GUARDED_EMPLOYEE_TOOLS."""
    for tool in tools:
        if tool.name in GUARDED_EMPLOYEE_TOOLS:
            _guard_employee_id_tool(tool)
    return tools


async def build_agent():
    """
    Connects to both MCP servers, discovers their tools, applies the
    employee_id guard to the relevant tools, and builds a LangGraph agent.
    """
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    tools = apply_employee_id_guard(tools)

    model = get_model()
    agent = create_react_agent(model, tools, system_prompt=AGENT_SYSTEM_PROMPT)
    return agent


async def invoke_with_retry(agent, messages, max_attempts: int = 3, delay: float = 3.0):
    """
    OpenRouter's free-tier models occasionally return a 504/429 packaged
    inside a 200 OK response body (application-level error, not an
    HTTP-level failure), which the underlying OpenAI SDK's built-in retry
    logic does not catch. This wrapper retries at the agent-invocation
    level to ride out that transient flakiness instead of failing the
    whole run on one bad response.

    Also sets the current-request context for the employee_id guard, based
    on the concatenated text of this request's user message(s).
    """
    user_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
    _current_user_message.set(user_text)

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
    """
    MCP tool results arrive as a list of content blocks, e.g.:
        [{"type": "text", "text": "<json string>", "id": "..."}]
    This unwraps that envelope and parses the inner JSON string into an
    actual dict, so the trace shows the real tool output instead of the
    MCP transport wrapper around it.
    """
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
    """
    Walks the full message history returned by agent.ainvoke() and extracts a
    structured operational trace: every tool the agent decided to call, the
    exact arguments it passed, the exact (unwrapped) output that tool
    returned, and whether that step represents an escalation (e.g. a
    confirmation-required irreversible action, or an employee_id guard
    rejection).
    """
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
    """
    Pretty-print the operational trace for terminal/demo-video visibility,
    including the loop's start and end so the full lifecycle is visible —
    not just the tool calls that happened in between.
    """
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

        # Regression test 1: previously caused a hallucinated employee_id.
        # Should now either answer from policy alone, or (if it still tries
        # to guess an ID) get blocked by the code-level guard rather than
        # silently succeeding with fabricated data.
        question = "How many floating holidays do I get?"

        result = await invoke_with_retry(
            agent,
            [{"role": "user", "content": question}],
        )

        trace = build_trace(result)
        print_trace(trace, question, result)

        print("FINAL ANSWER:\n")
        print(result["messages"][-1].content)

    asyncio.run(main())
