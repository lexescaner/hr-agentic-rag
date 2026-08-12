"""
Agent orchestrator — single LangGraph agent that discovers and calls tools
from both MCP servers (policy_mcp_server.py, hr_data_mcp_server.py).
Pattern adapted from real-estate-agent's real_estate_agent.py.

STATUS: core loop validated. Both required workflows verified end-to-end
(PTO Request Guidance, Remote Work Eligibility). Confirmation gate verified.
Operational trace logging implemented, including escalation detection.
"""

import os
import json
import asyncio
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

Always explain which tools you used and why in your final answer. Never take irreversible actions
(like creating a ticket) without the user's explicit confirmation first. If a tool response has
status "confirmation_required", stop and relay that request to the user — do not call the tool
again with confirmed=True unless the user has explicitly said yes in this conversation."""

# Explicit ceiling on agent loop iterations — a documented design decision rather
# than relying on the framework's undocumented default.
RECURSION_LIMIT = 10


async def build_agent():
    """
    Connects to both MCP servers, discovers their tools, and builds a
    LangGraph agent that can call them.
    """
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()

    model = get_model()
    agent = create_react_agent(model, tools, system_prompt=AGENT_SYSTEM_PROMPT)
    return agent


async def invoke_with_retry(agent, messages, max_attempts: int = 3, delay: float = 3.0):
    """
    OpenRouter's free-tier models occasionally return a 504 packaged inside a
    200 OK response body (application-level error, not an HTTP-level failure),
    which the underlying OpenAI SDK's built-in retry logic does not catch.
    This wrapper retries at the agent-invocation level to ride out that
    transient flakiness instead of failing the whole run on one bad response.
    """
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
    confirmation-required irreversible action).

    This is the rubric-required "visible or logged trace of agent reasoning
    steps" — architectural level (tool name, args, output, escalation),
    not the model's hidden chain-of-thought.
    """
    trace = []
    step_num = 0
    pending_calls = {}  # tool_call_id -> {"tool": name, "args": args}

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

            is_escalation = isinstance(output, dict) and output.get("status") == "confirmation_required"

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
        flag = "  [ESCALATION — confirmation required]" if step["escalation"] else ""
        print(f"\nStep {step['step']}: called `{step['tool']}`{flag}")
        print(f"  args:   {json.dumps(step['args'], indent=2)}")
        output_str = json.dumps(step["output"], indent=2) if isinstance(step["output"], (dict, list)) else str(step["output"])
        print(f"  output: {output_str}")

    # Determine why the loop actually ended. LangGraph's create_react_agent
    # terminates when the final AIMessage has no tool_calls — i.e. the model
    # decided it had everything it needed and produced a plain-text answer.
    # This is the one place in this codebase that makes that termination
    # condition explicit and visible, rather than leaving it as undocumented
    # framework behavior.
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
        print(f"  note: at least one step required user confirmation before proceeding")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":

    async def main():
        agent = await build_agent()

        question = (
            "I'm employee EMP001. Can I take 3 days of PTO next week? "
            "What do I need to know?"
        )

        result = await invoke_with_retry(
            agent,
            [{"role": "user", "content": question}],
        )

        trace = build_trace(result)
        print_trace(trace, question, result)

        print("FINAL ANSWER:\n")
        print(result["messages"][-1].content)

    asyncio.run(main())
