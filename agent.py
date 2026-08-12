"""
Agent orchestrator — single LangGraph agent that discovers and calls tools
from both MCP servers (policy_mcp_server.py, hr_data_mcp_server.py).
Pattern adapted from real-estate-agent's real_estate_agent.py.

STATUS: core loop validated (PTO Request Guidance workflow verified end-to-end).
Next: second workflow (Remote Work Eligibility), operational trace logging,
confirmation-gating verification for irreversible actions.
"""

import os
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
# than relying on the framework's undocumented default. Each tool call + LLM
# response pair counts as ~1-2 steps, so 10 comfortably covers a 2-tool workflow
# with room for a retry, while still bounding runaway loops.
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


if __name__ == "__main__":

    async def main():
        agent = await build_agent()

        # --- Test question: confirmation-gate check ---
        # Should trigger create_mock_hr_ticket, hit confirmed=False, and the
        # agent should STOP and relay the confirmation_required response to
        # the user rather than silently retrying with confirmed=True.
        # question = (
        #     "I'm employee EMP001. Please create an HR ticket to ask about "
        #     "my remote work options."
        # )
        question = (
            "I'm employee EMP003, currently based in Remote-EU. Can I work from a "
            "country outside the EU for 6 weeks? What do I need to know?"
        )

        result = await invoke_with_retry(
            agent,
            [{"role": "user", "content": question}],
        )
        print(result["messages"][-1].content)

    asyncio.run(main())
