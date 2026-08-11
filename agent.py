"""
Agent orchestrator — single LangGraph agent that discovers and calls tools
from both MCP servers (policy_mcp_server.py, hr_data_mcp_server.py).
Pattern adapted from real-estate-agent's real_estate_agent.py.

STATUS: skeleton only. Next session: fill in the two required workflows
(e.g. Remote Work Eligibility, PTO Request Guidance), operational trace
logging, and confirmation-gating for irreversible actions.
"""

import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
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
(like creating a ticket) without the user's explicit confirmation first."""


async def build_agent():
    """
    Connects to both MCP servers, discovers their tools, and builds a
    LangGraph agent that can call them. TODO next session: workflow logic,
    trace logging, failure handling.
    """
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()

    model = get_model()
    agent = create_react_agent(model, tools, prompt=AGENT_SYSTEM_PROMPT)
    return agent


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = await build_agent()
        # Placeholder smoke test — confirms agent + both MCP servers connect
        result = await agent.ainvoke(
            # {"messages": [{"role": "user", "content": "How many PTO days does employee EMP001 have left?"}]}
            {"messages": [{"role": "user", "content": "I'm employee EMP001. Can I take 3 days of PTO next week? What do I need to know?"}]}
        )
        print(result["messages"][-1].content)

    asyncio.run(main())
