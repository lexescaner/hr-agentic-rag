"""
Policy MCP Server
Exposes RAG-backed tools for searching and retrieving HR policy documents.
Mirrors the structure of property_data_mcp_server.py from real-estate-agent,
backed by a Chroma vector index (see rag/policy_rag.py).

Tools exposed:
  - search_policy_documents(query, top_k=4)
  - get_policy_section(doc_id, section)
  - check_policy_compliance(topic, employee_context)
"""

import sys
import os

# Allow importing rag/ as a sibling package when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP
from rag.policy_rag import search, get_section

mcp = FastMCP("policy-server")


@mcp.tool()
def search_policy_documents(query: str, top_k: int = 4) -> dict:
    """
    Search the HR policy corpus for chunks relevant to the query.
    Returns retrieved chunks with document_id, section, snippet, and distance
    (lower = more relevant) so the agent can cite sources in its final answer.
    """
    results = search(query, top_k=top_k)
    if not results:
        return {"results": [], "note": "No relevant policy content found for this query."}
    return {"results": results}


@mcp.tool()
def get_policy_section(doc_id: str, section: str) -> dict:
    """
    Fetch a specific section of a specific policy document by doc_id and
    section heading, for when the agent needs the full text rather than a
    top-k chunk (e.g. after search_policy_documents identifies the right doc).
    """
    result = get_section(doc_id, section)
    if result is None:
        return {"error": f"No section '{section}' found in document '{doc_id}'"}
    return result


@mcp.tool()
def check_policy_compliance(topic: str, employee_context: dict) -> dict:
    """
    Given a topic (e.g. 'remote_work', 'expense') and employee context
    (role, location, employment_type), retrieve relevant policy chunks and
    return them for the agent to reason over and produce a compliance verdict
    with citations. This tool retrieves evidence; the agent's LLM call
    synthesizes the actual compliant/non-compliant/needs-review judgment.
    """
    query = f"{topic} policy for {employee_context}"
    results = search(query, top_k=5)
    return {
        "topic": topic,
        "employee_context": employee_context,
        "retrieved_evidence": results,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
