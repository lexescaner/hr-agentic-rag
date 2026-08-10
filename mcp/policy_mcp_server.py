"""
Policy MCP Server
Exposes RAG-backed tools for searching and retrieving HR policy documents.
Mirrors the structure of property_data_mcp_server.py from real-estate-agent,
but backs onto a Chroma vector index instead of sqlite3.

Tools exposed:
  - search_policy_documents(query, top_k=4)
  - get_policy_section(doc_id, section)
  - check_policy_compliance(topic, employee_context)
"""

from mcp.server.fastmcp import FastMCP

# TODO (Day 2): import the RAG index built in rag/policy_rag.py
# from rag.policy_rag import search, get_section

mcp = FastMCP("policy-server")


@mcp.tool()
def search_policy_documents(query: str, top_k: int = 4) -> dict:
    """
    Search the HR policy corpus for chunks relevant to the query.
    Returns retrieved chunks with document_id, section, snippet, and score
    so the agent can cite sources in its final answer.
    """
    # TODO (Day 2): wire to Chroma retrieval
    raise NotImplementedError("Wire to RAG index in Day 2")


@mcp.tool()
def get_policy_section(doc_id: str, section: str) -> dict:
    """
    Fetch a specific section of a specific policy document by ID and section
    heading, for when the agent needs the full text rather than a chunk.
    """
    # TODO (Day 2)
    raise NotImplementedError("Wire to RAG index in Day 2")


@mcp.tool()
def check_policy_compliance(topic: str, employee_context: dict) -> dict:
    """
    Given a topic (e.g. 'remote_work', 'expense') and employee context
    (role, location, employment_type), retrieve relevant policy and return
    a structured compliant/non-compliant/needs-review verdict with citations.
    """
    # TODO (Day 2/3): combine retrieval + light rule evaluation
    raise NotImplementedError("Wire to RAG index in Day 2/3")


if __name__ == "__main__":
    mcp.run(transport="stdio")
