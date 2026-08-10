"""
HR Data MCP Server
Exposes tools over mock structured HR data (employees, PTO, benefits, tickets).
Mirrors the structure of crm_mcp_server.py from real-estate-agent, but reads
from local JSON files instead of sqlite3 (per project brief, JSON/CSV/SQLite
are all acceptable for mock data).

Tools exposed:
  - lookup_employee_profile(employee_id)
  - check_pto_balance(employee_id)
  - lookup_benefits_status(employee_id)
  - create_mock_hr_ticket(employee_id, subject, details)   # requires confirmation
  - draft_hr_email(to_employee_id, subject, body)          # mock action, not sent
"""

import json
import os
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")

mcp = FastMCP("hr-data-server")


def _load(filename: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        return json.load(f)


def _save(filename: str, data: list[dict]) -> None:
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)


@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict:
    """Look up an employee's profile by employee_id (role, location, manager, employment type)."""
    employees = _load("employees.json")
    match = next((e for e in employees if e["employee_id"] == employee_id), None)
    if not match:
        return {"error": f"No employee found with id {employee_id}"}
    return match


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict:
    """Check an employee's current PTO balance (total, used, remaining, pending requests)."""
    balances = _load("pto_balances.json")
    match = next((b for b in balances if b["employee_id"] == employee_id), None)
    if not match:
        return {"error": f"No PTO record found for {employee_id}"}
    return match


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict:
    """Look up an employee's benefits elections and eligibility."""
    benefits = _load("benefits.json")
    match = next((b for b in benefits if b["employee_id"] == employee_id), None)
    if not match:
        return {"error": f"No benefits record found for {employee_id}"}
    return match


@mcp.tool()
def create_mock_hr_ticket(employee_id: str, subject: str, details: str, confirmed: bool = False) -> dict:
    """
    Create a mock HR ticket. This is an irreversible-style action, so it must
    only be executed with confirmed=True, which the agent should only set
    after the user has explicitly confirmed the action.
    """
    if not confirmed:
        return {
            "status": "confirmation_required",
            "message": "This action creates an HR ticket. Please confirm before proceeding.",
            "preview": {"employee_id": employee_id, "subject": subject, "details": details},
        }
    tickets = _load("tickets.json")
    new_ticket = {
        "ticket_id": f"TCK{len(tickets) + 1:04d}",
        "employee_id": employee_id,
        "subject": subject,
        "details": details,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(new_ticket)
    _save("tickets.json", tickets)
    return {"status": "created", "ticket": new_ticket}


@mcp.tool()
def draft_hr_email(to_employee_id: str, subject: str, body: str) -> dict:
    """
    Draft (but do not send) an HR-related email. Always a mock/preview action —
    no email is actually sent by this tool.
    """
    return {
        "status": "draft_only",
        "to": to_employee_id,
        "subject": subject,
        "body": body,
        "note": "This is a mock draft. No email was sent.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
