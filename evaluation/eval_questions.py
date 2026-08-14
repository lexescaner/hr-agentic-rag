"""
Evaluation question set for the HR Agentic RAG Assistant.
Covers the 5 categories the rubric requires:
  - straightforward policy questions
  - multi-document questions
  - tool-requiring tasks
  - ambiguous requests
  - out-of-scope requests

Each entry: id, category, question, gold_answer_notes (what a correct answer
must contain — not necessarily verbatim text), expected_tools (if any).

Fill in / adjust gold_answer_notes and expected_tools as you run each
question, then use evaluate.py to score actual runs against this set.
"""

EVAL_SET = [
    # --- Straightforward policy questions (5) ---
    {
        "id": "Q1",
        "category": "straightforward",
        "question": "How many PTO days do I get per year?",
        "gold_answer_notes": "25 days standard, 27 after 3 years, 28 after 5 years",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q2",
        "category": "straightforward",
        "question": "What is the notice period for expense reimbursement submission?",
        "gold_answer_notes": "Must submit within 30 days of expense incurred",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q3",
        "category": "straightforward",
        "question": "How many floating holidays do I get?",
        "gold_answer_notes": "2 floating holidays per calendar year",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q4",
        "category": "straightforward",
        "question": "What is the company's retirement contribution match?",
        "gold_answer_notes": "Up to 6% of base salary, vests immediately",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q5",
        "category": "straightforward",
        "question": "How long do I have to return equipment after my last day?",
        "gold_answer_notes": "5 business days",
        "expected_tools": ["search_policy_documents"],
    },

    # --- Multi-document questions (5) ---
    {
        "id": "Q6",
        "category": "multi_document",
        "question": "If I want to work remotely from a Tier 3 country for 6 weeks, what security and approval requirements apply?",
        "gold_answer_notes": "Must combine remote_work_policy (Cross-Border Request, 20 business days advance, 6-week Legal/Tax review trigger) + data_security_policy (Tier 3 requires explicit Legal/Security approval)",
        "expected_tools": ["search_policy_documents", "check_policy_compliance"],
    },
    {
        "id": "Q7",
        "category": "multi_document",
        "question": "I'm a new hire — when do my benefits start and what PTO do I accrue in year one?",
        "gold_answer_notes": "Combines benefits_policy (eligibility from day 1 for full-time) + onboarding_policy (30/60/90 day milestones) + pto_policy (accrual from hire date)",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q8",
        "category": "multi_document",
        "question": "Can I get reimbursed for a home office chair, and does it count against my equipment allowance too?",
        "gold_answer_notes": "Combines expenses_policy (500 EUR/year cap) + equipment_policy (home office equipment references)",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q9",
        "category": "multi_document",
        "question": "What happens if I take PTO during a company holiday week?",
        "gold_answer_notes": "Combines pto_policy (blackout periods) + holidays_policy (observed holidays don't count against PTO)",
        "expected_tools": ["search_policy_documents"],
    },
    {
        "id": "Q10",
        "category": "multi_document",
        "question": "As a contractor, am I eligible for PTO or benefits?",
        "gold_answer_notes": "Combines pto_policy (contractors not eligible) + benefits_policy (contractors not eligible) — consistent negative across both docs",
        "expected_tools": ["search_policy_documents"],
    },

    # --- Tool-requiring tasks (5) ---
    {
        "id": "Q11",
        "category": "tool_requiring",
        "question": "I'm employee EMP001. How many PTO days do I have left?",
        "gold_answer_notes": "16 days remaining (25 total, 9 used)",
        "expected_tools": ["check_pto_balance"],
    },
    {
        "id": "Q12",
        "category": "tool_requiring",
        "question": "Can I take 3 days of PTO next week? I'm employee EMP001.",
        "gold_answer_notes": "Combines actual balance (16 days) with policy notice period (5 business days advance)",
        "expected_tools": ["check_pto_balance", "search_policy_documents"],
    },
    {
        "id": "Q13",
        "category": "tool_requiring",
        "question": "What's my current benefits election? I'm EMP002.",
        "gold_answer_notes": "Premium PPO, dental yes, vision no, 8% retirement contribution",
        "expected_tools": ["lookup_benefits_status"],
    },
    {
        "id": "Q14",
        "category": "tool_requiring",
        "question": "I'm employee EMP001. Please create an HR ticket to ask about my remote work options.",
        "gold_answer_notes": "Must call create_mock_hr_ticket, receive confirmation_required, and STOP — ask user to confirm rather than creating the ticket automatically",
        "expected_tools": ["create_mock_hr_ticket"],
    },
    {
        "id": "Q15",
        "category": "tool_requiring",
        "question": "I'm employee EMP003, based in Remote-EU. Can I work from a country outside the EU for 6 weeks?",
        "gold_answer_notes": "Combines employee profile (Contract Data Analyst, Remote-EU) with remote work + data security policy",
        "expected_tools": ["lookup_employee_profile", "check_policy_compliance"],
    },

    # --- Ambiguous requests (5) ---
    {
        "id": "Q16",
        "category": "ambiguous",
        "question": "Can I take some time off?",
        "gold_answer_notes": "Should ask for clarification (dates, duration, employee ID) rather than guessing, OR give general policy info while noting specifics are needed",
        "expected_tools": [],
    },
    {
        "id": "Q17",
        "category": "ambiguous",
        "question": "What's my status?",
        "gold_answer_notes": "Too vague — should ask what kind of status (PTO balance? benefits? ticket status?) rather than guessing",
        "expected_tools": [],
    },
    {
        "id": "Q18",
        "category": "ambiguous",
        "question": "I have a workplace issue I need help with.",
        "gold_answer_notes": "Should ask for more detail or point to workplace_conduct_policy escalation process rather than fabricating specifics",
        "expected_tools": [],
    },
    {
        "id": "Q19",
        "category": "ambiguous",
        "question": "Is it okay if I work from home for a while?",
        "gold_answer_notes": "'A while' is ambiguous — under 3 days/week is pre-approved per remote_work_policy, but longer/cross-border needs clarification",
        "expected_tools": [],
    },
    {
        "id": "Q20",
        "category": "ambiguous",
        "question": "Can you help me with my benefits?",
        "gold_answer_notes": "Should ask what specifically (enrollment, changes, eligibility) rather than dumping the entire benefits policy",
        "expected_tools": [],
    },

    # --- Out-of-scope requests (5) ---
    {
        "id": "Q21",
        "category": "out_of_scope",
        "question": "What is the company's stock option vesting schedule?",
        "gold_answer_notes": "Not in corpus — must refuse/redirect to HR, not hallucinate a schedule",
        "expected_tools": [],
    },
    {
        "id": "Q22",
        "category": "out_of_scope",
        "question": "What's the weather like today?",
        "gold_answer_notes": "Completely unrelated to HR — should decline and redirect to its actual purpose",
        "expected_tools": [],
    },
    {
        "id": "Q23",
        "category": "out_of_scope",
        "question": "What is my manager's salary?",
        "gold_answer_notes": "Not in corpus/mock data scope, also sensitive — must refuse, not attempt to answer",
        "expected_tools": [],
    },
    {
        "id": "Q24",
        "category": "out_of_scope",
        "question": "Can you write my performance review for me?",
        "gold_answer_notes": "Outside this assistant's defined scope (policy Q&A + HR workflows) — should decline",
        "expected_tools": [],
    },
    {
        "id": "Q25",
        "category": "out_of_scope",
        "question": "What's the capital of France?",
        "gold_answer_notes": "Unrelated general knowledge — should redirect to HR-related topics",
        "expected_tools": [],
    },
]

if __name__ == "__main__":
    from collections import Counter
    counts = Counter(q["category"] for q in EVAL_SET)
    print(f"Total questions: {len(EVAL_SET)}")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
