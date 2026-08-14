# AI Tooling

## Tools Used

**Claude (Anthropic)**, via the standard chat interface, was used throughout the entire build —
from Day 1 scaffolding through Day 5 evaluation and this documentation. No other AI coding tools
(Copilot, Cursor, etc.) were used; Claude was the sole AI assistant for this project.

## How It Was Used

**Code generation and scaffolding.** Claude wrote the initial project scaffold (corpus documents,
mock data, MCP server stubs), the RAG pipeline (`rag/policy_rag.py`, `rag/answer.py`), the agent
orchestrator (`agent.py`), the Flask web app (`app.py`), the CI/CD workflow and smoke tests, and
the deployment configuration (`Procfile`, `render.yaml`). Code was generated in response to
specific, incremental requests rather than a single large generation pass — each day's work was
built, tested, and verified before moving to the next.

**Debugging and root-cause diagnosis.** Claude diagnosed and fixed several real issues encountered
during the build:
- A `create_react_agent` import path deprecated mid-project, requiring a migration to
  `langchain.agents.create_agent` with a different keyword argument (`system_prompt=` vs `prompt=`)
- A Render deployment `Out of memory` crash, traced to `sentence-transformers`/PyTorch's memory
  footprint and fixed by switching to Chroma's lightweight ONNX embedding function
- Recurring OpenRouter free-tier rate limiting (504s and 429s), diagnosed as an application-level
  error not caught by the SDK's built-in retry logic, requiring a custom retry wrapper
- A hallucinated `employee_id` bug found through evaluation, first addressed with a prompt-level
  fix (measured at 1/3 success), then a code-level fix (measured at 6/6 success) — see
  `design-and-evaluation.md` Section 5 for the full case study

**Evaluation design and analysis.** Claude designed the 25-question evaluation set across all
5 required categories, built the evaluation runner and a systematic verification script (citation
validity + a groundedness heuristic), and — critically — reviewed its own heuristic's flagged
results by hand rather than reporting them uncritically; all 5 initially-flagged "groundedness"
issues were confirmed to be false positives from the heuristic itself (markdown list numbering
misread as factual claims), not real problems.

**Diagrams and documentation.** Claude designed and iteratively refined the architecture and
tool-call diagrams (SVG), including several rounds of correction based on visual review — fixing
disconnected arrows, cramped spacing, inconsistent color theming, and ambiguous labels. It also
produced `deployed.md`, `tool-call-breakdown.md`, `design-and-evaluation.md`, and this file.

## What Worked Well

- **Iterative, verify-before-proceeding development.** Each component (RAG retrieval, agent
  orchestration, deployment, guardrails) was built and tested before moving to the next, which
  caught real issues early rather than compounding them.
- **Rigorous handling of eval findings.** When evaluation surfaced the hallucinated-employee-ID
  issue, Claude didn't just apply one fix and declare it resolved — it measured the first fix's
  actual success rate (1/3), recognized that as insufficient, built a stronger code-level fix, and
  re-tested it multiple times (6/6) before considering it closed.
- **Willingness to correct its own claims.** Several early drafts of `design-and-evaluation.md`
  contained language that overstated what had actually been verified (e.g. "citations verified
  correct for all retrieval-based answers" when only a few had been spot-checked). These were
  caught during review and corrected to accurately reflect what was and wasn't systematically
  tested — that correction is what motivated actually building and running the systematic
  verification script.

## What Didn't Work Smoothly

- **OpenRouter free-tier instability required multiple rounds of diagnosis.** The same underlying
  category of problem (rate limits, timeouts, model delisting) surfaced repeatedly in different
  forms across Days 3-5, each requiring separate investigation before a durable fix (a dedicated
  API key with a credit top-up, plus a retry wrapper and request spacing) resolved it.
- **Diagram generation required several visual-review iterations.** SVG diagrams generated from
  text descriptions initially had real layout problems — disconnected arrow segments, boxes placed
  too close together, inconsistent color themes across diagrams built in different sessions — that
  weren't apparent from the SVG source code alone and needed to be caught by rendering and visually
  inspecting the output, then corrected in follow-up passes.
- **File path assumptions occasionally caused friction.** A few working-directory mismatches (e.g.
  a script's default relative output path creating a nested folder when run from inside that
  folder) required troubleshooting that a more defensive default would have avoided.
