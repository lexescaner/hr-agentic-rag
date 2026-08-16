# AI Tooling

## Tools Used

**Claude (Anthropic)**, via the standard chat interface, was used throughout the entire build —
from Day 1 scaffolding through final evaluation, security hardening, and this documentation. No
other AI coding tools (Copilot, Cursor, etc.) were used; Claude was the sole AI assistant for this
project.

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
- **A silent empty-answer failure mode**, distinct from the rate-limit errors above: the model
  would occasionally return a blank or corrupted completion (empty string, leaked chat-template
  tokens like `<pad>` or `<|channel|>`) with no exception raised, which the existing retry logic
  never caught since nothing actually raised. Fixed by extending `invoke_with_retry()` to treat an
  empty final answer as a retryable failure.
- **A tool-call-skipping bug in the confirmation gate**: the agent would sometimes describe
  creating an HR ticket in prose instead of actually calling `create_mock_hr_ticket`, meaning the
  code-level confirmation check never ran at all. Root-caused to the system prompt's "don't retry
  with `confirmed=True` unless the user said yes" instruction being over-applied to skip the
  *first* call too, not just a second one. Fixed by clarifying the prompt; reverified 3/3 clean.
- **A workflow regression contradicting an already-"verified" claim**: the Remote Work Eligibility
  workflow began skipping `lookup_employee_profile` in some runs, producing generic answers that
  never reflected the employee's actual contractor status — directly contradicting an earlier
  section of this same design doc that claimed the workflow was verified. Fixed with an explicit
  prompt instruction to check employment type before compliance questions; reverified 3/3 clean.

**Security hardening.** This gap was first surfaced by a structured review of the codebase against
SWEBOK v4's 18 knowledge areas (full citation in `design-and-evaluation.md`'s References) —
Software Security was the one area flagged as needing action, which motivated building and running
the adversarial suite described below. An adversarial test suite (8 attacks across 3 categories:
direct jailbreak, authorization bypass, system-prompt disclosure) found real gaps at baseline — 4
of 8 attacks succeeded, including a genuine authorization bypass where the original guard checked
only whether an employee ID was *mentioned* in a message, not whether the requester actually *was*
that employee (3/3 bypass attempts succeeded). Claude rewrote the guard to check a
bearer-token-resolved authenticated identity instead, added an output filter for the
system-prompt-disclosure gap, and added rate limiting (previously untested). Re-running the same
suite after these fixes: 0 of 8 attacks succeeded. A secondary, more subtle bug was found *during*
this fix — without knowing its own authenticated ID, the agent would guess a placeholder and only
self-correct by having the guess rejected and reading the real ID out of the error message; this
"fragile accidental side-channel" was closed by injecting the real authenticated ID directly into
the message context.

**Evaluation design, analysis, and re-verification.** Claude designed the 25-question evaluation
set across all 5 required categories, built the evaluation runner and a systematic verification
script (citation validity + a groundedness heuristic), and reviewed its own heuristic's flagged
results by hand rather than reporting them uncritically — every round of "flagged" groundedness
results was manually checked and confirmed to be heuristic false positives (markdown list
numbering misread as factual claims), not real problems. Tool-selection-match evolved across
several fix-and-reverify cycles as real issues were found (an eval-harness gap where
`run_eval.py` never sent auth tokens, producing misleading scores unrelated to actual agent
quality; the bugs above), each time re-running the full suite rather than assuming a fix worked.
Citation validity and answer-correctness checks were explicitly rerun against the final code state
rather than left standing from an earlier run, to confirm — not assume — they still held.

**CI/CD.** Claude built the GitHub Actions pipeline and smoke tests, and later found that the
`deploy` job's actual trigger had been left as a placeholder (`echo "TODO: replace with actual
deploy hook"`) rather than a real one — meaning deploys had been happening via Render's own
separate, ungated auto-deploy webhook, not through the gated CI pipeline the design doc described.
Fixed by wiring a real deploy hook via a GitHub secret, verified via a live test trigger and a
subsequent gated pipeline run.

**Diagrams and documentation.** Claude designed and iteratively refined the architecture and
tool-call diagrams (SVG and the `.drawio` source), including several rounds of correction based on
visual review — fixing disconnected arrows, cramped spacing, inconsistent color theming, ambiguous
labels, and a diagram that visually implied the pre-security-fix, unauthenticated flow was still
current. It also produced `deployed.md`, `tool-call-breakdown.md`, `design-and-evaluation.md`, and
this file.

## What Worked Well

- **Iterative, verify-before-proceeding development.** Each component (RAG retrieval, agent
  orchestration, deployment, guardrails, security) was built and tested before moving to the next,
  which caught real issues early rather than compounding them.
- **Rigorous handling of eval findings, applied repeatedly, not just once.** The
  hallucinated-employee-ID case (1/3 vs. 6/6) set the pattern — measure the fix, don't just apply
  it — and that same discipline was later applied to the Q14 confirmation-gate bug, the Q7
  empty-answer bug, and the Q15 workflow regression: each was found, root-caused, fixed, and
  reverified multiple times before being considered closed, not fixed once and assumed durable.
- **Willingness to correct its own claims, including ones already written into the document.**
  Several passes of `design-and-evaluation.md` contained language that overstated what had
  actually been verified — an early draft claiming "citations verified correct for all
  retrieval-based answers" when only a few had been spot-checked; later, a Section 3 claim that
  the Remote Work Eligibility workflow was verified, which Q15's regression directly contradicted
  until it was fixed and reverified; a stale "7 tools total" that undercounted the actual 8; a
  diagram showing an authentication flow that no longer matched the deployed system. Each was
  caught during review — several by the user's own close reading and direct questions — and
  corrected rather than left standing.
- **A genuine security finding, not a token gesture.** The authorization-bypass discovery (3/3
  succeeded at baseline) was a real gap in a real mechanism, found by actually attacking the
  system rather than assuming the guard worked because it looked reasonable in code review.

## What Didn't Work Smoothly

- **OpenRouter free-tier instability required multiple, distinct rounds of diagnosis.** The same
  general category of problem (rate limits, timeouts, model delisting, and — later — silent
  empty/corrupted completions) surfaced repeatedly in different forms across the build, each
  requiring separate investigation before a durable combination of fixes (a dedicated API key with
  a credit top-up, a retry wrapper, request spacing, an empty-answer retry extension, and
  eventually switching to the paid model tier for evaluation reliability) resolved it.
- **Diagram generation required several visual-review iterations, more than once.** SVG diagrams
  generated from text descriptions initially had real layout problems — disconnected arrow
  segments, boxes placed too close together, inconsistent color themes across diagrams built in
  different sessions, oversized arrowheads on short lines — that weren't apparent from the SVG
  source code alone and needed to be caught by rendering and visually inspecting the output. This
  happened in more than one pass: a diagram fixed for layout issues was later found to still
  contain a stale, pre-security-fix description of the system's authentication flow, requiring a
  second, separate round of correction.
- **File path assumptions occasionally caused friction.** A few working-directory mismatches (e.g.
  a script's default relative output path creating a nested folder when run from inside that
  folder) required troubleshooting that a more defensive default would have avoided.
