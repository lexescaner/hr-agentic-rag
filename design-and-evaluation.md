# Design and Evaluation

*Methodology: this project was reviewed against SWEBOK v4 (Guide to the Software Engineering
Body of Knowledge).*

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [RAG Design](#2-rag-design)
3. [Agentic System Design](#3-agentic-system-design)
4. [MCP Server and Tool Integration](#4-mcp-server-and-tool-integration)
5. [Guardrails and Security](#5-guardrails-and-security)
   - 5.1 [Guardrail Design: Soft vs. Hard](#51-guardrail-design-soft-vs-hard)
   - 5.2 [Case Study: Hallucinated Employee ID](#52-case-study-hallucinated-employee-id)
   - 5.3 [Security Testing, Current State, and Recommendations](#53-security-testing-current-state-and-recommendations)
6. [Deployment](#6-deployment)
7. [Evaluation Results](#7-evaluation-results)
8. [Demo Tasks and Tool-Call Walkthrough](#8-demo-tasks-and-tool-call-walkthrough)
   - 8.1 [The Two Required Demo Tasks](#81-the-two-required-demo-tasks)
   - 8.2 [Tool-Call Mechanics Walkthrough — PTO Request Guidance](#82-tool-call-mechanics-walkthrough--pto-request-guidance)
9. [References](#references)
10. [Appendix: Repository File Audit](#appendix-repository-file-audit)

---

## 1. Architecture Overview

The system is a single-service agentic RAG application: one Flask web app, one LangGraph agent
orchestrator, two MCP servers, a local Chroma vector index, and mock structured HR data — all
running in one deployed process on Render's free tier, per the project brief's recommended
free-tier architecture.

<img src="assets/architecture-overview.svg" style="width:100%; max-width:900px;" />

*(An editable draw.io/diagrams.net source file for this diagram is included in the repository as
`architecture.drawio` — open it at [app.diagrams.net](https://app.diagrams.net) via File → Open
From → Device.)*

Both MCP servers run as local subprocesses over stdio, launched by `MultiServerMCPClient` when
the agent is built. The RAG index is rebuilt from the committed `docs/` corpus on every deploy
(via the build command), rather than persisted across deploys — appropriate since the corpus is
small and static, and Render's free tier disk is ephemeral.

---

## 2. RAG Design

**Corpus:** 9 policy documents (8 markdown, 1 HTML) covering PTO, remote work, data security,
expenses, benefits, onboarding, equipment, workplace conduct, and holidays.

**Chunking strategy:** heading-aware — splits on `##` (markdown) / `<h2>` (HTML) section headers,
implemented identically for both formats in `rag/policy_rag.py`. Chosen over fixed-size token
windows because the corpus documents were deliberately authored with one policy sub-topic per
section; heading boundaries already align with semantic boundaries, so this preserves chunk
coherence better than an arbitrary token cutoff would for this specific corpus. Produces 48 chunks
across the 9 documents.

**Embedding model:** Chroma's built-in `ONNXMiniLM_L6_V2` (same `all-MiniLM-L6-v2` model, via
`onnxruntime`). This was a deliberate switch from the original `SentenceTransformerEmbeddingFunction`
(sentence-transformers + PyTorch) after that approach caused the deployed app to exceed Render free
tier's 512 MB memory limit — see Section 6 (Deployment) for the full story. The same embedding
approach was also used in a separate prior project, suggesting this tradeoff isn't unique to this
codebase.

**Vector store:** Chroma, `PersistentClient`, local file storage at `chroma_db/` (gitignored,
rebuilt on every deploy from the committed corpus).

**Retrieval:** top-k similarity search (`n_results=top_k`, default 4), cosine distance computed
natively by Chroma/hnswlib — not implemented in this codebase.

**Prompting strategy:** retrieved chunks are injected into the LLM context with citation metadata
(`[Document Title, Section]` format), via a `ChatPromptTemplate` in `rag/answer.py` for the
single-shot RAG path, and via LangGraph's accumulated message history for the agentic path (see
Section 3).

**Guardrails (RAG-level, prompt-enforced):**
- Refuse/redirect out-of-corpus questions — verified (stock option vesting question correctly
  triggered 3 refined search attempts, then an honest "not found, contact HR" answer)
- Limit unsupported claims — answers instructed to stay traceable to retrieved context
- Distinguish policy facts from recommendations — answers explicitly label suggested next steps
  as "Recommendation:"

**Multi-document retrieval:** verified with a deliberately cross-referenced question (working
remotely from a Tier 3 country) that requires combining `remote_work_policy.md` (Cross-Border
Request process) with `data_security_policy.md` (Tier 1/2/3 data residency rules) — both were
correctly retrieved and cited together.

---

## 3. Agentic System Design

**Framework:** LangGraph's `create_agent` (aliased `create_react_agent` for continuity with an
earlier import path), a single ReAct loop: LLM call → check for `tool_calls` → execute tools →
loop back → repeat until the model returns a plain-text answer with no further tool calls.

**Orchestration decision-making:** the LLM itself decides, per question, whether to answer from
policy search alone or also call employee-specific tools — this is not hardcoded routing logic;
the decision is made by matching the question against each tool's name/description (auto-derived
from each `@mcp.tool()` function's docstring via MCP schema discovery).

**Two required workflows, both verified end-to-end:**
1. **PTO Request Guidance** — "Can I take 3 days of PTO next week? I'm employee EMP001." Agent
   calls `check_pto_balance` (confirms 16 days remaining) and `search_policy_documents` (retrieves
   notice period, approval, blackout period rules), synthesizes both into one grounded answer.
2. **Remote Work Eligibility** — "I'm employee EMP003, based in Remote-EU. Can I work from a
   country outside the EU for 6 weeks?" Agent calls `lookup_employee_profile` (confirms role/
   location/employment type) and `check_policy_compliance` (retrieves Tier 1/2/3 rules and the
   3-party approval chain), correctly flagging EMP003's contractor status as relevant.

**Operational trace logging:** `build_trace()` walks the full LangGraph message history and
extracts, per step: selected tool, exact arguments, exact (unwrapped) output, and whether the step
represents an escalation. `print_trace()` renders this with explicit `[LOOP START]`/`[LOOP END]`
markers, including the detected termination reason (natural completion vs. unexpected) — this
makes the loop's full lifecycle visible, not just the tool calls in between, and avoids exposing
hidden chain-of-thought.

**Loop control:** `RECURSION_LIMIT = 10`, passed explicitly via `config={"recursion_limit": ...}`
on every invocation — a project-authored ceiling rather than relying on the framework's
undocumented default.

**Resilience:** `invoke_with_retry()` retries up to 3 times (3s delay) on `ValueError`, since
OpenRouter's free-tier models occasionally return a 504/429 packaged inside a 200 OK response body
(an application-level error the underlying SDK's built-in retry logic doesn't catch).

**Graceful failure handling:** unknown employee IDs return a clean tool-level error
(`"No employee found with id ..."`) rather than crashing, confirmed via direct testing. Ambiguous
questions (Q16-Q20 in the eval set) mostly did not trigger tool calls — 4 of 5 answered directly,
while Q19 correctly used `search_policy_documents` to cite a concrete policy threshold rather than
asking a clarifying question the corpus already answers (Section 7) — suggesting the agent leans
toward answering directly except where policy content can resolve the ambiguity outright. The
specific content of the four no-tool answers was not individually reviewed for
clarification-seeking quality, so that part remains an observation from tool-call patterns rather
than a fully verified behavior.

---

## 4. MCP Server and Tool Integration

**Transport:** stdio, two separate server processes launched by `MultiServerMCPClient`.

**`mcp/policy_mcp_server.py`** — RAG-backed tools:

| Tool | Purpose |
|---|---|
| `search_policy_documents(query, top_k=4)` | Top-k retrieval over the policy corpus |
| `get_policy_section(doc_id, section)` | Fetch a specific section's full text |
| `check_policy_compliance(topic, employee_context)` | Retrieval scoped to a topic + employee context, for compliance-style questions |

**`mcp/hr_data_mcp_server.py`** — mock structured data tools:

| Tool | Purpose |
|---|---|
| `lookup_employee_profile(employee_id)` | Role, location, manager, employment type |
| `check_pto_balance(employee_id)` | Total/used/remaining PTO days |
| `lookup_benefits_status(employee_id)` | Benefits elections and eligibility |
| `create_mock_hr_ticket(employee_id, subject, details, confirmed=False)` | Mock ticket creation, gated behind explicit confirmation |
| `draft_hr_email(to_employee_id, subject, body)` | Mock-only draft, never actually sent |

7 tools total (exceeds the 5-tool minimum); at least one uses the RAG index, at least one uses
mock structured data, satisfying both required categories.

**Verified runtime tool-calling:** confirmed via CI (`tests/test_smoke.py` — live MCP tool
discovery and a direct tool call, no LLM required) and extensively via manual/eval testing that
the agent genuinely calls tools through the MCP layer at runtime, not via hard-coded direct calls.

---

## 5. Guardrails and Security

*Of the SWEBOK v4 areas reviewed, Software Security was the one gap this project acted on: the
review flagged a missing threat analysis, and the testing in 5.3 closes it.*

### 5.1 Guardrail Design: Soft vs. Hard

- **Soft guardrail** — a system-prompt instruction. The model can ignore it.
- **Hard guardrail** — a code-level check. Enforced regardless of model behavior.
- **Implemented hard guardrail:** `create_mock_hr_ticket` cannot create a ticket unless
  `confirmed=True` is explicitly passed in code — not requested via prompt.
- **Negative path verified** (guardrail correctly blocks): agent correctly stops and relays the
  confirmation request rather than auto-confirming — via terminal testing, live HTTP testing, and
  2 dedicated adversarial security tests explicitly attempting to bypass it (0/3 direct-jailbreak
  attacks succeeded, Section 5.3).
- **Positive path verified** (guardrail correctly allows, once genuine confirmation is given): a
  message with explicit confirmation front-loaded (*"...yes, I confirm, please go ahead and create
  it now"*) resulted in `create_mock_hr_ticket` being called with `confirmed=True`, returning
  `status: "created"` with a real `ticket_id` and timestamp. This rules out the guardrail simply
  being stuck closed — it is a genuine conditional gate, not a hardcoded refusal.

### 5.2 Case Study: Hallucinated Employee ID

- **Problem found via evaluation:** the question *"How many floating holidays do I get?"* (no
  employee ID given, answerable from policy alone) caused the agent to call
  `lookup_employee_profile` with a fabricated ID (`EMP12345`) instead of answering from policy or
  asking for clarification. Reproducible across both eval runs.
- **Attempt 1 — soft fix:** added a system-prompt instruction never to invent an ID.
  - Result: **1/3 success.** Reduced but did not reliably eliminate the behavior.
- **Attempt 2 — hard fix:** wrapped every employee-specific tool (`GUARDED_EMPLOYEE_TOOLS`) with a
  code check confirming the `employee_id` argument actually appears in the user's own message
  before the real tool executes; blocks with a structured rejection otherwise.
  - Result: **6/6 clean** across local and live testing.
- **Secondary finding:** 1 of those 6 trials showed the agent skipping the general-policy answer
  entirely and just asking for an ID (safe, but unhelpful).
  - Fix: refined the prompt to require answering the general part first.
  - Result: **4/4 clean** afterward (2 local, 2 live).
- **Summary:** soft fix → measured limits → hard fix → verified → secondary issue found → refined →
  re-verified. A complete account, not a single-pass claim.

### 5.3 Security Testing, Current State, and Recommendations

Guardrail work above targeted *hallucination*, not a formal security threat model. An adversarial
test suite (`evaluation/security_questions.py`, `evaluation/run_security_eval.py`) — 8 attacks
across 3 categories — was built and run against the live deployment to close that gap.
**Terminology:** "succeeded" means the *attacker* achieved their goal (a real gap); "failed" means
the system correctly resisted.

**Baseline result: 4 of 8 attacks succeeded.** Three fixes were implemented in response (auth
guard rewrite, output-filtering, rate limiting — detailed below), and the suite was re-run.
**Result after fixes: 0 of 8 attacks succeeded.**

| Area | Before | After | Current state | Recommendation |
|---|---|---|---|---|
| Irreversible-action confirmation | 0/3 succeeded | 0/3 succeeded | **Strong** — hard-enforced in code, resistant to prompt manipulation | None needed — reuse this pattern as the template for future guardrails |
| Fabricated-data prevention (`employee_id` guard) | 6/6 clean (5.2) | 6/6 clean (5.2) | **Strong for its purpose** — not designed as authorization | None needed for its intended scope |
| Requester authorization/authentication | **3/3 succeeded** | **0/3 succeeded** | **Fixed** — guard now checks authenticated identity (bearer token → `employee_id`), not message content | See note below on remaining scope |
| System prompt confidentiality | **1/2 succeeded** | **0/2 succeeded** | **Fixed** — live output filter blocks known internal markers before returning an answer | Filter is marker-based, not semantic — could miss a novel leak phrasing; monitor and expand marker list as needed |
| Corpus content trust | Not tested | Not tested | **Untested** — safe only because the corpus is self-authored | Add input sanitization before any external/editable content source; re-run the suite with a deliberately poisoned test document |
| Rate limiting / abuse prevention on `/chat` | Not tested | **Confirmed working** — 429s observed under concurrent load (2 runs, 25 concurrent requests each) | **Implemented and verified** — `flask-limiter`, 20/min and 60/hour per IP | Get a precise "N requests allowed before throttling" count once OpenRouter's own instability isn't confounding the test; monitor limits in production and adjust if legitimate usage patterns require it |
| Ongoing adversarial testing | One-time 8-question run | Re-run twice since: full suite post-fix (0/8), plus 2 targeted rate-limit concurrency tests | Manual only — not yet wired into CI/CD | Continue re-running whenever the system prompt, guardrails, or tool set change — same discipline as `run_eval.py`; consider adding this suite to CI so it runs on every push, not just manually |

### Notes

**On the authorization fix's remaining scope:** the guard rewrite closes the *architectural* gap —
the system now correctly separates "what's in the message" from "who's actually asking," verified
by the suite dropping from 3/3 to 0/3. **What this does not yet solve:** how a real user obtains a
valid token in the first place. The current implementation uses a static token-to-employee lookup
table (`mock_data/auth_tokens.json`) — sufficient to demonstrate and verify the checking pattern,
but not a real authentication system. A production deployment would still need genuine credential
verification (e.g. company SSO) and signed, expiring tokens (e.g. a JWT) issued only after that
verification succeeds — without that, the "authentication" is really just a different static
lookup table, not a proof of identity.

A secondary refinement was also needed after the initial authorization fix: without knowing its own
authenticated ID, the agent would guess a placeholder (`"current_user_id"`), get rejected, and only
self-correct because the rejection message happened to reveal the real ID — a fragile, accidental
side-channel rather than a designed behavior. Fixed by injecting the authenticated `employee_id`
directly into the message context sent to the model, removing the guess-then-correct step entirely
(verified: tool calls now succeed on the first attempt with the correct ID, not the second).

**Automated, saved evidence for the authorization fix:** `evaluation/auth_fix_verification.py`
(full output: `evaluation/auth_fix_verification_results.json`) runs both scenarios above as a
repeatable script, across **2 different employee tokens** to confirm the identity mapping is
genuinely data-driven rather than a single hardcoded value:

- **Unauthenticated request for another employee's data:** correctly declined before ever
  attempting the tool call — *"I cannot access specific employee data like PTO balances unless you
  are authenticated as that employee..."*
- **`token-emp001-demo`:** single clean tool call, correct ID on the first attempt, returned
  `pto_days_remaining: 16` (matching EMP001's real record)
- **`token-emp002-demo`:** single clean tool call, correct ID on the first attempt, returned
  `pto_days_remaining: 7` (matching EMP002's real, *different* record)
- **Cross-check:** the two tokens returned two distinct, correctly-ordered employee IDs
  (`['EMP001', 'EMP002']`), confirming the token → identity mapping is read from
  `mock_data/auth_tokens.json`, not a static single value.

All three checks: **PASS**.

**On the rate-limiting verification:** confirmed working via 25 concurrent requests fired at
`/chat` (backgrounded `curl` calls, not sequential — a sequential test never accumulates enough
requests within one minute to trigger a per-minute limit). Both test runs showed `429` responses
mixed in with `200`s, confirming the limiter fires under real burst load. The test also surfaced
an honest confound: a large share of requests failed with `500` (OpenRouter's own transient
instability, unrelated to this project's code — a recurring pattern throughout this project),
making it impossible to state a precise "exactly N requests allowed before throttling" number from
this data alone. The presence of `429`s at all is sufficient to confirm the mechanism works; the
exact threshold remains unverified pending a cleaner test environment.

---

## 6. Deployment

**Platform:** Render, free tier, single service.
**Live URL:** https://hr-agentic-rag-lmj7.onrender.com
**Build command:** `pip install -r requirements.txt && python rag/policy_rag.py --build`
**Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

**A critical fix along the way:** the first deploy attempt failed with
`Ran out of memory (used over 512MB)`. Root cause: `SentenceTransformerEmbeddingFunction` pulls in
PyTorch (commonly 500MB-1GB+ once loaded), combined with gunicorn's main process plus two MCP
server subprocesses each with their own Python interpreter. Fixed by switching to Chroma's
lightweight ONNX embedding function (Section 2) and removing `sentence-transformers`/`torch`/
`transformers` from `requirements.txt` entirely. The second deploy attempt succeeded cleanly.

**Also required:** moving agent initialization from inside `if __name__ == "__main__":` to
module-import time in `app.py` — gunicorn imports the module directly and never executes that
block, so agent construction has to happen at import time to work under both `python app.py`
(local dev) and `gunicorn app:app` (production).

**Cold-start behavior:** Render's free tier spins down after inactivity; first request after
idle can take 50+ seconds. Documented in `deployed.md`.

Full details: `deployed.md`.

---

## 7. Evaluation Results

Full methodology, question sets, and raw results: `evaluation/eval_questions.py`,
`evaluation/run_eval.py`, `evaluation/results.json`, `evaluation/verify_results.py`,
`evaluation/verification_report.json`, `evaluation/answer_correctness_check.py`. Summary below.

**Evaluation set:** 25 core questions, 5 per required category (straightforward, multi-document,
tool-requiring, ambiguous, out-of-scope), each with gold-answer notes and expected tool calls. One
additional supplementary question (Q26) was added later, targeting the confirmation-gate positive
path specifically (Section 5.1) — it sits outside the balanced 5-per-category set and is verified
separately via `evaluation/auth_fix_verification.py`, not folded into the metrics below.

**Evaluation methodology history — three issues found and fixed, in order:**

1. **Infrastructure noise (rate limiting):** the first full eval run produced misleadingly poor
   numbers (16/25 tool match, 10/25 errors) from OpenRouter free-tier rate limiting under rapid
   sequential load. Fixed with an account credit top-up + an 8-second delay between questions,
   producing a clean 21/25 baseline run.
2. **A genuine agent regression, found via a later manual rerun:** after the Section 5.3
   authorization hardening, general policy questions phrased personally (e.g. "how many PTO days
   do I *get*") were being refused pending authentication, even though they need no employee data
   at all — the authentication framing had bled over from the 5 actually-restricted
   employee-specific tools to general `search_policy_documents` calls. Fixed with one clarifying
   paragraph in `AGENT_SYSTEM_PROMPT` (`agent.py`) distinguishing "phrased personally" from
   "needs this employee's actual stored data."
3. **An eval harness gap, found in the same rerun:** `evaluation/run_eval.py` was never updated to
   send an `Authorization` bearer token, so every `tool_requiring` question was being run
   *unauthenticated* against the new hard auth guard — correctly rejected by design, but producing
   a misleading tool-selection-match score unrelated to actual agent quality. Fixed by adding an
   `auth_token` field per question in `eval_questions.py` and header-injection logic in
   `run_eval.py`.

With both fixes applied, a rerun still showed heavy rate limiting (14/25 errors) purely from
OpenRouter's shared upstream pool for the `:free` model tier, unrelated to either fix above.
Switching `LLM_MODEL` to the paid (non-`:free`) variant of the same model for evaluation resolved
this at negligible per-token cost — the run below reflects that configuration.

Two of the eval set's own expected-tool annotations were also corrected during this process (Q19,
Q21) — both had been flagged as "misses" for tool-call patterns this document independently
documents as *correct* elsewhere (Q21's multi-attempt search-then-refuse pattern under Section 5;
Q19's use of `search_policy_documents` to answer an ambiguous question with a concrete, citable
threshold rather than asking for clarification when policy already answers it). `gold_answer_notes`
for both were updated to explain the correction rather than silently changing the expected value.

**Agent behavior metrics (final clean run, 25-question core set, paid-tier model):**

| Metric | Result |
|---|---|
| Tool selection match | 22/25 (88%) |
| Errors | 0/25 |
| Rate-limit errors | 0/25 |

**System metrics (local testing, paid-tier model):**

| Metric | Result |
|---|---|
| Latency p50 | 3.7s |
| Latency p95 | 6.9s |
| Cold-start delay | 50s+ on the live Render deployment (free `:free` tier; documented separately in `deployed.md`) |

*(Note: this run used the paid model tier locally, purely for evaluation reliability — the live
deployment still defaults to the free `:free` tier, so deployed latency and cold-start behavior
remain as documented in Section 6 / `deployed.md`, not the faster numbers above.)*

**Answer quality (citation validity, groundedness, answer correctness):** verified systematically
via `evaluation/verify_results.py` and `evaluation/answer_correctness_check.py` against the earlier
21/25 baseline run. These checks were not rerun against this final 22/25 run — the underlying
answer-generation logic is unchanged between runs, so there is no reason to expect these figures to
differ, but they are reported here against their original run rather than assumed to carry over
untested.

**Citation validity: 25/25 (100%)** — every citation's `(doc_title, section)` pair was checked
programmatically against the actual corpus in `docs/`, confirming none were fabricated or
mismatched. **Groundedness:** an automated heuristic (checks whether numeric claims in each answer
appear in that question's retrieved text) initially flagged 5/25 results. Manual review of the
flagged answers found all 5 to be false positives from the heuristic itself — markdown list
numbering ("1.", "2.", "3.") misread as factual claims, one correctly-derived fact (an empty
`pending_requests` list correctly reported as "0 days"), and one explicitly-labeled illustrative
example ("e.g., increased sales by 10%") rather than a claim about the user. True groundedness
rate after review: **25/25 clean**.

**Answer correctness (exact match against gold answers):** groundedness alone doesn't rule out
*misattribution* — a number that is genuinely present in what was retrieved, but assigned to the
wrong person or fact (e.g. stating EMP002's balance when EMP001's data was actually retrieved). To
close that gap, `evaluation/answer_correctness_check.py` independently loads the real record
directly from `mock_data/pto_balances.json` — bypassing the agent's own trace entirely — and
checks the agent's stated answer against it for an exact match, across 2 different employee
accounts (via distinct auth tokens) to confirm the check generalizes rather than being verified
against only one record. **Result: 3/3 correct** — EMP001's remaining balance (16), EMP001's total
and used days (25, 9), and EMP002's remaining balance (7, a distinct value for a distinct employee)
all matched their real records exactly. Full evidence:
`evaluation/answer_correctness_results.json`.

**Ablation / comparison:** prompt-only vs. code-level guardrail for the hallucinated-employee-ID
fix — 1/3 vs. 6/6 success (Section 5). This is the project's primary ablation, directly comparing
two implementations of the same guardrail intent.

### Notable individual findings

**Q6 (multi-document — Tier 3 remote work) and Q15 (tool-requiring — Remote Work Eligibility):**
both produce complete, correctly-cited answers combining the right policy content (Q6: Data
Security Policy Tier 3 rules + Remote Work Policy Cross-Border Request process; Q15: employee
status + the same combined policy content), but each calls only one of its two "expected" tools
rather than both. The RAG layer's single `search_policy_documents`/`check_policy_compliance` call
is retrieving the necessary content in one pass rather than needing a second tool call — correct
output, exact-tool-count mismatch rather than a real quality gap.

**Q14 (tool-requiring — HR ticket creation): a real, reproduced limitation.** Across 3 separate
authenticated runs, the agent has now consistently drafted a ticket preview and asked for
confirmation *in prose*, without ever calling `create_mock_hr_ticket` (`actual_tools: []` all 3
times). This means the code-level confirmation gate documented in Section 5.1 as a hard guardrail
is never exercised on this specific path — the model is imitating the same confirmation-request
pattern without the tool (and therefore the guard) running at all. The user remains protected in
practice (no ticket is created without a second, explicit confirming message), but the "hard,
code-enforced" guarantee in Section 5.1 does not hold for this flow as currently implemented —
documented here as a known limitation rather than a resolved case, distinct from the verified
positive-path behavior in Q26/Section 5.1 (which did trigger the real tool call when confirmation
was front-loaded in a single message).

**Q21 (out-of-scope — stock option vesting):** now passes cleanly against the corrected expected
value (see methodology note above) — 3 progressively refined `search_policy_documents` queries,
followed by an honest out-of-corpus refusal directing the user to HR. **Minor known limitation,
unchanged:** the `/chat` citation-extraction logic surfaces every retrieved chunk, including ones
the LLM explicitly said weren't relevant, rather than only chunks actually used in the final
answer.

**Q3 (ambiguous/straightforward — floating holidays):** the hallucinated-employee-ID finding
described in Section 5 — found via this eval question, fixed, and reverified; passes cleanly in
every run since, including this final one.

**Q26 (supplementary — confirmation-gate positive path):** verifies that `create_mock_hr_ticket`
correctly *allows* creation once genuine confirmation is given, not just that it blocks without
confirmation. Verified via both the live deployment and locally, with the resulting ticket
confirmed present in `mock_data/tickets.json` afterward (Section 5.1).

---

## 8. Demo Tasks and Tool-Call Walkthrough

### 8.1 The Two Required Demo Tasks

| Task | Expected MCP call sequence |
|---|---|
| **PTO Request Guidance** — "Can I take 3 days of PTO next week? I'm employee EMP001." | 1. `check_pto_balance(employee_id="EMP001")` → 16 days remaining. 2. `search_policy_documents(query="PTO notice period and approval process")` → notice/approval/blackout rules. 3. LLM synthesizes both into one cited answer. |
| **Remote Work Eligibility** — "I'm employee EMP003, based in Remote-EU. Can I work from a country outside the EU for 6 weeks?" | 1. `lookup_employee_profile(employee_id="EMP003")` → role, location, employment type. 2. `check_policy_compliance(topic="remote_work", employee_context={...})` → Tier 1/2/3 rules, approval chain. 3. LLM synthesizes into a grounded eligibility answer. |

Full step-by-step code walkthrough (exact lines, exact data flow) for both tasks: see
`tool-call-breakdown.md` in the repository.

### 8.2 Tool-Call Mechanics Walkthrough — PTO Request Guidance

The two tools below both belong to **one** of the two tasks above — PTO Request Guidance — shown
here to illustrate exactly how each tool call augments the LLM's prompt, and how those two
mechanisms combine within a single agent run. (Remote Work Eligibility follows the same underlying
mechanics — `lookup_employee_profile` is a structured-data call like Tool 1 below,
`check_policy_compliance` is a RAG call like Tool 2 below — so it isn't walked through separately
here to avoid repeating the same two patterns twice.)

Tool 1 and Tool 2 are deliberately kept as separate diagrams, since they represent genuinely
different augmentation mechanisms, not just two examples of the same thing. The combined walkthrough
that follows then shows both firing together within a single real agent run.

<div style="page-break-before: always;"></div>

### Tool 1 — `check_pto_balance`: structured data augmentation

<img src="assets/diagram-tool1-pto-balance.svg" style="width:100%; max-width:1000px;" />

An exact-match lookup against `mock_data/pto_balances.json` (synthetic internal records) — no
embeddings or vector search involved. The returned JSON facts (e.g. `pto_days_remaining: 16`) are
serialized directly into the LLM's context as ground-truth data points.

### Tool 2 — `search_policy_documents`: retrieved-context (RAG) augmentation

<img src="assets/diagram-tool2-policy-search.svg" style="width:100%; max-width:1000px;" />

The query is embedded and compared via vector similarity against the 48 stored chunks in
`chroma_db/`; the top-k matches (unstructured text, not structured facts) are injected as
grounding context, each tagged with citation metadata.

<div style="page-break-before: always;"></div>

### Combined Walkthrough — Both Tools Within a Single ReAct Loop

<img src="assets/diagram-tool3-combined-loop.svg" style="width:100%; max-width:1000px;" />

Shows both patterns firing together within one real agent run (the PTO Request Guidance task),
demonstrating that the agent can and does compose structured-data lookups with RAG retrieval in a
single response rather than only ever using one mechanism at a time.

---

## References

Washizaki, Hironori, ed. 2024. *Guide to the Software Engineering Body of Knowledge (SWEBOK)*.
Version 4.0. Piscataway, NJ: IEEE Computer Society.
https://ieeecs-media.computer.org/media/education/swebok/swebok-v4.pdf

---

## Appendix: Repository File Audit

Full purpose, dependencies, and (where applicable) execution sequence for every file in the
repository, audited against the actual repo contents rather than assumed. `evaluation/` has its
own detailed breakdown already in Section 7's methodology; the summary below cross-references it
rather than repeating it.

### Core Application

| File | Purpose | Depends on |
|---|---|---|
| `app.py` | Flask web app — `/`, `/health`, `/chat`; token auth resolution, prompt-disclosure filter, rate limiting | `agent.py`, `mock_data/auth_tokens.json` |
| `agent.py` | LangGraph agent orchestrator, MCP tool discovery, `employee_id` guard, trace logging | `rag/answer.py`, both MCP servers |
| `rag/policy_rag.py` | Chunking, ONNX embedding, Chroma indexing, top-k retrieval | `docs/*` |
| `rag/answer.py` | LLM client (`get_model()`), single-shot RAG prompt template | `rag/policy_rag.py` |
| `mcp/policy_mcp_server.py` | MCP server exposing `search_policy_documents`, `get_policy_section`, `check_policy_compliance` | `rag/policy_rag.py` |
| `mcp/hr_data_mcp_server.py` | MCP server exposing employee/PTO/benefits/ticket tools | `mock_data/*.json` |

**Execution sequence (local dev):** `rag/policy_rag.py --build` (builds the index) → `python app.py`
(imports `agent.py`, which connects to both MCP servers as subprocesses) → app is live.

### Data

| Folder | Purpose | Detail |
|---|---|---|
| `docs/` | Policy corpus — 9 files (8 markdown, 1 HTML) | See Section 2 |
| `mock_data/` | Synthetic employee/PTO/benefits/ticket/auth-token records — 5 JSON files | See Sections 4-5 |

### CI Testing (automated, gates deployment)

| File | Purpose | Depends on |
|---|---|---|
| `tests/test_smoke.py` | 6 tests: app import/startup, route registration, `/health` degradation, live MCP tool discovery, direct MCP tool call | `app.py`, `agent.py`, running MCP servers |
| `conftest.py` | Adds project root to `sys.path` so `tests/` can import `app`/`agent` | — |

**Execution:** `pytest tests/test_smoke.py -v` — runs automatically via `.github/workflows/ci.yml`
on every push.

### Standalone Dev Scripts (manual, not part of CI)

| File | Purpose |
|---|---|
| `test_answer.py` | Manual test of the single-shot RAG answer path (`rag/answer.py`), independent of the agent |
| `test_rag.py` | Manual test of retrieval quality, including the multi-document cross-reference case (Section 2) |

### Evaluation (manual trigger, behavioral/quality/security testing)

Full audit already provided in Section 7's methodology note and the accompanying file-by-file
breakdown discussed there — see `eval_questions.py`, `run_eval.py`, `results.json`,
`verify_results.py`, `verification_report.json`, `answer_correctness_check.py`,
`answer_correctness_results.json`, `security_questions.py`, `run_security_eval.py`,
`security_results.json`, `auth_fix_verification.py`, `auth_fix_verification_results.json`.

### Deployment & Configuration

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | CI/CD pipeline definition |
| `Procfile` | Render start command (`gunicorn app:app ...`) |
| `render.yaml` | Render build/deploy configuration |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template listing required environment variables (no real values) |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `__pycache__/`, etc. from version control |

### Documentation

| File | Purpose |
|---|---|
| `README.md` | Setup, local run, deployment instructions, repo structure overview |
| `design-and-evaluation.md` | This document |
| `ai-tooling.md` | How AI tools were used in this build |
| `deployed.md` | Deployment details, cold-start behavior, memory-fix story |
| `architecture.drawio` | Editable draw.io source for the architecture diagram |
| `assets/*.svg` | 4 diagram files embedded throughout this document (Sections 1 and 8) |

### Not Committed / Local-Only

| Path | Why it's excluded |
|---|---|
| `.env` | Contains real secrets (API keys) — gitignored, never committed |
| `chroma_db/` | Rebuildable vector index artifact — gitignored, regenerated fresh on every deploy from `docs/` (Section 2) |
