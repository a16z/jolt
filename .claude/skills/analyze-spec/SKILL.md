---
name: analyze-spec
description: Spec analysis with ambiguity scoring — interactive locally, single-pass remotely via label
argument-hint: "[spec file path]"
---

<Purpose>
Analyze a spec file using mathematical ambiguity scoring. The goal: ensure the spec is clear enough for a one-shot implementation with zero clarifying questions.

This skill operates in two modes:
- **Local mode** (invoked via `/analyze-spec` in Claude Code): Full interactive Socratic interview — one question at a time, iterative refinement with the spec author.
- **Remote mode** (triggered externally via the `claude-spec-review-request` label): Single-pass analysis — all questions posted at once as a single PR comment. Reads prior PR comments as context to account for already-answered questions.

Adapted from the Ouroboros-inspired deep interview methodology — specification quality is the primary bottleneck in AI-assisted development.
</Purpose>

<Execution_Policy>
- Gather codebase facts via `explore` agent BEFORE asking about them
- Cite repo evidence (file path, symbol, or pattern) instead of asking the spec author to rediscover it
- Score ambiguity transparently
- Do not approve until ambiguity ≤ threshold (default 0.2)
- Allow early approval with a clear warning if ambiguity is still high
</Execution_Policy>

<Mode_Detection>
Detect which mode to use:
- **Remote mode**: Running in a remote Claude instance triggered by the `claude-spec-review-request` label. Indicators: environment is non-interactive (no TTY), or running inside a GitHub PR context.
- **Local mode**: Running interactively in a terminal via `/analyze-spec`.

When in doubt, default to local mode (interactive).
</Mode_Detection>

<Steps>

## Phase 1: Initialize

1. **Locate the spec**:
   - If a path is provided in `{{ARGUMENTS}}`, use that directly.
   - Otherwise, detect the PR number:
     - Run `gh pr view --json number --jq .number` to get the current branch's PR number.
     - If that fails, list specs: `ls specs/` and ask the user which one.
   - Look for `specs/<PR#>-*.md` matching the PR number. This is the spec for this PR.
   - If no match, fall back to finding any `specs/*.md` file that is NOT `TEMPLATE.md`.
   - If multiple specs match, prefer the one matching the PR number. If still ambiguous, ask the user.
2. **Read the spec** thoroughly — understand all sections (Summary, Intent, Evaluation, Design, Execution).
3. **Explore the codebase**: Run `explore` agent to map codebase areas relevant to the spec's intent.
4. **Read prior context (remote mode)**: Read all existing PR comments via `gh pr view --json comments` to identify questions already asked and answers already given. Account for these when scoring — don't re-ask answered questions.

## Phase 2: Analyze

Score clarity across four dimensions (0.0–1.0 each):

| Dimension | Weight | What to assess |
|-----------|--------|---------------|
| Goal Clarity | 0.35 | Is the primary objective unambiguous? Can you state it in one sentence? Are key entities and relationships clear? |
| Constraint Clarity | 0.25 | Are boundaries, limitations, and non-goals clear? |
| Success Criteria | 0.25 | Could you write a test that verifies success? Are acceptance criteria concrete? |
| Context Clarity | 0.15 | Do we understand the existing system well enough to modify it safely? |

**Calculate ambiguity:**
`ambiguity = 1 - (goal × 0.35 + constraints × 0.25 + criteria × 0.25 + context × 0.15)`

For each dimension below 0.9, generate a targeted question that would improve it:
- Questions should expose ASSUMPTIONS, not gather feature lists
- If the scope is conceptually fuzzy, ask an ontology-style question about what the thing fundamentally IS
- Cite specific codebase context (files, types, patterns) when relevant

## Phase 3: Output (mode-dependent)

### Remote Mode — Single-Pass

Post a **single PR comment** with all findings:

```
**Spec Analysis: {spec title}**

| Dimension | Score | Gap |
|-----------|-------|-----|
| Goal | {s} | {gap or "Clear"} |
| Constraints | {s} | {gap or "Clear"} |
| Success Criteria | {s} | {gap or "Clear"} |
| Context | {s} | {gap or "Clear"} |
| **Ambiguity** | | **{score}%** |

{If ambiguity ≤ 20%:}
**Status: Approved** — The spec is clear enough for one-shot implementation.

**Summary:**
- {what will be built}
- {key invariants}
- {critical evaluation criteria}

**Next step:** Run `/implement-spec` to implement this spec:
- [Open in Claude Code (cloud)](https://claude.ai/code) — run `/implement-spec` on this branch
- Or run `/implement-spec` locally in Claude Code

{If ambiguity > 20%:}
**Status: Questions remain** — {n} ambiguities to resolve before implementation.

**Questions:**

**1. [{dimension}]** {question}

**2. [{dimension}]** {question}

...

> After addressing these questions, update the spec and re-add the `claude-spec-review-request` label.
```

If approved, add the label: `gh pr edit --add-label claude-spec-approved`

If NOT approved, do NOT add the label.

### Local Mode — Interactive Socratic Interview

Full iterative loop, one question at a time:

**Each round:**
1. Identify the dimension with the LOWEST clarity score
2. State why this dimension is the bottleneck
3. Ask ONE targeted question
4. Wait for the author's response
5. Re-score all dimensions
6. Report progress:

```
Round {n} complete.

| Dimension | Score | Weight | Weighted | Gap |
|-----------|-------|--------|----------|-----|
| Goal | {s} | 0.35 | {s*w} | {gap or "Clear"} |
| Constraints | {s} | 0.25 | {s*w} | {gap or "Clear"} |
| Success Criteria | {s} | 0.25 | {s*w} | {gap or "Clear"} |
| Context | {s} | 0.15 | {s*w} | {gap or "Clear"} |
| **Ambiguity** | | | **{score}%** | |

Next target: {weakest_dimension} — {rationale}
```

**Challenge modes** (local only):

- **Round 4+ — Contrarian:** Challenge the spec's core assumption. "What if this constraint doesn't exist?" Test whether the framing is correct or habitual.
- **Round 6+ — Simplifier:** Probe whether complexity can be removed. "What's the simplest version that satisfies the invariants?"
- **Round 8+ — Ontologist (if ambiguity > 0.3):** "What IS this, really?" — find the essence.

Each mode is used ONCE.

**Soft limits (local only):**
- **Round 3+**: Allow early approval if author says "enough", "looks good"
- **Round 10**: Soft warning about round count
- **Round 15**: Hard cap

**When approved (local):** Print the summary and offer to update the spec with any refinements discovered during the interview.

</Steps>

<Examples>
<Good>
Remote mode — single-pass with all questions:
```
**Spec Analysis: Streaming Commitments**

| Dimension | Score | Gap |
|-----------|-------|-----|
| Goal | 0.85 | Clear |
| Constraints | 0.70 | Memory budget undefined |
| Success Criteria | 0.50 | No new tests specified |
| Context | 0.60 | Unclear which Dory tier is streamed |
| **Ambiguity** | | **34%** |

**Status: Questions remain** — 3 ambiguities to resolve.

**Questions:**

**1. [Success Criteria]** The Evaluation section says "existing tests pass" but
doesn't specify new tests. What test would verify streaming produces identical
commitments to the non-streaming path? What's the performance target?

**2. [Context]** `DoryCommitmentScheme` uses a two-tier structure (tier-1 chunks
→ tier-2 aggregation in `poly/commitment/dory/`). Which tier is streamed?

**3. [Constraints]** What's the memory budget? The current non-streaming path
peaks at ~2GB for large traces — should streaming reduce this, and to what?

> After addressing these questions, update the spec and re-add the `claude-spec-review-request` label.
```
Why good: All questions posted at once, each tagged with the dimension it targets, clear next steps.
</Good>

<Good>
Local mode — interactive round:
```
Round 2 | Targeting: Success Criteria | Ambiguity: 38%

The Intent section defines a streaming commitment scheme, but the Evaluation
section only mentions "existing tests pass." What NEW test would verify that
streaming produces the same commitments as the non-streaming path? And what
is the acceptable performance threshold (e.g., memory usage, throughput)?
```
Why good: One question, targets weakest dimension, specific to the spec content.
</Good>

<Good>
Citing codebase context:
```
I found that `DoryCommitmentScheme` in `poly/commitment/dory/` uses a two-tier
structure (tier-1 chunks → tier-2 aggregation). The spec mentions "streaming
commitments" but doesn't specify which tier is streamed. Is this streaming
at tier-1 (chunk-level), tier-2 (aggregation), or both?
```
Why good: Explored first, cited specific types and files.
</Good>

<Bad>
Asking about codebase facts:
```
"What commitment scheme does Jolt use?"
```
Why bad: Should have explored the codebase to find this.
</Bad>
</Examples>

<Escalation_And_Stop_Conditions>
- **Remote mode**: Single pass — no escalation needed. Either approve or list remaining questions.
- **Local mode**:
  - Hard cap at 15 rounds
  - Soft warning at 10 rounds
  - Early approval allowed at round 3+
  - Ambiguity stalls (same score ±0.05 for 3 rounds): Activate Ontologist mode
  - All dimensions at 0.9+: Approve immediately
</Escalation_And_Stop_Conditions>

Task: Analyze the spec. {{ARGUMENTS}}
