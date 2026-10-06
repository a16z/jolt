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
3. **Read `jolt-eval/README.md`** so you understand the invariant/objective framework for scoring Success Criteria and generating questions.
4. **Explore the codebase**: Run `explore` agent to map codebase areas relevant to the spec's intent.
5. **Read prior context (remote mode)**: Read all existing PR comments via `gh pr view --json comments` to identify questions already asked and answers already given. Account for these when scoring — don't re-ask answered questions.

## Phase 2: Analyze

Score clarity across four dimensions (0.0–1.0 each):

| Dimension | Weight | What to assess |
|-----------|--------|---------------|
| Goal Clarity | 0.35 | Is the primary objective unambiguous? Can you state it in one sentence? Are key entities and relationships clear? |
| Constraint Clarity | 0.20 | Are boundaries, limitations, and non-goals clear? |
| Success Criteria | 0.30 | Could you write a test that verifies success? Are acceptance criteria concrete? Are relevant `jolt-eval` invariants/objectives described? |
| Context Clarity | 0.15 | Do we understand the existing system well enough to modify it safely? |

**Calculate ambiguity:**
`ambiguity = 1 - (goal × 0.35 + constraints × 0.20 + criteria × 0.30 + context × 0.15)`

For each dimension below 0.9, generate a targeted question that would improve it:
- Questions should expose ASSUMPTIONS, not gather feature lists
- If the scope is conceptually fuzzy, ask an ontology-style question about what the thing fundamentally IS
- Cite specific codebase context (files, types, patterns) when relevant

## Phase 3: Output (mode-dependent)

- **Remote mode**: single-pass PR comment with all findings — follow `references/remote-mode.md` (in this skill's directory) for the comment template and label handling.
- **Local mode**: interactive Socratic interview — follow `references/local-mode.md` for the round protocol, challenge modes, and stop conditions.

</Steps>

<Examples>
<Good>
Probing jolt-eval coverage:
```
The Intent → Invariants section says "streaming must produce the same
commitments as the non-streaming path." That looks like a binary property —
have you considered capturing it as a new `jolt-eval` invariant? The existing
`split_eq_bind_low_high` in `jolt-eval/src/invariant/` is a close model
(reference vs. optimized implementation comparison). If this is out of scope,
the Invariants section should say so explicitly.
```
Why good: Names a concrete existing invariant as a model, leaves the N/A
door open, doesn't force a fit.
</Good>

<Good>
Citing codebase context:
```
I found that `DoryScheme` in `crates/jolt-dory/src/scheme.rs` exposes both ordinary
and streaming commitment paths. The spec mentions "streaming commitments" but doesn't
say whether the new behavior belongs in the PCS adapter or in the prover's witness feed.
Which boundary should own it?
```
Why good: Explored first, cited specific types and files.
</Good>
</Examples>

Task: Analyze the spec. {{ARGUMENTS}}
