---
name: analyze-spec
description: Socratic deep-interview analysis of a spec file to ensure zero ambiguity before implementation
argument-hint: "[spec file path]"
---

<Purpose>
Analyze a spec file using Socratic questioning with mathematical ambiguity scoring. The goal: ensure the spec is clear enough for a one-shot implementation with zero clarifying questions. Post probing questions as PR review comments until all ambiguity is resolved, then add the `claude-approved` label.

Adapted from the Ouroboros-inspired deep interview methodology — specification quality is the primary bottleneck in AI-assisted development.
</Purpose>

<Execution_Policy>
- Ask ONE question at a time — never batch multiple questions
- Target the WEAKEST clarity dimension with each question
- Make weakest-dimension targeting explicit every round: name the weakest dimension, state its score/gap, and explain why the next question is aimed there
- Gather codebase facts via `explore` agent BEFORE asking about them
- Cite repo evidence (file path, symbol, or pattern) instead of asking the spec author to rediscover it
- Score ambiguity after every answer — display the score transparently
- Do not approve until ambiguity ≤ threshold (default 0.2)
- Allow early approval with a clear warning if ambiguity is still high
- Challenge agents activate at specific round thresholds to shift perspective
</Execution_Policy>

<Steps>

## Phase 1: Initialize

1. **Find the spec**: Locate the `specs/*.md` file in this PR (exclude `TEMPLATE.md`). If a path is provided in `{{ARGUMENTS}}`, use that.
2. **Read the spec** thoroughly — understand Intent, Evaluation, and Execution sections.
3. **Explore the codebase**: Run `explore` agent to map codebase areas relevant to the spec's intent. Store as `codebase_context`.
4. **Initial assessment**: Score each clarity dimension based on the spec alone (before any Q&A).

Announce:

> **Spec analysis started.** I'll ask targeted questions to ensure this spec is unambiguous enough for one-shot implementation. After each answer, I'll show the clarity score. Approval requires ambiguity below 20%.
>
> **Spec:** {spec title}
> **Current ambiguity:** {initial_score}%

## Phase 2: Interview Loop

Repeat until `ambiguity ≤ threshold` OR spec author signals completion:

### Step 2a: Generate Next Question

Identify the dimension with the LOWEST clarity score. Generate a question that specifically improves that dimension.

**Question targeting strategy:**
- Questions should expose ASSUMPTIONS, not gather feature lists
- If the scope is conceptually fuzzy (entities keep shifting, the core noun is unstable), ask an ontology-style question about what the thing fundamentally IS
- Always state why this dimension is the bottleneck before asking

**Clarity dimensions (brownfield weights — Jolt is always brownfield):**

| Dimension | Weight | Question Style |
|-----------|--------|---------------|
| Goal Clarity | 0.35 | "What exactly happens when...?" / "What invariant must hold?" |
| Constraint Clarity | 0.25 | "What are the boundaries?" / "What is explicitly out of scope?" |
| Success Criteria | 0.25 | "How do we know it works?" / "What test would verify this?" |
| Context Clarity | 0.15 | "How does this fit with existing {module}?" / "I found {pattern} in {file} — should this extend or diverge?" |

### Step 2b: Post the Question

Post as a PR review comment on the relevant line of the spec, or as a general comment if the question spans multiple sections:

```
**Round {n}** | Targeting: {weakest_dimension} | Ambiguity: {score}%

{question}
```

### Step 2c: Score Ambiguity

After receiving the spec author's response, score clarity across all dimensions (0.0–1.0 each):

1. **Goal Clarity**: Is the primary objective unambiguous? Can you state it in one sentence? Are key entities and their relationships clear?
2. **Constraint Clarity**: Are boundaries, limitations, and non-goals clear?
3. **Success Criteria**: Could you write a test that verifies success? Are acceptance criteria concrete?
4. **Context Clarity**: Do we understand the existing system well enough to modify it safely?

**Calculate ambiguity:**
`ambiguity = 1 - (goal × 0.35 + constraints × 0.25 + criteria × 0.25 + context × 0.15)`

### Step 2d: Report Progress

Post a comment with the updated scores:

```
**Round {n} complete.**

| Dimension | Score | Weight | Weighted | Gap |
|-----------|-------|--------|----------|-----|
| Goal | {s} | 0.35 | {s*w} | {gap or "Clear"} |
| Constraints | {s} | 0.25 | {s*w} | {gap or "Clear"} |
| Success Criteria | {s} | 0.25 | {s*w} | {gap or "Clear"} |
| Context | {s} | 0.15 | {s*w} | {gap or "Clear"} |
| **Ambiguity** | | | **{score}%** | |

**Next target:** {weakest_dimension} — {rationale}
```

### Step 2e: Check Soft Limits

- **Round 3+**: Allow early approval if author says "enough", "looks good", "approve it"
- **Round 10**: Soft warning: "We're at 10 rounds. Current ambiguity: {score}%. Continue or approve with current clarity?"
- **Round 15**: Hard cap: "Maximum rounds reached. Approving with current clarity level ({score}%)."

## Phase 3: Challenge Modes

At specific round thresholds, shift perspective. Each mode is used ONCE:

**Round 4+ — Contrarian:** Challenge the spec's core assumption. "What if this constraint doesn't actually exist?" or "What if the opposite approach is simpler?" Test whether the framing is correct or just habitual.

**Round 6+ — Simplifier:** Probe whether complexity can be removed. "What's the simplest version that satisfies the invariants?" or "Which constraints are necessary vs. assumed?"

**Round 8+ — Ontologist (if ambiguity > 0.3):** The ambiguity is still high, suggesting we may be addressing symptoms. "What IS this, really?" — find the essence by examining the key entities and their relationships.

## Phase 4: Approve or Summarize

### When ambiguity ≤ threshold (approved):

Post a summary comment:

```
**Spec analysis complete — approved.**

**Final ambiguity: {score}%**

| Dimension | Score |
|-----------|-------|
| Goal | {s} |
| Constraints | {s} |
| Success Criteria | {s} |
| Context | {s} |

**Summary:**
- {what will be built — one sentence}
- {key invariants that must hold}
- {critical evaluation criteria}

**Key entities:** {entity list with relationships}

The spec is clear enough for one-shot implementation.
```

Then add the label: `gh pr edit --add-label claude-approved`

### When ambiguity > threshold (not approved):

Post a summary:

```
**Spec analysis: {n} ambiguities remain.**

| Dimension | Score | Gap |
|-----------|-------|-----|
| ... | ... | ... |

**Current ambiguity: {score}% (threshold: 20%)**

Reply to the questions above and ask me to analyze again.
```

Do NOT add `claude-approved` until ambiguity is below threshold.

</Steps>

<Examples>
<Good>
Targeting weakest dimension:
```
Round 2 | Targeting: Success Criteria | Ambiguity: 38%

The Intent section defines a streaming commitment scheme, but the Evaluation
section only mentions "existing tests pass." What NEW test would verify that
streaming produces the same commitments as the non-streaming path? And what
is the acceptable performance threshold (e.g., memory usage, throughput)?
```
Why good: Identifies weakest dimension, asks a specific question tied to the spec content, suggests what a concrete criterion looks like.
</Good>

<Good>
Citing codebase context:
```
Round 3 | Targeting: Context Clarity | Ambiguity: 29%

I found that `DoryCommitmentScheme` in `poly/commitment/dory/` uses a two-tier
structure (tier-1 chunks → tier-2 aggregation). The spec mentions "streaming
commitments" but doesn't specify which tier is streamed. Is this streaming
at tier-1 (chunk-level), tier-2 (aggregation), or both?
```
Why good: Explored the codebase first, cited specific types and files, asked a precise question the spec should answer.
</Good>

<Good>
Contrarian challenge:
```
Round 5 | Contrarian Mode | Ambiguity: 25%

The spec assumes we need a new polynomial type for this. But `CompactPolynomial`
already handles small-scalar promotion to field elements on bind. What if we
extended `CompactPolynomial` instead of adding a new type? What property does
the new type have that `CompactPolynomial` can't express?
```
Why good: Challenges a structural assumption by pointing to an existing abstraction that might already suffice.
</Good>

<Bad>
Batching multiple questions:
```
"What's the memory budget? And how does this interact with BlindFold?
Also, should this work with HyperKZG too?"
```
Why bad: Three questions at once — causes shallow answers and makes scoring inaccurate.
</Bad>

<Bad>
Asking about codebase facts:
```
"What commitment scheme does Jolt use?"
```
Why bad: Should have explored the codebase to find this. Never ask the spec author what the code already tells you.
</Bad>
</Examples>

<Escalation_And_Stop_Conditions>
- **Hard cap at 15 rounds**: Approve with whatever clarity exists, noting the risk
- **Soft warning at 10 rounds**: Offer to continue or approve
- **Early approval (round 3+)**: Allow with warning if ambiguity > threshold
- **Author says "stop", "approve", "good enough"**: Respect their decision, note residual ambiguity
- **Ambiguity stalls** (same score +-0.05 for 3 rounds): Activate Ontologist mode to reframe
- **All dimensions at 0.9+**: Approve immediately
- **Codebase exploration fails**: Note the limitation, continue with spec-only analysis
</Escalation_And_Stop_Conditions>

Task: Analyze the spec in this PR. {{ARGUMENTS}}
