---
name: new-spec
description: Create a new spec through Socratic interview, filling each template section to zero ambiguity
arguments: "<feature-name>"
---

Create a new spec file in `specs/` by interviewing the user to fill each section of the template. The goal: produce a spec that would pass `/analyze-spec` with zero ambiguity on the first try.

## Steps

### 1. Initialize

1. Validate the argument: must be lowercase alphanumeric with dashes (e.g. `streaming-prover`). Reject otherwise.
2. Get the GitHub username: run `gh api user --jq .login`.
3. Get today's date in `YYYY-MM-DD` format.
4. Read `specs/TEMPLATE.md` to understand the required sections.
5. Explore the codebase to understand what areas the feature name suggests — run an `explore` agent to gather context. This informs your questions.

### 2. Interview — Section by Section

Walk through each template section with the user. For each section, ask targeted questions until you have enough to write it with zero ambiguity. Do NOT move to the next section until the current one is clear.

**One question at a time. Never batch.**

#### Summary
Ask: "In one paragraph, what problem does this solve and why does it matter?"

#### Intent — Goal
Ask: "What are we building? Can you state the primary objective in one sentence?" Follow up on abstractions, types, boundaries if unclear.

#### Intent — Invariants
Ask: "What properties must always hold? What would break if the implementation is wrong?" For ZK features, probe prover/verifier consistency. Cite codebase context — e.g., "I see `SumcheckInstanceParams` requires `input_claim` and `input_claim_constraint` to stay in sync. Does this feature touch sumcheck?"

#### Intent — Non-Goals
Ask: "What is explicitly out of scope?" Push back if non-goals are vague — "you said 'not performance-critical', but the hot path in `poly/` multiplies across thousands of rounds. Is there a concrete budget?"

#### Evaluation — Acceptance Criteria
Ask: "What test would prove this works? Give me concrete, checkable criteria." Each criterion must be a checkbox item. Push for specificity — "existing tests pass" is not enough; ask what NEW tests are needed.

#### Evaluation — Testing Strategy
Ask: "Which existing tests must keep passing? What new tests are needed? Does this need both `--features host` and `--features host,zk` coverage?"

#### Evaluation — Performance
Ask: "What are the performance expectations? Benchmarks, memory budgets, throughput targets? Or is 'no regression' sufficient — and if so, which benchmark verifies that?"

#### Design — Architecture
Ask: "How does this fit into the existing system?" Use codebase exploration to suggest which modules are affected. Ask about type parameters, trait implementations, module boundaries.

#### Design — Alternatives Considered
Ask: "What other approaches did you consider? Why this one?" If the user says "none", probe: "What's the simplest approach that could work? Why isn't that sufficient?"

#### Documentation
Ask: "Does this need Jolt book (`book/`) changes? New pages, updated sections, diagrams?"

#### Execution (optional)
Ask: "Any implementation direction you want to capture? Algorithmic approach, optimizations, modules to touch? Or should the implementer derive this from intent and evaluation?"

#### References
Ask: "Any papers, issues, PRs, or prior art to link?"

### 3. Write the Spec

Once all sections are filled, create `specs/<feature-name>.md` with the template fully populated:
- `[Feature Name]` → feature name from the interview
- `Author(s)` → `@<github-username>`
- `Created` → today's date
- `Status` → `proposed`
- `PR` → empty (filled by GitHub Action)
- All sections filled from interview answers

### 4. Review with User

Show the complete spec to the user. Ask: "Does this capture everything? Anything to add or change?" Make edits until the user is satisfied.

### 5. Next Steps

Print:
```
Spec created: specs/<feature-name>.md

Next steps:
1. Review the spec above
2. Open a PR — a GitHub Action will add the `spec` label
3. Add the `claude-spec-review-request` label for external analysis, or run `/analyze-spec` locally
```

Do NOT modify `TEMPLATE.md` itself.
