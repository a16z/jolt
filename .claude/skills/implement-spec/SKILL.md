---
name: implement-spec
description: Autonomous one-shot implementation from an approved spec (local/cloud only)
argument-hint: "[spec file path]"
---

<Purpose>
Take an approved spec and autonomously implement it: plan the work, execute in parallel where possible, run QA cycles until tests pass, and validate the result. Produces working, verified code from the spec in a single pass.

This skill runs locally or in Claude Code cloud (claude.ai/code) — NOT in CI. It needs write access to the repo to create commits and push to the PR branch.
</Purpose>

<Execution_Policy>
- The spec is the source of truth. Implement what it says, not more.
- Read CLAUDE.md for project conventions, testing requirements, and architecture.
- Each phase must complete before the next begins.
- Parallel execution within phases where possible.
- QA cycles repeat up to 5 times; if the same error persists 3 times, stop and report.
- If something in the spec is ambiguous, post a PR comment rather than guessing.
- Do not add features, refactor code, or make improvements beyond the spec.
</Execution_Policy>

<Steps>

## Phase 1: Plan

1. **Read the spec**: Find the `specs/*.md` file in this PR (exclude `TEMPLATE.md`). If a path is provided in `{{ARGUMENTS}}`, use that.
2. **Read CLAUDE.md**: Understand architecture, conventions, testing requirements.
3. **Explore relevant code**: Use `explore` agents to understand the modules, types, and patterns the implementation will touch.
4. **Create implementation plan**: Based on the spec's Intent and Execution sections, determine:
   - Files to create, modify, or remove
   - Order of changes (dependencies first)
   - How existing patterns and abstractions should be extended
   - Which tasks can run in parallel vs. sequential
5. **Post the plan** as a PR comment for visibility:

```
**Implementation plan for: {spec title}**

**Changes:**
1. {file/module} — {what changes and why}
2. ...

**Order:** {dependency chain}
**Parallel tasks:** {which can run simultaneously}
**Estimated scope:** {number of files, rough line count}
```

## Phase 2: Execute

1. **Implement** all changes from the plan.
   - Run independent tasks in parallel using agents where beneficial.
   - Follow project code style and conventions from CLAUDE.md.
   - Performance is critical — avoid regressions in hot paths.
2. **Commit** with clear, well-scoped messages as logical units complete.

## Phase 3: QA

Cycle until all checks pass (up to 5 cycles):

1. **Format**: `cargo fmt -q`
2. **Lint** (both modes):
   - `cargo clippy -p jolt-core --features host --message-format=short -q --all-targets -- -D warnings`
   - `cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings`
3. **Test**: Run evaluation criteria from the spec, plus:
   - `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host`
   - `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`
4. **Fix** any failures and repeat.

If the same error persists 3 times, stop and post a PR comment describing the fundamental issue.

## Phase 4: Validate

Run parallel validation:

1. **Correctness**: All spec evaluation criteria pass.
2. **Code review**: Self-review for consistency with existing patterns, missing edge cases, unnecessary changes beyond the spec.
3. **Security**: Check for OWASP top 10 patterns if the changes touch input handling or external data.

Fix any issues found and re-validate.

## Phase 5: Finalize

1. **Update spec status**: Change `Status` from `proposed`/`approved` to `implemented` in the spec file.
2. **Push** all commits to the PR branch.
3. **Post summary** as a PR comment:

```
**Implementation complete for: {spec title}**

**Changes made:**
- {file} — {summary}
- ...

**Evaluation results:**
- {criterion 1}: PASS
- {criterion 2}: PASS
- ...

**Tests:** All passing (host + zk modes)
**Lint:** Clean
```

</Steps>

<Examples>
<Good>
User: `/implement-spec`
Action: Reads the spec from the PR, creates a plan, implements it, runs QA, validates, pushes commits.
Why good: Full autonomous execution from spec to working code.
</Good>

<Bad>
User: `/implement-spec` on a spec without `claude-spec-approved`
Action: Should warn that the spec hasn't been analyzed yet, but proceed if the user insists.
Why bad situation: Implementation from an unanalyzed spec risks rework.
</Bad>
</Examples>

Task: Implement the spec in this PR. {{ARGUMENTS}}
