# Spec: Spec-Driven Development Workflow

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-04-02                     |
| Status      | proposed                       |
| PR          | #1370                          |

## Summary

Large PRs (1000+ lines) are expensive to review. Human review time is the scarce resource — code generation is cheap. This spec defines a workflow that front-loads review onto small, readable specs so that large implementations become mechanical follow-throughs.

## Intent

### Goal

Introduce a spec-driven development workflow where a single PR carries a feature from spec to implementation. Claude analyzes the spec for ambiguities (CI, read-only), and implementation happens locally or in Claude Code cloud after spec approval.

### Invariants

- Every spec PR gets auto-labeled (`spec` / `no-spec` / `implementation`) based on changed files.
- Spec files are auto-renamed to `<PR#>-<name>.md` on PR creation.
- Claude's analysis is read-only — CI never writes to the repo.
- The `claude-approved` label is only added when Claude's ambiguity score is below 20%.
- Specs are frozen point-in-time documents — the code is the source of truth for current behavior.

### Non-Goals

- Automated implementation via CI (implementation runs locally or in Claude Code cloud).
- Hard CI gates blocking PRs without specs (soft warning only).
- Persistent Claude sessions across PR comments (future enhancement via Channels or webhook server).

## Evaluation

### Acceptance Criteria

- [ ] A PR adding only a `specs/*.md` file gets the `spec` label automatically.
- [ ] A PR with no spec file gets the `no-spec` label.
- [ ] A PR with both spec and non-spec changes gets both `spec` and `implementation` labels.
- [ ] A PR exceeding 500 changed lines without a spec file gets a warning comment.
- [ ] The spec file is auto-renamed to include the PR number on PR open.
- [ ] `@claude analyze` on a spec PR triggers Claude to post a single-pass analysis comment.
- [ ] When Claude is satisfied, it adds `claude-approved` and posts a summary with link to Claude Code cloud.
- [ ] `/new-spec <name>` creates a valid spec file with the correct template, author, and date.
- [ ] `CONTRIBUTING.md` exists at repo root and explains the spec workflow.

### Testing Strategy

Workflow correctness is verified by observing label behavior and comment posting on actual PRs (this PR is the first test). No automated tests — these are GitHub Actions workflows.

### Performance

N/A — workflow tooling, not runtime code.

## Design

### Architecture

**Workflow lifecycle:**

```
new-spec.sh / /new-spec → open PR → [auto-rename + spec label]
→ @claude analyze (CI, read-only) → approve spec
→ /implement-spec (local/cloud) → review code → merge
```

**GitHub Actions:**
- `spec-tracking.yml`: Auto-label, auto-rename, large-PR warning
- `claude.yml`: `@claude` command handler (Opus model, read-only)

**Claude skills (`.claude/skills/`):**
- `analyze-spec.md`: Dual-mode — CI single-pass or local interactive Socratic interview
- `implement-spec.md`: Local/cloud autonomous implementation from approved spec
- `new-spec.md`: Create spec from template, offers analyze-spec as next step

**Labels:**

| Label | Meaning | Applied by |
|-------|---------|------------|
| `spec` | PR contains a spec file | Action (auto) |
| `no-spec` | PR has no spec file | Action (auto) |
| `implementation` | PR contains non-spec code alongside a spec | Action (auto) |
| `claude-approved` | Claude's analysis found no ambiguities | Claude (via `gh pr edit`) |

**Soft guardrails:** PRs exceeding 500 changed lines without a spec file get a warning comment linking to `CONTRIBUTING.md`.

### Alternatives Considered

- **Separate spec PR + implementation PR with tracking issue**: Rejected — adds overhead without benefit. The single-PR lifecycle keeps all discussion in one place.
- **Hard CI gate requiring specs for large PRs**: Rejected — too restrictive. Convention enforced by culture is sufficient.
- **Claude writes implementation in CI**: Rejected for now — security concern with `contents: write`. Implementation runs locally or in Claude Code cloud instead.
- **Persistent Claude sessions via GitHub Actions polling**: Rejected — wasteful CI minutes. Future enhancement via Claude Code Channels or webhook server.

## Documentation

No Jolt book changes required — this is developer workflow tooling with no user-facing API or behavior changes. The workflow is documented in `CONTRIBUTING.md`.

## Execution

### Files to create
- `.claude/skills/analyze-spec.md` — deep-interview spec analysis skill
- `.claude/skills/implement-spec.md` — local/cloud implementation skill
- `.claude/skills/new-spec.md` — create new spec from template
- `CONTRIBUTING.md` — contribution guide

### Files to modify
- `specs/TEMPLATE.md` — add Status, PR, structured sections
- `.github/workflows/claude.yml` — Opus model, env vars, read-only
- `.github/workflows/spec-tracking.yml` — replace issue-creation with labeling + rename + warning
- `.github/workflows/arch-tests.yml` — paths filter for tracer-only changes
- `.github/workflows/rust.yml` — paths-ignore for spec/docs changes

### Files to remove
- `specs/new-spec.sh` — replaced by `/new-spec` Claude skill
- `specs/README.md` — replaced by `CONTRIBUTING.md`

## References

- `anthropics/claude-code-action` — handles `@claude` commands in CI
- `specs/TEMPLATE.md` — spec template
- Claude Code Channels (research preview) — future path for persistent sessions
