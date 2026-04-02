# Spec: Spec-Driven Development Workflow

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-04-02                     |
| Status      | proposed                       |
| PR          |                                |

## Intent

Large PRs (1000+ lines) are expensive to review. Human review time is the scarce resource — code generation is cheap. This spec defines a workflow that front-loads review onto small, readable specs so that large implementations become mechanical follow-throughs.

### Workflow

A single PR carries a feature from spec to implementation:

1. Author creates `specs/<name>.md` using `new-spec.sh`, opens a PR.
2. A GitHub Action auto-renames the file to `specs/<PR#>-<name>.md` and adds the `spec` label.
3. A maintainer comments `@claude analyze` — Claude performs a deep-interview-style analysis, posting probing questions as review comments until it has zero ambiguity about the implementation. When satisfied, Claude adds the `claude-approved` label and posts a summary.
4. Maintainers review and approve the spec.
5. A maintainer comments `@claude implement` — Claude generates a one-shot implementation on the same branch. The Action adds the `implementation` label.
6. Maintainers review the implementation, merge the PR. Spec status becomes `implemented`.

### Labels

| Label | Meaning | Applied by |
|-------|---------|------------|
| `spec` | PR contains a spec file | Action (auto) |
| `no-spec` | PR has no spec file | Action (auto) |
| `implementation` | PR contains non-spec code alongside a spec | Action (auto) |
| `claude-approved` | Claude's analysis found no ambiguities | Claude |

Labels are additive — a PR starts as `spec`, later gains `implementation` when code lands.

### Soft Guardrails

PRs exceeding 500 changed lines without a spec file get an automated warning comment linking to `CONTRIBUTING.md`. No hard CI gate — the threshold for "needs a spec" is human judgment, documented as convention for major features and architectural changes.

### Spec Files

Specs live in `specs/` and are named `<PR#>-<feature-name>.md`. They are frozen point-in-time design documents — the code is the source of truth for current behavior; specs record the reasoning behind it.

The template has three sections following the project's existing philosophy:
- **Intent** — what and why, invariants, types, architectural boundaries
- **Evaluation** — tests, benchmarks, assertions that prove correctness
- **Execution** — optional implementation direction

Plus metadata: Author, Created, Status (`proposed` / `approved` / `implemented`), PR number.

### Claude Commands

**`@claude analyze`** — Deep-interview-style analysis of the spec. Claude reads the spec, explores relevant codebase areas, then posts targeted questions exposing ambiguities, missing invariants, unclear evaluation criteria, and hidden assumptions. Continues asking follow-up questions on subsequent `@claude` replies until satisfied. Posts a summary comment and adds `claude-approved` when done.

**`@claude implement`** — One-shot implementation from an approved spec. Claude reads the spec, generates the implementation, and pushes commits to the PR branch.

### CONTRIBUTING.md

A new `CONTRIBUTING.md` documents:
- The spec-driven workflow and its motivation
- How to create and submit a spec
- What qualifies as "needs a spec" (major features, architectural changes)
- How Claude analysis and implementation work
- That contributions are welcome

## Evaluation

### Workflow correctness
- A PR adding only a `specs/*.md` file gets the `spec` label automatically.
- A PR with no spec file gets the `no-spec` label.
- A PR with both spec and non-spec changes gets both `spec` and `implementation` labels.
- A PR exceeding 500 changed lines without a spec file gets a warning comment.
- The spec file is auto-renamed to include the PR number on PR open.

### Claude integration
- `@claude analyze` on a spec PR triggers Claude to post review comments with probing questions about the spec.
- When Claude is satisfied, it adds `claude-approved` and posts a summary.
- `@claude implement` triggers Claude to generate implementation commits on the PR branch.

### Spec template
- `new-spec.sh <name>` creates a valid spec file with the correct template, author, and date.
- Template includes Status and PR metadata fields.

### Documentation
- `CONTRIBUTING.md` exists at repo root and explains the spec workflow.
- `specs/README.md` reflects the updated workflow (no tracking issues).

## Execution

### Files to create
- `.claude/commands/analyze-spec.md` — prompt for deep-interview spec analysis
- `.claude/commands/implement-spec.md` — prompt for one-shot implementation
- `CONTRIBUTING.md` — contribution guide

### Files to modify
- `specs/TEMPLATE.md` — add Status and PR metadata fields
- `specs/README.md` — update workflow (remove tracking issue step)
- `specs/new-spec.sh` — no date prefix, just `<name>.md` (PR number added by Action)
- `.github/workflows/spec-tracking.yml` — replace issue-creation with labeling + rename + warning logic

### Files to remove
- Nothing removed; `spec-tracking.yml` is repurposed.

## Open Questions

- Should `@claude analyze` run automatically on spec PRs, or only when explicitly invoked?
- Should the `claude-approved` label be required before maintainer approval, or advisory?

## References

- Existing `claude.yml` workflow handles `@claude` commands via `anthropics/claude-code-action@v1`
- Spec template: `specs/TEMPLATE.md`
