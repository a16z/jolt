# Contributing to Jolt

We welcome contributions! Jolt uses a **spec-driven development** workflow for major features, and standard PRs for everything else.

## Quick Start

- **Bug fixes, small improvements, documentation**: Open a PR directly. No spec needed.
- **Major features, architectural changes**: Start with a spec (see below).

The rule of thumb: if the implementation will be hard to review (500+ lines of non-trivial changes), write a spec first. The spec is small and reviewable; the implementation follows mechanically from it.

## Spec-Driven Workflow

Large PRs are expensive to review. Code generation is cheap — human verification time is the scarce resource. We optimize for that by front-loading review onto small, readable specs.

### How It Works

A single PR carries a feature from spec to implementation:

1. **Create a spec** using `/new-spec <feature-name>` in Claude Code, or copy [`specs/TEMPLATE.md`](specs/TEMPLATE.md) manually.
2. **Open a PR** with just the spec file. A GitHub Action will add the `spec` label.
3. **Spec analysis**: Adding the `claude-spec-review-request` label triggers an external analysis. Claude performs a single-pass analysis, posting all questions at once — ambiguities, missing invariants, unclear evaluation criteria. When satisfied, Claude adds the `claude-spec-approved` label.
4. **Spec review**: Maintainers review and approve the spec.
5. **Implementation**: Run `/implement-spec` in [Claude Code cloud](https://claude.ai/code) or locally. Claude generates a one-shot implementation from the approved spec.
6. **Merge**: Maintainers review the implementation and merge.

### Writing a Good Spec

A spec has these sections (see [`specs/TEMPLATE.md`](specs/TEMPLATE.md)):

- **Summary**: One paragraph — what is this feature and why does it matter?
- **Intent**: Goal (one sentence), Invariants (correctness properties), Non-Goals (explicit scope boundaries)
- **Evaluation**: Acceptance Criteria (checkboxes), Testing Strategy (host + zk modes), Performance expectations
- **Design**: Architecture (how it fits the existing system), Alternatives Considered (what was rejected and why)
- **Documentation**: What changes to the Jolt book (`book/`) are required?
- **Execution** (optional): Implementation direction, algorithmic approach
- **References**: Papers, related specs, prior art

Focus on **intent and evaluation**. Execution is downstream — the implementer should be able to derive most of it from what you want (intent) and how you'll verify it (evaluation).

### Labels

| Label | Meaning | Applied by |
|-------|---------|------------|
| `spec` | PR contains a spec file | GitHub Action (auto) |
| `no-spec` | PR has no spec file | GitHub Action (auto) |
| `implementation` | PR contains code alongside a spec | GitHub Action (auto) |
| `claude-spec-review-request` | Triggers external Claude spec analysis | Maintainer (manual) |
| `claude-spec-approved` | Claude's analysis found no ambiguities | Claude |

### Soft Guardrails

PRs exceeding 500 changed lines without a spec file get an automated warning comment. This is a suggestion, not a gate — use your judgment.

## Claude Skills

These Claude Code skills are available in this repo:

| Skill | Description |
|-------|-------------|
| `/new-spec <name>` | Create a new spec file from the template |
| `/analyze-spec` | Interactive Socratic analysis of a spec (local) |
| `/implement-spec` | Autonomous implementation from an approved spec (local/cloud) |
| `/ci-code-review` | Deep PR code review with parallel analysis agents |

Spec analysis can also be triggered externally by adding the `claude-spec-review-request` label to a PR.

## Development Setup

### Prerequisites

- Rust toolchain (see `rust-toolchain.toml`)
- [cargo-nextest](https://nexte.st/) for running tests

### Key Commands

```bash
# Lint (must pass in both modes)
cargo clippy -p jolt-core --features host --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings

# Format
cargo fmt -q

# Test (always use nextest)
cargo nextest run --cargo-quiet

# Primary correctness check
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

## Code Style

- `cargo fmt` + `cargo clippy` with zero warnings
- Performance is critical — profile before optimizing, benchmark changes to hot paths
- The codebase uses `non_snake_case` for math variables: `log_T`, `ram_K`, etc.
- See `CLAUDE.md` for full development guidelines
