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
2. **Open a PR** with just the spec file.
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
| `claude-spec-review-request` | Triggers external Claude spec analysis | Maintainer (manual) |
| `claude-spec-approved` | Claude's analysis found no ambiguities | Claude |

## Claude Skills

These Claude Code skills are available in this repo:

| Skill | Description |
|-------|-------------|
| `/new-spec <name>` | Create a new spec file from the template |
| `/analyze-spec` | Interactive Socratic analysis of a spec (local) |
| `/implement-spec` | Autonomous implementation from an approved spec (local/cloud) |
| `/ci-code-review` | One-pass code review outside CI, before posting a review attestation |
| `/verifier-classifier` | Classify verifier impact and assign review-risk levels |

Spec analysis can also be triggered externally by adding the `claude-spec-review-request` label to a PR.

## AI Merge Gate

Once its admin setup is in place, merging into `main` requires both `ai-merge-gate` and
`ai-merge-gate/attestation`. The verifier classifier is the only model called by this gate.
It assigns `risk-1` for tests/scripts/CI/docs, `risk-2` for non-verifier code, `risk-3` for small,
scoped verifier-reaching dependency changes, or `risk-4` for protocol/verifier changes and
uncertainty. The highest applicable level wins. Risk-1 through risk-3 are eligible for native
auto-merge once the gate, attestation and required CI checks pass; core authorization and
gate-input protections still apply. Risk-4 also gets `needs-human-review` and needs a
current-head approval by a maintainer other than the PR author.

Run code review outside CI. Once it finds no issues, the Pika App bot or a member of the
configured Jolt maintainer team posts the exact standalone line below in a **PR comment**.
This is a format example using an existing commit, not an attestation of the current PR head;
replace the number, full SHA and review details with the actual reviewed values.

```text
I have reviewed pull request #2203 at commit f316de7b96be3846a36713175d9536d27d256855 using skill ci-code-review and model Astra with Max reasoning effort and found no issues.
```

Skill, model and effort must be nonempty and may contain spaces. Being the PR author alone
does not qualify. A new head requires a new attestation; editing away or deleting the line
invalidates it. Human approval, review bodies, old markers and cached CI reviews cannot replace
it, and the `ai-review` label starts no review. See [the gate guide](.github/ai-merge-gate.md)
for author permissions, core authorization, risk policy, admin setup and the off switch.

## Development Setup

### Prerequisites

- Rust toolchain (see `rust-toolchain.toml`)
- [cargo-nextest](https://nexte.st/) for running tests
- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit) for the dependency-advisory check

### Key Commands

```bash
# Lint (must pass in both modes)
cargo clippy -p jolt-prover --features prover-fixtures --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-prover --features prover-fixtures,zk --message-format=short -q --all-targets -- -D warnings

# Format
cargo fmt -q

# Test (always use nextest)
cargo nextest run --cargo-quiet

# Primary correctness check
cargo nextest run -p jolt-verifier standard_muldiv --cargo-quiet --features prover-fixtures
cargo nextest run -p jolt-prover zk_muldiv --cargo-quiet --features prover-fixtures,zk

# Dependency advisories (CI runs this nightly over every tracked Cargo.lock;
# see .github/workflows/cargo-audit.yml for the policy and triage steps)
cargo audit --deny unsound --deny yanked
```

## PR Titles and Commit Messages

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/) — CI enforces this (`.github/workflows/pr-title.yml`):

```
<type>[optional scope][optional !]: <description>
```

Allowed types: `feat` `fix` `chore` `docs` `refactor` `perf` `test` `build` `ci` `style` `revert` `spec`

Since PRs are squash-merged, only the PR title is checked — individual commit messages are not.

## Code Style

- `cargo fmt` + `cargo clippy` with zero warnings
- Performance is critical — profile before optimizing, benchmark changes to hot paths
- The codebase uses `non_snake_case` for math variables: `log_T`, `ram_K`, etc.
- See `CLAUDE.md` for full development guidelines
