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
2. **Open a PR** with just the spec file. A GitHub Action will rename it to `<PR#>-<name>.md` and add the `spec` label.
3. **Spec analysis**: A maintainer triggers `@claude analyze` on the PR. Claude performs a deep analysis, posting probing questions as review comments to identify ambiguities, missing invariants, and unclear evaluation criteria.
4. **Spec review**: Maintainers review and approve the spec. The spec is approved when both humans and Claude agree it's unambiguous.
5. **Implementation**: A maintainer triggers `@claude implement` for a one-shot implementation on the same branch, or implements manually.
6. **Merge**: Maintainers review the implementation and merge.

### Writing a Good Spec

A spec has three sections:

- **Intent**: What are we building and why? Define the types, invariants, abstractions, and architectural boundaries. Two engineers reading this should arrive at the same implementation.
- **Evaluation**: How do we know it works? Concrete, testable criteria. Tests, benchmarks, assertions. These verify the invariants from Intent, not the execution details.
- **Execution** (optional): Algorithmic direction, optimizations to consider, modules to touch. The implementer should be able to derive most of this from Intent and Evaluation.

Focus on **intent and evaluation**. Execution is downstream.

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
