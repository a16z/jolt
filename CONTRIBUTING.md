# Contributing to Jolt

Jolt is a security-critical zkVM. Every change to this codebase has the potential to compromise soundness, leak private inputs, or introduce exploits. We hold all contributions — whether written by humans or AI — to the same high standard.

## Getting Started

### Prerequisites

- **Rust toolchain**: Pinned in `rust-toolchain.toml` (currently 1.88)
- **cargo-nextest**: Required for running tests (`cargo install cargo-nextest`)
- **taplo**: TOML formatter (`cargo install taplo-cli`)
- **typos**: Spell checker (`cargo install typos-cli`)

### Git Hooks

We use [lefthook](https://github.com/evilmartians/lefthook) for local git hooks. Install it and activate:

```bash
brew install lefthook   # or: cargo install lefthook
lefthook install
```

This sets up:

| Hook | What Runs | Parallel |
|------|-----------|----------|
| **pre-commit** | `cargo fmt`, `clippy`, `taplo`, `typos`, `machete` | Yes |
| **commit-msg** | Conventional commits format validation | No |

To skip hooks: `git commit --no-verify` (do not make this a habit).

### Building

```bash
cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings

cargo build -p jolt-core --message-format=short -q
```

### Testing

Always use `cargo nextest`.

```bash
# Run all core tests
cargo nextest run --cargo-quiet -p jolt-core

# Primary correctness check — run this before every PR
cargo nextest run -p jolt-core muldiv --cargo-quiet

# Run a specific test
cargo nextest run -p [package_name] [test_name] --cargo-quiet
```

### Formatting and Linting

```bash
cargo fmt --all --check
cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings
cargo clippy --all --message-format=short -q --all-targets --no-default-features -- -D warnings
taplo fmt --check
typos
```

### Dependency DAG

The `crates/` directory enforces a strict dependency hierarchy. Do **not** introduce upward or circular dependencies:

```
jolt-field          (no internal deps)
jolt-transcript     (no internal deps)
jolt-poly           → jolt-field
jolt-openings       → jolt-field, jolt-poly, jolt-transcript
jolt-sumcheck       → jolt-field, jolt-poly, jolt-transcript
jolt-instructions   → jolt-field
jolt-spartan        → jolt-field, jolt-poly, jolt-transcript, jolt-sumcheck, jolt-openings
jolt-dory           → jolt-field, jolt-poly, jolt-transcript, jolt-openings
jolt-zkvm           → all of the above
```

## Code Standards

### Lint Configuration

All crates inherit workspace-level lints. Clippy runs in **pedantic mode**.

### Unsafe Code

This codebase uses `unsafe` in performance-critical paths. Every `unsafe` block **must** have a `// SAFETY:` comment explaining invariants:

```rust
// SAFETY: `len` was computed from `num_vars` which is validated at construction,
// guaranteeing `index < self.evaluations.len()`.
unsafe { *self.evaluations.get_unchecked(index) }
```

### Comments

**Delete these:**
- Section separators (`// ==========`, `// ----------`)
- Doc comments that restate the item name
- Obvious comments (`/// Returns the count` on `get_count()`)
- Commented-out code
- TODOs without issue links

**Keep these:**
- WHY something is done (when not obvious)
- WARNING comments for non-obvious gotchas
- SAFETY comments for unsafe blocks
- Complex algorithm explanations with paper references
- Public API docs that explain behavior, constraints, or invariants

### Error Handling

- Use `thiserror` for library error types, `eyre`/`anyhow` for application code
- Do **not** use `.unwrap()` or `.expect()` in library code — propagate errors with `?`
- Panics are acceptable only in cases that represent invariant violations.

### Performance

Performance is a top priority. This is a proving system where every nanosecond in the inner loop multiplies across millions of sumcheck rounds.

- Profile before optimizing — don't guess
- Pre-allocate vectors when size is known
- Avoid clones

### Dependencies

- Do **not** add new dependencies without justification in the PR description
- Prefer crates from the Rust standard library or already in the dependency tree
- All dependencies must have compatible licenses (MIT, Apache-2.0, BSD)
- Workspace dependencies are defined in the root `Cargo.toml` — use `workspace = true` in crate manifests

## Pull Requests

PR title must follow conventional commits: `feat(core): ...`, `fix(tracer): ...`, `refactor(poly): ...`

Valid scopes: `core`, `tracer`, `sdk`, `poly`, `spartan`, `dory`, `sumcheck`, `field`, `transcript`, `openings`, `instructions`, `zkvm`, `deps`, `ci`

A [PR template](/.github/pull_request_template.md) is provided — fill out every section.

## AI-Assisted Contributions

We welcome AI-assisted contributions, but they are held to the same standard as human-written code. AI-generated code is more likely to:

- Introduce `.unwrap()` calls and silent panics
- Add unnecessary dependencies
- Produce large, unfocused PRs
- Generate superficial comments and documentation
- Miss edge cases
- Introduce subtle correctness bugs that pass basic tests

Our CI is specifically designed to catch these patterns. The pedantic clippy configuration, mandatory `SAFETY` comments on `unsafe` blocks, and PR size limits exist specifically to maintain code quality regardless of authorship.
