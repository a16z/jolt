# scaffold-workspace: Create empty crate scaffolding for all new crates

**Scope:** crates/, Cargo.toml

**Depends:** none

**Verifier:** ./verifiers/default.sh

**Context:**

This is the first task in the jolt-core refactoring. Create the skeleton workspace structure for all 7 new crates. Each crate gets a `Cargo.toml` and an empty `src/lib.rs` with a top-level doc comment describing the crate's purpose.

`jolt-transcript` and `jolt-field` already exist and are complete — do not touch them.

### Crates to scaffold

| Crate | Purpose | Dependencies |
|-------|---------|-------------|
| `jolt-poly` | Polynomial types and operations | `jolt-field` |
| `jolt-openings` | Commitment scheme traits + opening accumulators | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-sumcheck` | Sumcheck protocol engine | `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-spartan` | R1CS + Spartan prover/verifier | `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript` |
| `jolt-instructions` | RISC-V instruction set + lookup tables | `jolt-field` |
| `jolt-dory` | Dory commitment scheme impl | `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs` |
| `jolt-zkvm` | zkVM prover/verifier orchestration | all `jolt-*` crates |

### Shared Cargo.toml conventions

Every crate must include:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
thiserror = "2"

[features]
default = ["parallel"]
parallel = ["rayon"]

[dependencies.rayon]
version = "1"
optional = true
```

The `serde` feature flag on `jolt-field` and `jolt-transcript` deps should be enabled.

### Workspace Cargo.toml

Add all new crates as workspace members. Use path dependencies for inter-crate references (e.g., `jolt-field = { path = "../jolt-field" }`).

### What each `src/lib.rs` contains

A single doc comment describing the crate per the table above, and nothing else. Example:

```rust
//! Polynomial types and operations for multilinear, univariate, and
//! specialized polynomials. Backend-agnostic and reusable outside Jolt.
```

**Acceptance:**

- All 7 new `Cargo.toml` files exist with correct dependencies
- All 7 new `src/lib.rs` files exist with doc comments
- Workspace `Cargo.toml` includes all new crates as members
- `cargo check --workspace` passes (empty crates compile)
- `cargo clippy --workspace` clean
- No modifications to `jolt-transcript` or `jolt-field`
- No modifications to `jolt-core/`
