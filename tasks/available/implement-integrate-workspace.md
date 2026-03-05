# integrate-workspace: Fix cross-crate integration after all implementations

**Scope:** entire workspace

**Depends:** impl-jolt-poly, impl-jolt-openings, impl-jolt-sumcheck, impl-jolt-spartan, impl-jolt-instructions, impl-jolt-dory, impl-jolt-zkvm, test-jolt-poly, test-jolt-openings, test-jolt-sumcheck, test-jolt-spartan, test-jolt-instructions, test-jolt-dory, test-jolt-zkvm

**Verifier:** ./verifiers/default.sh

**Context:**

All crate implementations and their tests are complete. This task ensures the entire workspace compiles and tests pass together, and wires the new crates into the existing workspace alongside `jolt-core`.

### Tasks

#### 1. Workspace compilation

- `cargo check --workspace` must pass
- Resolve any cross-crate type mismatches, trait bound issues, or version conflicts
- Ensure all inter-crate path dependencies in `Cargo.toml` are correct

#### 2. Cross-crate trait coherence

Verify that traits defined in one crate are correctly implemented in another:
- `jolt-dory` implements `CommitmentScheme` + `HomomorphicCommitmentScheme` + `StreamingCommitmentScheme` from `jolt-openings`
- `jolt-spartan` uses `SumcheckInstanceProver` from `jolt-sumcheck`
- `jolt-zkvm` sub-protocols implement `SumcheckInstanceProver` from `jolt-sumcheck`
- All `Field` bounds resolve to `jolt-field::Field`

#### 3. Feature flag consistency

- `parallel` feature propagates correctly (enabling it on `jolt-zkvm` enables it on all deps)
- Building without `parallel` works on all crates
- `serde` feature works correctly across all crate boundaries

#### 4. Wire into existing workspace

- Add all new crates to the workspace root `Cargo.toml` members list
- Ensure `jolt-core` and the new crates coexist (no naming conflicts)
- `jolt-sdk` does NOT depend on new crates yet (that's a future step)
- No circular dependencies

#### 5. Full test suite

- `cargo nextest run --workspace` passes
- No test flakiness (run twice to verify)
- Clippy clean across entire workspace: `cargo clippy --workspace`
- Format check: `cargo fmt --all -- --check`

#### 6. Dependency audit

- No unnecessary dependencies introduced
- No duplicate dependency versions where avoidable
- `cargo tree` shows clean dependency graph
- Serde is used consistently (no `CanonicalSerialize` in new crates' public APIs)

**Acceptance:**

- `cargo check --workspace` passes
- `cargo nextest run --workspace` passes
- `cargo clippy --workspace` clean
- `cargo fmt --all -- --check` passes
- No circular dependencies
- Feature flags (`parallel`, `serde`) propagate correctly
- `jolt-core` and new crates coexist without conflicts
- Dependency graph is clean (no unnecessary deps, no arkworks leaking into public APIs)
