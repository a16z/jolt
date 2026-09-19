# CLAUDE.md

## Project Overview

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V (RV64IMAC) that efficiently proves and verifies program execution. It uses sumcheck-based protocols, multilinear polynomial commitments (Dory), and the Twist/Shout lookup argument.

## Essential Commands

### Linting and Formatting

```bash
# Must pass in both standard and ZK modes
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```

### Testing

```bash
# Always cargo nextest, never cargo test
cargo nextest run --cargo-quiet

# Run specific test in specific package
cargo nextest run -p [package_name] [test_name] --cargo-quiet

# Prover acceptance suites (mirror CI)
cargo nextest run -p jolt-verifier standard_muldiv --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk --cargo-quiet
cargo nextest run -p jolt-prover --features akita,prover-fixtures --cargo-quiet
```

### Building

```bash
# Prefer clippy over build for validation. Only build when preparing to execute a binary.
cargo build -p jolt-prover -q

# After pulling changes, reinstall the jolt CLI or guest builds may fail.
cargo install --path . --locked
```

### Profiling

```bash
# Modular prover (primary): emits benchmark-runs/{timestamp}_modular_{name}_{scale}/ containing trace.json
# (Perfetto UI / trace_processor SQL), summary.json (machine-queryable), and memory.html,
# with benchmark-runs/latest_modular_{name}_{scale} symlinked to the newest successful run.
cargo run --release -p jolt-prover --features profiling -- profile --name fibonacci --format chrome
# --name options (default scale): fibonacci (16), sha2-chain (22), sha3-chain (22), btreemap (20)
# --scale <log2 trace length> overrides; --format none = no-subscriber Instant baseline
# --backend reference (default, naive test oracle) | optimized (performance tier);
# optimized artifacts get an _optimized suffix on the run dir and latest_ symlink

# Canonical summary queries (no Perfetto UI needed) — see book/src/usage/profiling/zkvm_profiling.md
jq '.stages | map({label, s: (.wall_time_ns/1e9)})' benchmark-runs/latest_modular_fibonacci_16/summary.json
jq '.spans | to_entries | sort_by(-.value.total_ns) | .[:10]' benchmark-runs/latest_modular_fibonacci_16/summary.json

# Multi-scale sweep (one profile subprocess per run; results in benchmark-runs/modular_timings.csv,
# rendered by scripts/benchmark_summary.py, plot_benchmarks.py, plot_memory_usage.py)
cargo run --release -p jolt-prover --features profiling -- benchmark --min-scale 18 --max-scale 21 --resume

# Per-batch heap snapshots (*.folded in the run directory, exact bytes; totals in summary.json's .heap; rendered by memory.html)
cargo run --release -p jolt-prover --features profiling,allocative -- profile --name fibonacci --format chrome

# jolt-eval telemetry objectives over the same summary (grammar: telemetry:<workload>:<metric>)
cargo run -p jolt-eval --bin measure-objectives -- --objective telemetry:fibonacci:prover_time_s

```

The span taxonomy (versioned, normative) lives in `crates/jolt-profiling/src/taxonomy.rs` — renaming a span is a schema change (summary keys and `telemetry:*` objectives break; the profiling smoke test enforces label presence, but it is not yet CI-wired — run it explicitly after taxonomy changes, see the NOTE in `.github/workflows/rust.yml`).

## Architecture

### Crate Structure

The proof system is split into focused crates under `crates/`. Top-level crates include `tracer`, `jolt-sdk`, `jolt-inlines`, and `common`.

Arkworks dependencies use a fork: `a16z/arkworks-algebra` branch `dev/twist-shout`, pinned in the root `Cargo.toml`.

**jolt-prover** — Prover orchestration for the staged Jolt protocol; Dory is the default PCS, `akita` selects the packed lattice path, and `zk` enables BlindFold.

**jolt-verifier / jolt-claims** — Verifier staging and the symbolic protocol relations shared with the prover.

**jolt-kernels / jolt-witness** — Reference and optimized prover kernels plus trace-backed witness construction.

**jolt-host / jolt-program** — Guest builds, tracing entry points, bytecode expansion, and shared program preprocessing.

**jolt-field / jolt-poly / jolt-sumcheck / jolt-openings** — Field, polynomial, sumcheck, and PCS abstractions.

**tracer** — RISC-V emulator producing execution traces (`Cycle` per instruction)

**jolt-sdk** — `#[jolt::provable]` macro generating prove/verify/analyze/preprocess functions

**jolt-inlines** — Optimized cryptographic primitives (sha2, blake3, bigint, secp256k1, etc.) replacing guest-side computation with efficient constraint-native implementations

**common** — Shared constants (`XLEN`, `REGISTER_COUNT`, thresholds) and `JoltDevice`/`MemoryLayout` types

The SDK's `host` feature enables native build/prove APIs. On `jolt-prover`, `akita` selects the packed lattice protocol and is mutually exclusive with `zk`.

### Key Type Parameters

The staged prover and verifier are generic over the PCS, vector commitment,
and transcript. Dory uses `Fr`; the `akita` build uses `AkitaField`.

```
PCS: CommitmentScheme
VC: VectorCommitment<Field = PCS::Field>
T: Transcript<Challenge = PCS::Field>
```

### Prover Pipeline

1. **Preprocess**: `jolt-program` expands bytecode and builds the program/RAM view; the selected PCS builds prover and verifier preprocessing.
2. **Trace and witness**: `jolt-host`/`tracer` execute the guest; `jolt-witness` exposes trace-backed committed and virtual polynomials.
3. **Commit**: stage 0 commits the trace objects. Dory streams individual polynomials; Akita packs the one-hot trace and auxiliary objects.
4. **Reduce**: `jolt-prover/src/stages/` proves stages 1–7 from the symbolic relations in `jolt-claims`.
5. **Open**: stage 8 reduces the remaining claims and dispatches the joint PCS opening.
6. **BlindFold**: ZK builds prove the committed sumcheck transcript with `jolt-blindfold`.

### Protocol Ownership

- `jolt-claims` owns relation IDs, opening geometry, protocol dimensions, and every symbolic input/output expression.
- `jolt-verifier` owns proof/preprocessing wire types, transcript validation, and the stage verifier.
- `jolt-prover` owns proving orchestration and backend-specific stage 0/stage 8 integration; it consumes the shared claims rather than restating verifier formulas.
- `jolt-witness` owns trace-backed witness construction. `jolt-kernels` owns reference and optimized evaluation kernels.
- `jolt-poly`, `jolt-sumcheck`, `jolt-openings`, `jolt-dory`, and `jolt-akita` own the reusable polynomial, sumcheck, and PCS layers.
- `jolt-r1cs` owns the RV64 constraint matrices and variable layout. `jolt-blindfold` owns the generic zero-knowledge proof over recorded sumchecks.

Committed trace polynomials are identified by `JoltCommittedPolynomial`; virtual
polynomials by `JoltVirtualPolynomial`. Do not recreate their ordering or opening
points outside `jolt-claims`.

### ZK Feature Gate

The `zk` feature selects committed sumcheck recorders and the BlindFold tail at
compile time. Clear proofs carry `JoltProofClaims::Clear`; ZK proofs carry
`JoltProofClaims::Zk`. `JoltProof::protocol` still self-describes the mode, and
the verifier rejects a build/proof mismatch.

`jolt-prover/src/recorder.rs` is the mode seam. Stage recipes are shared: clear
recorders expose round polynomials, while ZK recorders retain committed witnesses.
After stage 8, `jolt-prover/src/blindfold.rs` replays the assembled shell through
the verifier's own stages and lowers those outputs with
`jolt-verifier/src/stages/zk/`. It checks that the replay transcript exactly equals
the forward prover transcript before constructing the BlindFold witness.

**Critical invariant:** a relation's input and output expressions have one owner
in `jolt-claims`. Prover evaluation, verifier checking, and BlindFold lowering must
all consume those expressions. Never add a parallel claim formula for one mode.
Changes to stage order, public-value derivation, or transcript absorption require
clear, ZK, and verifier-fixture tests.

## Development Guidelines

### Performance

- Profile before optimizing
- Benchmark changes to `poly/` code — small regressions multiply across thousands of sumcheck rounds
- Use `#[inline]` judiciously in hot paths
- Pre-allocate vectors unsafely when size is known; avoid clones in hot paths
- Hot trace paths get one pass and one owner of trace-sized storage. Produce or share derived rows during the existing pass instead of walking or materializing the trace again.

### Prover Hot Paths

- Sumcheck inner loop dominates: polynomial bind, sumcheck_evals, eq_poly evals
- Keep compact witness values in their native scalar types until field arithmetic is required.
- Preserve the shared lazy-RA kernels; do not materialize one dense field vector per RA polynomial.

### Code Style Invariants

- Use `non_snake_case` for math variables: `log_T`, `ram_K`, `log_K`, etc.
- **Machine-checked, repo-wide:** one `cfg_attr` per predicate per item; fold adjacent `#[cfg_attr(P, A)]` `#[cfg_attr(P, B)]` into `#[cfg_attr(P, A, B)]`.
- **Machine-checked, repo-wide:** `#[allocative(visit = ...)]` never decorates a container of primitives (`Vec<u32>`, `Vec<Vec<usize>>`, `Vec<Option<u8>>`, ...) or of field elements (`Vec<F>`, `Vec<Vec<F>>`, `[Vec<F>; N]`). Native impls report element types and unused capacity, and `JoltField: MaybeAllocative` makes them reachable for every scalar table. In-crate element types built from primitives and `F` derive `Allocative`; `jolt_kernels::backend::visit_heap_free_elements` is only for element types that own no heap but carry no `Allocative` impl — the witness rows, selectors, opening ids, and prefix evaluations whose jolt-claims, jolt-lookup-tables, and jolt-witness types would gain the derive for nothing.
- **Machine-checked on added lines:** import types, traits, enums, constants, and PascalCase macros; reference them by short name. Keep enum variants qualified by the imported enum type (`Kind::ADD`, not bare `ADD`). Import singleton paths directly (`use x::Kind`, never `use x::Kind::{self}`). Lowercase namespace free functions and macros (`std::mem::take`, `tracing::info!`) may stay qualified. Fully qualified paths remain valid for ambiguity, attribute arguments (`allocative(visit = crate::backend::visit_heap_free_elements)`), and macro bodies. Once a path is imported, never spell it qualified in the same file.
- Alias an instruction-kind enum as `Kind` at emitter call sites and write `Kind::INSTRUCTION`; never qualify emitted instructions with `SourceInstructionKind`, `JoltInstructionKind`, or a module path.
- Give each protocol formula, geometry or sizing law, schedule, and state transition one owner. Consumers call the canonical implementation instead of mirroring or open-coding it. Tests use independent ground truth or the production computation; never a second implementation of the same rule as their oracle.
- Encode correlated state and absence with typed requests, structs, enums, and `Option`; never value sentinels, decomposed arguments, or parallel options that can disagree.
- Serialized or ordinal enums are append-only; keep feature-gated and test-only variants last so feature selection cannot shift real discriminants.
- Enforce public-boundary invariants at the point of fault in release builds with typed errors. Use `debug_assert` only for properties already pinned by types or release checks; keep recoverable capability gaps separate from invariant violations.
- Derive trait impls instead of hand-rolling; exhaust derive escapes (`#[allocative(bound = "F: JoltField")]`, `visit = ...`, `skip`) first. Hand-write only what a derive cannot express, keep it local to the one type that needs it, and size buffers by `capacity()`, not `len()`.
- A free function is pure or shared across ≥2 callers. Otherwise make it a method on the type whose state it uses; inline it when no type owns the behavior.
- No public API, abstraction, mode, or state container without an in-repo production caller or documented external contract; add it with its first use. Lazy-init globals and error slots are valid, but speculative lifecycle guards and unreachable transitions are not.
- State enforcement honestly in docs: never describe a property as constraint-enforced when it holds only for the honest encoder or under a `debug_assert`; name the mechanism and location that pins each invariant. If reviewers independently misread a deliberate gap, the missing argument belongs in a comment.
- Make names track current semantics: rename vocabulary when an encoding changes (`UnsignedIncMsb` → `BalancedIncCarry`); keep no compatibility names.
- Add `cfg`/`cfg_attr` gates only where the build requires them.
- Before PR handoff, audit every added test and helper. Remove development-only probes, ignored tests, temporary benchmarks, diagnostic counters or histograms, and one-off fuzz or parity scaffolding. Keep permanent tests only when they add a distinct failure signal beyond existing tests, golden fixtures, or CI. Make a worthwhile manual diagnostic an intentional tool or benchmark with a documented command.

### Testing Guidelines

- Do not add old-vs-new equivalence tests that reimplement the pre-change logic as the oracle. Transition-validation belongs in the PR process, not the permanent suite. Permanent tests must assert against independent ground truth: spec vectors, frozen wire digests, verifier fixtures, `jolt-kernels`' reference tier, or algebraic properties. If the old code is deleted, its reimplementation in a test is dead weight — delete the test rather than keep the old logic alive inside it. A `#[cfg(test)]` copy of superseded production code "kept as the oracle" is the same anti-pattern.

### Lint Policy

- Workspace enforces `allow_attributes = "deny"` — use `#[expect(...)]` instead of `#[allow(...)]`
- The jolt-verifier runtime closure (18 crates, listed in `specs/verifier-closure-lints.md`) carries stricter crate-root lints: panic-source denies (`indexing_slicing` in control-plane crates, `panic_in_result_fn`, `wildcard_enum_match_arm`, ...), `forbid(unsafe_code)` where a crate has no unsafe, and numeric-discipline denies in jolt-verifier itself — which additionally denies `unreachable`, the only abort macro that escapes both `panic` and `panic_in_result_fn`. New code in those crates must fix the lint or add `#[expect(clippy::..., reason = "...")]` at the narrowest scope with a real justification
- `.unwrap()` and `.expect()` are fine in tests. In non-test code, avoid them unless the alternative significantly hurts readability (e.g., infallible fixed-size array conversions). When used, annotate the function with `#[expect(clippy::unwrap_used)]` or `#[expect(clippy::expect_used)]`
- Use `#[expect(clippy::...)]` on test modules to blanket-suppress test-inappropriate lints rather than per-function annotations

### Comments

Match the codebase's low comment density. Worth writing: WHY comments, WARNING for non-obvious gotchas, SAFETY on unsafe blocks, algorithm explanations (link to paper if applicable), public API docs stating behavior or invariants.
Do not narrate code or test assertions. If a comment only restates an expression, make the code self-documenting instead.
