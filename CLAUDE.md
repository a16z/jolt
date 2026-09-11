# CLAUDE.md

## Project Overview

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V (RV64IMAC) that efficiently proves and verifies program execution. It uses sumcheck-based protocols, multilinear polynomial commitments (Dory), and the Twist/Shout lookup argument.

## Agent Workflow

- Carry an authorized request through implementation and verification. Resolve routine choices from repository conventions; ask only when missing information would materially change correctness or scope. Continue independent work while an answer is pending.
- User instructions take precedence over skill guidance. Reuse authorization and answers already supplied. If a skill blocks requested work, cite its exact file and instruction and explain the conflict.
- Read relevant code before asking about repository facts. Use `rg` for searches; batch independent reads when supported, and wait for required results before dependent work. Run Cargo commands sequentially, including across agents.
- Treat follow-up messages as updates to the active task unless the user cancels or replaces it.
- Lead with the result. Keep explanations concise and concrete; report checks actually run, their results, and any remaining blocker. Preserve required output formats in automated workflows.

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

# Primary correctness check — run muldiv e2e test in both modes
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk

# Modular prover acceptance suites (mirror CI): clear-mode byte-diff ratchets
# vs the legacy prover, and the modular ZK e2e (muldiv accept, tamper reject,
# advice, committed program)
cargo nextest run -p jolt-prover --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk --cargo-quiet
```

### Building

```bash
# Prefer clippy over build for validation. Only build when preparing to execute a binary.
cargo build -p jolt-prover-legacy -q

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
# --backend reference (default, naive test oracle) | optimized (performance tier, legacy-parity);
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

# Legacy prover
cargo run --release -p jolt-prover-legacy profile --name sha3 --format chrome
# --name options: sha2, sha3, sha2-chain, sha3-chain, fibonacci, btreemap
RUST_LOG=debug cargo run --release --features allocative -p jolt-prover-legacy profile --name sha3 --format chrome
```

The span taxonomy (versioned, normative) lives in `crates/jolt-profiling/src/taxonomy.rs` — renaming a span is a schema change (summary keys and `telemetry:*` objectives break; the profiling smoke test enforces label presence, but it is not yet CI-wired — run it explicitly after taxonomy changes, see the NOTE in `.github/workflows/rust.yml`).

## Architecture

### Crate Structure

The workspace is mid-decomposition: `crates/` holds the modular stack (jolt-verifier, jolt-prover, jolt-sumcheck, jolt-poly, jolt-blindfold, jolt-witness, jolt-openings, jolt-r1cs, jolt-dory, jolt-transcript, jolt-utils, …26 crates), while **crates/jolt-prover-legacy** is the legacy monolith mapped below. Top-level crates: `tracer`, `jolt-sdk`, `jolt-inlines`, `common`.

Arkworks dependencies use a fork: `a16z/arkworks-algebra` branch `dev/twist-shout`, pinned in the root `Cargo.toml`.

**jolt-prover-legacy** — Legacy core proving system

- `host/`: Guest ELF compilation and program analysis (feature-gated behind `host`)
- `zkvm/`: Jolt PIOP — prover, verifier, R1CS/Spartan, memory checking, instruction lookups
- `poly/`: Polynomial types, commitment schemes (Dory, Hyrax, Pedersen), opening proofs
- `field/`: `JoltField` trait and BN254 scalar field implementation
- `subprotocols/`: Sumcheck (batched, streaming, univariate skip), booleanity checks, BlindFold ZK protocol
- `msm/`: Multi-scalar multiplication
- `transcripts/`: Fiat-Shamir transcripts (Blake2b, Keccak)

**tracer** — RISC-V emulator producing execution traces (`Cycle` per instruction)

**jolt-sdk** — `#[jolt::provable]` macro generating prove/verify/analyze/preprocess functions

**jolt-inlines** — Optimized cryptographic primitives (sha2, blake3, bigint, secp256k1, etc.) replacing guest-side computation with efficient constraint-native implementations

**common** — Shared constants (`XLEN`, `REGISTER_COUNT`, thresholds) and `JoltDevice`/`MemoryLayout` types

Feature flag hierarchy: `host` ⊃ `prover` ⊃ `minimal`. Most code is unconditional; `host/` is the main gated module. The `akita` feature selects the packed (lattice/Akita) commitment mode — mutually exclusive with `zk` (compile error on the combination).

### Key Type Parameters

Most core types are generic over three parameters:

```
F: JoltField                              — scalar field (BN254 Fr)
PCS: CommitmentScheme<Field = F>          — polynomial commitment (DoryCommitmentScheme)
ProofTranscript: Transcript               — Fiat-Shamir transcript (Blake2bTranscript)
```

### Prover Pipeline

1. **Trace**: Execute guest ELF in tracer emulator → `Vec<Cycle>` + `JoltDevice` (I/O)
2. **Witness gen**: Trace → committed polynomials (Inc, Ra one-hot, advice)
3. **Streaming commitment**: Dory tier-1 chunks → tier-2 aggregation → final commitments
4. **Spartan**: R1CS constraint satisfaction via univariate skip + outer/product sumchecks
5. **Sumcheck rounds**: Batched sumchecks for instruction lookups, bytecode, RAM/register read-write checking, Hamming booleanity, claim reductions
6. **Opening proofs**: Batched Dory opening proofs via `ProverOpeningAccumulator`
7. **BlindFold**: ZK proof over all sumcheck stages (see BlindFold section below)

### Polynomial Types (poly/)

- `DensePolynomial<F>`: Full field-element coefficients
- `CompactPolynomial<T>`: Small scalar coefficients (u8–i128), promoted to field on bind
- `RaPolynomial`: Lazy materialization via Round1→Round2→Round3→RoundN state machine
- `SharedRaPolynomials`: Shares eq tables across N polynomials for memory efficiency
- `PrefixSuffixDecomposition`: Splits polynomial as `Σ P_i(prefix) · Q_i(suffix)` for efficient sumcheck
- `MultilinearPolynomial<F>`: Enum dispatching over all scalar types + OneHot/RLC variants

### Witness Polynomials (zkvm/witness.rs)

Committed: `RdInc`, `RamInc`, `InstructionRa(d)`, `BytecodeRa(d)`, `RamRa(d)`, `TrustedAdvice`, `UntrustedAdvice`

Virtual (derived during proving): PC, register values, RAM values, instruction flags, lookup operands/outputs

### zkvm/ Submodules

- `spartan/`: Spartan IOP — outer sumcheck, product virtual sumcheck, shift, instruction input constraints
- `r1cs/`: R1CS constraint system and `UniformSpartanKey`
- `ram/`: RAM read-write checking, val evaluation, val final, output check, Hamming booleanity, RAF evaluation
- `registers/`: Register read-write checking, val evaluation
- `instruction_lookups/`: RA virtual sumcheck, read-RAF checking
- `claim_reductions/`: Advice, Hamming weight, increment, instruction lookups, register, RAM RA reductions
- `bytecode/`: Bytecode preprocessing and PC mapping, read-RAF checking
- `config.rs`: `OneHotParams`, `OneHotConfig`, `ReadWriteConfig` — control proof structure (chunk sizes, phase rounds)

### ZK Feature Gate

The `zk` Cargo feature (`cfg(feature = "zk")`) controls zero-knowledge mode:

| Aspect | Standard (`--features host`) | ZK (`--features host,zk`) |
|---|---|---|
| Sumcheck proving | `BatchedSumcheck::prove` — cleartext round polys | `BatchedSumcheck::prove_zk` — Pedersen-committed |
| Uni-skip | `prove_uniskip_round` | `prove_uniskip_round_zk` |
| Proof contains | `Claims<F>` (all opening claims) | `BlindFoldProof` (no cleartext claims) |
| `input_claim()` | Called, appended to Fiat-Shamir transcript | Skipped; `input_claim_constraint()` used by BlindFold |
| Output claim check | Explicit equality check | Skipped; verified by BlindFold R1CS |
| Opening proof | `bind_opening_inputs` (raw eval) | `bind_opening_inputs_zk` (committed eval) |

**Key cfg-gated items:**
- `JoltProof::opening_claims: Claims<F>` — `#[cfg(not(feature = "zk"))]`
- `JoltProof::blindfold_proof: BlindFoldProof` — `#[cfg(feature = "zk")]`
- Prover uses `#[cfg(feature = "zk")]` / `#[cfg(not(feature = "zk"))]` blocks — compile-time path selection, no runtime `zk_mode` field
- Verifier zk mode is fixed at compile time (`zk` feature → `JOLT_VERIFIER_CONFIG` in `crates/jolt-verifier/src/config.rs`); the proof self-describes its protocol (`JoltProof::protocol: JoltProtocolConfig`) and `validate_proof_config` rejects a mismatch fail-closed

**CRITICAL — Verifier `new_from_verifier` must support both modes:**

In ZK mode, `input_claim()` is never called so verifier params can use partial values (e.g., `init_eval = init_eval_public`). In standard mode, `input_claim()` IS called and the values must match the prover exactly. Any verifier param that decomposes a value for BlindFold constraints must reconstruct the full value for standard mode. Use `ram::reconstruct_full_eval()` to add advice contributions back.

### BlindFold Zero-Knowledge Protocol (subprotocols/blindfold/)

BlindFold makes all sumcheck proofs zero-knowledge without SNARK composition. Instead of revealing sumcheck round polynomial coefficients, the prover sends Pedersen commitments. Sumcheck verifier checks are encoded into a small verifier R1CS, proved via Nova folding + Spartan. (The modular prover has full ZK support: `crates/jolt-blindfold` plus `crates/jolt-prover/src/blindfold.rs` and `recorder.rs`, behind jolt-prover's compile-time `zk` feature — see `specs/jolt-prover-blindfold.md`. The module map below is the legacy implementation.)

**Module structure:**
- `mod.rs`: `StageConfig`, `BakedPublicInputs`, `HyraxParams`, R1CS primitives (`Variable`, `LinearCombination`, `Constraint`)
- `r1cs.rs`: `VerifierR1CS`, `VerifierR1CSBuilder` — sparse R1CS encoding of sumcheck verification
- `protocol.rs`: `BlindFoldProver`, `BlindFoldVerifier`, `BlindFoldProof`
- `folding.rs`: Nova folding — cross-term computation, random instance sampling
- `spartan.rs`: Spartan outer + inner sumcheck over the folded R1CS
- `relaxed_r1cs.rs`: Relaxed R1CS instance/witness with Hyrax grid layout
- `witness.rs`: `BlindFoldWitness` — witness assignment from sumcheck stage data
- `output_constraint.rs`: `InputClaimConstraint`, `OutputClaimConstraint`, `ValueSource`, `ProductTerm` — constraint types for claim binding
- `layout.rs`: `LayoutStep`, `ConstraintKind`, `compute_witness_layout` — witness grid layout computation

**Protocol flow:**
1. During stages 1–7, `prove_zk` commits each sumcheck round's coefficients via Pedersen and caches them in `ProverOpeningAccumulator`
2. At stage 8, prover and verifier build the same `VerifierR1CS` from `StageConfig`s and `BakedPublicInputs` (Fiat-Shamir-derived values baked into matrix coefficients)
3. Nova folds the real instance with a random satisfying instance to hide the witness
4. Spartan outer sumcheck proves relaxed R1CS satisfaction; inner sumcheck reduces to a single witness evaluation
5. Hyrax-style openings verify W(ry) and E(rx) against folded row commitments

**Supporting changes:**
- `poly/commitment/pedersen.rs`: Pedersen commitment scheme for small vectors (round polynomials)
- `curve.rs`: `JoltCurve`/`JoltGroupElement` traits for elliptic curve abstractions
- `poly/commitment/dory/commitment_scheme.rs`: ZK evaluation commitments (`y_com`) — Dory proves evaluation correctness without revealing the evaluation value
- `sumcheck.rs` / `univariate_skip.rs`: `prove_zk`/`verify_zk` variants

**CRITICAL INVARIANT — Sumcheck claim/constraint synchronization:**

Every sumcheck instance implements `SumcheckInstanceParams` which defines both the claim computation AND the corresponding BlindFold constraint. These must stay in sync:

- `input_claim(accumulator)` computes the input claim value from polynomial openings
- `input_claim_constraint()` returns an `InputClaimConstraint` describing the same formula as a sum-of-products over `ValueSource::{Opening, Challenge, Constant}` terms
- `input_constraint_challenge_values(accumulator)` returns the public challenge values the constraint evaluates against
- `output_claim_constraint()` / `output_constraint_challenge_values()` — same pattern for output claims

**Any change to how a sumcheck's input or output claim is derived requires a matching update to its constraint.** If you modify `input_claim()` to include a new term, you must add a corresponding `ProductTerm` to `input_claim_constraint()` and supply any new challenge values. Failure to synchronize causes BlindFold R1CS unsatisfiability — the `muldiv` e2e test will catch this.

**Corollary — prover/verifier `input_claim()` consistency:** When a value is decomposed for BlindFold constraints (e.g., `init_eval` split into `init_eval_public` + advice terms), the verifier's `new_from_verifier` must reconstruct the full value for `input_claim()` in standard mode. If only the public portion is stored, the verifier computes a different `input_claim` than the prover, causing a Fiat-Shamir transcript mismatch. The `advice` e2e tests catch this (they exercise non-ZK mode with advice polynomials).

Concrete implementations: `OuterRemainingSumcheckParams` (spartan/outer.rs), `RamReadWriteCheckingParams` (ram/read_write_checking.rs), `InstructionRaSumcheckParams` (instruction_lookups/ra_virtual.rs), and all claim reduction params.

## Development Guidelines

### Performance

- Profile before optimizing
- Benchmark changes to `poly/` code — small regressions multiply across thousands of sumcheck rounds
- Use `#[inline]` judiciously in hot paths
- Pre-allocate vectors unsafely when size is known; avoid clones in hot paths
- Hot trace paths get one pass and one owner of trace-sized storage. Produce or share derived rows during the existing pass instead of walking or materializing the trace again.

### Prover Hot Paths

- Sumcheck inner loop dominates: polynomial bind, sumcheck_evals, eq_poly evals
- `CompactPolynomial` bind converts small scalars to field elements — keep scalars small
- `SharedRaPolynomials` avoids per-polynomial memory duplication for RA indices

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

- Match verification to the changed behavior and complete the applicable checks above and in the invoked workflow. After they pass, repeat or broaden checks only for further changes, failures, or unresolved concerns.
- Add tests for distinct failure signals, not to mirror the implementation or require new tests for every low-impact edit.
- Do not add old-vs-new equivalence tests that reimplement the pre-change logic as the oracle. Transition-validation belongs in the PR process (byte-parity CI vs a living reference, one-off scripts), not the permanent suite. Permanent tests must assert against independent ground truth: spec vectors, golden fixtures, live reference paths (e.g. `jolt-kernels`' reference tier, the legacy-prover byte-parity suites), or properties. If the old code is deleted, its reimplementation in a test is dead weight — delete the test rather than keep the old logic alive inside it. A `#[cfg(test)]` copy of superseded production code "kept as the oracle" is the same anti-pattern.

### Lint Policy

- Workspace enforces `allow_attributes = "deny"` — use `#[expect(...)]` instead of `#[allow(...)]`
- The jolt-verifier runtime closure (18 crates, listed in `specs/verifier-closure-lints.md`) carries stricter crate-root lints: panic-source denies (`indexing_slicing` in control-plane crates, `panic_in_result_fn`, `wildcard_enum_match_arm`, ...), `forbid(unsafe_code)` where a crate has no unsafe, and numeric-discipline denies in jolt-verifier itself — which additionally denies `unreachable`, the only abort macro that escapes both `panic` and `panic_in_result_fn`. New code in those crates must fix the lint or add `#[expect(clippy::..., reason = "...")]` at the narrowest scope with a real justification
- `.unwrap()` and `.expect()` are fine in tests. In non-test code, avoid them unless the alternative significantly hurts readability (e.g., infallible fixed-size array conversions). When used, annotate the function with `#[expect(clippy::unwrap_used)]` or `#[expect(clippy::expect_used)]`
- Use `#[expect(clippy::...)]` on test modules to blanket-suppress test-inappropriate lints rather than per-function annotations

### Comments

Match the codebase's low comment density. Worth writing: WHY comments, WARNING for non-obvious gotchas, SAFETY on unsafe blocks, algorithm explanations (link to paper if applicable), public API docs stating behavior or invariants.
Do not narrate code or test assertions. If a comment only restates an expression, make the code self-documenting instead.
