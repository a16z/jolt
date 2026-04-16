# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task Loop Protocol

TASKS.md is the program counter. Read it at the start of every session and after
every context compression. Follow this loop:

```
1. Read TASKS.md — find the first unchecked violation
2. Read the relevant source code (file + line range from the violation)
3. Understand what the handler currently does and WHY it violates the philosophy
4. Design the fix:
   - What primitive ops does the compiler need to emit?
   - What backend methods or existing ops handle the lowered form?
   - Does this require new Op variants? New ComputeBackend methods? Neither?
5. Implement the NEW path alongside the old one (dual-path):
   a. Add new Op variants / compiler emission / runtime handlers
   b. In the runtime handler, run BOTH old and new paths
   c. Assert the new path produces identical results to the old path
   d. Add a jolt-equivalence test if the assertion is non-trivial
6. Run the test gate from TASKS.md (includes transcript_divergence)
7. Pass → the new path is proven correct. Now:
   a. Remove the old path (delete old handler code, old Op variants)
   b. Wire the runtime to use only the new path
   c. Run the test gate again to confirm transcript parity
8. Pass → mark the violation DONE in TASKS.md with a one-line summary, commit
9. Fail → diagnose, fix, go to 6
10. Go to 1
```

**Dual-path validation**: jolt-equivalence is the correctness sandbox. When
replacing a complex handler, ALWAYS keep the old code running alongside the new
code and assert equality before deleting. This catches subtle bugs (endianness,
off-by-one in bit extraction, wrong binding order) that transcript_divergence
would catch too late. Add targeted tests in `crates/jolt-equivalence/tests/`
freely — they're cheap and survive refactors.

**When stuck on design**: The abstraction is wrong. Don't compromise by leaving
complex handlers. Step back and ask: "What would make this handler ≤ 30 LOC
while keeping the runtime protocol-unaware?" Write your thinking to the Notes
section of TASKS.md, then try the new approach.

**Philosophy checkpoint**: After every change, verify the handler you touched is
≤ 30 LOC and doesn't interpret any protocol-specific data structures. If it does,
the compiler hasn't lowered far enough — go back to step 4.

**North star**: `crates/jolt-zkvm/ARCHITECTURE.md`

## Perf Loop Protocol

Parallel to the Task Loop Protocol above. `PERF_TASKS.md` is the program
counter for performance work. Read it at the start of every perf session
and after every context compression. The loop drives modular-stack
prove time toward jolt-core parity WITHOUT collapsing the ML-compiler
abstraction — handlers stay ≤ 30 LOC and protocol-unaware.

Per iteration:

```
1. Read PERF_TASKS.md — current phase, current log_t, first unchecked P-item
2. MEASURE: cargo run --release -p jolt-bench -- --program muldiv \
     --iters 1 --warmup 1 --log-t <current> --json perf/last-iter.json
3. STOP CHECK: Phase 3 green (modular ≤ core at log_t ∈ {18, 20}) for
   3 consecutive iters → exit. Nothing else exits.
4. PICK HYPOTHESIS: first unchecked P-item. Empty queue or stall ≥ 5 →
   run with --profile, analyze Perfetto trace, append new P-items.
5. IMPLEMENT: respect ≤30 LOC handler rule + declared abstraction risk.
6. CORRECTNESS GATE (hard — fail reverts + continues):
   cargo nextest run -p jolt-equivalence transcript_divergence
   cargo nextest run -p jolt-equivalence zkvm_proof_accepted
   cargo nextest run -p jolt-equivalence
   cargo clippy ... -- -D warnings  (see PERF_TASKS.md for full set)
7. PERF GATE: re-measure, compare vs perf/baseline-modular-best.json.
   Accept: ≥5% faster → update ratchet. Reject: ≥5% slower → revert.
   Inconclusive band ±5% → one rerun; still in band → revert as flat.
8. COMMIT ALWAYS — one commit per iter:
   - Improvement: `perf(<scope>): P<n> <name> (-X% prove_ms on muldiv @ log_T=<n>)`
   - Flat/reverted: `journal: P<n> reverted (<reason>)` — only bookkeeping
9. GRADUATE if phase graduation condition met (see PERF_TASKS.md table).
10. Schedule next tick via ScheduleWakeup. GOTO 1.
```

**Never exit except via step 3.** Reverts, flat results, empty hypothesis
queue, stall mode — all continue. The loop is perpetual until Phase 3
parity is achieved.

**Commit discipline**: Every iteration produces exactly one commit. Reverts
still commit the bookkeeping update so dead ends aren't rediscovered on
resume. `git log --grep='^perf'` is the changelog of real wins.

**State files** live in `perf/`:
- `baseline-core.json` — frozen jolt-core reference
- `baseline-modular-best.json` — monotone ratchet
- `last-iter.json` — most recent bench output
- `history.jsonl` — append-only per-iteration log

## Project Overview

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V (RV64IMAC) that efficiently proves and verifies program execution. It uses sumcheck-based protocols, multilinear polynomial commitments (Dory), and the Twist/Shout lookup argument.

## Essential Commands

### Linting and Formatting

```bash
# CRITICAL: Must pass in BOTH standard and ZK modes
cargo clippy -p jolt-core --features host --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings
cargo fmt -q
```

### Testing

```bash
# CRITICAL: Always use cargo nextest, never cargo test
cargo nextest run --cargo-quiet

# Run specific test in specific package
cargo nextest run -p [package_name] [test_name] --cargo-quiet

# CRITICAL: Primary correctness check — run muldiv e2e test in BOTH modes
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk

```

### Building

```bash
# Prefer clippy over build for validation. Only build when preparing to execute a binary.
cargo build -p jolt-core --message-format=short -q

# After pulling changes, reinstall the jolt CLI or guest builds may fail.
cargo install --path . --locked
```

### Profiling

```bash
# Execution trace (viewable in Perfetto)
cargo run --release -p jolt-core profile --name sha3 --format chrome
# --name options: sha2, sha3, sha2-chain, fibonacci, btreemap

# With CPU/memory monitoring (adds counter tracks to Perfetto trace)
cargo run --release --features monitor -p jolt-core profile --name sha3 --format chrome

# Memory profiling (outputs SVG flamegraphs)
RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3 --format chrome
```

## Architecture

### Crate Structure

Arkworks dependencies use a fork: `a16z/arkworks-algebra` branch `dev/twist-shout` (patched via `[patch.crates-io]` in root `Cargo.toml`).

**jolt-core** — Core proving system

- `host/`: Guest ELF compilation and program analysis (feature-gated behind `host`)
- `zkvm/`: Jolt PIOP — prover, verifier, R1CS/Spartan, memory checking, instruction lookups
- `poly/`: Polynomial types, commitment schemes (Dory, HyperKZG), opening proofs
- `field/`: `JoltField` trait and BN254 scalar field implementation
- `subprotocols/`: Sumcheck (batched, streaming, univariate skip), booleanity checks, BlindFold ZK protocol
- `msm/`: Multi-scalar multiplication
- `transcripts/`: Fiat-Shamir transcripts (Blake2b, Keccak)

**tracer** — RISC-V emulator producing execution traces (`Cycle` per instruction)

**jolt-sdk** — `#[jolt::provable]` macro generating prove/verify/analyze/preprocess functions

**jolt-inlines** — Optimized cryptographic primitives (sha2, blake3, bigint, secp256k1, etc.) replacing guest-side computation with efficient constraint-native implementations

**common** — Shared constants (`XLEN`, `REGISTER_COUNT`, thresholds) and `JoltDevice`/`MemoryLayout` types

Feature flag hierarchy: `host` ⊃ `prover` ⊃ `minimal`. Most code is unconditional; `host/` is the main gated module.

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
- Verifier detects mode from proof at runtime: `proof.stage1_sumcheck_proof.is_zk()` — stored as `VerifierOpeningAccumulator::zk_mode`

**CRITICAL — Verifier `new_from_verifier` must support both modes:**

In ZK mode, `input_claim()` is never called so verifier params can use partial values (e.g., `init_eval = init_eval_public`). In standard mode, `input_claim()` IS called and the values must match the prover exactly. Any verifier param that decomposes a value for BlindFold constraints must reconstruct the full value for standard mode. Use `ram::reconstruct_full_eval()` to add advice contributions back.

**Opening accumulator transcript changes (vs main):**

On this branch, `ProverOpeningAccumulator::append_*` and `VerifierOpeningAccumulator::append_*` do NOT append claims to the Fiat-Shamir transcript (the `transcript` parameter was removed). Both sides are consistent. On main, these methods DO append `opening_claim` scalars.

### BlindFold Zero-Knowledge Protocol (subprotocols/blindfold/)

BlindFold makes all sumcheck proofs zero-knowledge without SNARK composition. Instead of revealing sumcheck round polynomial coefficients, the prover sends Pedersen commitments. Sumcheck verifier checks are encoded into a small verifier R1CS, proved via Nova folding + Spartan.

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

- PERFORMANCE IS CRITICAL AND TOP PRIORITY
- Profile before optimizing
- Benchmark changes to `poly/` code — small regressions multiply across thousands of sumcheck rounds
- Use `#[inline]` judiciously in hot paths
- Pre-allocate vectors unsafely when size is known; avoid clones in hot paths

### Prover Hot Paths

- Sumcheck inner loop dominates: polynomial bind, sumcheck_evals, eq_poly evals
- `CompactPolynomial` bind converts small scalars to field elements — keep scalars small
- `SharedRaPolynomials` avoids per-polynomial memory duplication for RA indices

### Code Style

- `cargo fmt` + `cargo clippy` with zero warnings
- Codebase uses `non_snake_case` convention for math variables: `log_T`, `ram_K`, `log_K`, etc.

### Comment Policy

**Delete these comment types:**
- Section separators (`// ==========`, `// ----------`)
- Doc comments that restate the item name (`/// Sumcheck prover for X` on `XProver`)
- Obvious comments (`/// Returns the count` on `get_count()`)
- Commented-out code
- TODOs without issue links

**Keep these comment types:**
- WHY something is done (when not obvious)
- WARNING comments for non-obvious gotchas
- SAFETY comments for unsafe blocks
- Complex algorithm explanations (link to paper if applicable)
- Public API docs that explain behavior, constraints, or invariants

### Testing

- Always use `cargo nextest` (never `cargo test`)
- Run `muldiv` e2e test as primary correctness check
