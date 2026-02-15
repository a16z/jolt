# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V (RV64IMAC) that efficiently proves and verifies program execution. It uses sumcheck-based protocols, multilinear polynomial commitments (Dory), and the Twist/Shout lookup argument.

## Essential Commands

### Linting and Formatting

```bash
cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings
cargo fmt -q
```

### Testing

```bash
# CRITICAL: Always use cargo nextest, never cargo test
cargo nextest run --cargo-quiet

# Run specific test in specific package
cargo nextest run -p [package_name] [test_name] --cargo-quiet

# CRITICAL: Primary correctness check — run muldiv e2e test
cargo nextest run -p jolt-core muldiv --cargo-quiet
```

### Building

```bash
# Prefer clippy over build for validation. Only build when preparing to execute a binary.
cargo build -p jolt-core --message-format=short -q
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
- `subprotocols/`: Sumcheck (batched, streaming, univariate skip), booleanity checks
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
