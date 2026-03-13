# Refactor Gap Tracking

Status as of 2026-03-09. Branch: `refactor/crates`.

## Legend

- ✗ **MISSING** — Not implemented, must be added
- ⚠ **PARTIAL** — Exists but incomplete or needs rework
- ○ **DEFERRED** — Intentionally postponed (GPU, arkworks migration, etc.)
- ✓ **DONE** — Complete, no issues

---

## Foundation Crates (All ✓)

| Crate | Status | Notes |
|-------|--------|-------|
| jolt-field | ✓ | BN254 field, challenges, accumulators, SignedBigInt |
| jolt-transcript | ✓ | Blake2b, Keccak |
| jolt-crypto | ✓ | JoltGroup, PairingGroup, Pedersen, BN254 types |
| jolt-poly | ✓ | Dense, Compact, Eq, UnivariatePoly, Identity |
| jolt-openings | ✓ | CommitmentScheme, ProverClaim/VerifierClaim, RlcReduction |
| jolt-dory | ✓ | Streaming, GLV, batch add, instance-local params |
| jolt-sumcheck | ✓ | Batched, streaming, SumcheckReduction, SplitEqEvaluator |
| jolt-spartan | ✓ | Standard + Uniform, uni-skip, prove_with_challenges |
| jolt-instructions | ✓ | ~105 opcodes (RV64IMAC + virtual extension set) |
| jolt-blindfold | ✓ | Committed handlers, Nova folding, verifier R1CS |
| jolt-compute | ✓ | ComputeBackend trait, CpuBackend |
| jolt-cpu-kernels | ✓ | ProductSum + Custom compilers |
| jolt-ir | ✓ | ExprBuilder, ClaimDefinition, 4 visitors (eval, R1CS, Lean4, circuit) |
| jolt-wrapper | ✓ | SymbolicField, GnarkAstEmitter, 121 tests |
| jolt-verifier | ✓ | Proof types, JoltVerifier orchestrator, VerifierStage trait |
| jolt-profiling | ✓ | Tracing, Perfetto, monitor, pprof, allocative |

---

## jolt-zkvm — Critical Gaps

### ✗ 1. Bytecode Checking (Missing Entirely)

**jolt-core source:** `zkvm/bytecode/mod.rs`, `zkvm/bytecode/read_raf_checking.rs`

**What's needed:**
- `BytecodePreprocessing` — preprocesses ELF bytecode into lookup-friendly form
- `BytecodePCMapper` — maps program counter to bytecode address
- Bytecode read-RAF checking stage — verifies PC updates are consistent with the program
- Bytecode claim definitions in `claims/`
- Bytecode witness types in `witnesses/`

**Why critical:** Without bytecode checking, the prover cannot verify that instruction fetches match the committed program.

---

### ✗ 2. Instruction Lookup Read-RAF Checking (Missing)

**jolt-core source:** `zkvm/instruction_lookups/read_raf_checking.rs`

**What's needed:**
- Instruction read-RAF stage implementation
- RAF evaluation claim definitions
- Integration with Stage 2 (RA virtual) output

**Why critical:** Instruction lookup soundness requires verifying that read-address function evaluations are consistent with committed RA polynomials.

---

### ✗ 3. Advice Claim Reduction (Missing)

**jolt-core source:** `zkvm/claim_reductions/advice.rs`

**What's needed:**
- Two-phase advice reduction (modeled as two composed `SumcheckReduction`s per spec)
- `advice_claim_reduction()` in `claims/reductions.rs` (only `advice_claim_reduction_address()` exists)
- Integration into Stage 3 (ClaimReductionStage)

**Why critical:** Untrusted advice polynomials require a dedicated reduction to ensure they are consistent with the claimed values.

---

### ✗ 4. Fiat-Shamir Preamble (Missing)

**jolt-core source:** `zkvm/mod.rs:169` — `fiat_shamir_preamble()`

**What's needed:**
- Function that absorbs Dory tier-2 commitments into the transcript before any sumcheck begins
- Called once at the start of the proving pipeline (before Stage 1)

**Why critical:** Binds polynomial commitments to the Fiat-Shamir transcript. Without it, commitments are not tied to the proof — soundness issue.

---

### ✗ 5. Config Management (Missing)

**jolt-core source:** `zkvm/config.rs`

**What's needed:**
- `OneHotParams` — controls one-hot encoding chunk sizes and indexing
- `OneHotConfig` — serialized minimal config for proof
- `ReadWriteConfig` — controls read-write checking phase structure
- `get_instruction_sumcheck_phases()` — derives phase config from params
- Stages need these configs to construct correct witnesses and claims

**Why critical:** Stages produce incorrect witnesses without proper configuration. Config determines polynomial dimensions, chunk decomposition, and phase structure.

---

### ⚠ 6. Witness Generation from Traces (Partial)

**jolt-core source:** `zkvm/witness.rs` — `CommittedPolynomial` enum, `generate_witness()`

**What exists:** `WitnessStore<F>` (flat BTreeMap keyed by PolynomialTag)

**What's missing:**
- `CommittedPolynomial` enum (RdInc, RamInc, InstructionRa, BytecodeRa, RamRa, TrustedAdvice, UntrustedAdvice)
- `VirtualPolynomial` enum (PC, RamVal, RamReadValue, register values, instruction flags, lookup operands)
- `generate_witness()` entry point: trace → all polynomial evaluation tables
- Bytecode-related witness generation (BytecodeRa, PC mapping)
- Streaming commitment integration (tier-1 chunking during witness gen)

**Why critical:** Bridges the gap between tracer output (execution trace) and prover input (polynomial tables). Without this, only synthetic test traces work.

---

### ⚠ 7. ZK Feature Gates in Proof Types (Partial)

**jolt-core source:** `zkvm/proof_serialization.rs` — `#[cfg(feature = "zk")]` gates

**What exists:** `proof.rs` (43 lines) with `JoltProof<F, PCS>` containing spartan_proof + sumcheck_proofs + opening_proofs

**What's missing:**
- `#[cfg(not(feature = "zk"))]` field: `opening_claims: Claims<F>` (all polynomial evaluations)
- `#[cfg(feature = "zk")]` field: `blindfold_proof: BlindFoldProof` (no cleartext claims)
- Decision needed: compile-time cfg gates (matching jolt-core) vs runtime enum dispatch

**Why critical:** BlindFold integration cannot proceed without proof type support for ZK mode.

---

### ⚠ 8. Verifier Stage Implementations (Skeleton Only)

**What exists:** `jolt-verifier` has `VerifierStage` trait and `JoltVerifier` orchestrator

**What's missing:**
- Concrete `VerifierStage` implementations for stages 2–7
- Each mirrors the corresponding ProverStage: rebuild claims from IR, verify sumcheck, extract opening claims
- Challenge reconstruction logic for each stage

**Why critical:** Proofs can be generated but not verified until verifier stages are implemented.

---

## jolt-zkvm — Optimization Gaps

### ○ 9. SharedRaPolynomials (Removed — Optimization Deferred)

**jolt-core source:** Shared eq table across N RA polynomials for memory efficiency

**Current state:** RA polynomials constructed independently in `witnesses/ra_poly.rs`

**Risk:** Memory bloat at production scale. Each RA polynomial allocates its own eq table.

**Action:** Re-add sharing mechanism in `ra_virtual.rs` when benchmarking at scale.

---

### ○ 10. Streaming Commitment Wiring (Trait Exists, Not Used)

**What exists:** `StreamingCommitment` trait in jolt-openings, implemented by Dory

**What's missing:** jolt-zkvm stages commit witnesses whole (via WitnessStore), not streamed

**Risk:** OOM on large traces. Streaming is essential for production-size polynomials.

**Action:** Wire `StreamingCommitment::begin()/feed()/finish()` into witness generation pipeline.

---

## Deferred Work (By Design)

### ○ 11. GPU Backends

- `jolt-metal` — Apple Metal compute shaders
- `jolt-cuda` — NVIDIA CUDA kernels
- `jolt-webgpu` — Browser-compatible WebGPU

All depend on `jolt-compute` ComputeBackend trait (done). Kernel descriptors from `jolt-ir` (done). Implementation deferred.

### ○ 12. Arkworks Fork Migration (6 Phases)

See `crates/migration.md`. Phases:
1. SignedBigInt → jolt-field (**HIGH** risk — hot path)
2. Field arithmetic → jolt-field
3. Typed MSMs → jolt-dory/msm (**HIGH** risk — perf critical)
4. Optimizations (GLV, batch add) → jolt-dory
5. Switch to upstream arkworks v0.5.0
6. Validate Montgomery mul performance

### ○ 13. IR GPU Kernel Codegen (5th Visitor)

`jolt-ir` has 4 of 5 planned visitors. GPU kernel generation (Expr → CUDA/Metal shader source) deferred until GPU backends exist.

### ○ 14. Host/ELF Module Extraction

jolt-core `host/` module (ELF compilation, program analysis, bytecode preprocessing) not yet extracted into a standalone crate. Needed for real RISC-V trace → proof pipeline.

### ○ 15. jolt-sdk Integration

`jolt-sdk` (`#[jolt::provable]` macro) still calls into jolt-core. Needs updating to call new modular crates once witness generation and host module are extracted.

---

## Suggested Implementation Order

| Priority | Item | Dependency |
|----------|------|------------|
| 1 | Bytecode stage + instruction RAF (#1, #2) | Completes proving pipeline |
| 2 | Advice claim reduction (#3) | Completes claim reductions |
| 3 | Fiat-Shamir preamble (#4) | Soundness requirement |
| 4 | Config types (#5) | Stages need correct dimensions |
| 5 | ZK feature gates in proof.rs (#7) | Enables BlindFold integration |
| 6 | Witness generation (#6) | Bridges tracer → prover |
| 7 | Verifier stages (#8) | Enables proof verification |
| 8 | SharedRaPolynomials (#9) | Memory optimization |
| 9 | Streaming commitment (#10) | Large-trace support |
