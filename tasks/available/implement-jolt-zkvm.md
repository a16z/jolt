# impl-jolt-zkvm: Clean-room implementation of jolt-zkvm

**Scope:** crates/jolt-zkvm/

**Depends:** impl-jolt-spartan, impl-jolt-instructions, impl-jolt-dory

**Verifier:** ./verifiers/scoped.sh /workdir jolt-zkvm

**Context:**

Implement the `jolt-zkvm` crate — the top-level zkVM that orchestrates all sub-crates into a complete proving system. This replaces the old `jolt-core` and is the last implementation crate to be built.

**This is a clean-room rewrite.** Study `jolt-core/src/zkvm/` for algorithmic reference — especially `prover.rs`, `verifier.rs`, and the sub-protocol modules (RAM, registers, bytecode, claim reductions, instruction lookups). Write from scratch.

**Dependencies:** All `jolt-*` crates.

### Reference material

The old code lives in `jolt-core/src/zkvm/` (17 subdirectories):
- `prover.rs` (2,467 LOC) — `JoltCpuProver`, main proving algorithm
- `verifier.rs` (862 LOC) — `JoltVerifier`, verification
- `config.rs` (11.6 KB) — `ProverConfig`
- `proof_serialization.rs` (20.9 KB) — `JoltProof`
- `witness.rs` (10.3 KB) — `TraceWitness`
- `ram/` — RAM read/write consistency checking
- `registers/` — Register read/write consistency
- `bytecode/` — Program code verification
- `claim_reductions/` — Batching claims from different sub-protocols
- `instruction_lookups/` — Connecting execution trace to lookup tables

Also read the Jolt Book architecture overview: https://jolt.a16zcrypto.com/ — this covers the full proving flow.

### Public API contract

```rust
pub struct JoltProver<PCS: CommitmentScheme> { ... }

impl<PCS: HomomorphicCommitmentScheme> JoltProver<PCS> {
    pub fn new(config: ProverConfig, pcs_setup: PCS::ProverSetup) -> Self;

    pub fn prove<T: Transcript>(
        &self,
        trace: ExecutionTrace,
        transcript: &mut T,
    ) -> Result<JoltProof<PCS>, JoltError>;
}

pub struct JoltVerifier<PCS: CommitmentScheme> { ... }

impl<PCS: HomomorphicCommitmentScheme> JoltVerifier<PCS> {
    pub fn new(pcs_setup: PCS::VerifierSetup) -> Self;

    pub fn verify<T: Transcript>(
        &self,
        proof: &JoltProof<PCS>,
        transcript: &mut T,
    ) -> Result<(), JoltError>;
}

pub struct ProverConfig {
    pub memory_layout: MemoryLayout,
    pub first_round_strategy: FirstRoundStrategy,
    // ... other config
}

pub struct JoltProof<PCS: CommitmentScheme> { ... }

pub struct ExecutionTrace { ... }

#[derive(Debug, thiserror::Error)]
pub enum JoltError {
    #[error("spartan error: {0}")]
    Spartan(#[from] SpartanError),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumcheckError),
    #[error("opening error: {0}")]
    Opening(#[from] OpeningsError),
    #[error("instruction error: {0}")]
    Instruction(String),
    #[error("memory error: {0}")]
    Memory(String),
}
```

### Internal modules — each implements `SumcheckInstanceProver`

The key architectural insight: each zkVM sub-protocol (RAM, registers, bytecode, etc.) implements the `SumcheckInstanceProver<F>` trait from `jolt-sumcheck`. The zkVM prover orchestrates these as a batched sumcheck.

**RAM checking:**
- Implements read/write memory consistency via multiset hash arguments
- Proves that every memory read returns the value from the most recent write to that address
- Implements `SumcheckInstanceProver` for the RAM consistency polynomial

**Register checking:**
- Same pattern as RAM but for the 32 RISC-V registers
- Simpler because register addresses are small (5 bits)

**Bytecode checking:**
- Verifies the program counter trace matches the committed bytecode
- Read-only memory (never written during execution)

**Claim reductions:**
- Batches claims from different sub-protocols into a unified sumcheck
- Handles advice columns, Hamming weight, increments, instruction lookups

**Instruction lookups:**
- Connects the execution trace to `jolt-instructions` lookup tables
- Verifies that each instruction's lookup decomposition is consistent

### Implementation notes

The prover flow:
1. Receive `ExecutionTrace` from the tracer
2. Commit to witness polynomials (registers, RAM, bytecode, instruction lookups) via PCS
3. Run batched sumcheck over all sub-protocols
4. Accumulate opening claims in `ProverOpeningAccumulator`
5. Reduce and produce opening proofs
6. Package everything into `JoltProof`

The verifier flow:
1. Receive `JoltProof`
2. Recompute commitments / check commitment consistency
3. Verify batched sumcheck
4. Accumulate opening claims in `VerifierOpeningAccumulator`
5. Batch-verify opening proofs

### File structure

```
jolt-zkvm/src/
├── lib.rs
├── prover.rs           # JoltProver
├── verifier.rs         # JoltVerifier
├── config.rs           # ProverConfig
├── proof.rs            # JoltProof, serialization
├── trace.rs            # ExecutionTrace
├── error.rs            # JoltError
├── ram/                # RAM consistency checking
│   ├── mod.rs
│   ├── read_write.rs
│   └── output.rs
├── registers/          # Register checking
│   ├── mod.rs
│   └── read_write.rs
├── bytecode/           # Bytecode verification
│   ├── mod.rs
│   └── read_checking.rs
├── claim_reductions/   # Claim batching
│   ├── mod.rs
│   ├── advice.rs
│   ├── hamming.rs
│   ├── increments.rs
│   └── lookups.rs
└── instruction_lookups/ # Lookup table integration
    ├── mod.rs
    └── checking.rs
```

**Acceptance:**

- `JoltProver` produces a valid `JoltProof` for a simple execution trace
- `JoltVerifier` accepts valid proofs and rejects invalid ones
- RAM, register, bytecode, and instruction lookup sub-protocols each implement `SumcheckInstanceProver`
- Batched sumcheck orchestration works across all sub-protocols
- Opening accumulator correctly reduces to PCS proofs
- `JoltProof` is `Serialize + Deserialize`
- Error types compose correctly from all sub-crate errors
- `parallel` feature enables rayon in prover hot paths
- No file exceeds 500 lines
- Rustdoc on all public items
- `cargo clippy` clean
- Integration tests with small execution traces
