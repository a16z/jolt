# Task 12: BlindFold Protocol Orchestrator

**Status:** Pending
**Phase:** BlindFold Layer 4
**Dependencies:** Task 09 (relaxed Spartan), Task 10 (verifier R1CS), Task 11 (Nova folding)
**Blocks:** jolt-zkvm ZK mode integration

## Objective

Implement the full BlindFold protocol orchestrator that ties all layers together: committed sumcheck (Layer 1) → verifier R1CS (Layer 2) → Nova folding (Layer 3) → relaxed Spartan proof.

## Context

After all sumcheck stages run in committed mode, the `BlindFoldAccumulator` contains the private round data (coefficients, blinding factors, challenges). The protocol orchestrator:

1. Builds the verifier R1CS from the stage configurations
2. Assigns the witness from accumulated data
3. Creates a real relaxed R1CS instance
4. Samples a random satisfying instance (one-time pad)
5. Computes and commits to the cross-term
6. Folds the instances
7. Proves the folded instance via relaxed Spartan
8. Includes opening proofs for the witness and error commitments

## Deliverables

### `BlindFoldProver`

```rust
pub struct BlindFoldProver;

impl BlindFoldProver {
    /// Produces a BlindFold proof from accumulated committed sumcheck data.
    ///
    /// Generic over:
    /// - `VC: JoltCommitment` — round polynomial commitment scheme
    /// - `PCS: CommitmentScheme` — opening proof scheme for relaxed Spartan
    ///
    /// `VC::Commitment: JoltGroup` required for Nova folding.
    pub fn prove<F, VC, PCS, T>(
        accumulator: BlindFoldAccumulator<F, VC>,
        stage_configs: &[StageConfig<F>],
        vc_setup: &VC::Setup,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
        rng: &mut impl CryptoRngCore,
    ) -> Result<BlindFoldProof<F, VC, PCS>, BlindFoldError>
    where
        F: Field,
        VC: JoltCommitment,
        VC::Commitment: JoltGroup,
        PCS: CommitmentScheme<Field = F>,
        T: Transcript;
}
```

### `BlindFoldVerifier`

```rust
pub struct BlindFoldVerifier;

impl BlindFoldVerifier {
    /// Verifies a BlindFold proof.
    ///
    /// Reconstructs the verifier R1CS from stage configs and baked
    /// inputs (derived from the transcript), folds public instances,
    /// and verifies the relaxed Spartan proof.
    pub fn verify<F, VC, PCS, T>(
        proof: &BlindFoldProof<F, VC, PCS>,
        stage_configs: &[StageConfig<F>],
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), BlindFoldError>
    where
        F: Field,
        VC: JoltCommitment,
        VC::Commitment: JoltGroup,
        PCS: CommitmentScheme<Field = F>,
        T: Transcript;
}
```

### `BlindFoldProof`

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlindFoldProof<F: Field, VC: JoltCommitment, PCS: CommitmentScheme> {
    /// Commitment to the cross-term T.
    pub cross_term_commitment: VC::Commitment,
    /// Commitment to the random instance's witness.
    pub random_w_commitment: VC::Commitment,
    /// Commitment to the random instance's error vector.
    pub random_e_commitment: VC::Commitment,
    /// Relaxed Spartan proof over the folded instance.
    pub spartan_proof: SpartanProof<F, PCS>,
    // Opening proofs for W and E (scheme-dependent)
}
```

### `BlindFoldError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum BlindFoldError {
    #[error("stage config mismatch: expected {expected} stages, got {actual}")]
    StageCountMismatch { expected: usize, actual: usize },

    #[error("verifier R1CS construction failed")]
    R1csConstruction,

    #[error("witness assignment failed: {0}")]
    WitnessAssignment(String),

    #[error("Nova folding failed: {0}")]
    FoldingError(String),

    #[error("relaxed Spartan verification failed: {0}")]
    Spartan(#[from] SpartanError),

    #[error("opening proof failed")]
    OpeningProof,
}
```

### Protocol flow (prover)

```
accumulator.into_stages()
    │
    ▼
build_verifier_r1cs(stages, baked_inputs)
    │
    ▼
assign_witness(stages, round_data) → witness
    │
    ▼
commit(witness) → w_commitment, create real RelaxedInstance (u=1, E=0)
    │
    ▼
sample_random_instance(r1cs, vc_setup, rng) → (random_inst, random_wit)
    │
    ▼
compute_cross_term(r1cs, real_wit, random_wit, u1, u2) → T
    │
    ▼
commit(T) → t_commitment, append to transcript
    │
    ▼
transcript.challenge() → folding_challenge r
    │
    ▼
fold_instances(real_inst, random_inst, t_commitment, r) → folded_inst
fold_witnesses(real_wit, random_wit, T, r) → folded_wit
    │
    ▼
SpartanProver::prove_relaxed(folded_inst, folded_wit, pcs_setup, transcript)
    │
    ▼
BlindFoldProof { cross_term_commitment, spartan_proof, ... }
```

### Protocol flow (verifier)

```
proof.cross_term_commitment → append to transcript
    │
    ▼
transcript.challenge() → folding_challenge r (same as prover)
    │
    ▼
build_verifier_r1cs(stages, baked_inputs) (same R1CS as prover)
    │
    ▼
fold public instances: real_public_inst + random commitments from proof
    │
    ▼
SpartanVerifier::verify_relaxed(folded_inst, proof.spartan_proof, pcs_setup, transcript)
```

## Testing

### End-to-end test

1. Define a simple polynomial witness (e.g., degree-1 sumcheck with 4 evaluations)
2. Run committed sumcheck via `CommittedRoundHandler` + `SumcheckProver::prove_with_handler`
3. Push round data to `BlindFoldAccumulator`
4. Call `BlindFoldProver::prove`
5. Call `BlindFoldVerifier::verify` → success

### Negative tests

- Tamper with `cross_term_commitment` → verification fails
- Modify `stage_configs` between prover and verifier → verification fails
- Corrupt one round polynomial coefficient in the proof → verification fails

### Backend-agnostic test

- Run with `Pedersen<Bn254G1>` as `VC` and `MockCommitmentScheme` as `PCS`
- (Future: run with hash-based VC once available)

### Multi-stage test

- Two sumcheck stages with different num_rounds and degrees
- Both stages committed, accumulated, proved as one BlindFold proof

## Files

| File | Change |
|------|--------|
| `jolt-blindfold/src/protocol.rs` | New: `BlindFoldProver`, `BlindFoldVerifier` |
| `jolt-blindfold/src/error.rs` | Add error variants |
| `jolt-blindfold/src/lib.rs` | Re-exports for Layer 4 types |
| `jolt-blindfold/tests/e2e.rs` | End-to-end integration tests |
| `jolt-blindfold/Cargo.toml` | Add `jolt-spartan` dependency (for `SpartanProver/Verifier`) |

## Reference

- `jolt-core/src/subprotocols/blindfold/protocol.rs` — `BlindFoldProver`, `BlindFoldVerifier`
- `jolt-core/src/subprotocols/blindfold/spartan.rs` — relaxed Spartan over folded instance
- `jolt-core/src/subprotocols/blindfold/witness.rs` — witness assignment
- Spec §4.12 — jolt-blindfold protocol flow
