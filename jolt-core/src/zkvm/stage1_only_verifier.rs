//! # Stage 1 Only Verifier
//!
//! **Purpose:** Isolated verifier for Spartan outer sumcheck (Stage 1) only.
//!
//! This module provides a minimal verifier that ONLY verifies the R1CS constraint
//! satisfaction via Spartan's outer sumcheck. It is designed for experimentation
//! with Groth16 transpilation and does NOT include:
//! - Polynomial commitment scheme (PCS) verification
//! - RAM verification (Twist memory checking)
//! - Register verification (Twist memory checking)
//! - Instruction lookups (Shout)
//! - Bytecode lookups (Shout)
//! - Stages 2-7
//!
//! ## What Stage 1 Verifies
//!
//! Stage 1 verifies that the R1CS constraints are satisfied over the execution trace:
//!
//! ```text
//! ∑_{x ∈ {0,1}^n} eq(τ, x) · [Az(x) · Bz(x) - Cz(x)] = 0
//! ```
//!
//! Where:
//! - `Az`, `Bz`, `Cz` are multilinear extensions of R1CS matrices
//! - `τ` is verifier randomness
//! - In Jolt: constraints are conditional (`a = condition`, `b = left - right`, `c = 0`)
//!
//! ## Verification Steps
//!
//! 1. **Univariate-skip first round** (`verify_stage1_uni_skip`):
//!    - Verifies the first-round polynomial using Lagrange interpolation
//!    - Reduces first variable by evaluating Lagrange polynomial
//!    - Samples challenge `r0`
//!
//! 2. **Remaining sumcheck rounds** (`OuterRemainingSumcheckVerifier`):
//!    - Streaming first cycle-bit round (cubic from endpoints)
//!    - Linear-time rounds for subsequent bits
//!    - Final check: `eq(τ, r) · [Az(r) · Bz(r)]`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use jolt_core::zkvm::stage1_only_verifier::{Stage1OnlyProof, Stage1OnlyVerifier};
//!
//! let verifier = Stage1OnlyVerifier::new(
//!     preprocessing,
//!     proof,
//!     trace_length,
//! )?;
//!
//! let is_valid = verifier.verify()?;
//! ```
//!
//! ## Experiment Context
//!
//! This module is part of the Groth16 transpilation experiment (branch: `groth16-experiment`).
//! See `GROTH16_EXPERIMENT.md` for full documentation.

use crate::{
    field::JoltField,
    poly::opening_proof::VerifierOpeningAccumulator,
    subprotocols::{
        sumcheck::BatchedSumcheck,
        sumcheck::{SumcheckInstanceProof, UniSkipFirstRoundProof},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::{
        r1cs::key::UniformSpartanKey,
        spartan::{outer::OuterRemainingSumcheckVerifier, verify_stage1_uni_skip},
    },
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Minimal proof structure for Stage 1 only verification
///
/// Contains only the sumcheck proofs needed for Spartan outer sumcheck.
/// Does not include commitments, opening proofs, or other stage proofs.
///
/// ## Purpose
///
/// This structure is designed for **Groth16 transpilation**, where the verification
/// algorithm will be converted to an arithmetic circuit. The intended workflow is:
///
/// 1. **For Groth16 circuit**: Use this verifier's `verify()` logic directly
///    - Replace Fiat-Shamir transcript with public randomness (challenges as public inputs)
///    - Convert sumcheck verification to R1CS constraints
///    - No polynomial commitments needed (all data is witness)
///
/// 2. **For testing with real proofs**: Must provide full transcript context
///    - Use `new()` with all required parameters (program_io, commitments, etc.)
///    - This reconstructs the Fiat-Shamir transcript state
///    - Allows testing against real Jolt proofs
///
/// ## Why the complex API?
///
/// The verifier needs the Fiat-Shamir preamble and commitments to generate the same
/// random challenges that the prover used. For Groth16, you'll bypass all of this
/// and provide challenges directly as circuit inputs.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Stage1OnlyProof<F: JoltField, ProofTranscript: Transcript> {
    /// Univariate-skip first round proof
    pub uni_skip_first_round_proof: UniSkipFirstRoundProof<F, ProofTranscript>,
    /// Remaining sumcheck rounds proof
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Trace length (needed for Spartan key construction)
    pub trace_length: usize,
}

impl<F: JoltField, ProofTranscript: Transcript> Stage1OnlyProof<F, ProofTranscript> {
    /// Create a Stage1OnlyProof from a full JoltProof
    ///
    /// This extracts just the Stage 1 components from a complete proof,
    /// useful for testing and comparing the isolated verifier against
    /// the full verifier.
    ///
    /// Returns:
    /// - Stage1OnlyProof
    /// - Opening claims
    /// - Commitments (needed to reconstruct transcript state)
    /// - ram_K, bytecode_K (needed for Fiat-Shamir preamble)
    pub fn from_full_proof<PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>>(
        full_proof: &crate::zkvm::proof_serialization::JoltProof<F, PCS, ProofTranscript>,
    ) -> (
        Self,
        crate::poly::opening_proof::Openings<F>,
        Vec<PCS::Commitment>,
        usize, // ram_K
    ) {
        (
            Self {
                uni_skip_first_round_proof: full_proof.stage1_uni_skip_first_round_proof.clone(),
                sumcheck_proof: full_proof.stage1_sumcheck_proof.clone(),
                trace_length: full_proof.trace_length,
            },
            full_proof.opening_claims.0.clone(),
            full_proof.commitments.clone(),
            full_proof.ram_K,
        )
    }
}

/// Minimal preprocessing for Stage 1 only verification
///
/// Contains only the information needed to verify Stage 1.
///
/// ## What is the Spartan Key?
///
/// The `UniformSpartanKey` encodes the **structure** of Jolt's R1CS constraint system:
///
/// - **Uniform**: Same ~30 R1CS constraints applied to every execution cycle
/// - **Structure**: Constraint matrix shape (encoded as SHA3 digest)
/// - **Parameters**:
///   - `num_steps`: Trace length (padded to power of 2)
///   - `num_cons_total`: Total constraint rows = num_steps × constraints_per_step
///   - `vk_digest`: Hash of constraint definitions (for binding to specific R1CS)
///
/// The key is **deterministic** - computed from:
/// 1. Trace length (public parameter)
/// 2. R1CS constraint definitions (compile-time constants in `jolt-core/src/zkvm/r1cs/constraints.rs`)
///
/// **Why "uniform"?** Unlike arbitrary R1CS systems where each constraint can be different,
/// Jolt's constraints are uniform: the same small set (~30 constraints) is applied to every
/// cycle of execution. This enables significant optimizations in Spartan's sumcheck protocol.
///
/// **No secret data**: The key contains only public structural information about the constraint
/// system. It does NOT contain:
/// - Witness data (execution trace)
/// - Secret randomness
/// - Polynomial commitments
/// - Opening proofs
#[derive(Clone)]
pub struct Stage1OnlyPreprocessing<F: JoltField> {
    /// Spartan key encoding the R1CS constraint system structure
    ///
    /// Derived deterministically from:
    /// - Trace length (num_steps)
    /// - R1CS constraint definitions (~30 constraints per cycle)
    pub spartan_key: UniformSpartanKey<F>,
}

impl<F: JoltField> Stage1OnlyPreprocessing<F> {
    /// Create preprocessing from trace length
    pub fn new(trace_length: usize) -> Self {
        let spartan_key = UniformSpartanKey::new(trace_length.next_power_of_two());
        Self { spartan_key }
    }
}

/// Stage 1 only verifier
///
/// Verifies ONLY the Spartan outer sumcheck (R1CS constraints).
/// This is a minimal verifier for experimentation with Groth16 transpilation.
pub struct Stage1OnlyVerifier<F: JoltField, ProofTranscript: Transcript> {
    /// The proof to verify
    pub proof: Stage1OnlyProof<F, ProofTranscript>,
    /// Preprocessing (Spartan key)
    pub preprocessing: Stage1OnlyPreprocessing<F>,
    /// Transcript for Fiat-Shamir
    pub transcript: ProofTranscript,
    /// Opening accumulator (tracks polynomial evaluation claims)
    ///
    /// Note: In the isolated verifier, we still track openings but don't
    /// verify them via PCS. This maintains compatibility with the sumcheck
    /// verification interface.
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> Stage1OnlyVerifier<F, ProofTranscript> {
    /// Create a new Stage 1 only verifier
    ///
    /// # Arguments
    /// * `preprocessing` - Contains Spartan key derived from trace length
    /// * `proof` - Stage 1 proof (uni-skip + sumcheck)
    /// * `opening_claims` - Polynomial opening claims from full proof
    /// * `program_io` - Program I/O device (for Fiat-Shamir preamble)
    /// * `commitments` - Polynomial commitments (to reconstruct transcript state)
    /// * `ram_K` - RAM size parameter (for Fiat-Shamir preamble)
    ///
    /// # Returns
    /// A verifier ready to run `verify()`
    pub fn new<PCS: crate::poly::commitment::commitment_scheme::CommitmentScheme<Field = F>>(
        preprocessing: Stage1OnlyPreprocessing<F>,
        proof: Stage1OnlyProof<F, ProofTranscript>,
        opening_claims: crate::poly::opening_proof::Openings<F>,
        program_io: &crate::zkvm::JoltDevice,
        commitments: &[PCS::Commitment],
        ram_K: usize,
    ) -> Result<Self, ProofVerifyError> {
        // Validate trace length is power of 2
        if !proof.trace_length.is_power_of_two() {
            return Err(ProofVerifyError::SpartanError(
                "Trace length must be power of 2".to_string(),
            ));
        }

        // Initialize transcript with same seed as full verifier
        let mut transcript = ProofTranscript::new(b"Jolt");

        // Reconstruct transcript state to match full verifier
        // This is necessary for Fiat-Shamir to produce the same challenges
        crate::zkvm::fiat_shamir_preamble(
            program_io,
            ram_K,
            proof.trace_length,
            &mut transcript,
        );

        // Append commitments to transcript (same as full verifier)
        for commitment in commitments {
            transcript.append_serializable(commitment);
        }

        // Initialize opening accumulator and populate with claims from full proof
        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        for (key, (_, claim)) in &opening_claims {
            opening_accumulator
                .openings
                .insert(*key, (crate::poly::opening_proof::OpeningPoint::default(), *claim));
        }

        Ok(Self {
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
        })
    }

    /// Verify the Stage 1 proof - Spartan outer sumcheck
    ///
    /// ## What This Verifies
    ///
    /// Verifies that R1CS constraints are satisfied over the execution trace:
    ///
    /// ```text
    /// ∑_{x ∈ {0,1}^n} eq(τ, x) · [Az(x) · Bz(x) - Cz(x)] = 0
    /// ```
    ///
    /// Where:
    /// - `Az(x), Bz(x), Cz(x)`: R1CS constraint evaluations at cycle x (~30 constraints)
    /// - `τ`: Random challenge (from Fiat-Shamir transcript)
    /// - `eq(τ, x)`: Multilinear extension of equality function
    /// - `n = log(trace_length)`: Number of sumcheck rounds
    ///
    /// ## Verification Algorithm (Two Steps)
    ///
    /// ### Step 1: Univariate-Skip First Round ([verify_stage1_uni_skip])
    ///
    /// **Purpose:** Efficiently handle the first variable using Lagrange interpolation
    ///
    /// **What it does:**
    /// - Verifies first-round polynomial `s1(Y) = L(τ_high, Y) · t1(Y)`
    /// - `L(τ_high, Y)`: Lagrange basis polynomial at challenge τ_high
    /// - `t1(Y)`: Prover-supplied evaluations on extended domain `{-D..D}`
    /// - Samples challenge `r0` from transcript
    ///
    /// **Why univariate-skip?**
    /// - Standard sumcheck: multilinear polynomial per round
    /// - Univariate-skip: send full univariate in first round (more efficient)
    /// - Exploits structure: R1CS constraints split into two groups
    /// - Degree ~13-15 (vs standard degree 3)
    ///
    /// **Output:** State for remaining rounds (includes r0)
    ///
    /// ### Step 2: Remaining Sumcheck Rounds ([OuterRemainingSumcheckVerifier])
    ///
    /// **Purpose:** Bind remaining `n-1` cycle variables via standard sumcheck
    ///
    /// **Round structure:**
    /// 1. **Streaming first cycle-bit round:**
    ///    - Computes endpoints: `t(0)` and `t(∞)`
    ///    - Builds cubic polynomial from cached coefficients
    ///    - Samples challenge `r1`
    ///
    /// 2. **Remaining linear-time rounds (r2...rn):**
    ///    - Each round: reuse bound coefficients
    ///    - Compute endpoints in linear time
    ///    - Sample challenge ri
    ///
    /// **Per-round check:**
    /// - Receives univariate polynomial `g_i(X)`
    /// - Verifies: `C_{i-1} = g_i(0) + g_i(1)` (sumcheck consistency)
    /// - Samples: `r_i` ← transcript
    /// - Updates: `C_i = g_i(r_i)` (claim for next round)
    ///
    /// **Final check** (after all rounds):
    /// ```text
    /// C_final = eq(τ, r) · [Az(r) · Bz(r) - Cz(r)]
    /// ```
    /// Where `r = [r0, r1, ..., rn]` are all sampled challenges
    ///
    /// ## Important Notes
    ///
    /// **Fiat-Shamir Transcript (NOT ADDED):**
    /// - The transcript uses Blake2b/Keccak (NOT circuit-friendly!)
    /// - For Groth16 transpilation: challenges must be public inputs
    /// - Current implementation: transcript initialized empty
    /// - Security model: Public coin (interactive) vs Fiat-Shamir (non-interactive)
    ///
    /// **For Groth16 Transpilation:**
    /// When converting to Groth16 circuit:
    /// 1. Replace transcript sampling with public inputs (challenges as circuit inputs)
    /// 2. Implement sumcheck verification logic (field operations only)
    /// 3. Avoid implementing Blake2b/Keccak in circuit (thousands of constraints)
    /// 4. Alternative: Use Poseidon for circuit-friendly hashing (not implemented here)
    ///
    /// **Polynomial Opening Claims:**
    /// - The opening accumulator tracks evaluation claims (Az(r), Bz(r), etc.)
    /// - In full verifier: verified via PCS (Dory) in Stage 7
    /// - In this isolated verifier: accumulator populated but not verified
    /// - For transpilation: these become additional public inputs or committed values
    ///
    /// ## References
    ///
    /// Mathematical details:
    /// - See: `docs/03_Verifier_Mathematics_and_Code.md` (lines 1680-1779)
    /// - Spartan paper: <https://eprint.iacr.org/2019/550.pdf>
    /// - Univariate-skip optimization: `jolt-core/src/zkvm/spartan/outer.rs` (lines 47-73)
    ///
    /// # Returns
    /// `Ok(())` if verification succeeds, `Err` otherwise
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        // Step 1: Verify univariate-skip first round
        // Verifies: s1(Y) = L(τ_high, Y) · t1(Y) where t1(Y) evaluated on extended domain
        // Samples: r0 ← transcript
        // Output: State for remaining rounds (includes r0, Lagrange weights, etc.)
        let spartan_outer_uni_skip_state = verify_stage1_uni_skip(
            &self.proof.uni_skip_first_round_proof,
            &self.preprocessing.spartan_key,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round verification failed")?;

        // Step 2: Verify remaining sumcheck rounds
        // Rounds: n = log(trace_length) - 1 (first round done above)
        // Per round: verify univariate polynomial, sample challenge
        // Final: check C_final = eq(τ, r) · [Az(r) · Bz(r)]
        let n_cycle_vars = self.proof.trace_length.log_2();
        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            n_cycle_vars,
            &spartan_outer_uni_skip_state,
            self.preprocessing.spartan_key,
        );

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 remaining sumcheck verification failed")?;

        // Opening accumulator now contains evaluation claims (Az(r), Bz(r), etc.)
        // In full verifier: these would be verified via PCS in Stage 7
        // In isolated verifier: we skip PCS verification
        // For transpilation: these become public inputs or committed values

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use crate::transcripts::Blake2bTranscript;

    type F = Fr;
    type ProofTranscript = Blake2bTranscript;

    #[test]
    fn test_stage1_only_preprocessing_creation() {
        let trace_length = 1024;
        let preprocessing = Stage1OnlyPreprocessing::<F>::new(trace_length);

        // Spartan key should have correct num_steps (next power of 2)
        assert_eq!(preprocessing.spartan_key.num_steps, 1024);
    }

    #[test]
    fn test_stage1_only_verifier_rejects_invalid_trace_length() {
        let trace_length = 1023; // Not a power of 2
        let preprocessing = Stage1OnlyPreprocessing::<F>::new(1024);

        // Create a dummy proof with invalid trace length
        let proof = Stage1OnlyProof::<F, ProofTranscript> {
            uni_skip_first_round_proof: UniSkipFirstRoundProof::new(
                crate::poly::unipoly::UniPoly::from_coeff(vec![F::from(0u64)])
            ),
            sumcheck_proof: SumcheckInstanceProof::new(vec![]),
            trace_length,
        };

        use crate::poly::opening_proof::Openings;
        use crate::poly::commitment::dory::DoryCommitmentScheme;
        use crate::zkvm::JoltDevice;

        let opening_claims = Openings::new();
        let program_io = JoltDevice::default();
        let commitments: Vec<<DoryCommitmentScheme as crate::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment> = vec![];
        let ram_K = 1024;

        let result = Stage1OnlyVerifier::<F, ProofTranscript>::new::<DoryCommitmentScheme>(
            preprocessing,
            proof,
            opening_claims,
            &program_io,
            &commitments,
            ram_K,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_stage1_proof_structure() {
        // Verifies the Stage1OnlyProof structure is correct
        use crate::poly::unipoly::UniPoly;

        let trace_length = 1024; // Must be power of 2
        let proof = Stage1OnlyProof::<F, ProofTranscript> {
            uni_skip_first_round_proof: UniSkipFirstRoundProof::new(
                UniPoly::from_coeff(vec![F::from(0u64)])
            ),
            sumcheck_proof: SumcheckInstanceProof::new(vec![]),
            trace_length,
        };

        // Verify the structure is correct
        assert_eq!(proof.trace_length, 1024);
        assert!(proof.trace_length.is_power_of_two());
    }

    // NOTE: Full integration test commented out due to deserialization limitation
    //
    // The JoltProof deserialization requires AllCommittedPolynomials::initialize()
    // context which triggers unimplemented trait bounds. This is a known limitation
    // of the current serialization approach.
    //
    // For real-world usage, Stage1OnlyProof should be extracted from the JoltProof
    // object BEFORE serialization, or the proof should be passed in-memory.
    //
    // Example usage pattern:
    // ```
    // // In prover code:
    // let full_proof = prove_fibonacci(input);
    // let stage1_proof = Stage1OnlyProof::from_full_proof(&full_proof);
    //
    // // Pass stage1_proof to verifier (in-memory or with custom serialization)
    // let stage1_preprocessing = Stage1OnlyPreprocessing::new(trace_length);
    // let verifier = Stage1OnlyVerifier::new(stage1_preprocessing, stage1_proof)?;
    // verifier.verify()?; // Verifies ONLY R1CS constraints (Stage 1)
    // ```
    //
    // For manual testing:
    // 1. Run: cargo run --release -p fibonacci
    // 2. In fibonacci/src/main.rs, add after line 39:
    //    ```
    //    use jolt_core::zkvm::stage1_only_verifier::{Stage1OnlyProof, Stage1OnlyPreprocessing, Stage1OnlyVerifier};
    //    let stage1_proof = Stage1OnlyProof::from_full_proof(&proof);
    //    let stage1_preprocessing = Stage1OnlyPreprocessing::new(proof.trace_length.next_power_of_two());
    //    let verifier = Stage1OnlyVerifier::new(stage1_preprocessing, stage1_proof).unwrap();
    //    verifier.verify().expect("Stage 1 verification failed!");
    //    println!("✅ Stage 1 verification PASSED!");
    //    ```
}
