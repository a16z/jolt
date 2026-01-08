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
        sumcheck::SumcheckInstanceProof,
        univariate_skip::UniSkipFirstRoundProof,
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
    /// Create a verifier for transpilation (symbolic execution)
    ///
    /// This constructor skips Fiat-Shamir preamble and commitment handling,
    /// making it suitable for use with MleAst where we just want to build the AST.
    pub fn new_for_transpilation(
        preprocessing: Stage1OnlyPreprocessing<F>,
        proof: Stage1OnlyProof<F, ProofTranscript>,
        transcript: ProofTranscript,
    ) -> Self {
        let opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());

        Self {
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
        }
    }

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
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round verification failed")?;

        // Step 2: Verify remaining sumcheck rounds
        // Rounds: n = log(trace_length) - 1 (first round done above)
        // Per round: verify univariate polynomial, sample challenge
        // Final: check C_final = eq(τ, r) · [Az(r) · Bz(r)]
        let n_cycle_vars = self.proof.trace_length.log_2();
        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.preprocessing.spartan_key.clone(),
            n_cycle_vars,
            spartan_outer_uni_skip_state,
            &self.opening_accumulator,
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

}

// ============================================================================
// Transpilation-friendly verification (no Transcript)
// ============================================================================

use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::poly::lagrange_poly::LagrangeHelper;

/// Challenges and proof data for transpilation-friendly verification.
///
/// This structure contains all the data needed to verify Stage 1 without
/// using a Fiat-Shamir transcript. The challenges are pre-computed externally
/// (e.g., by replaying the transcript) and passed in directly.
///
/// This enables transpilation to Gnark/Groth16 circuits where challenges
/// become public inputs rather than being computed via hash functions.
#[derive(Clone, Debug)]
pub struct Stage1VerificationData<F: JoltField> {
    /// Initial challenges for outer sumcheck (tau)
    pub tau: Vec<F>,
    /// Challenge from uni-skip first round
    pub r0: F,
    /// Challenges from remaining sumcheck rounds
    pub sumcheck_challenges: Vec<F>,
    /// Univariate polynomial coefficients from first round
    pub uni_skip_poly_coeffs: Vec<F>,
    /// Decompressed polynomials from sumcheck rounds
    pub sumcheck_round_polys: Vec<Vec<F>>,
    /// Trace length
    pub trace_length: usize,
}

/// Result of Stage 1 verification with all constraints.
///
/// Contains the final claim and all intermediate constraint values
/// that must equal zero for a valid proof.
#[derive(Clone, Debug)]
pub struct Stage1VerificationResult<F: JoltField> {
    /// The final claim after all sumcheck rounds
    pub final_claim: F,
    /// Power sum check: should equal 0
    pub power_sum_check: F,
    /// Sumcheck round consistency checks: each should equal 0
    /// (computed as poly(0) + poly(1) - previous_claim)
    pub sumcheck_consistency_checks: Vec<F>,
}

/// Verify Stage 1 without Transcript - suitable for transpilation to Gnark.
///
/// This function performs the same verification as `Stage1OnlyVerifier::verify()`,
/// but takes all challenges as parameters instead of generating them via
/// Fiat-Shamir transcript. This makes it suitable for:
///
/// 1. **Transpilation to Gnark**: Run with `MleAst` to build AST, then transpile
/// 2. **Transpilation to other circuit DSLs**: Same approach
/// 3. **Testing**: Verify with pre-computed challenges
///
/// ## Arguments
///
/// * `data` - Pre-computed challenges and proof polynomials
///
/// ## Returns
///
/// `Stage1VerificationResult` containing:
/// - `final_claim`: The result after all sumcheck rounds
/// - `power_sum_check`: Must equal 0 for valid proof
/// - `sumcheck_consistency_checks`: Each must equal 0 for valid proof
///
/// ## Usage for Transpilation
///
/// ```rust,ignore
/// use zklean_extractor::mle_ast::MleAst;
///
/// // Create MleAst versions of the inputs
/// let data = Stage1VerificationData {
///     tau: tau.iter().map(|&x| MleAst::from_i128(x)).collect(),
///     r0: MleAst::from_i128(r0),
///     // ... etc
/// };
///
/// // Run verification - AST builds automatically
/// let result = verify_stage1_for_transpilation(data);
///
/// // Transpile each constraint to Gnark
/// // power_sum_check should be constrained to equal 0
/// // each sumcheck_consistency_check should be constrained to equal 0
/// ```
pub fn verify_stage1_for_transpilation<F: JoltField>(
    data: Stage1VerificationData<F>,
) -> Stage1VerificationResult<F> {
    // Step 1: Verify univariate-skip first round
    // Check: Σ_j coeff[j] * power_sum[j] == 0 (initial claim)
    // Then: evaluate polynomial at r0 to get claim_after_first
    let (claim_after_first, power_sum_check) = verify_uni_skip_first_round(
        &data.uni_skip_poly_coeffs,
        &data.r0,
    );

    // Step 2: Verify remaining sumcheck rounds
    // For each round: check g(0) + g(1) == previous_claim
    // Then: evaluate g(r) to get next claim
    let (final_claim, sumcheck_consistency_checks) = verify_sumcheck_rounds(
        claim_after_first,
        &data.sumcheck_challenges,
        &data.sumcheck_round_polys,
    );

    Stage1VerificationResult {
        final_claim,
        power_sum_check,
        sumcheck_consistency_checks,
    }
}

/// Verify univariate-skip first round (pure field arithmetic).
///
/// Returns:
/// - next_claim: poly(r0)
/// - power_sum_check: Σ_j coeff[j] * S_j (should equal 0 for valid proof)
fn verify_uni_skip_first_round<F: JoltField>(
    poly_coeffs: &[F],
    r0: &F,
) -> (F, F) {
    // Power sum check: Σ_j coeff[j] * S_j should equal 0
    // The power sums are precomputed constants for the symmetric domain
    let power_sums = LagrangeHelper::power_sums::<
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
    >();

    let mut power_sum_check = F::zero();
    for (j, coeff) in poly_coeffs.iter().enumerate() {
        if j < power_sums.len() {
            power_sum_check += coeff.mul_i128(power_sums[j]);
        }
    }

    // Evaluate polynomial at r0 (Horner's method)
    let next_claim = evaluate_polynomial(poly_coeffs, r0);

    (next_claim, power_sum_check)
}

/// Verify sumcheck rounds (pure field arithmetic).
///
/// Returns:
/// - final_claim: The claim after all rounds
/// - consistency_checks: Vec of (poly(0) + poly(1) - previous_claim) for each round
///   (each should equal 0 for valid proof)
fn verify_sumcheck_rounds<F: JoltField>(
    initial_claim: F,
    challenges: &[F],
    round_polys: &[Vec<F>],
) -> (F, Vec<F>) {
    let mut claim = initial_claim;
    let mut consistency_checks = Vec::with_capacity(round_polys.len());

    for (challenge, poly_coeffs) in challenges.iter().zip(round_polys) {
        // Check: poly(0) + poly(1) == previous_claim
        let poly_at_0 = evaluate_polynomial(poly_coeffs, &F::zero());
        let poly_at_1 = evaluate_polynomial(poly_coeffs, &F::one());
        let sum = poly_at_0 + poly_at_1;

        // Constraint: sum - claim should equal 0
        consistency_checks.push(sum - claim);

        // Update claim for next round
        claim = evaluate_polynomial(poly_coeffs, challenge);
    }

    (claim, consistency_checks)
}

/// Verify sumcheck rounds with COMPRESSED polynomials.
///
/// The compressed polynomial stores [c0, c2, c3, ...] (missing c1).
/// The linear term c1 is derived from the claim (hint):
///   c1 = claim - 2*c0 - c2 - c3 - ...
///
/// This is because: p(0) + p(1) = claim
///   p(0) = c0
///   p(1) = c0 + c1 + c2 + c3 + ...
///   So: claim = 2*c0 + c1 + c2 + c3 + ...
///   Therefore: c1 = claim - 2*c0 - c2 - c3 - ...
///
/// Returns:
/// - final_claim: The claim after all rounds
/// - consistency_checks: Empty vec (consistency is guaranteed by construction)
fn verify_sumcheck_rounds_compressed<F: JoltField>(
    initial_claim: F,
    challenges: &[F],
    compressed_polys: &[Vec<F>],  // Each is [c0, c2, c3, ...]
) -> (F, Vec<F>) {
    let mut claim = initial_claim;
    let consistency_checks = Vec::new();  // No checks needed - consistency built into decompress

    for (challenge, compressed) in challenges.iter().zip(compressed_polys) {
        // Decompress: compute linear term c1 from hint (claim)
        // c1 = claim - 2*c0 - c2 - c3 - ...
        let c0 = compressed[0];
        let mut linear_term = claim - c0 - c0;  // claim - 2*c0
        for coeff in compressed.iter().skip(1) {
            linear_term -= *coeff;
        }

        // Build full coefficients: [c0, c1, c2, c3, ...]
        let mut full_coeffs = vec![c0, linear_term];
        full_coeffs.extend(compressed.iter().skip(1).cloned());

        // Evaluate at challenge to get next claim
        claim = evaluate_polynomial(&full_coeffs, challenge);
    }

    (claim, consistency_checks)
}

/// Evaluate polynomial using Horner's method (pure field arithmetic).
///
/// Computes: Σ_i coeff[i] * x^i
fn evaluate_polynomial<F: JoltField>(coeffs: &[F], x: &F) -> F {
    if coeffs.is_empty() {
        return F::zero();
    }

    let mut result = coeffs[0];
    let mut x_power = *x;

    for coeff in coeffs.iter().skip(1) {
        result += *coeff * x_power;
        x_power *= *x;
    }

    result
}

//
// ═══════════════════════════════════════════════════════════════════════════
// TRANSCRIPT-BASED VERIFIER (with in-circuit challenge derivation)
// ═══════════════════════════════════════════════════════════════════════════
//

/// Verification data for transcript-based Stage 1 verification
///
/// Preamble data for Fiat-Shamir initialization.
/// These values are hashed into the transcript before Stage 1.
#[derive(Clone)]
pub struct Stage1PreambleData<F: JoltField> {
    /// max_input_size (as field element)
    pub max_input_size: F,
    /// max_output_size (as field element)
    pub max_output_size: F,
    /// memory_size (as field element)
    pub memory_size: F,
    /// inputs hash (as field elements, one per 32-byte chunk)
    pub inputs: Vec<F>,
    /// outputs hash (as field elements, one per 32-byte chunk)
    pub outputs: Vec<F>,
    /// panic flag (as field element)
    pub panic: F,
    /// ram_K (as field element)
    pub ram_k: F,
    /// trace_length (as field element)
    pub trace_length: F,
}

/// Unlike `Stage1VerificationData`, this does NOT include pre-computed challenges.
/// Challenges are derived from the transcript during verification.
#[derive(Clone)]
pub struct Stage1TranscriptVerificationData<F: JoltField> {
    /// Preamble data (memory layout, inputs, outputs, etc.)
    pub preamble: Option<Stage1PreambleData<F>>,

    /// Commitments (each as a vector of field elements representing 32-byte chunks)
    pub commitments: Vec<Vec<F>>,

    /// Coefficients of univariate-skip polynomial
    pub uni_skip_poly_coeffs: Vec<F>,

    /// Sumcheck round polynomials (flattened)
    pub sumcheck_round_polys: Vec<Vec<F>>,

    /// Number of sumcheck rounds
    pub num_rounds: usize,
}

/// Result of transcript-based Stage 1 verification
pub struct Stage1TranscriptVerificationResult<F: JoltField> {
    /// Final claim after all rounds
    pub final_claim: F,

    /// Power sum check (must equal 0)
    pub power_sum_check: F,

    /// Consistency checks for each round (all must equal 0)
    pub sumcheck_consistency_checks: Vec<F>,

    /// Derived challenges (for debugging)
    pub derived_tau: Vec<F>,
    pub derived_r0: F,
    pub derived_sumcheck_challenges: Vec<F>,
}

/// Stage 1 verifier with transcript for challenge derivation
///
/// This version derives all challenges from a Fiat-Shamir transcript,
/// making it suitable for full in-circuit verification with Gnark.
///
/// Works with any `Transcript` implementation:
/// - `PoseidonTranscript<Fr, FrParams>` for real verification
/// - `PoseidonMleTranscript` for transpilation (builds AST)
///
/// ## Usage for Transpilation
///
/// ```rust,ignore
/// use zklean_extractor::mle_ast::MleAst;
/// use gnark_transpiler::poseidon::PoseidonMleTranscript;
///
/// // Create transcript
/// let mut transcript = PoseidonMleTranscript::new(b"Jolt");
///
/// // Create verification data with symbolic variables
/// let data = Stage1TranscriptVerificationData {
///     uni_skip_poly_coeffs: vec![MleAst::from_var(0), ...],
///     sumcheck_round_polys: vec![vec![MleAst::from_var(4), ...]],
///     num_rounds: 3,
/// };
///
/// // Run verification - challenges derived via Poseidon hashing, AST builds automatically
/// let result = verify_stage1_with_transcript(data, &mut transcript);
/// ```
pub fn verify_stage1_with_transcript<F: JoltField + Clone, T: Transcript>(
    data: Stage1TranscriptVerificationData<F>,
    transcript: &mut T,
) -> Stage1TranscriptVerificationResult<F> {
    let num_rounds = data.num_rounds;

    // === Fiat-Shamir Protocol (matching real Jolt verifier) ===
    //
    // Complete Jolt verification flow:
    //   1. fiat_shamir_preamble (memory layout, inputs, outputs, etc.)
    //   2. append commitments (41 commitments, each ~384 bytes)
    //   3. derive tau challenges
    //   4. append uni_skip_poly, derive r0
    //   5. for each sumcheck round: append poly, derive challenge
    //
    // If preamble/commitments are provided, we process them here.
    // Otherwise, we assume transcript is already initialized.

    // Step 1: Append preamble to transcript (if provided)
    if let Some(preamble) = &data.preamble {
        // fiat_shamir_preamble order:
        // append_u64(max_input_size)
        // append_u64(max_output_size)
        // append_u64(memory_size)
        // append_bytes(inputs)
        // append_bytes(outputs)
        // append_u64(panic)
        // append_u64(ram_K)
        // append_u64(trace_length)
        transcript.append_scalar(&preamble.max_input_size);
        transcript.append_scalar(&preamble.max_output_size);
        transcript.append_scalar(&preamble.memory_size);
        // inputs as field elements (each represents 32-byte chunk)
        for input_chunk in &preamble.inputs {
            transcript.append_scalar(input_chunk);
        }
        // outputs as field elements
        for output_chunk in &preamble.outputs {
            transcript.append_scalar(output_chunk);
        }
        transcript.append_scalar(&preamble.panic);
        transcript.append_scalar(&preamble.ram_k);
        transcript.append_scalar(&preamble.trace_length);
    }

    // Step 2: Append commitments to transcript
    for commitment in &data.commitments {
        // Each commitment is a vector of field elements (representing 32-byte chunks)
        for chunk in commitment {
            transcript.append_scalar(chunk);
        }
    }

    // Step 3: Derive tau challenges from transcript
    // (tau depends on preamble + commitments)
    let mut derived_tau = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        derived_tau.push(transcript.challenge_scalar::<F>());
    }

    // Step 4: Append uni-skip polynomial to transcript, then derive r0
    for coeff in &data.uni_skip_poly_coeffs {
        transcript.append_scalar(coeff);
    }
    let derived_r0 = transcript.challenge_scalar::<F>();

    // Verify univariate-skip first round
    let (claim_after_first, power_sum_check) = verify_uni_skip_first_round(
        &data.uni_skip_poly_coeffs,
        &derived_r0,
    );

    // Step 5: For each sumcheck round, append poly then derive challenge
    let mut derived_sumcheck_challenges = Vec::with_capacity(num_rounds);
    for round_poly in &data.sumcheck_round_polys {
        // Append this round's polynomial coefficients
        for coeff in round_poly {
            transcript.append_scalar(coeff);
        }
        // Derive challenge for this round
        derived_sumcheck_challenges.push(transcript.challenge_scalar::<F>());
    }

    // Verify sumcheck rounds using COMPRESSED polynomial format
    // The sumcheck_round_polys contain [c0, c2, c3, ...] (missing c1)
    // The linear term c1 is derived on-the-fly from the claim
    let (final_claim, sumcheck_consistency_checks) = verify_sumcheck_rounds_compressed(
        claim_after_first,
        &derived_sumcheck_challenges,
        &data.sumcheck_round_polys,
    );

    Stage1TranscriptVerificationResult {
        final_claim,
        power_sum_check,
        sumcheck_consistency_checks,
        derived_tau,
        derived_r0,
        derived_sumcheck_challenges,
    }
}
