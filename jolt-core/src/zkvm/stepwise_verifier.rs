//! # Stepwise Stage 1 Verifier
//!
//! **Purpose:** Complete Stage 1 verifier for transpilation to Gnark/Groth16.
//!
//! This module implements the FULL Stage 1 verification logic, matching the real
//! Jolt verifier exactly. Unlike `stage1_only_verifier.rs` which is simplified,
//! this version includes all components needed for a complete verification:
//!
//! 1. **Tau derivation** from transcript (after preamble + commitments)
//! 2. **Univariate-skip first round** with power sum check
//! 3. **Batched sumcheck** with proper input_claim handling
//! 4. **expected_output_claim** computation:
//!    - LagrangeKernel(tau_high, r0)
//!    - EqPolynomial::mle(tau_low, r_tail)
//!    - Inner sum product Az(r) * Bz(r)
//! 5. **Final claim check**: output_claim == expected_output_claim
//!
//! ## Architecture for Transpilation
//!
//! The verifier is split into discrete steps, each producing intermediate values
//! that can be constrained in a Gnark circuit:
//!
//! ```text
//! Step 1: Transcript Setup
//!   - Append preamble to transcript
//!   - Append commitments to transcript
//!   - Derive tau = [tau_0, ..., tau_{n-1}, tau_high]
//!
//! Step 2: Univariate-Skip First Round
//!   - Append uni_skip_poly coefficients to transcript
//!   - Derive r0 = transcript.challenge_scalar()
//!   - CHECK: power_sum == 0
//!   - Compute claim_after_first = poly(r0)
//!
//! Step 3: Batched Sumcheck Setup
//!   - Append input_claim to transcript
//!   - Derive batching_coeff (trivial for single instance)
//!   - initial_claim = claim_after_first * batching_coeff
//!
//! Step 4: Sumcheck Rounds (n rounds)
//!   For each round i:
//!   - Append compressed_poly[i] to transcript
//!   - Derive r_i = transcript.challenge_scalar()
//!   - Decompress: c1 = claim - 2*c0 - c2 - c3
//!   - Compute next_claim = poly(r_i)
//!   - claim = next_claim
//!
//! Step 5: Expected Output Claim
//!   - tau_high_bound_r0 = LagrangeKernel(tau_high, r0)
//!   - tau_bound_r_tail = EqPolynomial::mle(tau_low, r_tail_reversed)
//!   - inner_sum_prod = Az(r) * Bz(r)  via evaluate_inner_sum_product_at_point
//!   - expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
//!
//! Step 6: Final Check
//!   - CHECK: final_claim == expected_output_claim
//! ```

use ark_serialize::CanonicalSerialize;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::transcripts::Transcript;
use crate::zkvm::r1cs::constraints::{
    R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::inputs::NUM_R1CS_INPUTS;

// Re-export for external use
pub use crate::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
pub use crate::zkvm::witness::VirtualPolynomial;

/// Number of sumcheck rounds for a trace of length 2^11 = 2048
/// This is configurable based on trace_length.log_2() + 1
pub const DEFAULT_NUM_SUMCHECK_ROUNDS: usize = 12;

/// Preamble data for Fiat-Shamir initialization
///
/// This struct holds the public statement data that is appended to the transcript
/// at the beginning of verification. These are CONSTANTS for a given proof/program,
/// not variables. The transcript methods `append_u64` and `append_bytes` are used
/// to hash them correctly.
#[derive(Clone, Debug)]
pub struct PreambleData {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub memory_size: u64,
    /// Raw input bytes
    pub inputs: Vec<u8>,
    /// Raw output bytes
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub ram_k: usize,
    pub trace_length: usize,
}

/// Complete Stage 1 verification data
///
/// This struct holds all inputs needed for Stage 1 verification:
/// - Preamble: public statement (constants)
/// - Commitments: proof commitments (generic type C that implements CanonicalSerialize)
/// - Polynomials: sumcheck proof data
/// - R1CS evaluations: opening claims for R1CS constraint check
///
/// The type parameter `F` is used for field elements (Fr or MleAst).
/// The type parameter `C` is used for commitments:
/// - For real verification: `PCS::Commitment` (e.g., G1Affine)
/// - For symbolic transpilation: `MleCommitment` (wrapper for 12 MleAst chunks)
///
/// This generic design ensures the verification code is identical to the real verifier,
/// following the "NEVER TWO PATHS, ALWAYS GENERICS" principle.
#[derive(Clone, Debug)]
pub struct Stage1FullVerificationData<F: JoltField, C: CanonicalSerialize> {
    /// Preamble data (public constants - max_input_size, memory_size, etc.)
    pub preamble: PreambleData,

    /// Commitments to append to transcript.
    /// The real verifier calls `transcript.append_serializable(commitment)` for each.
    /// For symbolic execution, C = MleCommitment which stores 12 MleAst chunks.
    pub commitments: Vec<C>,

    /// Univariate-skip polynomial coefficients
    pub uni_skip_poly_coeffs: Vec<F>,

    /// Compressed sumcheck round polynomials [c0, c2, c3] for each round
    /// Number of rounds = 1 + num_cycle_vars (streaming round + cycle rounds)
    pub sumcheck_round_polys: Vec<Vec<F>>,

    /// R1CS input evaluations at the final point r
    /// These are the evaluations of all R1CS input polynomials at the sumcheck point.
    /// In the real verifier, these come from opening_accumulator.
    /// Length: NUM_R1CS_INPUTS (36 elements)
    /// Order: as defined in ALL_R1CS_INPUTS
    pub r1cs_input_evals: [F; NUM_R1CS_INPUTS],

    /// Number of cycle variables = log2(trace_length)
    /// Note: total sumcheck rounds = num_cycle_vars + 1 (streaming + cycle)
    /// Note: tau length = num_cycle_vars + 2 (num_rows_bits)
    pub num_cycle_vars: usize,
}

/// Result of Stage 1 verification with all intermediate values
#[derive(Clone, Debug)]
pub struct Stage1FullVerificationResult<F: JoltField> {
    // === Derived Challenges ===
    // NOTE: tau, r0, and sumcheck_challenges use F::Challenge to match the real verifier.
    // For Fr, F::Challenge = MontU128Challenge<Fr> (128-bit optimized).
    // For MleAst, F::Challenge = MleAst (same type).
    /// Tau challenges (length = num_rounds + 2)
    pub tau: Vec<F::Challenge>,
    /// Challenge from univariate-skip first round
    pub r0: F::Challenge,
    /// Batching coefficient (for single instance, derived from transcript)
    /// NOTE: This uses challenge_scalar (not _optimized) to match BatchedSumcheck::verify
    pub batching_coeff: F,
    /// Challenges from sumcheck rounds
    pub sumcheck_challenges: Vec<F::Challenge>,

    // === Intermediate Claims ===
    /// Claim after univariate-skip first round
    pub claim_after_uni_skip: F,
    /// Claims after each sumcheck round
    pub claims_after_round: Vec<F>,
    /// Final claim from sumcheck
    pub final_claim: F,

    // === Expected Output Claim Components ===
    /// L(tau_high, r0) - Lagrange kernel
    pub tau_high_bound_r0: F,
    /// eq(tau_low, r_tail_reversed) - Equality polynomial
    pub tau_bound_r_tail: F,
    /// Az(r) * Bz(r) - R1CS inner product (computed via evaluate_inner_sum_product_at_point)
    pub inner_sum_prod: F,
    /// Expected output claim
    pub expected_output_claim: F,

    // === Constraint Checks ===
    /// Power sum check: should equal 0
    pub power_sum_check: F,
    /// Final check: final_claim - expected_output_claim (should equal 0)
    pub final_check: F,
}

/// Power sums for the univariate-skip domain
/// These are precomputed constants: S_j = Σ_{i ∈ domain} i^j
fn compute_power_sums<const N: usize, const NUM_COEFFS: usize>() -> [i128; NUM_COEFFS] {
    use crate::poly::lagrange_poly::LagrangeHelper;
    LagrangeHelper::power_sums::<N, NUM_COEFFS>()
}

/// Evaluate polynomial at a point using Horner's method
/// poly(x) = c0 + c1*x + c2*x^2 + c3*x^3 + ...
pub fn evaluate_polynomial<F: JoltField>(coeffs: &[F], x: &F) -> F {
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

/// Evaluate polynomial at a challenge point.
/// Similar to evaluate_polynomial but takes F::Challenge for the point.
/// F::Challenge * F -> F via Into trait.
pub fn evaluate_polynomial_challenge<F: JoltField>(coeffs: &[F], x: &F::Challenge) -> F {
    if coeffs.is_empty() {
        return F::zero();
    }

    let mut result = coeffs[0];
    let mut x_power: F = (*x).into(); // Convert Challenge to F

    for coeff in coeffs.iter().skip(1) {
        result += *coeff * x_power;
        x_power = x_power * x; // F * Challenge -> F
    }

    result
}

/// Verify univariate-skip first round
///
/// Returns:
/// - next_claim: poly(r0)
/// - power_sum_check: Σ_j coeff[j] * S_j (should equal 0 for valid proof)
fn verify_uni_skip_first_round<F: JoltField, const N: usize, const NUM_COEFFS: usize>(
    poly_coeffs: &[F],
    r0: &F::Challenge,
) -> (F, F) {
    let power_sums = compute_power_sums::<N, NUM_COEFFS>();

    // Power sum check: Σ_j coeff[j] * S_j should equal 0 (input_claim)
    let mut power_sum_check = F::zero();
    for (j, coeff) in poly_coeffs.iter().enumerate() {
        if j < power_sums.len() {
            power_sum_check += coeff.mul_i128(power_sums[j]);
        }
    }

    // Evaluate polynomial at r0 (using F::Challenge)
    let next_claim = evaluate_polynomial_challenge(poly_coeffs, r0);

    (next_claim, power_sum_check)
}

/// Decompress and evaluate a compressed polynomial
///
/// Compressed format: [c0, c2, c3, ...]
/// Full format: [c0, c1, c2, c3, ...] where c1 = claim - 2*c0 - c2 - c3 - ...
pub fn decompress_and_evaluate<F: JoltField>(compressed: &[F], claim: &F, challenge: &F::Challenge) -> F {
    // Decompress: c1 = claim - 2*c0 - c2 - c3 - ...
    let c0 = compressed[0];
    let mut c1 = *claim - c0 - c0; // claim - 2*c0
    for coeff in compressed.iter().skip(1) {
        c1 -= *coeff;
    }

    // Build full coefficients: [c0, c1, c2, c3, ...]
    let mut full_coeffs = vec![c0, c1];
    full_coeffs.extend(compressed.iter().skip(1).cloned());

    // Evaluate at challenge (using F::Challenge)
    evaluate_polynomial_challenge(&full_coeffs, challenge)
}

/// Compute LagrangeKernel(tau_high, r0)
///
/// This is L(τ, r) = Σ_i L_i(τ) * L_i(r) where L_i are Lagrange basis polynomials
/// over the univariate-skip domain.
/// NOTE: Both tau_high and r0 are F::Challenge to match the real verifier.
pub fn lagrange_kernel<F: JoltField>(tau_high: &F::Challenge, r0: &F::Challenge) -> F {
    LagrangePolynomial::<F>::lagrange_kernel::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(tau_high, r0)
}

/// Compute EqPolynomial::mle(tau_low, r_tail_reversed)
///
/// eq(x, y) = Π_i (x_i * y_i + (1-x_i)(1-y_i))
/// NOTE: Uses F::Challenge for both inputs to match the real verifier.
pub fn eq_polynomial_mle<F: JoltField>(tau_low: &[F::Challenge], r_tail_reversed: &[F::Challenge]) -> F {
    EqPolynomial::<F>::mle(tau_low, r_tail_reversed)
}

/// Evaluate Az(r) * Bz(r) using the R1CS constraint structure
///
/// This is a direct port of `UniformSpartanKey::evaluate_inner_sum_product_at_point`
/// from `jolt-core/src/zkvm/r1cs/key.rs`.
///
/// ## Arguments
/// * `rx_constr` - Row binding randomness [r_stream, r0]
/// * `r1cs_input_evals` - Evaluations of all R1CS input polynomials at point r
///
/// ## Returns
/// Az(r) * Bz(r)
pub fn evaluate_inner_sum_product_at_point<F: JoltField>(
    rx_constr: &[F],
    r1cs_input_evals: [F; NUM_R1CS_INPUTS],
) -> F {
    // Row axis: r_constr = [r_stream, r0]
    debug_assert!(rx_constr.len() >= 2);
    let r_stream = rx_constr[0];
    let r0 = rx_constr[1];

    // Lagrange weights over the univariate-skip base domain at r0
    let w = LagrangePolynomial::<F>::evals::<F, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(&r0);

    // Build z(r_cycle) vector with a trailing 1 for the constant column
    let z_const_col = NUM_R1CS_INPUTS;
    let mut z = r1cs_input_evals.to_vec();
    z.push(F::one());

    // Helper function to compute dot product of LC with z vector
    fn lc_dot_product<F: JoltField>(
        lc: &crate::zkvm::r1cs::ops::LC,
        z: &[F],
        const_col: usize,
    ) -> F {
        lc.dot_product(z, const_col)
    }

    // Group 0: First group of R1CS constraints (boolean-guarded, ~64-bit Bz)
    let mut az_g0 = F::zero();
    let mut bz_g0 = F::zero();
    for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
        let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
        let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
        az_g0 += w[i] * lc_dot_product(lc_a, &z, z_const_col);
        bz_g0 += w[i] * lc_dot_product(lc_b, &z, z_const_col);
    }

    // Group 1: Second group of R1CS constraints (may have wider Bz)
    let mut az_g1 = F::zero();
    let mut bz_g1 = F::zero();
    let g2_len = core::cmp::min(
        R1CS_CONSTRAINTS_SECOND_GROUP.len(),
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    );
    for i in 0..g2_len {
        let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
        let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
        az_g1 += w[i] * lc_dot_product(lc_a, &z, z_const_col);
        bz_g1 += w[i] * lc_dot_product(lc_b, &z, z_const_col);
    }

    // Bind by r_stream to match the outer streaming combination used for final Az,Bz
    let az_final = az_g0 + r_stream * (az_g1 - az_g0);
    let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

    az_final * bz_final
}

/// Complete Stage 1 verification with transcript
///
/// This function performs the FULL Stage 1 verification, matching the real
/// Jolt verifier exactly. It derives all challenges from the transcript and
/// computes the expected_output_claim.
///
/// ## Arguments
/// * `data` - Complete verification data including preamble, commitments, polynomials
/// * `transcript` - Fiat-Shamir transcript
///
/// ## Returns
/// `Stage1FullVerificationResult` containing all intermediate values and checks
pub fn verify_stage1_full<F: JoltField + Clone, C: CanonicalSerialize, T: Transcript>(
    data: Stage1FullVerificationData<F, C>,
    transcript: &mut T,
) -> Stage1FullVerificationResult<F> {
    let num_cycle_vars = data.num_cycle_vars;

    // Derived values matching the real verifier:
    // - num_rows_bits = num_cycle_vars + 2 (tau length)
    // - remaining sumcheck rounds = 1 + num_cycle_vars (streaming + cycle)
    let tau_len = num_cycle_vars + 2;
    let num_sumcheck_rounds = 1 + num_cycle_vars;

    // =========================================================================
    // STEP 1: Transcript Setup - Append preamble and commitments
    // =========================================================================

    // Append preamble to transcript using the SAME methods as fiat_shamir_preamble
    // in zkvm/mod.rs. This ensures exact transcript matching.
    //
    // For symbolic transpilation (PoseidonMleTranscript), these methods are
    // implemented to produce the correct AST nodes (with MulTwoPow192 for append_u64).
    transcript.append_u64(data.preamble.max_input_size);
    transcript.append_u64(data.preamble.max_output_size);
    transcript.append_u64(data.preamble.memory_size);
    transcript.append_bytes(&data.preamble.inputs);
    transcript.append_bytes(&data.preamble.outputs);
    transcript.append_u64(data.preamble.panic as u64);
    transcript.append_u64(data.preamble.ram_k as u64);
    transcript.append_u64(data.preamble.trace_length as u64);

    // Append commitments to transcript - IDENTICAL to real verifier (verifier.rs:159-161)
    // The real verifier does: for commitment in &self.proof.commitments {
    //     self.transcript.append_serializable(commitment);
    // }
    // For symbolic execution with MleCommitment, append_serializable triggers
    // the thread-local mechanism that performs 12-hash chaining.
    for commitment in &data.commitments {
        transcript.append_serializable(commitment);
    }

    // Derive tau challenges
    // tau has length = num_rows_bits = num_cycle_vars + 2
    // MUST use challenge_vector_optimized to match real verifier (outer.rs:91)
    let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(tau_len);

    // =========================================================================
    // STEP 2: Univariate-Skip First Round
    // =========================================================================

    // Append uni_skip_poly coefficients to transcript
    // This MUST match UniPoly::append_to_transcript exactly!
    // See: jolt-core/src/poly/unipoly.rs lines 469-475
    transcript.append_message(b"UncompressedUniPoly_begin");
    for coeff in &data.uni_skip_poly_coeffs {
        transcript.append_scalar(coeff);
    }
    transcript.append_message(b"UncompressedUniPoly_end");

    // Derive r0
    // MUST use challenge_scalar_optimized to match real verifier (univariate_skip.rs:138)
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();

    // Verify univariate-skip first round
    // Using domain size 10 and 28 coefficients
    let (claim_after_uni_skip, power_sum_check) =
        verify_uni_skip_first_round::<F, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, 28>(
            &data.uni_skip_poly_coeffs,
            &r0,
        );

    // IMPORTANT: After computing the uni-skip claim, append it to transcript
    // This matches what happens in cache_openings (opening_proof.rs:912)
    // The claim gets appended TWICE:
    // 1. Here (from cache_openings in uni-skip verification)
    // 2. Below (from BatchedSumcheck::verify's input_claim append)
    transcript.append_scalar(&claim_after_uni_skip);

    // =========================================================================
    // STEP 3: Batched Sumcheck Setup
    // =========================================================================

    // In BatchedSumcheck::verify, we first append the input_claim (again!)
    // For a single sumcheck instance, input_claim = claim_after_uni_skip
    // This is the SECOND append of the same claim.
    transcript.append_scalar(&claim_after_uni_skip);

    // Derive batching coefficient (for single instance)
    let batching_coeff = transcript.challenge_scalar::<F>();

    // Initial claim = input_claim * batching_coeff
    let initial_claim = claim_after_uni_skip * batching_coeff;

    // =========================================================================
    // STEP 4: Sumcheck Rounds
    // =========================================================================

    let mut claim = initial_claim;
    let mut sumcheck_challenges: Vec<F::Challenge> = Vec::with_capacity(num_sumcheck_rounds);
    let mut claims_after_round = Vec::with_capacity(num_sumcheck_rounds);

    for round in 0..num_sumcheck_rounds {
        // Append compressed polynomial coefficients to transcript
        // This MUST match CompressedUniPoly::append_to_transcript exactly!
        // See: jolt-core/src/poly/unipoly.rs lines 479-485
        transcript.append_message(b"UniPoly_begin");
        for coeff in &data.sumcheck_round_polys[round] {
            transcript.append_scalar(coeff);
        }
        transcript.append_message(b"UniPoly_end");

        // Derive challenge for this round
        // MUST use challenge_scalar_optimized to match real verifier (sumcheck.rs:302)
        let challenge: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        sumcheck_challenges.push(challenge);

        // Decompress and evaluate at challenge (using F::Challenge)
        let next_claim = decompress_and_evaluate(&data.sumcheck_round_polys[round], &claim, &challenge);
        claims_after_round.push(next_claim);

        claim = next_claim;
    }

    let final_claim = claim;

    // =========================================================================
    // STEP 5: Expected Output Claim
    // =========================================================================
    //
    // This matches OuterRemainingSumcheckVerifier::expected_output_claim exactly.
    // See: jolt-core/src/zkvm/spartan/outer.rs lines 407-436
    //
    // Formula:
    //   expected = L(tau_high, r0) * eq(tau_low, r_tail_reversed) * Az(r)*Bz(r)
    //
    // Where:
    //   tau_high = tau[tau.len() - 1]
    //   tau_low = tau[..tau.len() - 1]
    //   r_tail_reversed = sumcheck_challenges.iter().rev()

    // tau_high is the last element of tau (F::Challenge type)
    let tau_high: F::Challenge = tau[tau.len() - 1];

    // tau_low is all elements except the last (F::Challenge type)
    let tau_low: Vec<F::Challenge> = tau[..tau.len() - 1].to_vec();

    // r_tail_reversed: reverse the sumcheck challenges (F::Challenge type)
    let r_tail_reversed: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();

    // Compute L(tau_high, r0) - Lagrange kernel
    let tau_high_bound_r0: F = lagrange_kernel::<F>(&tau_high, &r0);

    // Compute eq(tau_low, r_tail_reversed)
    // Note: tau_low has (num_cycle_vars + 1) elements
    //       r_tail_reversed has (num_cycle_vars + 1) elements
    // They should match! The real verifier uses the full tau_low.
    debug_assert_eq!(
        tau_low.len(),
        r_tail_reversed.len(),
        "tau_low and r_tail_reversed must have the same length"
    );
    let tau_bound_r_tail: F = eq_polynomial_mle::<F>(&tau_low, &r_tail_reversed);

    // Compute inner_sum_prod = Az(r) * Bz(r) using R1CS constraints
    // rx_constr = [r_stream, r0] where r_stream is the first sumcheck challenge
    // Note: These need to be converted to F for the R1CS evaluation
    debug_assert!(
        !sumcheck_challenges.is_empty(),
        "sumcheck_challenges must not be empty"
    );
    let r_stream: F = sumcheck_challenges[0].into();
    let r0_f: F = r0.into();
    let rx_constr = vec![r_stream, r0_f];
    let inner_sum_prod = evaluate_inner_sum_product_at_point(&rx_constr, data.r1cs_input_evals);

    // Expected output claim = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
    // The batching coefficient is applied by BatchedSumcheck::verify AFTER getting
    // the expected_output_claim from each instance. See sumcheck.rs:236.
    let expected_output_claim_unbatched = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod;
    let expected_output_claim = expected_output_claim_unbatched * batching_coeff;

    // =========================================================================
    // STEP 6: Final Check
    // =========================================================================

    let final_check = final_claim - expected_output_claim;

    Stage1FullVerificationResult {
        // Derived challenges
        tau,
        r0,
        batching_coeff,
        sumcheck_challenges,

        // Intermediate claims
        claim_after_uni_skip,
        claims_after_round,
        final_claim,

        // Expected output claim components
        tau_high_bound_r0,
        tau_bound_r_tail,
        inner_sum_prod,
        expected_output_claim,

        // Constraint checks
        power_sum_check,
        final_check,
    }
}

/// Simplified verification without expected_output_claim computation
///
/// This version matches `stage1_only_verifier.rs` behavior - it only verifies
/// the sumcheck protocol without the R1CS evaluation step. Useful for testing
/// transcript and sumcheck logic in isolation.
pub fn verify_stage1_sumcheck_only<F: JoltField + Clone, C: CanonicalSerialize, T: Transcript>(
    preamble: PreambleData,
    commitments: Vec<C>,
    uni_skip_poly_coeffs: Vec<F>,
    sumcheck_round_polys: Vec<Vec<F>>,
    num_rounds: usize,
    transcript: &mut T,
) -> Stage1FullVerificationResult<F> {
    // Step 1: Append preamble - MUST match fiat_shamir_preamble exactly!
    transcript.append_u64(preamble.max_input_size);
    transcript.append_u64(preamble.max_output_size);
    transcript.append_u64(preamble.memory_size);
    transcript.append_bytes(&preamble.inputs);
    transcript.append_bytes(&preamble.outputs);
    transcript.append_u64(preamble.panic as u64);
    transcript.append_u64(preamble.ram_k as u64);
    transcript.append_u64(preamble.trace_length as u64);

    // Step 2: Append commitments - IDENTICAL to real verifier
    for commitment in &commitments {
        transcript.append_serializable(commitment);
    }

    // Step 3: Derive tau (using _optimized to match real verifier)
    let tau_len = num_rounds + 2;
    let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(tau_len);

    // Step 4: Univariate-skip first round
    // Match UniPoly::append_to_transcript exactly
    transcript.append_message(b"UncompressedUniPoly_begin");
    for coeff in &uni_skip_poly_coeffs {
        transcript.append_scalar(coeff);
    }
    transcript.append_message(b"UncompressedUniPoly_end");
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();

    let (claim_after_uni_skip, power_sum_check) =
        verify_uni_skip_first_round::<F, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, 28>(
            &uni_skip_poly_coeffs,
            &r0,
        );

    // Step 5: Sumcheck rounds (simplified - no batching)
    let mut claim = claim_after_uni_skip;
    let mut sumcheck_challenges: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
    let mut claims_after_round = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        // Match CompressedUniPoly::append_to_transcript exactly
        transcript.append_message(b"UniPoly_begin");
        for coeff in &sumcheck_round_polys[round] {
            transcript.append_scalar(coeff);
        }
        transcript.append_message(b"UniPoly_end");

        let challenge: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        sumcheck_challenges.push(challenge);

        let next_claim = decompress_and_evaluate(&sumcheck_round_polys[round], &claim, &challenge);
        claims_after_round.push(next_claim);

        claim = next_claim;
    }

    Stage1FullVerificationResult {
        tau,
        r0,
        batching_coeff: F::one(), // Not used in simplified version
        sumcheck_challenges,

        claim_after_uni_skip,
        claims_after_round,
        final_claim: claim,

        // These are not computed in simplified version
        tau_high_bound_r0: F::zero(),
        tau_bound_r_tail: F::zero(),
        inner_sum_prod: F::zero(),
        expected_output_claim: F::zero(),

        power_sum_check,
        final_check: F::zero(), // Not computed
    }
}

// ============================================================================
// STAGE 2: Product Virtualization + Batched Sumchecks
// ============================================================================
//
// Stage 2 consists of:
// 1. Uni-skip first round for product virtualization
// 2. Batched sumcheck with 5 instances:
//    - ProductVirtualRemainderVerifier
//    - RamRafEvaluationSumcheckVerifier
//    - RamReadWriteCheckingVerifier
//    - OutputSumcheckVerifier
//    - InstructionLookupsClaimReductionSumcheckVerifier
//
// Stage 2 depends heavily on claims computed in Stage 1:
// - r_cycle (point from SpartanOuter sumcheck)
// - base_evals (claims of Product, WriteLookupOutputToRD, etc.)
// - Various other virtual polynomial claims
//
// For transpilation, these claims become additional circuit inputs.

use common::jolt_device::MemoryLayout;

/// Claims from Stage 1 that Stage 2 needs
///
/// These are the opening points and claims that were populated in the
/// `opening_accumulator` during Stage 1 verification.
#[derive(Clone, Debug)]
pub struct Stage1Claims<F: JoltField> {
    /// r_cycle: Opening point for Product @ SpartanOuter
    /// This is the concatenated cycle challenges from Stage 1
    pub r_cycle: Vec<F::Challenge>,

    /// Base evaluations for the 5 product constraints at r_cycle:
    /// [Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    pub product_base_evals: [F; 5],

    /// LookupOutput claim at r_cycle (for InstructionLookupsClaimReduction)
    pub lookup_output_claim: F,

    /// r_spartan opening point (same as r_cycle for our purposes)
    pub r_spartan: Vec<F::Challenge>,
}

/// Stage 2 verification data
///
/// Contains all inputs needed for Stage 2 verification:
/// - Claims from Stage 1 (via opening accumulator)
/// - Proof polynomials (uni-skip and sumcheck)
/// - Public parameters
#[derive(Clone, Debug)]
pub struct Stage2VerificationData<F: JoltField> {
    /// Claims from Stage 1
    pub stage1_claims: Stage1Claims<F>,

    /// Uni-skip polynomial coefficients for product virtualization
    pub uni_skip_poly_coeffs: Vec<F>,

    /// Compressed sumcheck round polynomials for the batched sumcheck
    /// Number of rounds = num_cycle_vars (for ProductVirtualRemainder)
    /// Each sumcheck may have different numbers of rounds - we use max
    pub sumcheck_round_polys: Vec<Vec<F>>,

    /// Trace length (needed for several verifiers)
    pub trace_length: usize,

    /// RAM K (size of RAM in words)
    pub ram_k: usize,

    /// Memory layout (needed for RamRafEvaluation and OutputSumcheck)
    pub memory_layout: MemoryLayout,

    /// One-hot parameters (needed for RAM sumchecks)
    /// For now, we'll compute these from trace_length
    pub one_hot_d: usize,

    /// Opening claims that will be produced by Stage 2 verifiers
    /// These are the claims that Stage 2's expected_output_claim will use
    pub stage2_opening_claims: Stage2OpeningClaims<F>,
}

/// Opening claims needed for Stage 2's expected_output_claim computation
///
/// These correspond to the claims that `cache_openings` would store.
#[derive(Clone, Debug)]
pub struct Stage2OpeningClaims<F: JoltField> {
    // === ProductVirtualRemainder claims ===
    /// LeftInstructionInput claim at r_product
    pub left_instruction_input: F,
    /// RightInstructionInput claim at r_product
    pub right_instruction_input: F,
    /// IsRdNotZero flag claim
    pub is_rd_not_zero: F,
    /// WriteLookupOutputToRD flag claim
    pub write_lookup_output_to_rd_flag: F,
    /// Jump flag claim
    pub jump_flag: F,
    /// LookupOutput claim at r_product
    pub lookup_output: F,
    /// Branch flag claim
    pub branch_flag: F,
    /// NextIsNoop claim
    pub next_is_noop: F,

    // === RamRafEvaluation claims ===
    /// RamRa claim for RAF evaluation
    pub ram_ra_raf: F,

    // === RamReadWriteChecking claims ===
    /// RamVal claim
    pub ram_val: F,
    /// RamRa claim for read/write checking
    pub ram_ra_rw: F,
    /// RamInc claim
    pub ram_inc: F,

    // === OutputSumcheck claims ===
    /// RamValFinal claim
    pub ram_val_final: F,
    /// RamValInit claim
    pub ram_val_init: F,

    // === InstructionLookupsClaimReduction claims ===
    /// LookupOutput claim at r_instruction
    pub lookup_output_instruction: F,
    /// LeftLookupOperand claim
    pub left_lookup_operand: F,
    /// RightLookupOperand claim
    pub right_lookup_operand: F,
}

/// Result of Stage 2 verification
#[derive(Clone, Debug)]
pub struct Stage2VerificationResult<F: JoltField> {
    // === Uni-skip challenges ===
    /// tau_high challenge (appended to tau from Stage 1)
    pub tau_high: F::Challenge,

    /// r0 challenge from uni-skip
    pub r0: F::Challenge,

    /// Claim after uni-skip
    pub claim_after_uni_skip: F,

    // === Batched sumcheck ===
    /// Batching coefficient
    pub batching_coeff: F,

    /// Sumcheck challenges
    pub sumcheck_challenges: Vec<F::Challenge>,

    /// Final claim
    pub final_claim: F,

    /// Expected output claim
    pub expected_output_claim: F,

    // === Constraint checks ===
    /// Power sum check (should be 0)
    pub power_sum_check: F,

    /// Final check: final_claim - expected_output_claim (should be 0)
    pub final_check: F,
}

// TODO: Implement verify_stage2_full when we have:
// 1. Access to program_io data for OutputSumcheck
// 2. Access to one_hot_params for RAM sumchecks
// 3. Full understanding of the claim flow between stages

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use crate::field::JoltField;

    type F = Fr;
    type C = <Fr as JoltField>::Challenge;

    #[test]
    fn test_evaluate_polynomial() {
        // poly(x) = 1 + 2x + 3x^2
        let coeffs = vec![F::from(1u64), F::from(2u64), F::from(3u64)];

        // poly(0) = 1
        assert_eq!(evaluate_polynomial(&coeffs, &F::from(0u64)), F::from(1u64));

        // poly(1) = 1 + 2 + 3 = 6
        assert_eq!(evaluate_polynomial(&coeffs, &F::from(1u64)), F::from(6u64));

        // poly(2) = 1 + 4 + 12 = 17
        assert_eq!(evaluate_polynomial(&coeffs, &F::from(2u64)), F::from(17u64));
    }

    #[test]
    fn test_decompress_and_evaluate() {
        // Test decompress_and_evaluate produces consistent results
        // The key property is that the decompressed polynomial is well-defined
        // and evaluates consistently at different points

        let c0 = F::from(1u64);
        let c2 = F::from(3u64);
        let c3 = F::from(4u64);
        // c1 = 10, claim = 2*1 + 10 + 3 + 4 = 19
        let claim = F::from(19u64);
        let compressed = vec![c0, c2, c3];

        // Evaluate at a few different challenge points
        let challenge_a: C = C::from(12345u128);
        let challenge_b: C = C::from(67890u128);

        let poly_at_a = decompress_and_evaluate(&compressed, &claim, &challenge_a);
        let poly_at_b = decompress_and_evaluate(&compressed, &claim, &challenge_b);

        // Different challenges should (with overwhelming probability) give different results
        assert_ne!(poly_at_a, poly_at_b);

        // Same challenge should give same result (deterministic)
        let poly_at_a_again = decompress_and_evaluate(&compressed, &claim, &challenge_a);
        assert_eq!(poly_at_a, poly_at_a_again);

        // Result should not be zero (with overwhelming probability for random-ish challenge)
        assert_ne!(poly_at_a, F::zero());
    }

    #[test]
    fn test_eq_polynomial_mle() {
        // Test the fundamental properties of eq polynomial
        // eq(x, y) = Π_i (x_i * y_i + (1-x_i)(1-y_i))

        // Property 1: eq(x, x) should give non-zero result for any x
        let x: Vec<C> = vec![C::from(12345u128)];
        let eq_xx = eq_polynomial_mle::<F>(&x, &x);
        assert_ne!(eq_xx, F::zero());

        // Property 2: eq is symmetric: eq(x, y) = eq(y, x)
        let a: Vec<C> = vec![C::from(111u128)];
        let b: Vec<C> = vec![C::from(222u128)];
        let eq_ab = eq_polynomial_mle::<F>(&a, &b);
        let eq_ba = eq_polynomial_mle::<F>(&b, &a);
        assert_eq!(eq_ab, eq_ba);

        // Property 3: Empty vectors should give 1 (empty product)
        let empty: Vec<C> = vec![];
        assert_eq!(eq_polynomial_mle::<F>(&empty, &empty), F::from(1u64));
    }

    #[test]
    fn test_evaluate_inner_sum_product_basic() {
        // Test with all-zero R1CS inputs - should give a deterministic result
        let r1cs_input_evals = [F::zero(); NUM_R1CS_INPUTS];
        let rx_constr = vec![F::from(1u64), F::from(2u64)];

        // This should not panic and should return some value
        let result = evaluate_inner_sum_product_at_point(&rx_constr, r1cs_input_evals);

        // With all zeros, the result depends only on constant terms in constraints
        // We just verify it runs without panicking
        let _ = result;
    }
}
