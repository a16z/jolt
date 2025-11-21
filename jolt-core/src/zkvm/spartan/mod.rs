use crate::field::JoltField;
use crate::poly::opening_proof::{
    OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::subprotocols::sumcheck::UniSkipFirstRoundProof;
use crate::subprotocols::univariate_skip::{prove_uniskip_round, UniSkipState};
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::outer::OuterUniSkipInstanceProver;
use crate::zkvm::spartan::product::{
    ProductVirtualUniSkipInstanceParams, ProductVirtualUniSkipInstanceProver,
};
use crate::zkvm::witness::VirtualPolynomial;

use tracer::instruction::Cycle;

pub mod instruction_input;
pub mod outer;
pub mod product;
pub mod shift;

// Stage 1: Outer sumcheck with uni-skip first round
pub fn prove_stage1_uni_skip<F: JoltField, T: Transcript>(
    trace: &[Cycle],
    bytecode_preprocessing: &BytecodePreprocessing,
    key: &UniformSpartanKey<F>,
    transcript: &mut T,
) -> (UniSkipState<F>, UniSkipFirstRoundProof<F, T>) {
    let num_rounds_x: usize = key.num_rows_bits();

    // Transcript and tau
    let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);

    // Prove uni-skip first round
    let mut uniskip_instance = OuterUniSkipInstanceProver::gen(trace, bytecode_preprocessing, &tau);
    let (first_round_proof, r0, claim_after_first) =
        prove_uniskip_round(&mut uniskip_instance, transcript);

    let uni_skip_state = UniSkipState {
        claim_after_first,
        r0,
        tau,
    };

    (uni_skip_state, first_round_proof)
}

/// Stage 1a: Verify first round of Spartan outer sum-check with univariate skip
pub fn verify_stage1_uni_skip<F: JoltField, T: Transcript>(
    proof: &UniSkipFirstRoundProof<F, T>,
    key: &UniformSpartanKey<F>,
    transcript: &mut T,
) -> Result<UniSkipState<F>, anyhow::Error> {
    let num_rounds_x = key.num_rows_bits();

    let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);

    let input_claim = F::zero();
    let (r0, claim_after_first) = proof
        .verify::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_FIRST_ROUND_POLY_NUM_COEFFS>(
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS - 1,
            input_claim,
            transcript,
        )
        .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

    Ok(UniSkipState {
        claim_after_first,
        r0,
        tau,
    })
}

// Stage 2: Product virtualization uni-skip first round
pub fn prove_stage2_uni_skip<F: JoltField, T: Transcript>(
    trace: &[Cycle],
    opening_accumulator: &ProverOpeningAccumulator<F>,
    key: &UniformSpartanKey<F>,
    transcript: &mut T,
) -> (UniSkipState<F>, UniSkipFirstRoundProof<F, T>) {
    let num_cycle_vars: usize = key.num_cycle_vars();

    // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
        .0
        .r;
    debug_assert_eq!(r_cycle.len(), num_cycle_vars);
    let tau_high = transcript.challenge_scalar_optimized::<F>();
    let mut tau = r_cycle;
    tau.push(tau_high);

    let mut uniskip_instance =
        ProductVirtualUniSkipInstanceProver::gen(trace, opening_accumulator, &tau);
    let (first_round_proof, r0, claim_after_first) =
        prove_uniskip_round(&mut uniskip_instance, transcript);

    let uni_skip_state = UniSkipState {
        claim_after_first,
        r0,
        tau,
    };

    (uni_skip_state, first_round_proof)
}

pub fn verify_stage2_uni_skip<F: JoltField, T: Transcript>(
    proof: &UniSkipFirstRoundProof<F, T>,
    key: &UniformSpartanKey<F>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<UniSkipState<F>, anyhow::Error> {
    let num_cycle_vars: usize = key.num_cycle_vars();

    // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
        .0
        .r;
    debug_assert_eq!(r_cycle.len(), num_cycle_vars);
    let tau_high: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    let mut tau: Vec<F::Challenge> = r_cycle;
    tau.push(tau_high);

    let uniskip_params = ProductVirtualUniSkipInstanceParams::new(opening_accumulator, &tau);
    let input_claim = uniskip_params.input_claim();
    let (r0, claim_after_first) = proof
        .verify::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS>(
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1,
            input_claim,
            transcript,
        )
        .map_err(|_| anyhow::anyhow!("ProductVirtual uni-skip first-round verification failed"))?;

    Ok(UniSkipState {
        claim_after_first,
        r0,
        tau,
    })
}
