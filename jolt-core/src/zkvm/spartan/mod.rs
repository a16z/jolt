use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::subprotocols::sumcheck::UniSkipFirstRoundProof;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::subprotocols::univariate_skip::{prove_uniskip_round, UniSkipState};
use crate::transcripts::Transcript;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::{SumcheckStagesProver, SumcheckStagesVerifier};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::{InnerSumcheckProver, InnerSumcheckVerifier};
use crate::zkvm::spartan::instruction_input::{
    InstructionInputSumcheckProver, InstructionInputSumcheckVerifier,
};
use crate::zkvm::spartan::outer::{
    OuterRemainingSumcheckProver, OuterRemainingSumcheckVerifier, OuterUniSkipInstanceProver,
};
use crate::zkvm::spartan::product::{
    ProductVirtualInnerProver, ProductVirtualInnerVerifier, ProductVirtualRemainderProver,
    ProductVirtualRemainderVerifier, ProductVirtualUniSkipInstanceParams,
};
use crate::zkvm::spartan::shift::{ShiftSumcheckProver, ShiftSumcheckVerifier};
use crate::zkvm::witness::VirtualPolynomial;

use product::{
    ProductVirtualUniSkipInstanceProver, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};

pub mod inner;
pub mod instruction_input;
pub mod outer;
pub mod product;
pub mod shift;

pub struct SpartanDagProver<F: JoltField> {
    state: SpartanDagState<F>,
}

impl<F: JoltField> SpartanDagProver<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        Self {
            state: SpartanDagState::new(padded_trace_length),
        }
    }

    // Stage 1: Outer sumcheck with uni-skip first round
    pub fn stage1_uni_skip<T: Transcript>(
        &mut self,
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UniSkipFirstRoundProof<F, T> {
        let num_rounds_x: usize = self.state.key.num_rows_bits();

        // Transcript and tau
        let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);

        // Prove uni-skip first round
        let mut uniskip_instance = OuterUniSkipInstanceProver::gen(state_manager, &tau);
        let (first_round_proof, r0, claim_after_first) =
            prove_uniskip_round(&mut uniskip_instance, transcript);

        self.state.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });

        first_round_proof
    }

    // Stage 2: Product virtualization uni-skip first round
    pub fn stage2_uni_skip<T: Transcript>(
        &mut self,
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UniSkipFirstRoundProof<F, T> {
        let num_cycle_vars: usize = self.state.key.num_cycle_vars();

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
            ProductVirtualUniSkipInstanceProver::gen(state_manager, opening_accumulator, &tau);
        let (first_round_proof, r0, claim_after_first) =
            prove_uniskip_round(&mut uniskip_instance, transcript);

        self.state.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });
        first_round_proof
    }
}

impl<F, ProofTranscript, PCS> SumcheckStagesProver<F, ProofTranscript, PCS> for SpartanDagProver<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage1_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.state.uni_skip_state.take() {
            let n_cycles = self.state.key.num_cycle_vars();
            let outer_remaining = OuterRemainingSumcheckProver::gen(state_manager, n_cycles, &st);
            instances.push(Box::new(outer_remaining));
        }
        instances
    }

    fn stage2_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        // Stage 2 remainder: inner + product remainder
        let key = self.state.key.clone();
        let inner_sumcheck = InnerSumcheckProver::gen(opening_accumulator, key, transcript);

        let st = self
            .state
            .uni_skip_state
            .take()
            .expect("stage2_prover_uni_skip must run before stage2_prover_instances");
        let n_cycle_vars = self.state.key.num_cycle_vars();
        let product_virtual_remainder =
            ProductVirtualRemainderProver::gen(state_manager, n_cycle_vars, &st);

        vec![
            Box::new(inner_sumcheck),
            Box::new(product_virtual_remainder),
        ]
    }

    fn stage3_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let key = self.state.key.clone();
        let shift_sumcheck =
            ShiftSumcheckProver::gen(state_manager, opening_accumulator, key, transcript);
        let instruction_input_sumcheck =
            InstructionInputSumcheckProver::gen(state_manager, opening_accumulator, transcript);
        let product_virtual_claim_check =
            ProductVirtualInnerProver::new(opening_accumulator, transcript);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan ShiftSumcheck", &shift_sumcheck);
            print_data_structure_heap_usage(
                "InstructionInputSumcheck",
                &instruction_input_sumcheck,
            );
        }

        vec![
            Box::new(shift_sumcheck),
            Box::new(instruction_input_sumcheck),
            Box::new(product_virtual_claim_check),
        ]
    }
}

pub struct SpartanDagVerifier<F: JoltField> {
    state: SpartanDagState<F>,
}

impl<F: JoltField> SpartanDagVerifier<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        Self {
            state: SpartanDagState::new(padded_trace_length),
        }
    }

    /// Stage 1a: Verify first round of Spartan outer sum-check with univariate skip
    pub fn stage1_uni_skip<T: Transcript>(
        &mut self,
        proof: &UniSkipFirstRoundProof<F, T>,
        transcript: &mut T,
    ) -> Result<(), anyhow::Error> {
        let key = self.state.key.clone();
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

        self.state.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });

        Ok(())
    }

    pub fn stage2_uni_skip<T: Transcript>(
        &mut self,
        proof: &UniSkipFirstRoundProof<F, T>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), anyhow::Error> {
        let num_cycle_vars: usize = self.state.key.num_cycle_vars();

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

        self.state.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });
        Ok(())
    }
}

impl<F, ProofTranscript, PCS> SumcheckStagesVerifier<F, ProofTranscript, PCS>
    for SpartanDagVerifier<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage1_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining (verifier)
        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.state.uni_skip_state.take() {
            let num_cycles_bits = self.state.key.num_steps.ilog2() as usize;
            let outer_remaining = OuterRemainingSumcheckVerifier::new(num_cycles_bits, &st);
            instances.push(Box::new(outer_remaining));
        }
        instances
    }

    fn stage2_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        // Stage 2 remainder (verifier side)
        let num_cycle_vars = self.state.key.num_cycle_vars();
        let inner_sumcheck = InnerSumcheckVerifier::new(self.state.key.clone(), transcript);

        let st = self
            .state
            .uni_skip_state
            .take()
            .expect("stage2_uni_skip must run before stage2_verifier_instances");
        let product_virtual_remainder = ProductVirtualRemainderVerifier::new(num_cycle_vars, &st);

        vec![
            Box::new(inner_sumcheck),
            Box::new(product_virtual_remainder),
        ]
    }

    fn stage3_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let key = self.state.key.clone();
        let shift_sumcheck = ShiftSumcheckVerifier::new(opening_accumulator, key, transcript);
        let instruction_input_sumcheck =
            InstructionInputSumcheckVerifier::new(opening_accumulator, transcript);
        let product_virtual_claim_check =
            ProductVirtualInnerVerifier::new(opening_accumulator, transcript);
        vec![
            Box::new(shift_sumcheck),
            Box::new(instruction_input_sumcheck),
            Box::new(product_virtual_claim_check),
        ]
    }
}

struct SpartanDagState<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
    /// Handoff state from univariate skip first round (shared by prover and verifier)
    /// Consists of the `tau` vector for Lagrange / eq evals, the claim from univariate skip round,
    /// and the challenge r0 from the univariate skip round
    /// This is first used in stage 1 and then reused in stage 2
    uni_skip_state: Option<UniSkipState<F>>,
}

impl<F: JoltField> SpartanDagState<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanKey::new(padded_trace_length));
        Self {
            key,
            uni_skip_state: None,
        }
    }
}
