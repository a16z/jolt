use instruction_input::InstructionInputSumcheck;
use std::sync::Arc;
// use tracing::{span, Level};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::SumcheckId;
use crate::subprotocols::sumcheck::{prove_uniskip_round, UniSkipFirstRoundInstance};
use crate::subprotocols::univariate_skip::UniSkipState;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheck;
use crate::zkvm::spartan::outer::{OuterRemainingSumcheck, OuterUniSkipInstance};
use crate::zkvm::spartan::product::{
    ProductVirtualInner, ProductVirtualRemainder, ProductVirtualUniSkipInstance,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::spartan::shift::ShiftSumcheck;
use crate::zkvm::witness::VirtualPolynomial;

use crate::transcripts::Transcript;

use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::zkvm::r1cs::constraints::{FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DOMAIN_SIZE};

pub mod inner;
pub mod instruction_input;
pub mod outer;
pub mod product;
pub mod shift;

pub struct SpartanDag<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
    /// Handoff state from univariate skip first round (shared by prover and verifier)
    /// Consists of the `tau` vector for Lagrange / eq evals, the claim from univariate skip round,
    /// and the challenge r0 from the univariate skip round
    /// This is first used in stage 1 and then reused in stage 2
    uni_skip_state: Option<UniSkipState<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanKey::new(padded_trace_length));
        Self {
            key,
            uni_skip_state: None,
        }
    }
}

impl<F, ProofTranscript, PCS> SumcheckStages<F, ProofTranscript, PCS> for SpartanDag<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    // Stage 1: Outer sumcheck with uni-skip first round
    fn stage1_prover_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();
        let num_rounds_x: usize = key.num_rows_bits();

        // Transcript and tau
        let transcript = state_manager.get_transcript();
        let tau: Vec<F::Challenge> = transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Prove uni-skip first round
        let mut uniskip_instance = OuterUniSkipInstance::<F>::new_prover(state_manager, &tau);
        let (first_round_proof, r0, claim_after_first) = prove_uniskip_round::<F, ProofTranscript, _>(
            &mut uniskip_instance,
            &mut *transcript.borrow_mut(),
        );

        // Store proof and handoff state
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1UniSkipFirstRound,
            ProofData::UniSkipFirstRoundProof(first_round_proof),
        );

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });

        Ok(())
    }

    fn stage1_verifier_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();
        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Load and verify uni-skip first round proof
        let first_round = {
            let proofs = state_manager.proofs.borrow();
            match proofs
                .get(&ProofKeys::Stage1UniSkipFirstRound)
                .expect("missing Stage1UniSkipFirstRound")
            {
                ProofData::UniSkipFirstRoundProof(fr) => fr.clone(),
                _ => panic!("unexpected proof type for Stage1UniSkipFirstRound"),
            }
        };

        let uniskip_instance = OuterUniSkipInstance::<F>::new_verifier(&tau);
        let input_claim = <OuterUniSkipInstance<F> as UniSkipFirstRoundInstance<
            F,
            ProofTranscript,
        >>::input_claim(&uniskip_instance);
        let (r0, claim_after_first) = first_round
            .verify::<UNIVARIATE_SKIP_DOMAIN_SIZE, FIRST_ROUND_POLY_NUM_COEFFS>(
                FIRST_ROUND_POLY_NUM_COEFFS - 1,
                input_claim,
                &mut *state_manager.transcript.borrow_mut(),
            )
            .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });

        Ok(())
    }

    fn stage1_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.uni_skip_state.take() {
            let outer_remaining =
                OuterRemainingSumcheck::new_prover(state_manager, self.key.num_cycle_vars(), &st);
            instances.push(Box::new(outer_remaining));
        }
        instances
    }

    fn stage1_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining (verifier)
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.uni_skip_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let outer_remaining = OuterRemainingSumcheck::new_verifier(num_cycles_bits, st);
            instances.push(Box::new(outer_remaining));
        }
        instances
    }

    // Stage 2: Product virtualization uni-skip first round
    fn stage2_prover_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let num_cycle_vars: usize = self.key.num_cycle_vars();
        let transcript = state_manager.get_transcript();

        // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
        let r_cycle: Vec<F::Challenge> = {
            let acc = state_manager.get_prover_accumulator();
            let (outer_opening, _eval) = acc.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::Product,
                SumcheckId::SpartanOuter,
            );
            outer_opening.r
        };
        debug_assert_eq!(r_cycle.len(), num_cycle_vars);
        let tau_high: F::Challenge = transcript.borrow_mut().challenge_scalar_optimized::<F>();
        let mut tau: Vec<F::Challenge> = r_cycle;
        tau.push(tau_high);

        let mut uniskip_instance =
            ProductVirtualUniSkipInstance::<F>::new_prover(state_manager, &tau);
        let (first_round_proof, r0, claim_after_first) =
            prove_uniskip_round::<F, ProofTranscript, ProductVirtualUniSkipInstance<F>>(
                &mut uniskip_instance,
                &mut *transcript.borrow_mut(),
            );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2UniSkipFirstRound,
            ProofData::UniSkipFirstRoundProof(first_round_proof),
        );

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });
        Ok(())
    }

    fn stage2_verifier_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let num_cycle_vars: usize = self.key.num_cycle_vars();

        // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
        let r_cycle: Vec<F::Challenge> = {
            let acc = state_manager.get_verifier_accumulator();
            let (outer_opening, _eval) = acc.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::Product,
                SumcheckId::SpartanOuter,
            );
            outer_opening.r
        };
        debug_assert_eq!(r_cycle.len(), num_cycle_vars);
        let tau_high: F::Challenge = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_optimized::<F>();
        let mut tau: Vec<F::Challenge> = r_cycle;
        tau.push(tau_high);

        let first_round = {
            let proofs = state_manager.proofs.borrow();
            match proofs
                .get(&ProofKeys::Stage2UniSkipFirstRound)
                .expect("missing Stage2UniSkipFirstRound")
            {
                ProofData::UniSkipFirstRoundProof(fr) => fr.clone(),
                _ => panic!("unexpected proof type for Stage2UniSkipFirstRound"),
            }
        };

        let uniskip_instance =
            ProductVirtualUniSkipInstance::<F>::new_verifier(state_manager, &tau);
        let input_claim = <ProductVirtualUniSkipInstance<F> as UniSkipFirstRoundInstance<
            F,
            ProofTranscript,
        >>::input_claim(&uniskip_instance);
        let (r0, claim_after_first) = first_round
            .verify::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS>(
                PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1,
                input_claim,
                &mut *state_manager.transcript.borrow_mut(),
            )
            .map_err(|_| anyhow::anyhow!("ProductVirtual uni-skip first-round verification failed"))?;

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });
        Ok(())
    }

    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 2 remainder: inner + product remainder
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::new_prover(state_manager, key);

        let st = self
            .uni_skip_state
            .take()
            .expect("stage2_prover_uni_skip must run before stage2_prover_instances");
        let num_cycle_vars = self.key.num_cycle_vars();
        let product_virtual_remainder =
            ProductVirtualRemainder::new_prover(state_manager, num_cycle_vars, &st);

        vec![
            Box::new(inner_sumcheck),
            Box::new(product_virtual_remainder),
        ]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 2 remainder (verifier side)
        let num_cycle_vars = self.key.num_cycle_vars();
        let inner_sumcheck = InnerSumcheck::<F>::new_verifier(state_manager, self.key.clone());

        let st = self
            .uni_skip_state
            .take()
            .expect("stage2_verifier_uni_skip must run before stage2_verifier_instances");
        let product_virtual_remainder = ProductVirtualRemainder::new_verifier(num_cycle_vars, st);

        vec![
            Box::new(inner_sumcheck),
            Box::new(product_virtual_remainder),
        ]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let key = self.key.clone();
        let shift_sumcheck = ShiftSumcheck::<F>::new_prover(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_prover(state_manager);
        let product_factors_order_check = ProductVirtualInner::new_prover(state_manager);

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
            Box::new(product_factors_order_check),
        ]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let key = self.key.clone();
        let shift_sumcheck = ShiftSumcheck::<F>::new_verifier(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_verifier(state_manager);
        let product_factors_order_check = ProductVirtualInner::new_verifier(state_manager);
        vec![
            Box::new(shift_sumcheck),
            Box::new(instruction_input_sumcheck),
            Box::new(product_factors_order_check),
        ]
    }
}
