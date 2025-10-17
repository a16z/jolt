use instruction_input::InstructionInputSumcheck;
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::SumcheckId;
use crate::subprotocols::sumcheck::prove_uniskip_round;
use crate::subprotocols::univariate_skip::UniSkipState;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheck;
use crate::zkvm::spartan::outer::{OuterRemainingSumcheck, OuterUniSkipInstance};
use crate::zkvm::spartan::pc::PCSumcheck;
use crate::zkvm::spartan::product::{
    ProductVirtualRemainder, ProductVirtualUniSkipInstance,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::witness::VirtualPolynomial;

use crate::transcripts::Transcript;

use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::zkvm::r1cs::constraints::{FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DOMAIN_SIZE};

pub mod inner;
pub mod instruction_input;
pub mod outer;
pub mod pc;
pub mod product;

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
    // Stage 1: Outer sumcheck with first round as univariate skip, and remaining rounds as a
    // batched sumcheck instance (currently only the remaining outer sumcheck rounds, but could be
    // extended to include other sumchecks)

    fn stage1_prover_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        // Stage 1a: Uni-skip first round only
        let key = self.key.clone();
        let num_rounds_x: usize = key.num_rows_bits();

        // Capture transcript and accumulator handles up-front to avoid borrowing state_manager later
        let transcript = state_manager.get_transcript();

        let tau: Vec<F::Challenge> = transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Uni-skip first round using canonical instance + helper
        let mut uniskip_instance = OuterUniSkipInstance::<F>::new_prover(state_manager, &tau);

        let (first_round_proof, r0, claim_after_first) = prove_uniskip_round::<F, ProofTranscript, _>(
            &mut uniskip_instance,
            &mut *transcript.borrow_mut(),
        );

        // Store first round
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1UniSkipFirstRound,
            ProofData::UniSkipFirstRoundProof(first_round_proof),
        );

        // Cache remainder construction state for instances method
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

        // Load first round
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

        let (r0, claim_after_first) = first_round
            .verify::<UNIVARIATE_SKIP_DOMAIN_SIZE, FIRST_ROUND_POLY_NUM_COEFFS>(
                FIRST_ROUND_POLY_NUM_COEFFS - 1,
                &mut *state_manager.transcript.borrow_mut(),
            )
            .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

        // Cache info needed to build verifier-side remainder instance later
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
        // Stage 1 remainder: outer-remaining + extras.
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.uni_skip_state.take() {
            let outer_remaining =
                OuterRemainingSumcheck::new_prover(state_manager, self.key.num_cycle_vars(), &st);
            instances.push(Box::new(outer_remaining));
        }
        // TODO: append extras when available
        instances
    }

    fn stage1_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining + extras (verifier side).
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.uni_skip_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let outer_remaining = OuterRemainingSumcheck::new_verifier(num_cycles_bits, st);
            instances.push(Box::new(outer_remaining));
        }
        // TODO: append extras when available
        instances
    }

    /* Sumcheck 2: Inner sumcheck + Product Virtualization
        This stage proves two things in parallel:
        1) Inner sumcheck: claim_Az + r * claim_Bz = sum_y (A_small(rx, y) + r * B_small(rx, y)) * z(y)
        2) Product virtualization (single protocol):
           - Univariate-skip first round over a size-5 domain Y to compress five product constraints
           - A remainder sumcheck over cycle variables that binds one Left/Right pair defined by
             Lagrange weights at r0

        Notation (for product virtualization):
        • Variables: r = (r0, r_tail), where r0 is the uni-skip challenge and r_tail are cycle bits;
          τ = (τ_low, τ_high) with τ_low the cycle eq point and τ_high the uni-skip binding point.
        • Lagrange weights: w_i = L_i(r0), i ∈ {0..4}, on the size-5 base domain.
        • Five product terms P_i(x) over cycle assignment x:
          P0(x) = LeftInstructionInput(x) * RightInstructionInput(x)
          P1(x) = RdWa(x) * OpFlags_WriteLookupOutputToRD(x)
          P2(x) = RdWa(x) * OpFlags_Jump(x)
          P3(x) = LookupOutput(x) * InstructionFlags_Branch(x)
          P4(x) = OpFlags_Jump(x) * (1 - NextIsNoop(x))
        • Weighted Left/Right:
          Left(x)  = Σ_i w_i · Left_i(x)
          Right(x) = Σ_i w_i · Right_i^eff(x)  (with Right_4^eff(x) = 1 - NextIsNoop(x))

        Prover sends:
        - Uni-skip first round s1(Y) = L(τ_high, Y) · t1(Y)
        - Remainder rounds binding r_tail with a degree-3 sumcheck whose endpoints depend on
          Left/Right as defined above.

        Verifier checks:
        - L(τ_high, r0) · Eq(τ_low, r_tail^rev) · Left(r_cycle) · Right(r_cycle), where r_cycle = r_tail.
    */

    fn stage2_prover_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        // Univariate-skip first round for product virtualization (Stage 2a)
        let num_cycle_vars: usize = self.key.num_cycle_vars();
        let transcript = state_manager.get_transcript();

        // Reuse r_cycle from Stage 1 (Spartan outer) for τ_low, and sample a single τ_high
        let r_cycle: Vec<F::Challenge> = {
            let acc = state_manager.get_prover_accumulator();
            let (outer_opening, _eval) = acc.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::Product,
                SumcheckId::SpartanOuter,
            );
            // Outer stored only r_cycle for these witness openings
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

    /* Sumcheck 2: Inner sumcheck + Product Virtualization
       Verification perspective.
       1) Inner sumcheck: verify claim_Az + r * claim_Bz = (A_small(rx, ry) + r * B_small(rx, ry)) * z(ry)

       2) Product virtualization: define the five product terms and weights as above,
          with w_i = L_i(r0). Let
            Left(r_cycle)  = Σ_i w_i · Left_i(r_cycle)
            Right(r_cycle) = Σ_i w_i · Right_i^eff(r_cycle)  (Right_4^eff = 1 - NextIsNoop)
          Then verify that the uni-skip + remainder messages imply the final claim
            L(τ_high, r0) · Eq(τ_low, r_tail^rev) · Left(r_cycle) · Right(r_cycle).
    */

    fn stage2_verifier_uni_skip(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let num_cycle_vars: usize = self.key.num_cycle_vars();

        // Reuse r_cycle from Stage 1 (Spartan outer) for τ_low, and sample a single τ_high
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

        let (r0, claim_after_first) = first_round
            .verify::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS>(
                PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1,
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
        // Sumcheck 2: Inner sumcheck + Product Virtualization remainder
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
        // Sumcheck 2: Inner sumcheck + Product Virtualization remainder (verifier)
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
        let pc_sumcheck = PCSumcheck::<F>::new_prover(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan PCSumcheck", &pc_sumcheck);
            print_data_structure_heap_usage(
                "InstructionInputSumcheck",
                &instruction_input_sumcheck,
            );
        }

        vec![Box::new(pc_sumcheck), Box::new(instruction_input_sumcheck)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /* Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
           Verifies the batched constraint for both NextUnexpandedPC and NextPC
        */
        let key = self.key.clone();
        let pc_sumcheck = PCSumcheck::<F>::new_verifier(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_verifier(state_manager);
        vec![Box::new(pc_sumcheck), Box::new(instruction_input_sumcheck)]
    }
}
