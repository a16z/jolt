use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::prove_uniskip_round;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheck;
use crate::zkvm::spartan::outer::{OuterRemainingSumcheck, OuterUniSkipInstance, SpartanInterleavedPoly};
use crate::zkvm::spartan::pc::PCSumcheck;
use crate::zkvm::spartan::product::ProductVirtualizationSumcheck;

use crate::transcripts::Transcript;

use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::zkvm::r1cs::constraints::{FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DOMAIN_SIZE};

pub mod inner;
pub mod outer;
pub mod pc;
pub mod product;

pub struct SpartanDag<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
    /// Cached state for Stage 1 uniskip remainder (prover)
    stage1_prover_state: Option<SpartanStage1ProverState<F>>,
    /// Cached state for Stage 1 uniskip remainder (verifier)
    stage1_verifier_state: Option<SpartanStage1VerifierState<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanKey::new(padded_trace_length));
        Self { key, stage1_prover_state: None, stage1_verifier_state: None }
    }
}

struct SpartanStage1ProverState<F: JoltField> {
    claim_after_first: F,
    r0: F::Challenge,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    total_rounds_remainder: usize,
}

struct SpartanStage1VerifierState<F: JoltField> {
    claim_after_first: F,
    r0: F::Challenge,
    tau: Vec<<F as JoltField>::Challenge>,
    total_rounds_remainder: usize,
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

    fn stage1_first_round_prove(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        // Stage 1a: Uni-skip first round only
        let key = self.key.clone();
        let num_rounds_x = key.num_rows_bits();

        // Capture transcript and accumulator handles up-front to avoid borrowing state_manager later
        let transcript_rc = state_manager.get_transcript();
        let _accumulator = state_manager.get_prover_accumulator();

        let tau: Vec<F::Challenge> = transcript_rc
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Uni-skip first round using canonical instance + helper
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let mut uniskip_instance = OuterUniSkipInstance::<F>::new(
            &preprocessing.shared,
            trace,
            &tau,
        );
        let (first_round_proof, r0, claim_after_first) = prove_uniskip_round::<F, ProofTranscript, _>(
            &mut uniskip_instance,
            &mut *transcript_rc.borrow_mut(),
        );
        let tau_low: Vec<F::Challenge> = tau[0..tau.len() - 1].to_vec();
        let split_eq_poly = GruenSplitEqPolynomial::new(&tau_low, BindingOrder::LowToHigh);

        // Store first round
        state_manager
            .proofs
            .borrow_mut()
            .insert(ProofKeys::Stage1UniSkipFirstRoundProof, ProofData::UniSkipFirstRoundProof(first_round_proof));

        // Cache remainder construction state for instances method
        self.stage1_prover_state = Some(SpartanStage1ProverState {
            claim_after_first,
            r0,
            split_eq_poly,
            total_rounds_remainder: num_rounds_x - 1,
        });

        Ok(())
    }

    fn stage1_first_round_verify(
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
                .get(&ProofKeys::Stage1UniSkipFirstRoundProof)
                .expect("missing Stage1UniSkipFirstRoundProof")
            {
                ProofData::UniSkipFirstRoundProof(fr) => fr.clone(),
                _ => panic!("unexpected proof type for Stage1UniSkipFirstRoundProof"),
            }
        };

        let (r0, claim_after_first) = first_round
            .verify::<UNIVARIATE_SKIP_DOMAIN_SIZE, FIRST_ROUND_POLY_NUM_COEFFS>(
                FIRST_ROUND_POLY_NUM_COEFFS - 1,
                &mut *state_manager.transcript.borrow_mut(),
            )
            .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

        // Cache info needed to build verifier-side remainder instance later
        self.stage1_verifier_state = Some(SpartanStage1VerifierState {
            claim_after_first,
            r0,
            tau: tau.clone(),
            total_rounds_remainder: num_rounds_x - 1,
        });

        Ok(())
    }


    fn stage1_remainder_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining + extras.
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.stage1_prover_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let (preprocessing, trace, _, _) = state_manager.get_prover_data();
            let outer_remaining = OuterRemainingSumcheck::new_prover(
                st.claim_after_first,
                &preprocessing.shared,
                trace,
                st.split_eq_poly,
                st.r0,
                st.total_rounds_remainder,
                num_cycles_bits,
            );
            instances.push(Box::new(outer_remaining));
        }
        // TODO: append extras when available
        instances
    }

    fn stage1_remainder_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining + extras (verifier side).
        let mut instances: Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> = Vec::new();
        if let Some(st) = self.stage1_verifier_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let outer_remaining = OuterRemainingSumcheck::new_verifier(
                st.claim_after_first,
                GruenSplitEqPolynomial::new(&st.tau[0..st.tau.len() - 1], BindingOrder::LowToHigh),
                SpartanInterleavedPoly::new(),
                st.r0,
                st.total_rounds_remainder,
                st.tau,
                num_cycles_bits,
            );
            instances.push(Box::new(outer_remaining));
        }
        // TODO: append extras when available
        instances
    }


    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /* Sumcheck 2: Inner sumcheck
            Proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)

            Evaluates the uniform constraint matrices A_small, B_small, C_small at the point
            determined by the outer sumcheck.
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::new_prover(state_manager, key);

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Spartan InnerSumcheck", &inner_sumcheck);

        vec![Box::new(inner_sumcheck)]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /* Sumcheck 2: Inner sumcheck
           Verifies: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::<F>::new_verifier(state_manager, key);
        vec![Box::new(inner_sumcheck)]
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
        /* Also performs the product virtualization sumcheck */
        let product_sumcheck = ProductVirtualizationSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan PCSumcheck", &pc_sumcheck);
            print_data_structure_heap_usage(
                "Spartan ProductVirtualizationSumcheck",
                &product_sumcheck,
            );
        }

        vec![Box::new(pc_sumcheck), Box::new(product_sumcheck)]
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
        /* Also verifies the product virtualization sumcheck */
        let product_sumcheck = ProductVirtualizationSumcheck::new_verifier(state_manager);

        vec![Box::new(pc_sumcheck), Box::new(product_sumcheck)]
    }
}
