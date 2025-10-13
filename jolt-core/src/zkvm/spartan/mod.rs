use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheck;
use crate::zkvm::spartan::outer::{OuterRemainingSumcheck, OuterSumcheck, SpartanInterleavedPoly};
use crate::subprotocols::sumcheck::{BatchedSumcheck, UniSkipFirstRound};
use crate::zkvm::spartan::pc::PCSumcheck;
use crate::zkvm::spartan::product::ProductVirtualizationSumcheck;

use crate::transcripts::Transcript;

use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::zkvm::r1cs::constraints::{FIRST_ROUND_POLY_NUM_COEFFS, UNIVARIATE_SKIP_DOMAIN_SIZE};
use tracer::instruction::Cycle;

pub mod inner;
pub mod outer;
pub mod pc;
pub mod product;

pub struct SpartanDag<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new<ProofTranscript: Transcript>(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanKey::new(padded_trace_length));
        Self { key }
    }
}

impl<F, ProofTranscript, PCS> SumcheckStages<F, ProofTranscript, PCS> for SpartanDag<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    // Stage 1: Outer sumcheck (via legacy prove/verify) + additional sumchecks (via instances)
    // The outer sumcheck uses UniSkipSumcheck and must be handled specially via stage1_prove/verify.
    // Additional sumchecks (e.g. HammingWeightSumcheck) can be added via stage1_*_instances.
    
    fn stage1_prover_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // Stage 1 extras (e.g. HammingWeightSumcheck) can be returned here in the future.
        // The Outer sumcheck is handled directly in stage1_prove via OuterSumcheck::prove_with_state_manager.
        vec![]
    }

    fn stage1_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        // For verification, we don't actually need to create instances since we already have the proof
        // The verification is handled directly in stage1_verify by calling SingleSumcheck::verify
        // This method exists for consistency with the trait but returns empty for stage 1
        vec![]
    }

    fn stage1_prove(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        // Stage 1: Uni-skip only, then batch OuterRemaining + extras
        let key = self.key.clone();
        let num_rounds_x = key.num_rows_bits();

        // Gather any extra Stage 1 instances first (avoids mutable borrow conflicts later)
        let mut extras = self.stage1_prover_instances(state_manager);

        // Capture transcript and accumulator handles up-front to avoid borrowing state_manager later
        let transcript_rc = state_manager.get_transcript();
        let accumulator = state_manager.get_prover_accumulator();

        let tau: Vec<F::Challenge> = transcript_rc
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Uni-skip first round
        let (first_poly, r0, claim_after_first, split_eq_poly, interleaved_poly);
        // Uni-skip round
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let (fp, r0_, claim1, split_eq, inter_poly) = OuterSumcheck::prove_univariate_skip_round(
            &preprocessing.shared,
            trace,
            &tau,
            &mut *transcript_rc.borrow_mut(),
        );
        first_poly = fp;
        r0 = r0_;
        claim_after_first = claim1;
        split_eq_poly = split_eq;
        interleaved_poly = inter_poly;

        // Prove batched remainder inside a narrow scope to drop borrows before storing proofs
        let batched_proof = {
            let (preprocessing, trace, _, _) = state_manager.get_prover_data();
            let mut outer_remaining = OuterRemainingSumcheck::new_prover(
                claim_after_first,
                &preprocessing.shared,
                trace,
                split_eq_poly,
                interleaved_poly,
                r0,
                num_rounds_x - 1,
            );

            let mut instances: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> = Vec::new();
            instances.push(&mut outer_remaining);
            for inst in extras.iter_mut() {
                instances.push(&mut **inst);
            }

            let (batched_proof, _r_batched) = BatchedSumcheck::prove::<F, ProofTranscript>(
                instances,
                Some(accumulator.clone()),
                &mut *transcript_rc.borrow_mut(),
            );
            batched_proof
        };

        // Now store under existing Stage1Sumcheck as a combined payload
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1Sumcheck,
            ProofData::Stage1Combined {
                first_round: UniSkipFirstRound::new(first_poly),
                batched_remainder: batched_proof,
            },
        );

        Ok(())
    }

    fn stage1_verify(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();
        let num_rounds_x = key.num_rows_bits();

        let (preprocessing, _, _trace_length) = state_manager.get_verifier_data();
        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Fetch combined Stage 1 and verify only first round to get r0 and claim
        let (first_round, batched_remainder_cloned) = {
            let proofs = state_manager.proofs.borrow();
            match proofs.get(&ProofKeys::Stage1Sumcheck).expect("missing Stage1Sumcheck") {
                ProofData::Stage1Combined { first_round, batched_remainder } => {
                    (first_round.clone(), batched_remainder.clone())
                }
                _ => panic!("unexpected proof type for Stage1Sumcheck"),
            }
        };

        let (r0, claim_after_first) = first_round
            .verify::<UNIVARIATE_SKIP_DOMAIN_SIZE, FIRST_ROUND_POLY_NUM_COEFFS>(
                FIRST_ROUND_POLY_NUM_COEFFS - 1,
                &mut *state_manager.transcript.borrow_mut(),
            )
            .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

        // Build outer_remaining verifier instance
        let dummy_trace: Vec<Cycle> = vec![];
        let outer_remaining = OuterRemainingSumcheck::new_verifier(
            claim_after_first,
            &preprocessing.shared,
            &dummy_trace,
            GruenSplitEqPolynomial::new(&tau[0..tau.len() - 1], BindingOrder::LowToHigh),
            SpartanInterleavedPoly::new(),
            r0,
            num_rounds_x - 1,
            tau.clone(),
        );

        // Collect any extra verifier instances
        let extras = self.stage1_verifier_instances(state_manager);
        let instances: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = {
            let mut v: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = Vec::new();
            v.push(&outer_remaining);
            for inst in extras.iter() {
                v.push(&**inst);
            }
            v
        };

        let opening_accumulator = state_manager.get_verifier_accumulator();
        let _r = BatchedSumcheck::verify::<F, ProofTranscript>(
            &batched_remainder_cloned,
            instances,
            Some(opening_accumulator.clone()),
            &mut *state_manager.transcript.borrow_mut(),
        )?;

        Ok(())
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
