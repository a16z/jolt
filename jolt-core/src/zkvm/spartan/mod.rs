use instruction_input::InstructionInputSumcheck;
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
use crate::zkvm::spartan::outer::{OuterRemainingSumcheck, OuterUniSkipInstance};
use crate::zkvm::spartan::pc::PCSumcheck;
use crate::zkvm::spartan::product::ProductVirtualizationSumcheck;

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
    /// Stage 1 handoff state from univariate skip first round (shared by prover and verifier)
    stage1_state: Option<Stage1State<F>>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        let key = Arc::new(UniformSpartanKey::new(padded_trace_length));
        Self {
            key,
            stage1_state: None,
        }
    }
}

struct Stage1State<F: JoltField> {
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
        let mut uniskip_instance =
            OuterUniSkipInstance::<F>::new(&preprocessing.shared, trace, &tau);
        let (first_round_proof, r0, claim_after_first) = prove_uniskip_round::<F, ProofTranscript, _>(
            &mut uniskip_instance,
            &mut *transcript_rc.borrow_mut(),
        );

        // Store first round
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1UniSkipFirstRoundProof,
            ProofData::UniSkipFirstRoundProof(first_round_proof),
        );

        // Cache remainder construction state for instances method
        self.stage1_state = Some(Stage1State {
            claim_after_first,
            r0,
            tau,
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
        self.stage1_state = Some(Stage1State {
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
        if let Some(st) = self.stage1_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let tau_low: Vec<F::Challenge> = st.tau[0..st.tau.len() - 1].to_vec();
            let outer_remaining = OuterRemainingSumcheck::new_prover(
                state_manager,
                st.claim_after_first,
                &tau_low,
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
        if let Some(st) = self.stage1_state.take() {
            let num_cycles_bits = self.key.num_steps.ilog2() as usize;
            let outer_remaining = OuterRemainingSumcheck::new_verifier(
                st.claim_after_first,
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
        /* Sumcheck 2: Inner sumcheck + ShouldJump/ShouldBranch/WritePCtoRD/WriteLookupOutputToRD product virtualization
            - Inner sumcheck proves: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    \sum_y (A_small(rx, y) + r * B_small(rx, y) + r^2 * C_small(rx, y)) * z(y)
            - ShouldJump sumcheck proves: ShouldJump(r_cycle) = Jump_flag(r_cycle) × (1 - NextIsNoop(r_cycle))
            - ShouldBranch sumcheck proves: ShouldBranch(r_cycle) = lookup_output(r_cycle) × Branch_flag(r_cycle)
            - WritePCtoRD sumcheck proves: WritePCtoRD(r_cycle) = rd_addr(r_cycle) × Jump_flag(r_cycle)
            - WriteLookupOutputToRD sumcheck proves: WriteLookupOutputToRD(r_cycle) = rd_addr(r_cycle) × WriteLookupOutputToRD_flag(r_cycle)
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::new_prover(state_manager, key);

        let should_jump_sumcheck = ProductVirtualizationSumcheck::new_prover(
            product::VirtualProductType::ShouldJump,
            state_manager,
        );

        let should_branch_sumcheck = ProductVirtualizationSumcheck::new_prover(
            product::VirtualProductType::ShouldBranch,
            state_manager,
        );

        let write_pc_to_rd_sumcheck = ProductVirtualizationSumcheck::new_prover(
            product::VirtualProductType::WritePCtoRD,
            state_manager,
        );

        let write_lookup_output_to_rd_sumcheck = ProductVirtualizationSumcheck::new_prover(
            product::VirtualProductType::WriteLookupOutputToRD,
            state_manager,
        );

        let product_sumcheck = ProductVirtualizationSumcheck::new_prover(
            product::VirtualProductType::Instruction,
            state_manager,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan InnerSumcheck", &inner_sumcheck);
            print_data_structure_heap_usage(
                "Spartan ShouldJump ProductVirtualizationSumcheck",
                &should_jump_sumcheck,
            );
            print_data_structure_heap_usage(
                "Spartan ShouldBranch ProductVirtualizationSumcheck",
                &should_branch_sumcheck,
            );
            print_data_structure_heap_usage(
                "Spartan WritePCtoRD ProductVirtualizationSumcheck",
                &write_pc_to_rd_sumcheck,
            );
            print_data_structure_heap_usage(
                "Spartan WriteLookupOutputToRD ProductVirtualizationSumcheck",
                &write_lookup_output_to_rd_sumcheck,
            );
            print_data_structure_heap_usage(
                "Spartan ProductVirtualizationSumcheck",
                &product_sumcheck,
            );
        }

        vec![
            Box::new(inner_sumcheck),
            Box::new(should_jump_sumcheck),
            Box::new(should_branch_sumcheck),
            Box::new(write_pc_to_rd_sumcheck),
            Box::new(write_lookup_output_to_rd_sumcheck),
            Box::new(product_sumcheck),
        ]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        /* Sumcheck 2: Inner sumcheck + ShouldJump/ShouldBranch/WritePCtoRD/WriteLookupOutputToRD product virtualization
           - Inner sumcheck verifies: claim_Az + r * claim_Bz + r^2 * claim_Cz =
                    (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)
           - ShouldJump sumcheck verifies: ShouldJump(r_cycle) = Jump_flag(r_cycle) × (1 - NextIsNoop(r_cycle))
           - ShouldBranch sumcheck verifies: ShouldBranch(r_cycle) = lookup_output(r_cycle) × Branch_flag(r_cycle)
           - WritePCtoRD sumcheck verifies: WritePCtoRD(r_cycle) = rd_addr(r_cycle) × Jump_flag(r_cycle)
           - WriteLookupOutputToRD sumcheck verifies: WriteLookupOutputToRD(r_cycle) = rd_addr(r_cycle) × WriteLookupOutputToRD_flag(r_cycle)
        */
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheck::<F>::new_verifier(state_manager, key);

        let should_jump_sumcheck = ProductVirtualizationSumcheck::new_verifier(
            product::VirtualProductType::ShouldJump,
            state_manager,
        );

        let should_branch_sumcheck = ProductVirtualizationSumcheck::new_verifier(
            product::VirtualProductType::ShouldBranch,
            state_manager,
        );

        let write_pc_to_rd_sumcheck = ProductVirtualizationSumcheck::new_verifier(
            product::VirtualProductType::WritePCtoRD,
            state_manager,
        );

        let write_lookup_output_to_rd_sumcheck = ProductVirtualizationSumcheck::new_verifier(
            product::VirtualProductType::WriteLookupOutputToRD,
            state_manager,
        );

        let product_sumcheck = ProductVirtualizationSumcheck::new_verifier(
            product::VirtualProductType::Instruction,
            state_manager,
        );

        vec![
            Box::new(inner_sumcheck),
            Box::new(should_jump_sumcheck),
            Box::new(should_branch_sumcheck),
            Box::new(write_pc_to_rd_sumcheck),
            Box::new(write_lookup_output_to_rd_sumcheck),
            Box::new(product_sumcheck),
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
