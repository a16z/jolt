use instruction_input::InstructionInputSumcheck;
use std::sync::Arc;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{OpeningPoint, SumcheckId};
use crate::poly::spartan_interleaved_poly::NUM_SVO_ROUNDS;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::inputs::{compute_claimed_witness_evals, ALL_R1CS_INPUTS};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheck;
use crate::zkvm::spartan::product::ProductVirtualizationSumcheck;
use crate::zkvm::spartan::shift::ShiftSumcheck;
use crate::zkvm::witness::VirtualPolynomial;

use crate::transcripts::Transcript;

use crate::subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof};

pub mod inner;
pub mod instruction_input;
pub mod product;
pub mod shift;

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
    fn stage1_prove(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        /* Sumcheck 1: Outer sumcheck
           Proves: \sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x)) = 0

           The matrices A, B, C have a block-diagonal structure with repeated blocks
           A_small, B_small, C_small corresponding to the uniform constraints.
        */
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        let key = self.key.clone();

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        let transcript = &mut *state_manager.transcript.borrow_mut();
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::<F, ProofTranscript>::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
                &preprocessing.shared,
                trace,
                num_rounds_x,
                &tau,
                transcript,
            );

        let outer_sumcheck_r: Vec<F::Challenge> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);

        // Store Az, Bz, Cz claims with the outer sumcheck point
        let accumulator = state_manager.get_prover_accumulator();
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[0],
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[1],
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            OpeningPoint::new(outer_sumcheck_r.clone()),
            outer_sumcheck_claims[2],
        );

        // Append the outer sumcheck proof to the state manager
        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage1Sumcheck,
            ProofData::SumcheckProof(outer_sumcheck_proof),
        );

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Compute claimed witness evals at r_cycle via streaming (fast)
        let claimed_witness_evals = {
            let _guard = span!(Level::INFO, "claimed_witness_evals_fast").entered();
            compute_claimed_witness_evals::<F>(&preprocessing.shared, trace, r_cycle)
        };

        // Add virtual polynomial evaluations to the accumulator
        // These are needed by the verifier for future sumchecks and are not part of the PCS opening proof
        for (input, eval) in ALL_R1CS_INPUTS.iter().zip(claimed_witness_evals.iter()) {
            accumulator.borrow_mut().append_virtual(
                transcript,
                input.into(),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
                *eval,
            );
        }

        Ok(())
    }

    fn stage1_verify(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = self.key.clone();

        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Get the outer sumcheck proof
        let proofs = state_manager.proofs.borrow();
        let proof_data = {
            proofs
                .get(&ProofKeys::Stage1Sumcheck)
                .expect("Outer sumcheck proof not found")
        };

        let outer_sumcheck_proof = match proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof data type"),
        };

        // Get the claims:
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let (_, claim_Az) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        drop(accumulator_ref);
        let outer_sumcheck_claims = [claim_Az, claim_Bz, claim_Cz];

        let transcript = &mut *state_manager.transcript.borrow_mut();

        // Run the main sumcheck verifier:
        let (claim_outer_final, outer_sumcheck_r_original) = {
            match outer_sumcheck_proof.verify(F::zero(), num_rounds_x, 3, transcript) {
                Ok(result) => result,
                Err(_) => return Err(anyhow::anyhow!("Outer sumcheck verification failed")),
            }
        };

        // Outer sumcheck is bound from the top, reverse the challenge
        // TODO(markosg04): Make use of Endianness here?
        let outer_sumcheck_r_reversed: Vec<F::Challenge> =
            outer_sumcheck_r_original.iter().rev().cloned().collect();
        let opening_point = OpeningPoint::new(outer_sumcheck_r_reversed.clone());

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims[..]);

        // Populate the opening points for Az, Bz, Cz claims now that we have outer_sumcheck_r
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::SpartanCz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
        );

        let tau_bound_rx = EqPolynomial::<F>::mle(&tau, &outer_sumcheck_r_reversed);
        let claim_outer_final_expected = tau_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(anyhow::anyhow!("Invalid outer sumcheck claim"));
        }

        // Add the commitments to verifier accumulator
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        let (r_cycle, _rx_var) = outer_sumcheck_r_reversed.split_at(num_cycles_bits);

        let accumulator = state_manager.get_verifier_accumulator();

        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.borrow_mut().append_virtual(
                transcript,
                input.into(),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });

        Ok(())
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
        let pc_sumcheck = ShiftSumcheck::<F>::new_prover(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_prover(state_manager);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan ShiftSumcheck", &pc_sumcheck);
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
        let pc_sumcheck = ShiftSumcheck::<F>::new_verifier(state_manager, key);
        let instruction_input_sumcheck = InstructionInputSumcheck::new_verifier(state_manager);
        vec![Box::new(pc_sumcheck), Box::new(instruction_input_sumcheck)]
    }
}
