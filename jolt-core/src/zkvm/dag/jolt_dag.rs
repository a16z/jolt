use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance};
use crate::transcripts::Transcript;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::write_flamegraph_svg;
use crate::utils::thread::drop_in_background_thread;
use crate::zkvm::bytecode::BytecodeDag;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::dag::stage::SumcheckStages;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::instruction_lookups::LookupsDag;
use crate::zkvm::ram::RamDag;
use crate::zkvm::registers::RegistersDag;
use crate::zkvm::spartan::SpartanDag;
use crate::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use crate::zkvm::ProverDebugInfo;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use anyhow::Context;

pub enum JoltDAG {}

impl JoltDAG {
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "JoltDAG::prove")]
    pub fn prove<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        mut state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<
        (
            JoltProof<F, PCS, ProofTranscript>,
            Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
        ),
        anyhow::Error,
    > {
        state_manager.fiat_shamir_preamble();

        // Initialize DoryGlobals at the beginning to keep it alive for the entire proof
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();

        tracing::info!("bytecode size: {}", preprocessing.shared.bytecode.code_size);

        let ram_K = state_manager.ram_K;
        let bytecode_d = preprocessing.shared.bytecode.d;

        // Commit to untrusted_advice
        let _untrusted_advice_opening_proof_hints =
            if !state_manager.program_io.untrusted_advice.is_empty() {
                let _guard = DoryGlobals::initialize(
                    1,
                    state_manager
                        .program_io
                        .memory_layout
                        .max_untrusted_advice_size as usize
                        / 8,
                );
                let hints = Self::commit_untrusted_advice(&mut state_manager);
                Some(hints)
            } else {
                None
            };

        let _guard = (
            DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
            AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d),
        );

        // Generate and commit to all witness polynomials
        let opening_proof_hints = Self::generate_and_commit_polynomials(&mut state_manager)?;

        // Append commitments to transcript
        let commitments = state_manager.get_commitments();
        let transcript = state_manager.get_transcript();
        for commitment in commitments.borrow().iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }
        drop(commitments);

        // Append untrusted_advice commitment to transcript if it exists
        if let Some(ref untrusted_advice_commitment) = state_manager.untrusted_advice_commitment {
            transcript
                .borrow_mut()
                .append_serializable(untrusted_advice_commitment);
        }

        if !state_manager.program_io.trusted_advice.is_empty() {
            Self::compute_trusted_advice_poly(&mut state_manager);
            transcript
                .borrow_mut()
                .append_serializable(state_manager.trusted_advice_commitment.as_ref().unwrap());
        }

        // Stage 1:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 1 sumchecks");
        let _guard = span.enter();

        let (_, trace, _, _) = state_manager.get_prover_data();
        let padded_trace_length = trace.len().next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_prover(&state_manager);
        let mut bytecode_dag = BytecodeDag::default();

        tracing::info!("Stage 1 proving");
        spartan_dag
            .stage1_prove(&mut state_manager)
            .context("Stage 1")?;

        drop(_guard);
        drop(span);

        // Stage 2:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 2 sumchecks");
        let _guard = span.enter();

        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_prover_instances(&mut state_manager))
            .chain(registers_dag.stage2_prover_instances(&mut state_manager))
            .chain(ram_dag.stage2_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage2_prover_instances(&mut state_manager))
            .chain(bytecode_dag.stage2_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage2_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage2_start_flamechart.svg");
        }

        let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
            stage2_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, ProofTranscript>)
                .collect();

        let transcript = state_manager.get_transcript();
        let accumulator = state_manager.get_prover_accumulator();
        tracing::info!("Stage 2 proving");
        let (stage2_proof, _r_stage2) = BatchedSumcheck::prove(
            stage2_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2Sumcheck,
            ProofData::SumcheckProof(stage2_proof),
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage2_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage2_end_flamechart.svg");
        }

        drop_in_background_thread(stage2_instances);

        drop(_guard);
        drop(span);

        // Stage 3:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 3 sumchecks");
        let _guard = span.enter();

        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_prover_instances(&mut state_manager))
            .chain(registers_dag.stage3_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage3_prover_instances(&mut state_manager))
            .chain(ram_dag.stage3_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage3_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage3_start_flamechart.svg");
        }

        let stage3_instances_mut: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
            stage3_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, ProofTranscript>)
                .collect();

        tracing::info!("Stage 3 proving");
        let (stage3_proof, _r_stage3) = BatchedSumcheck::prove(
            stage3_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::SumcheckProof(stage3_proof),
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage3_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage3_end_flamechart.svg");
        }

        drop_in_background_thread(stage3_instances);

        drop(_guard);
        drop(span);

        // Stage 4:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 4 sumchecks");
        let _guard = span.enter();

        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(registers_dag.stage4_prover_instances(&mut state_manager))
            .chain(ram_dag.stage4_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage4_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage4_start_flamechart.svg");
        }

        let stage4_instances_mut: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
            stage4_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, ProofTranscript>)
                .collect();

        tracing::info!("Stage 4 proving");
        let (stage4_proof, _r_stage4) = BatchedSumcheck::prove(
            stage4_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage4Sumcheck,
            ProofData::SumcheckProof(stage4_proof),
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage4_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage4_end_flamechart.svg");
        }

        drop_in_background_thread(stage4_instances);

        drop(_guard);
        drop(span);

        // Stage 5:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 5 sumchecks");
        let _guard = span.enter();

        let mut stage5_instances: Vec<_> = std::iter::empty()
            .chain(registers_dag.stage5_prover_instances(&mut state_manager))
            .chain(ram_dag.stage5_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage5_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage5_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage5_start_flamechart.svg");
        }

        let stage5_instances_mut: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
            stage5_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, ProofTranscript>)
                .collect();

        tracing::info!("Stage 5 proving");
        let (stage5_proof, _r_stage5) = BatchedSumcheck::prove(
            stage5_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage5Sumcheck,
            ProofData::SumcheckProof(stage5_proof),
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage5_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage5_end_flamechart.svg");
        }

        drop_in_background_thread(stage5_instances);

        drop(_guard);
        drop(span);

        // Stage 6:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 6 sumchecks");
        let _guard = span.enter();

        let mut stage6_instances: Vec<_> = std::iter::empty()
            .chain(bytecode_dag.stage6_prover_instances(&mut state_manager))
            .chain(ram_dag.stage6_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage6_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage6_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage6_start_flamechart.svg");
        }

        let stage6_instances_mut: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>> =
            stage6_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, ProofTranscript>)
                .collect();

        tracing::info!("Stage 6 proving");
        let (stage6_proof, _r_stage6) = BatchedSumcheck::prove(
            stage6_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage6Sumcheck,
            ProofData::SumcheckProof(stage6_proof),
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage6_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage6_end_flamechart.svg");
        }

        drop_in_background_thread(stage6_instances);

        drop(_guard);
        drop(span);

        // Batch-prove all openings (Stage 7)
        let (_, trace, _, _) = state_manager.get_prover_data();

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let polynomials_map =
            CommittedPolynomial::generate_witness_batch(&all_polys, preprocessing, trace);

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Committed polynomials map", &polynomials_map);

        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 7 baseline");

        tracing::info!("Stage 7 proving");

        // Generate trusted_advice opening proofs
        if !state_manager.program_io.trusted_advice.is_empty() {
            let proof = Self::generate_trusted_advice_proof(
                &mut state_manager,
                &preprocessing.generators,
                &mut *transcript.borrow_mut(),
            );
            state_manager.proofs.borrow_mut().insert(
                ProofKeys::TrustedAdviceProof,
                ProofData::OpeningProof(proof),
            );
        }

        // Generate untrusted_advice opening proofs
        if !state_manager.program_io.untrusted_advice.is_empty() {
            let proof = Self::generate_untrusted_advice_proof(
                &mut state_manager,
                &preprocessing.generators,
                &mut *transcript.borrow_mut(),
            );
            state_manager.proofs.borrow_mut().insert(
                ProofKeys::UntrustedAdviceProof,
                ProofData::OpeningProof(proof),
            );
        }

        let opening_proof = accumulator.borrow_mut().reduce_and_prove(
            polynomials_map,
            opening_proof_hints,
            &preprocessing.generators,
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(opening_proof),
        );

        #[cfg(test)]
        assert!(
            state_manager
                .get_prover_accumulator()
                .borrow()
                .appended_virtual_openings
                .borrow()
                .is_empty(),
            "Not all virtual openings have been proven, missing: {:?}",
            state_manager
                .get_prover_accumulator()
                .borrow()
                .appended_virtual_openings
                .borrow()
        );

        #[cfg(test)]
        let debug_info = {
            let transcript = state_manager.transcript.take();
            let opening_accumulator = state_manager.get_prover_accumulator().borrow().clone();
            Some(ProverDebugInfo {
                transcript,
                opening_accumulator,
                prover_setup: preprocessing.generators.clone(),
            })
        };
        #[cfg(not(test))]
        let debug_info = None;

        let proof = JoltProof::from_prover_state_manager(state_manager);

        Ok((proof, debug_info))
    }

    #[tracing::instrument(skip_all, name = "JoltDAG::verify")]
    pub fn verify<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        mut state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        state_manager.fiat_shamir_preamble();

        let ram_K = state_manager.ram_K;
        let bytecode_d = state_manager.get_verifier_data().0.shared.bytecode.d;
        let _guard = AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d);

        // Append commitments to transcript
        let commitments = state_manager.get_commitments();
        let transcript = state_manager.get_transcript();
        for commitment in commitments.borrow().iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = state_manager.untrusted_advice_commitment {
            transcript
                .borrow_mut()
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = state_manager.trusted_advice_commitment {
            transcript
                .borrow_mut()
                .append_serializable(trusted_advice_commitment);
        }

        // Stage 1:
        let (preprocessing, _, trace_length) = state_manager.get_verifier_data();
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut registers_dag = RegistersDag::default();
        let mut ram_dag = RamDag::new_verifier(&state_manager);
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_verify(&mut state_manager)
            .context("Stage 1")?;

        // Stage 2:
        let stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage2_verifier_instances(&mut state_manager))
            .chain(bytecode_dag.stage2_verifier_instances(&mut state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage2_proof_data = proofs
            .get(&ProofKeys::Stage2Sumcheck)
            .expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = state_manager.get_transcript();
        let opening_accumulator = state_manager.get_verifier_accumulator();
        let _r_stage2 = BatchedSumcheck::verify(
            stage2_proof,
            stage2_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 2")?;

        drop(proofs);

        // Stage 3:
        let stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage3_verifier_instances(&mut state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage3_proof_data = proofs
            .get(&ProofKeys::Stage3Sumcheck)
            .expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let _r_stage3 = BatchedSumcheck::verify(
            stage3_proof,
            stage3_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 3")?;

        drop(proofs);

        // Stage 4:
        let stage4_instances: Vec<_> = std::iter::empty()
            .chain(registers_dag.stage4_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage4_verifier_instances(&mut state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage4_proof_data = proofs
            .get(&ProofKeys::Stage4Sumcheck)
            .expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let _r_stage4 = BatchedSumcheck::verify(
            stage4_proof,
            stage4_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 4")?;

        drop(proofs);

        // Stage 5:
        let stage5_instances: Vec<_> = std::iter::empty()
            .chain(registers_dag.stage5_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage5_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage5_verifier_instances(&mut state_manager))
            .collect();
        let stage5_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage5_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage5_proof_data = proofs
            .get(&ProofKeys::Stage5Sumcheck)
            .expect("Stage 5 sumcheck proof not found");
        let stage5_proof = match stage5_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 5"),
        };

        let _r_stage5 = BatchedSumcheck::verify(
            stage5_proof,
            stage5_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 5")?;

        drop(proofs);

        // Stage 6:
        let stage6_instances: Vec<_> = std::iter::empty()
            .chain(bytecode_dag.stage6_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage6_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage6_verifier_instances(&mut state_manager))
            .collect();
        let stage6_instances_ref: Vec<&dyn SumcheckInstance<F, ProofTranscript>> = stage6_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F, ProofTranscript>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage6_proof_data = proofs
            .get(&ProofKeys::Stage6Sumcheck)
            .expect("Stage 6 sumcheck proof not found");
        let stage6_proof = match stage6_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 6"),
        };

        let _r_stage6 = BatchedSumcheck::verify(
            stage6_proof,
            stage6_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 6")?;

        // Verify trusted_advice opening proofs
        if state_manager.trusted_advice_commitment.is_some() {
            Self::verify_trusted_advice_proofs(
                &state_manager,
                &preprocessing.generators,
                &mut *transcript.borrow_mut(),
            )
            .context("Trusted advice proofs")?;
        }

        // Verify untrusted_advice opening proofs
        if state_manager.untrusted_advice_commitment.is_some() {
            Self::verify_untrusted_advice_proofs(
                &state_manager,
                &preprocessing.generators,
                &mut *transcript.borrow_mut(),
            )
            .context("Untrusted advice proofs")?;
        }

        // Batch-prove all openings (Stage 7)
        let batched_opening_proof = proofs
            .get(&ProofKeys::ReducedOpeningProof)
            .expect("Reduced opening proof not found");
        let batched_opening_proof = match batched_opening_proof {
            ProofData::ReducedOpeningProof(proof) => proof,
            _ => panic!("Invalid proof type for opening reduction"),
        };

        let mut commitments_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            commitments_map.insert(
                *polynomial,
                commitments.borrow()[polynomial.to_index()].clone(),
            );
        }
        let accumulator = state_manager.get_verifier_accumulator();
        accumulator
            .borrow_mut()
            .reduce_and_verify(
                &preprocessing.generators,
                &mut commitments_map,
                batched_opening_proof,
                &mut *transcript.borrow_mut(),
            )
            .context("Stage 7")?;

        Ok(())
    }

    // Prover utility to commit to all the polynomials for the PCS
    #[tracing::instrument(skip_all)]
    fn generate_and_commit_polynomials<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        prover_state_manager: &mut StateManager<F, ProofTranscript, PCS>,
    ) -> Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>, anyhow::Error> {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            prover_state_manager.get_prover_data();

        let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
        let mut all_polys =
            CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);

        let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
            .filter_map(|poly| all_polys.remove(poly))
            .collect();

        let span = tracing::span!(tracing::Level::INFO, "commit to polynomials");
        let _guard = span.enter();

        let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);

        let (commitments, hints): (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) =
            commit_results.into_iter().unzip();
        drop(_guard);
        drop(span);
        let mut hint_map = HashMap::with_capacity(committed_polys.len());
        for (poly, hint) in AllCommittedPolynomials::iter().zip(hints) {
            hint_map.insert(*poly, hint);
        }

        prover_state_manager.set_commitments(commitments);

        drop_in_background_thread(committed_polys);

        Ok(hint_map)
    }

    fn commit_untrusted_advice<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &mut StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Option<PCS::OpeningProofHint> {
        let (preprocessing, _, program_io, _) = state_manager.get_prover_data();

        if program_io.untrusted_advice.is_empty() {
            return None;
        }

        let mut initial_memory_state =
            vec![0; program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        let mut index = 1;
        for chunk in program_io.untrusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            index += 1;
        }

        let poly = MultilinearPolynomial::from(initial_memory_state);
        let (commitment, hint) = PCS::commit(&poly, &preprocessing.generators);

        if let Some(ref mut prover_state) = state_manager.prover_state {
            prover_state.untrusted_advice_polynomial = Some(poly);
        }

        state_manager.untrusted_advice_commitment = Some(commitment);
        Some(hint)
    }

    fn compute_trusted_advice_poly<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &mut StateManager<'a, F, ProofTranscript, PCS>,
    ) {
        let (_, _, program_io, _) = state_manager.get_prover_data();

        if program_io.trusted_advice.is_empty() {
            return;
        }

        let mut initial_memory_state =
            vec![0; program_io.memory_layout.max_trusted_advice_size as usize / 8];

        let mut index = 1;
        for chunk in program_io.trusted_advice.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            initial_memory_state[index] = word;
            index += 1;
        }

        let poly = MultilinearPolynomial::from(initial_memory_state);

        if let Some(ref mut prover_state) = state_manager.prover_state {
            prover_state.trusted_advice_polynomial = Some(poly);
        }
    }

    fn generate_trusted_advice_proof<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        generators: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> PCS::Proof {
        let prover_state = state_manager.prover_state.as_ref().unwrap();
        let trusted_advice_poly = prover_state.trusted_advice_polynomial.as_ref().unwrap();
        let accumulator = state_manager.get_prover_accumulator();
        let (point, _) = accumulator.borrow().get_trusted_advice_opening().unwrap();
        PCS::prove_without_hint(generators, trusted_advice_poly, &point.r, transcript)
    }

    fn generate_untrusted_advice_proof<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        generators: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> PCS::Proof {
        let prover_state = state_manager.prover_state.as_ref().unwrap();
        let untrusted_advice_poly = prover_state.untrusted_advice_polynomial.as_ref().unwrap();
        let accumulator = state_manager.get_prover_accumulator();
        let (point, _) = accumulator.borrow().get_untrusted_advice_opening().unwrap();
        PCS::prove_without_hint(generators, untrusted_advice_poly, &point.r, transcript)
    }

    fn verify_trusted_advice_proofs<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), anyhow::Error> {
        let trusted_advice_commitment = state_manager.trusted_advice_commitment.as_ref().unwrap();
        let accumulator = state_manager.get_verifier_accumulator();

        let (point, eval) = accumulator.borrow().get_trusted_advice_opening().unwrap();
        let proof = match state_manager
            .proofs
            .borrow()
            .get(&ProofKeys::TrustedAdviceProof)
        {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Trusted advice proof not found")),
        };

        PCS::verify(
            &proof,
            verifier_setup,
            transcript,
            &point.r,
            &eval,
            trusted_advice_commitment,
        )
        .map_err(|e| anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}"))?;

        Ok(())
    }

    fn verify_untrusted_advice_proofs<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), anyhow::Error> {
        let untrusted_advice_commitment =
            state_manager.untrusted_advice_commitment.as_ref().unwrap();
        let accumulator = state_manager.get_verifier_accumulator();

        let (point, eval) = accumulator.borrow().get_untrusted_advice_opening().unwrap();
        let proof = match state_manager
            .proofs
            .borrow()
            .get(&ProofKeys::UntrustedAdviceProof)
        {
            Some(ProofData::OpeningProof(proof)) => proof.clone(),
            _ => return Err(anyhow::anyhow!("Untrusted advice proof not found")),
        };

        PCS::verify(
            &proof,
            verifier_setup,
            transcript,
            &point.r,
            &eval,
            untrusted_advice_commitment,
        )
        .map_err(|e| {
            anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
        })?;

        Ok(())
    }
}
