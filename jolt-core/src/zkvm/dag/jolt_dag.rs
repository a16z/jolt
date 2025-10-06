use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
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
use crate::zkvm::ram::{remap_address, RamDag};
use crate::zkvm::registers::RegistersDag;
use crate::zkvm::spartan::SpartanDag;
use crate::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use crate::zkvm::ProverDebugInfo;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use anyhow::Context;
use rayon::prelude::*;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use common::jolt_device::JoltDevice;

/// Commits to the private inputs of a JoltDevice using the Dory commitment scheme.
/// 
/// This function extracts the private inputs from the JoltDevice, converts them into
/// a multilinear polynomial, and commits to them using the specified commitment scheme.
///
/// # Arguments
/// * `jolt_device` - The JoltDevice containing the private inputs to commit to
/// * `setup` - The PCS prover setup
/// * `ram_K` - The RAM size parameter K (polynomial will have log2(K) variables)
///
/// # Returns
/// A tuple containing:
/// * The multilinear polynomial of private inputs
/// * The commitment to the private inputs
/// * The opening proof hint (for later use in opening proofs)
fn commit_to_private_inputs<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    jolt_device: &JoltDevice,
    setup: &PCS::ProverSetup,
    ram_K: usize,
) -> (MultilinearPolynomial<F>, PCS::Commitment, PCS::OpeningProofHint) {
    tracing::info!("commit_to_private_inputs: ram_K = {}", ram_K);
    
    let mut initial_memory_state = vec![0; ram_K];

    let mut index = remap_address(
        jolt_device.memory_layout.private_input_start,
        &jolt_device.memory_layout,
    )
    .unwrap() as usize;
    
    tracing::info!("Private input start index = {}", index);
    tracing::info!("Private input size = {} bytes", jolt_device.private_inputs.len());

    // Convert input bytes into words and populate
    // `initial_memory_state` and `final_memory_state`
    for chunk in jolt_device.private_inputs.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = u64::from_le_bytes(word);
        initial_memory_state[index] = word;
        index += 1;
    }

    
    // for chunk in jolt_device.private_inputs.chunks(8) {
    //     let mut word = [0u8; 8];
    //     for (i, byte) in chunk.iter().enumerate() {
    //         word[i] = *byte;
    //     }
    //     let word = u64::from_le_bytes(word);
    //     coeffs.push(F::from_u64(word));
    // }
    
    // // Pad with zeros to reach the required size (K)
    // // The polynomial needs K coefficients to have log2(K) variables
    // while coeffs.len() < ram_K {
    //     coeffs.push(F::zero());
    // }
    
    // // Ensure we don't exceed K
    // coeffs.truncate(ram_K);
    
    // Pad the initial memory state to match Dory's expected matrix dimensions
    // Dory views polynomials as matrices with fixed dimensions based on its global initialization
    let num_columns = DoryGlobals::get_num_columns();
    let num_rows = DoryGlobals::get_max_num_rows();
    let expected_size = num_columns * num_rows;
    
    tracing::info!("Dory expects matrix: {} rows x {} cols = {} total", num_rows, num_columns, expected_size);
    tracing::info!("Initial state size before padding: {}", initial_memory_state.len());
    
    if initial_memory_state.len() < expected_size {
        initial_memory_state.resize(expected_size, 0u64);
        tracing::info!("Padded initial state to size: {}", initial_memory_state.len());
    }
    
    // Create a multilinear polynomial from the padded field elements
    let poly = MultilinearPolynomial::from(initial_memory_state);
    tracing::info!("Polynomial len: {}, num_vars: {}", poly.len(), poly.get_num_vars());
    
    tracing::info!(
        "Created private input polynomial with {} coefficients (log2({}) = {} variables)",
        poly.len(),
        poly.len(),
        poly.get_num_vars()
    );
    
    // Commit to the polynomial using the existing setup that's already configured for Dory
    let (commitment, hint) = PCS::commit(&poly, setup);
    
    (poly, commitment, hint)
}

pub enum JoltDAG {}

impl JoltDAG {
    #[allow(clippy::type_complexity)]
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

        let (preprocessing, _, program_io, _) = state_manager.get_prover_data();
        let (private_input_poly, private_input_commitment, private_input_hint) = 
            commit_to_private_inputs::<F, PCS>(program_io, &preprocessing.generators, state_manager.ram_K);
        
        // Store the polynomial and hint in the prover state for later evaluation
        if let Some(ref mut prover_state) = state_manager.prover_state {
            prover_state.private_input_polynomial = Some(private_input_poly);
            prover_state.private_input_hint = Some(private_input_hint);
        }
        
        // Append the private input commitment to the transcript for Fiat-Shamir
        state_manager
            .get_transcript()
            .borrow_mut()
            .append_serializable(&private_input_commitment);
        
        // Store the commitment in the state manager for later use in the proof
        state_manager.private_input_commitment = Some(private_input_commitment);

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
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage2_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage2_start_flamechart.svg");
        }

        let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
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

        let stage3_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
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
            .chain(ram_dag.stage4_prover_instances(&mut state_manager))
            .chain(bytecode_dag.stage4_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage4_prover_instances(&mut state_manager))
            .collect();

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            for sumcheck in stage4_instances.iter() {
                sumcheck.update_flamegraph(&mut flamegraph);
            }
            write_flamegraph_svg(flamegraph, "stage4_start_flamechart.svg");
        }

        let stage4_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
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

        // Batch-prove all openings
        let (_, trace, _, _) = state_manager.get_prover_data();

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let polynomials_map =
            CommittedPolynomial::generate_witness_batch(&all_polys, preprocessing, trace);

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Committed polynomials map", &polynomials_map);

        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");

        tracing::info!("Stage 5 proving");
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
        
        // Append private input commitment to transcript for Fiat-Shamir
        if let Some(ref private_input_commitment) = state_manager.private_input_commitment {
            transcript.borrow_mut().append_serializable(private_input_commitment);
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
            .chain(registers_dag.stage2_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage2_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage2_verifier_instances(&mut state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
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
            .chain(registers_dag.stage3_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut state_manager))
            .chain(ram_dag.stage3_verifier_instances(&mut state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
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
            .chain(ram_dag.stage4_verifier_instances(&mut state_manager))
            .chain(bytecode_dag.stage4_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage4_verifier_instances(&mut state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
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

        // Batch-prove all openings
        let batched_opening_proof = proofs
            .get(&ProofKeys::ReducedOpeningProof)
            .expect("Reduced opening proof not found");
        let batched_opening_proof = match batched_opening_proof {
            ProofData::ReducedOpeningProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
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
            .context("Stage 5")?;

        Ok(())
    }

    // Prover utility to commit to all the polynomials for the PCS
    #[tracing::instrument(skip_all)]
    fn generate_and_commit_polynomials<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        prover_state_manager: &mut StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>, anyhow::Error> {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            prover_state_manager.get_prover_data();

        let mut all_polys = CommittedPolynomial::generate_witness_batch(
            &AllCommittedPolynomials::iter().copied().collect::<Vec<_>>(),
            preprocessing,
            trace,
        );
        let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
            .filter_map(|poly| all_polys.remove(poly))
            .collect();

        let (commitments, hints): (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) =
            committed_polys
                .par_iter()
                .map(|poly| PCS::commit(poly, &preprocessing.generators))
                .unzip();
        let mut hint_map = HashMap::with_capacity(committed_polys.len());
        for (poly, hint) in AllCommittedPolynomials::iter().zip(hints) {
            hint_map.insert(*poly, hint);
        }

        prover_state_manager.set_commitments(commitments);

        drop_in_background_thread(committed_polys);

        Ok(hint_map)
    }
}
