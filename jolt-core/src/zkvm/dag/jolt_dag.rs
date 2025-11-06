use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::transcripts::Transcript;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::write_flamegraph_svg;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transpose;
use crate::zkvm::bytecode::BytecodeDagProver;
use crate::zkvm::bytecode::BytecodeDagVerifier;
use crate::zkvm::dag::proof_serialization::Claims;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::dag::stage::SumcheckStagesProver;
use crate::zkvm::dag::stage::SumcheckStagesVerifier;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction_lookups::LookupsDagProver;
use crate::zkvm::instruction_lookups::LookupsDagVerifier;
use crate::zkvm::ram::{populate_memory_states, RamDagProver, RamDagVerifier};
use crate::zkvm::registers::RegistersDagProver;
use crate::zkvm::registers::RegistersDagVerifier;
use crate::zkvm::spartan::SpartanDagProver;
use crate::zkvm::spartan::SpartanDagVerifier;
use crate::zkvm::witness::{AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K};
use crate::zkvm::ProverDebugInfo;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use anyhow::Context;
use itertools::Itertools;
use rayon::prelude::*;
use tracer::instruction::Cycle;
use tracer::ChunksIterator;

#[allow(clippy::type_complexity)]
#[tracing::instrument(skip_all)]
pub fn prove_jolt_dag<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: StreamingCommitmentScheme<Field = F>,
>(
    mut state_manager: StateManager<'_, F, PCS>,
    mut opening_accumulator: ProverOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
) -> Result<
    (
        JoltProof<F, PCS, ProofTranscript>,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ),
    anyhow::Error,
> {
    state_manager.fiat_shamir_preamble(transcript);

    // Initialize DoryGlobals at the beginning to keep it alive for the entire proof
    let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();

    tracing::info!("bytecode size: {}", preprocessing.shared.bytecode.code_size);

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
            let hints = commit_untrusted_advice(&mut state_manager);
            Some(hints)
        } else {
            None
        };

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(state_manager.ram_K, bytecode_d),
    );

    // Generate and commit to all witness polynomials
    let (commitments, opening_proof_hints) = generate_and_commit_polynomials(&mut state_manager);

    // Append commitments to transcript
    for commitment in &commitments {
        transcript.append_serializable(commitment);
    }

    // Append untrusted_advice commitment to transcript if it exists
    if let Some(ref untrusted_advice_commitment) = state_manager.untrusted_advice_commitment {
        transcript.append_serializable(untrusted_advice_commitment);
    }

    if !state_manager.program_io.trusted_advice.is_empty() {
        compute_trusted_advice_poly(&mut state_manager);
        transcript.append_serializable(state_manager.trusted_advice_commitment.as_ref().unwrap());
    }

    // Stage 1:
    #[cfg(not(target_arch = "wasm32"))]
    print_current_memory_usage("Stage 1 baseline");
    let span = tracing::span!(tracing::Level::INFO, "Stage 1 sumchecks");
    let _guard = span.enter();

    let (_, _, trace, _, _) = state_manager.get_prover_data();
    let padded_trace_length = trace.len().next_power_of_two();
    let mut spartan_dag = SpartanDagProver::new(padded_trace_length);
    let mut lookups_dag = LookupsDagProver::new();
    let mut registers_dag = RegistersDagProver;
    let mut ram_dag = RamDagProver::new(&state_manager);
    let mut bytecode_dag = BytecodeDagProver;

    tracing::info!("Stage 1 proving (univariate skip first round)");
    let stage1_uni_skip_first_round_proof =
        spartan_dag.stage1_uni_skip(&mut state_manager, &mut opening_accumulator, transcript);

    // Batch the stage1 remainder instances (outer-remaining + extras)
    let mut remainder_instances: Vec<_> = spartan_dag
        .stage1_instances(&mut state_manager, &mut opening_accumulator, transcript)
        .into_iter()
        .collect();
    let remainder_instances_mut = remainder_instances
        .iter_mut()
        .map(|instance| &mut **instance as _)
        .collect();

    tracing::info!("Stage 1 proving (remainder batch)");
    let (stage1_sumcheck_proof, _r_stage1) = BatchedSumcheck::prove(
        remainder_instances_mut,
        &mut opening_accumulator,
        transcript,
    );

    drop(_guard);
    drop(span);

    // Stage 2:
    #[cfg(not(target_arch = "wasm32"))]
    print_current_memory_usage("Stage 2 baseline");
    let span = tracing::span!(tracing::Level::INFO, "Stage 2 sumchecks");
    let _guard = span.enter();

    // Stage 2a: Prove univariate-skip first round for product virtualization
    let stage2_uni_skip_first_round_proof =
        spartan_dag.stage2_uni_skip(&mut state_manager, &mut opening_accumulator, transcript);

    let mut stage2_instances: Vec<_> = std::iter::empty()
        .chain(spartan_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(registers_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage2_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(bytecode_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();

    #[cfg(feature = "allocative")]
    {
        let mut flamegraph = FlameGraphBuilder::default();
        for sumcheck in stage2_instances.iter() {
            sumcheck.update_flamegraph(&mut flamegraph);
        }
        write_flamegraph_svg(flamegraph, "stage2_start_flamechart.svg");
    }

    let stage2_instances_mut = stage2_instances
        .iter_mut()
        .map(|instance| &mut **instance as _)
        .collect();

    tracing::info!("Stage 2 proving");
    let (stage2_sumcheck_proof, _r_stage2) =
        BatchedSumcheck::prove(stage2_instances_mut, &mut opening_accumulator, transcript);

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
        .chain(spartan_dag.stage3_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(registers_dag.stage3_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(lookups_dag.stage3_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage3_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .collect();

    #[cfg(feature = "allocative")]
    {
        let mut flamegraph = FlameGraphBuilder::default();
        for sumcheck in stage3_instances.iter() {
            sumcheck.update_flamegraph(&mut flamegraph);
        }
        write_flamegraph_svg(flamegraph, "stage3_start_flamechart.svg");
    }

    let stage3_instances_mut = stage3_instances
        .iter_mut()
        .map(|instance| &mut **instance as _)
        .collect();

    tracing::info!("Stage 3 proving");
    let (stage3_sumcheck_proof, _r_stage3) =
        BatchedSumcheck::prove(stage3_instances_mut, &mut opening_accumulator, transcript);

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
        .chain(registers_dag.stage4_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage4_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .collect();

    #[cfg(feature = "allocative")]
    {
        let mut flamegraph = FlameGraphBuilder::default();
        for sumcheck in stage4_instances.iter() {
            sumcheck.update_flamegraph(&mut flamegraph);
        }
        write_flamegraph_svg(flamegraph, "stage4_start_flamechart.svg");
    }

    let stage4_instances_mut = stage4_instances
        .iter_mut()
        .map(|instance| &mut **instance as _)
        .collect();

    tracing::info!("Stage 4 proving");
    let (stage4_sumcheck_proof, _r_stage4) =
        BatchedSumcheck::prove(stage4_instances_mut, &mut opening_accumulator, transcript);

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
        .chain(registers_dag.stage5_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage5_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage5_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();

    #[cfg(feature = "allocative")]
    {
        let mut flamegraph = FlameGraphBuilder::default();
        for sumcheck in stage5_instances.iter() {
            sumcheck.update_flamegraph(&mut flamegraph);
        }
        write_flamegraph_svg(flamegraph, "stage5_start_flamechart.svg");
    }

    let stage5_instances_mut = stage5_instances
        .iter_mut()
        .map(|instance| &mut **instance as _)
        .collect();

    tracing::info!("Stage 5 proving");
    let (stage5_sumcheck_proof, _r_stage5) =
        BatchedSumcheck::prove(stage5_instances_mut, &mut opening_accumulator, transcript);

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
        .chain(bytecode_dag.stage6_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage6_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage6_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();

    #[cfg(feature = "allocative")]
    {
        let mut flamegraph = FlameGraphBuilder::default();
        for sumcheck in stage6_instances.iter() {
            sumcheck.update_flamegraph(&mut flamegraph);
        }
        write_flamegraph_svg(flamegraph, "stage6_start_flamechart.svg");
    }

    let stage6_instances_mut = stage6_instances
        .iter_mut()
        .map(|inst| &mut **inst as &mut _)
        .collect();

    tracing::info!("Stage 6 proving");
    let (stage6_sumcheck_proof, _r_stage6) =
        BatchedSumcheck::prove(stage6_instances_mut, &mut opening_accumulator, transcript);

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
    let (_, _, trace, _, _) = state_manager.get_prover_data();

    let all_polys: Vec<CommittedPolynomial> = AllCommittedPolynomials::iter().copied().collect();
    let polynomials_map =
        CommittedPolynomial::generate_witness_batch(&all_polys, preprocessing, trace);

    #[cfg(feature = "allocative")]
    print_data_structure_heap_usage("Committed polynomials map", &polynomials_map);

    #[cfg(not(target_arch = "wasm32"))]
    print_current_memory_usage("Stage 7 baseline");

    tracing::info!("Stage 7 proving");

    // Generate trusted_advice opening proofs
    let trusted_advice_proof = (!state_manager.program_io.trusted_advice.is_empty()).then(|| {
        generate_trusted_advice_proof(
            &mut state_manager,
            &opening_accumulator,
            transcript,
            &preprocessing.generators,
        )
    });

    // Generate untrusted_advice opening proofs
    let untrusted_advice_proof =
        (!state_manager.program_io.untrusted_advice.is_empty()).then(|| {
            generate_untrusted_advice_proof(
                &mut state_manager,
                &opening_accumulator,
                transcript,
                &preprocessing.generators,
            )
        });

    let reduced_opening_proof = opening_accumulator.reduce_and_prove(
        polynomials_map,
        opening_proof_hints,
        &preprocessing.generators,
        transcript,
    );

    #[cfg(test)]
    assert!(
        opening_accumulator
            .appended_virtual_openings
            .borrow()
            .is_empty(),
        "Not all virtual openings have been proven, missing: {:?}",
        opening_accumulator.appended_virtual_openings.borrow()
    );

    #[cfg(test)]
    let debug_info = Some(ProverDebugInfo {
        transcript: transcript.clone(),
        opening_accumulator: opening_accumulator.clone(),
        prover_setup: preprocessing.generators.clone(),
    });
    #[cfg(not(test))]
    let debug_info = None;

    let prover_state = state_manager.prover_state.as_mut().unwrap();
    let proof = JoltProof {
        opening_claims: Claims(opening_accumulator.openings),
        commitments,
        untrusted_advice_commitment: state_manager.untrusted_advice_commitment,
        stage1_uni_skip_first_round_proof,
        stage1_sumcheck_proof,
        stage2_uni_skip_first_round_proof,
        stage2_sumcheck_proof,
        stage3_sumcheck_proof,
        stage4_sumcheck_proof,
        stage5_sumcheck_proof,
        stage6_sumcheck_proof,
        trusted_advice_proof,
        untrusted_advice_proof,
        reduced_opening_proof,
        trace_length: prover_state.trace.len(),
        ram_K: state_manager.ram_K,
        bytecode_d: prover_state.preprocessing.shared.bytecode.d,
        twist_sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
    };

    Ok((proof, debug_info))
}

#[tracing::instrument(skip_all)]
pub fn verify_jolt_dag<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    proof: &JoltProof<F, PCS, ProofTranscript>,
    mut state_manager: StateManager<'a, F, PCS>,
    mut opening_accumulator: VerifierOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
) -> Result<(), anyhow::Error> {
    state_manager.fiat_shamir_preamble(transcript);

    let ram_K = state_manager.ram_K;
    let bytecode_d = state_manager.get_verifier_data().0.shared.bytecode.d;
    let _guard = AllCommittedPolynomials::initialize(ram_K, bytecode_d);

    // Append commitments to transcript
    for commitment in &proof.commitments {
        transcript.append_serializable(commitment);
    }

    // Append untrusted advice commitment to transcript
    if let Some(ref untrusted_advice_commitment) = state_manager.untrusted_advice_commitment {
        transcript.append_serializable(untrusted_advice_commitment);
    }
    // Append trusted advice commitment to transcript
    if let Some(ref trusted_advice_commitment) = state_manager.trusted_advice_commitment {
        transcript.append_serializable(trusted_advice_commitment);
    }

    // Initialize Dags
    let (preprocessing, _, trace_length) = state_manager.get_verifier_data();
    let padded_trace_length = trace_length.next_power_of_two();
    let mut spartan_dag = SpartanDagVerifier::new(padded_trace_length);
    let mut lookups_dag = LookupsDagVerifier;
    let mut registers_dag = RegistersDagVerifier;
    let mut ram_dag = RamDagVerifier::new(&state_manager);
    let mut bytecode_dag = BytecodeDagVerifier;

    // Stage 1:
    spartan_dag
        .stage1_uni_skip(&proof.stage1_uni_skip_first_round_proof, transcript)
        .context("Stage 1 univariate skip first round")?;

    let stage1_remainder_instances: Vec<_> = spartan_dag
        .stage1_instances(&mut state_manager, &mut opening_accumulator, transcript)
        .into_iter()
        .collect();
    let stage1_remainder_instances_ref = stage1_remainder_instances
        .iter()
        .map(|instance| &**instance as _)
        .collect();

    let _r_stage1 = BatchedSumcheck::verify(
        &proof.stage1_sumcheck_proof,
        stage1_remainder_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 1 remainder")?;

    // Stage 2:
    // Stage 2a: Verify univariate-skip first round for product virtualization
    spartan_dag
        .stage2_uni_skip(
            &proof.stage2_uni_skip_first_round_proof,
            &mut opening_accumulator,
            transcript,
        )
        .context("Stage 2 univariate skip first round")?;

    let stage2_instances: Vec<_> = std::iter::empty()
        .chain(spartan_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(registers_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage2_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(bytecode_dag.stage2_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();
    let stage2_instances_ref = stage2_instances.iter().map(|inst| &**inst as _).collect();

    let _r_stage2 = BatchedSumcheck::verify(
        &proof.stage2_sumcheck_proof,
        stage2_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 2")?;

    // Stage 3:
    let stage3_instances: Vec<_> = std::iter::empty()
        .chain(spartan_dag.stage3_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(lookups_dag.stage3_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage3_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .collect();
    let stage3_instances_ref = stage3_instances.iter().map(|inst| &**inst as _).collect();

    let _r_stage3 = BatchedSumcheck::verify(
        &proof.stage3_sumcheck_proof,
        stage3_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 3")?;

    // Stage 4:
    let stage4_instances: Vec<_> = std::iter::empty()
        .chain(registers_dag.stage4_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage4_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .collect();
    let stage4_instances_ref = stage4_instances
        .iter()
        .map(|instance| &**instance as _)
        .collect();

    let _r_stage4 = BatchedSumcheck::verify(
        &proof.stage4_sumcheck_proof,
        stage4_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 4")?;

    // Stage 5:
    let stage5_instances: Vec<_> = std::iter::empty()
        .chain(registers_dag.stage5_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage5_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage5_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();
    let stage5_instances_ref = stage5_instances.iter().map(|inst| &**inst as _).collect();

    let _r_stage5 = BatchedSumcheck::verify(
        &proof.stage5_sumcheck_proof,
        stage5_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 5")?;

    // Stage 6:
    let stage6_instances: Vec<_> = std::iter::empty()
        .chain(bytecode_dag.stage6_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .chain(ram_dag.stage6_instances(&mut state_manager, &mut opening_accumulator, transcript))
        .chain(lookups_dag.stage6_instances(
            &mut state_manager,
            &mut opening_accumulator,
            transcript,
        ))
        .collect();
    let stage6_instances_ref = stage6_instances.iter().map(|inst| &**inst as _).collect();

    let _r_stage6 = BatchedSumcheck::verify(
        &proof.stage6_sumcheck_proof,
        stage6_instances_ref,
        &mut opening_accumulator,
        transcript,
    )
    .context("Stage 6")?;

    // Verify trusted_advice opening proofs
    if let Some(ref commitment) = state_manager.trusted_advice_commitment {
        let Some(ref proof) = proof.trusted_advice_proof else {
            return Err(anyhow::anyhow!("Trusted advice proof not found"));
        };
        let Some((point, eval)) = opening_accumulator.get_trusted_advice_opening() else {
            return Err(anyhow::anyhow!("Trusted advice opening not found"));
        };
        PCS::verify(
            proof,
            &preprocessing.generators,
            transcript,
            &point.r,
            &eval,
            commitment,
        )
        .map_err(|e| anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}"))?;
    }

    // Verify untrusted_advice opening proofs
    if let Some(ref commitment) = state_manager.untrusted_advice_commitment {
        let Some(ref proof) = proof.untrusted_advice_proof else {
            return Err(anyhow::anyhow!("Untrusted advice proof not found"));
        };
        let Some((point, eval)) = opening_accumulator.get_untrusted_advice_opening() else {
            return Err(anyhow::anyhow!("Untrusted advice opening not found"));
        };
        PCS::verify(
            proof,
            &preprocessing.generators,
            transcript,
            &point.r,
            &eval,
            commitment,
        )
        .map_err(|e| {
            anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
        })?;
    }

    // Batch-prove all openings (Stage 7)
    let mut commitments_map = HashMap::new();
    for (polynomial, commitment) in AllCommittedPolynomials::iter().zip_eq(&proof.commitments) {
        commitments_map.insert(*polynomial, commitment.clone());
    }
    opening_accumulator
        .reduce_and_verify(
            &preprocessing.generators,
            &mut commitments_map,
            &proof.reduced_opening_proof,
            transcript,
        )
        .context("Stage 7")?;

    Ok(())
}

// Prover utility to commit to all the polynomials for the PCS
#[tracing::instrument(skip_all)]
fn generate_and_commit_polynomials<F: JoltField, PCS: StreamingCommitmentScheme<Field = F>>(
    prover_state_manager: &mut StateManager<F, PCS>,
) -> (
    Vec<PCS::Commitment>,
    HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
) {
    let (preprocessing, lazy_trace, _trace, _program_io, _final_memory_state) =
        prover_state_manager.get_prover_data();

    let T = DoryGlobals::get_T();

    let cached_data = PCS::prepare_cached_data(&preprocessing.generators);

    let polys: Vec<_> = AllCommittedPolynomials::iter().collect();
    let row_len = DoryGlobals::get_num_columns();
    let mut row_commitments: Vec<Vec<<PCS>::ChunkState>> =
        vec![vec![]; T / DoryGlobals::get_max_num_rows()];

    lazy_trace
        .as_ref()
        .expect("Lazy trace not found!")
        .clone()
        .pad_using(T, |_| Cycle::NoOp)
        .iter_chunks(row_len)
        .zip(row_commitments.iter_mut())
        .par_bridge()
        .for_each(|(chunk, row_commitments)| {
            let res: Vec<_> = polys
                .par_iter()
                .map(|poly| {
                    poly.generate_witness_and_commit_row::<_, PCS>(
                        &cached_data,
                        preprocessing,
                        &chunk,
                        prover_state_manager.ram_d,
                    )
                })
                .collect();
            *row_commitments = res;
        });

    let (commitments, hints): (Vec<_>, Vec<_>) = transpose(row_commitments)
        .into_par_iter()
        .zip(polys.into_par_iter())
        .map(|(rc, poly)| PCS::finalize(&cached_data, poly.get_onehot_k(preprocessing), &rc))
        .unzip();

    let mut hint_map = HashMap::with_capacity(AllCommittedPolynomials::len());
    for (poly, hint) in AllCommittedPolynomials::iter().zip(hints) {
        hint_map.insert(*poly, hint);
    }

    (commitments, hint_map)
}

fn commit_untrusted_advice<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
    state_manager: &mut StateManager<'a, F, PCS>,
) -> Option<PCS::OpeningProofHint> {
    let (preprocessing, _, _, program_io, _) = state_manager.get_prover_data();

    if program_io.untrusted_advice.is_empty() {
        return None;
    }

    let mut untrusted_advice_vec =
        vec![0; program_io.memory_layout.max_untrusted_advice_size as usize / 8];

    populate_memory_states(
        0,
        &program_io.untrusted_advice,
        Some(&mut untrusted_advice_vec),
        None,
    );

    let poly = MultilinearPolynomial::from(untrusted_advice_vec);
    let (commitment, hint) = PCS::commit(&poly, &preprocessing.generators);

    if let Some(ref mut prover_state) = state_manager.prover_state {
        prover_state.untrusted_advice_polynomial = Some(poly);
    }

    state_manager.untrusted_advice_commitment = Some(commitment);
    Some(hint)
}

fn compute_trusted_advice_poly<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
    state_manager: &mut StateManager<'a, F, PCS>,
) {
    let (_, _, _, program_io, _) = state_manager.get_prover_data();

    if program_io.trusted_advice.is_empty() {
        return;
    }

    let mut trusted_advice_vec =
        vec![0; program_io.memory_layout.max_trusted_advice_size as usize / 8];

    populate_memory_states(
        0,
        &program_io.trusted_advice,
        Some(&mut trusted_advice_vec),
        None,
    );

    let poly = MultilinearPolynomial::from(trusted_advice_vec);

    if let Some(ref mut prover_state) = state_manager.prover_state {
        prover_state.trusted_advice_polynomial = Some(poly);
    }
}

fn generate_trusted_advice_proof<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    state_manager: &mut StateManager<'_, F, PCS>,
    opening_accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
    generators: &PCS::ProverSetup,
) -> PCS::Proof {
    let prover_state = state_manager.prover_state.as_ref().unwrap();
    let trusted_advice_poly = prover_state.trusted_advice_polynomial.as_ref().unwrap();
    let (point, _) = opening_accumulator.get_trusted_advice_opening().unwrap();
    PCS::prove_without_hint(generators, trusted_advice_poly, &point.r, transcript)
}

fn generate_untrusted_advice_proof<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    state_manager: &mut StateManager<'_, F, PCS>,
    opening_accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
    generators: &PCS::ProverSetup,
) -> PCS::Proof {
    let prover_state = state_manager.prover_state.as_ref().unwrap();
    let untrusted_advice_poly = prover_state.untrusted_advice_polynomial.as_ref().unwrap();
    let (point, _) = opening_accumulator.get_untrusted_advice_opening().unwrap();
    PCS::prove_without_hint(generators, untrusted_advice_poly, &point.r, transcript)
}
