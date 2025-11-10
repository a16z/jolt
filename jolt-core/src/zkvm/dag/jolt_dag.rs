use std::collections::HashMap;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::write_flamegraph_svg;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transpose;
use crate::zkvm::bytecode;
use crate::zkvm::bytecode::read_raf_checking::ReadRafSumcheckVerifier as BytecodeReadRafSumcheckVerifier;
use crate::zkvm::bytecode::BytecodeDagProver;
use crate::zkvm::dag::proof_serialization::Claims;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::dag::stage::SumcheckStagesProver;
use crate::zkvm::dag::state_manager::fiat_shamir_preamble;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction_lookups;
use crate::zkvm::instruction_lookups::ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier;
use crate::zkvm::instruction_lookups::read_raf_checking::ReadRafSumcheckVerifier as LookupsReadRafSumcheckVerifier;
use crate::zkvm::instruction_lookups::LookupsDagProver;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::ram;
use crate::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheckVerifier;
use crate::zkvm::ram::output_check::OutputSumcheckVerifier;
use crate::zkvm::ram::output_check::ValFinalSumcheckVerifier;
use crate::zkvm::ram::ra_virtual::RaSumcheckVerifier as RamRaSumcheckVerifier;
use crate::zkvm::ram::raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier;
use crate::zkvm::ram::read_write_checking::RamReadWriteCheckingVerifier;
use crate::zkvm::ram::val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier;
use crate::zkvm::ram::verifier_accumulate_advice;
use crate::zkvm::ram::RamDagProver;
use crate::zkvm::registers::read_write_checking::RegistersReadWriteCheckingVerifier;
use crate::zkvm::registers::val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier;
use crate::zkvm::registers::RegistersDagProver;
use crate::zkvm::spartan::inner::InnerSumcheckVerifier;
use crate::zkvm::spartan::instruction_input::InstructionInputSumcheckVerifier;
use crate::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
use crate::zkvm::spartan::product::ProductVirtualInnerVerifier;
use crate::zkvm::spartan::product::ProductVirtualRemainderVerifier;
use crate::zkvm::spartan::shift::ShiftSumcheckVerifier;
use crate::zkvm::spartan::verify_stage1_uni_skip;
use crate::zkvm::spartan::verify_stage2_uni_skip;
use crate::zkvm::spartan::SpartanDagProver;
use crate::zkvm::witness::{AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K};
use crate::zkvm::JoltVerifierPreprocessing;
use crate::zkvm::ProverDebugInfo;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use anyhow::Context;
use itertools::Itertools;
use rayon::prelude::*;
use tracer::instruction::Cycle;
use tracer::ChunksIterator;
use tracer::JoltDevice;

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
    // Initialize DoryGlobals at the beginning to keep it alive for the entire proof
    let (preprocessing, _, trace, program_io, _) = state_manager.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();

    fiat_shamir_preamble(program_io, state_manager.ram_K, trace_length, transcript);

    tracing::info!("bytecode size: {}", preprocessing.bytecode.code_size);

    let bytecode_d = preprocessing.bytecode.d;

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

    tracing::info!("Stage 1 proving");
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
        bytecode_d: prover_state.preprocessing.bytecode.d,
        twist_sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
    };

    Ok((proof, debug_info))
}

pub struct DagVerifier<
    'a,
    'b,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    pub transcript: &'a mut ProofTranscript,
    pub preprocessing: &'b JoltVerifierPreprocessing<F, PCS>,
}

impl<'a, 'b, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    DagVerifier<'a, 'b, F, ProofTranscript, PCS>
{
    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            self.transcript,
        );

        let ram_K = self.proof.ram_K;
        let bytecode_d = self.preprocessing.bytecode.d;
        let _guard = AllCommittedPolynomials::initialize(ram_K, bytecode_d);

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript.append_serializable(commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(trusted_advice_commitment);
        }

        let spartan_key = UniformSpartanKey::new(self.proof.trace_length.next_power_of_two());

        self.verify_stage1(&spartan_key)?;
        self.verify_stage2(&spartan_key)?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        self.verify_stage6()?;
        self.verify_trusted_advice_opening_proofs()?;
        self.verify_untrusted_advice_opening_proofs()?;
        self.verify_stage7()?;

        Ok(())
    }

    fn verify_stage1(&mut self, spartan_key: &UniformSpartanKey<F>) -> Result<(), anyhow::Error> {
        let spartan_outer_uni_skip_state = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            spartan_key,
            self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let n_cycle_vars = self.proof.trace_length.log_2();
        let spartan_outer_remaining =
            OuterRemainingSumcheckVerifier::new(n_cycle_vars, &spartan_outer_uni_skip_state);

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    fn verify_stage2(&mut self, spartan_key: &UniformSpartanKey<F>) -> Result<(), anyhow::Error> {
        let product_virtual_uni_skip_state = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            spartan_key,
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_inner = InnerSumcheckVerifier::new(spartan_key, self.transcript);
        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length.log_2(),
            &product_virtual_uni_skip_state,
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io,
            self.proof.ram_K,
            &self.opening_accumulator,
        );
        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            self.proof.ram_K,
            self.proof.trace_length,
            self.proof.twist_sumcheck_switch_index,
            &self.opening_accumulator,
            self.transcript,
        );
        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, self.transcript);

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &spartan_inner as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &spartan_product_virtual_remainder,
                &ram_raf_evaluation,
                &ram_read_write_checking,
                &ram_output_check,
            ],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
        let spartan_shift_sumcheck = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, self.transcript);
        let spartan_product_virtual_claim_check =
            ProductVirtualInnerVerifier::new(&self.opening_accumulator, self.transcript);
        let lookups_ra_hamming_weight =
            instruction_lookups::new_ra_hamming_weight_verifier(self.transcript);

        let _r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            vec![
                &spartan_shift_sumcheck as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &spartan_instruction_input,
                &spartan_product_virtual_claim_check,
                &lookups_ra_hamming_weight,
            ],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.twist_sumcheck_switch_index,
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            self.transcript,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            self.transcript,
        );
        let ram_ra_booleanity = ram::new_ra_booleanity_verifier(
            self.proof.ram_K,
            self.proof.trace_length.log_2(),
            self.transcript,
        );
        let initial_ram_state = ram::gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.ram,
            &self.program_io,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_ra_booleanity,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let registers_val_evaluation = RegistersValEvaluationSumcheckVerifier::new(n_cycle_vars);
        let ram_hamming_booleanity = HammingBooleanitySumcheckVerifier::new(n_cycle_vars);
        let ram_ra_virtual = RamRaSumcheckVerifier::new(
            self.proof.trace_length,
            self.proof.ram_K,
            &self.opening_accumulator,
            self.transcript,
        );
        let lookups_read_raf = LookupsReadRafSumcheckVerifier::new(n_cycle_vars, self.transcript);

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_hamming_booleanity,
                &ram_ra_virtual,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
            &self.preprocessing.bytecode,
            n_cycle_vars,
            &self.opening_accumulator,
            self.transcript,
        );
        let (bytecode_hamming_weight, bytecode_booleanity) = bytecode::new_ra_one_hot_verifiers(
            &self.preprocessing.bytecode,
            n_cycle_vars,
            self.transcript,
        );
        let ram_hamming_weight =
            ram::new_ra_hamming_weight_verifier(self.proof.ram_K, self.transcript);
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(&self.opening_accumulator);
        let lookups_booleanity =
            instruction_lookups::new_ra_booleanity_verifier(n_cycle_vars, self.transcript);

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            vec![
                &bytecode_read_raf as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &bytecode_hamming_weight,
                &bytecode_booleanity,
                &ram_hamming_weight,
                &lookups_ra_virtual,
                &lookups_booleanity,
            ],
            &mut self.opening_accumulator,
            self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    fn verify_trusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        if let Some(ref commitment) = self.trusted_advice_commitment {
            let Some(ref proof) = self.proof.trusted_advice_proof else {
                return Err(anyhow::anyhow!("Trusted advice proof not found"));
            };
            let Some((point, eval)) = self.opening_accumulator.get_trusted_advice_opening() else {
                return Err(anyhow::anyhow!("Trusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Trusted advice opening proof verification failed: {e:?}")
            })?;
        }

        Ok(())
    }

    fn verify_untrusted_advice_opening_proofs(&mut self) -> Result<(), anyhow::Error> {
        // Verify untrusted_advice opening proofs
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            let Some(ref proof) = self.proof.untrusted_advice_proof else {
                return Err(anyhow::anyhow!("Untrusted advice proof not found"));
            };
            let Some((point, eval)) = self.opening_accumulator.get_untrusted_advice_opening()
            else {
                return Err(anyhow::anyhow!("Untrusted advice opening not found"));
            };
            PCS::verify(
                proof,
                &self.preprocessing.generators,
                self.transcript,
                &point.r,
                &eval,
                commitment,
            )
            .map_err(|e| {
                anyhow::anyhow!("Untrusted advice opening proof verification failed: {e:?}")
            })?;
        }

        Ok(())
    }

    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Batch-prove all openings (Stage 7)
        let mut commitments_map = HashMap::new();
        for (polynomial, commitment) in
            AllCommittedPolynomials::iter().zip_eq(&self.proof.commitments)
        {
            commitments_map.insert(*polynomial, commitment.clone());
        }

        self.opening_accumulator
            .reduce_and_verify(
                &self.preprocessing.generators,
                &mut commitments_map,
                &self.proof.reduced_opening_proof,
                self.transcript,
            )
            .context("Stage 7")?;

        Ok(())
    }
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

fn compute_trusted_advice_poly<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
    state_manager: &mut StateManager<'a, F, PCS>,
) {
    let (_, _, _, program_io, _) = state_manager.get_prover_data();

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
