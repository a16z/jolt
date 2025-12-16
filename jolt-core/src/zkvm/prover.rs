use crate::subprotocols::streaming_schedule::LinearOnlySchedule;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::{print_data_structure_heap_usage, write_flamegraph_svg};
use crate::{
    field::JoltField,
    guest,
    poly::{
        commitment::{
            commitment_scheme::StreamingCommitmentScheme,
            dory::{DoryContext, DoryGlobals},
        },
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::ProverOpeningAccumulator,
        rlc_polynomial::RLCStreamingData,
    },
    pprof_scope,
    subprotocols::{
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round, UniSkipFirstRoundProof},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
    zkvm::{
        bytecode::read_raf_checking::ReadRafSumcheckParams as BytecodeReadRafParams,
        config::{get_log_k_chunk, OneHotParams},
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckParams,
            read_raf_checking::ReadRafSumcheckParams as InstructionReadRafParams,
        },
        ram::{
            hamming_booleanity::HammingBooleanityParams,
            output_check::OutputSumcheckParams,
            populate_memory_states,
            ra_reduction::{RaReductionParams, RamRaReductionSumcheckProver},
            ra_virtual::RamRaVirtualParams,
            raf_evaluation::RafEvaluationSumcheckParams,
            read_write_checking::RamReadWriteCheckingParams,
            val_evaluation::{
                ValEvaluationSumcheckParams,
                ValEvaluationSumcheckProver as RamValEvaluationSumcheckProver,
            },
            val_final::{ValFinalSumcheckParams, ValFinalSumcheckProver},
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingParams,
            val_evaluation::RegistersValEvaluationSumcheckParams,
        },
        spartan::{
            claim_reductions::{
                InstructionLookupsClaimReductionSumcheckParams,
                RegistersClaimReductionSumcheckParams, RegistersClaimReductionSumcheckProver,
            },
            instruction_input::InstructionInputParams,
            outer::{OuterUniSkipParams, OuterUniSkipProver},
            product::{
                ProductVirtualRemainderParams, ProductVirtualUniSkipParams,
                ProductVirtualUniSkipProver,
            },
            shift::ShiftSumcheckParams,
        },
        verifier::JoltVerifierPreprocessing,
        witness::all_committed_polynomials,
    },
};
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    zkvm::{
        bytecode::{
            self, read_raf_checking::ReadRafSumcheckProver as BytecodeReadRafSumcheckProver,
            BytecodePreprocessing,
        },
        fiat_shamir_preamble,
        instruction_lookups::{
            self, ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver,
            read_raf_checking::ReadRafSumcheckProver as LookupsReadRafSumcheckProver,
        },
        proof_serialization::{Claims, JoltProof},
        r1cs::key::UniformSpartanKey,
        ram::{
            self, gen_ram_memory_states, hamming_booleanity::HammingBooleanitySumcheckProver,
            output_check::OutputSumcheckProver, prover_accumulate_advice,
            ra_virtual::RamRaVirtualSumcheckProver,
            raf_evaluation::RafEvaluationSumcheckProver as RamRafEvaluationSumcheckProver,
            read_write_checking::RamReadWriteCheckingProver, RAMPreprocessing,
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingProver,
            val_evaluation::ValEvaluationSumcheckProver as RegistersValEvaluationSumcheckProver,
        },
        spartan::{
            claim_reductions::InstructionLookupsClaimReductionSumcheckProver,
            instruction_input::InstructionInputSumcheckProver,
            outer::{OuterRemainingStreamingSumcheck, OuterSharedState},
            product::ProductVirtualRemainderProver,
            shift::ShiftSumcheckProver,
        },
        witness::CommittedPolynomial,
        ProverDebugInfo, Serializable,
    },
};
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::{MemoryConfig, MemoryLayout};
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use tracer::{
    emulator::memory::Memory,
    instruction::{Cycle, Instruction},
    ChunksIterator, JoltDevice, LazyTraceIterator,
};

/// Jolt CPU prover for RV64IMAC.
pub struct JoltCpuProver<
    'a,
    F: JoltField,
    PCS: StreamingCommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub program_io: JoltDevice,
    pub lazy_trace: LazyTraceIterator,
    pub trace: Arc<Vec<Cycle>>,
    pub advice: JoltAdvice<F, PCS>,
    pub unpadded_trace_len: usize,
    pub padded_trace_len: usize,
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub initial_ram_state: Vec<u64>,
    pub final_ram_state: Vec<u64>,
    pub one_hot_params: OneHotParams,
    /// Joint commitment for testing (computed in Stage 7, used in proof for verification)
    #[cfg(test)]
    joint_commitment_for_test: Option<PCS::Commitment>,
}
impl<'a, F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, ProofTranscript: Transcript>
    JoltCpuProver<'a, F, PCS, ProofTranscript>
{
    pub fn gen_from_elf(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<PCS::Commitment>,
    ) -> Self {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: preprocessing.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.memory_layout.max_input_size,
            max_output_size: preprocessing.memory_layout.max_output_size,
            stack_size: preprocessing.memory_layout.stack_size,
            memory_size: preprocessing.memory_layout.memory_size,
            program_size: Some(preprocessing.memory_layout.program_size),
        };

        let (lazy_trace, trace, final_memory_state, program_io) = {
            let _pprof_trace = pprof_scope!("trace");
            guest::program::trace(
                elf_contents,
                None,
                inputs,
                untrusted_advice,
                trusted_advice,
                &memory_config,
            )
        };

        let num_riscv_cycles: usize = trace
            .par_iter()
            .map(|cycle| {
                // Count the cycle if the instruction is not part of a inline sequence
                // (`virtual_sequence_remaining` is `None`) or if it's the first instruction
                // in a inline sequence (`virtual_sequence_remaining` is `Some(0)`)
                if let Some(virtual_sequence_remaining) =
                    cycle.instruction().normalize().virtual_sequence_remaining
                {
                    if virtual_sequence_remaining > 0 {
                        return 0;
                    }
                }
                1
            })
            .sum();
        tracing::info!(
            "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
            trace.len() - num_riscv_cycles,
            trace.len(),
        );

        Self::gen_from_trace(
            preprocessing,
            lazy_trace,
            trace,
            program_io,
            trusted_advice_commitment,
            final_memory_state,
        )
    }

    pub fn gen_from_trace(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        mut trace: Vec<Cycle>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        final_memory_state: Memory,
    ) -> Self {
        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // Setup trace length and padding
        let unpadded_trace_len = trace.len();
        let padded_trace_len = if unpadded_trace_len < 256 {
            256 // ensures that T >= k^{1/D}
        } else {
            (trace.len() + 1).next_power_of_two()
        };
        trace.resize(padded_trace_len, Cycle::NoOp);

        // Calculate K for DoryGlobals initialization
        let ram_K = trace
            .par_iter()
            .filter_map(|cycle| {
                crate::zkvm::ram::remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.memory_layout,
                )
            })
            .max()
            .unwrap_or(0)
            .max(
                crate::zkvm::ram::remap_address(
                    preprocessing.ram.min_bytecode_address,
                    &preprocessing.memory_layout,
                )
                .unwrap_or(0)
                    + preprocessing.ram.bytecode_words.len() as u64
                    + 1,
            )
            .next_power_of_two() as usize;

        let transcript = ProofTranscript::new(b"Jolt");
        let opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());

        let spartan_key = UniformSpartanKey::new(trace.len());

        let (initial_ram_state, final_ram_state) =
            gen_ram_memory_states::<F>(ram_K, &preprocessing.ram, &program_io, &final_memory_state);

        Self {
            preprocessing,
            program_io,
            lazy_trace,
            trace: trace.into(),
            advice: JoltAdvice {
                untrusted_advice_polynomial: None,
                trusted_advice_commitment,
                trusted_advice_polynomial: None,
            },
            unpadded_trace_len,
            padded_trace_len,
            transcript,
            opening_accumulator,
            spartan_key,
            initial_ram_state,
            final_ram_state,
            one_hot_params: OneHotParams::new(
                padded_trace_len.log_2(),
                preprocessing.bytecode.code_size,
                ram_K,
            ),
            #[cfg(test)]
            joint_commitment_for_test: None,
        }
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub fn prove(
        mut self,
    ) -> (
        JoltProof<F, PCS, ProofTranscript>,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) {
        let _pprof_prove = pprof_scope!("prove");

        let start = Instant::now();
        fiat_shamir_preamble(
            &self.program_io,
            self.one_hot_params.ram_k,
            self.trace.len(),
            &mut self.transcript,
        );

        tracing::info!("bytecode size: {}", self.preprocessing.bytecode.code_size);

        let (commitments, opening_proof_hints) = self.generate_and_commit_witness_polynomials();
        let untrusted_advice_commitment = self.generate_and_commit_untrusted_advice();
        self.generate_and_commit_trusted_advice();
        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof) = self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof) = self.prove_stage2();
        let stage3_sumcheck_proof = self.prove_stage3();
        let stage4_sumcheck_proof = self.prove_stage4();
        let stage5_sumcheck_proof = self.prove_stage5();
        let stage6_sumcheck_proof = self.prove_stage6();
        let (trusted_advice_val_evaluation_proof, trusted_advice_val_final_proof) =
            self.prove_trusted_advice();
        let (untrusted_advice_val_evaluation_proof, untrusted_advice_val_final_proof) =
            self.prove_untrusted_advice();
        let stage7_sumcheck_proof = self.prove_stage7();
        let joint_opening_proof = self.prove_stage8(opening_proof_hints);

        #[cfg(test)]
        assert!(
            self.opening_accumulator
                .appended_virtual_openings
                .borrow()
                .is_empty(),
            "Not all virtual openings have been proven, missing: {:?}",
            self.opening_accumulator.appended_virtual_openings.borrow()
        );

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: self.transcript.clone(),
            opening_accumulator: self.opening_accumulator.clone(),
            prover_setup: self.preprocessing.generators.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;

        let opening_state = self
            .opening_accumulator
            .opening_reduction_state
            .as_ref()
            .unwrap();
        let proof = JoltProof {
            opening_claims: Claims(self.opening_accumulator.openings.clone()),
            commitments,
            untrusted_advice_commitment,
            stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6_sumcheck_proof,
            trusted_advice_val_evaluation_proof,
            trusted_advice_val_final_proof,
            untrusted_advice_val_evaluation_proof,
            untrusted_advice_val_final_proof,
            stage7_sumcheck_proof,
            stage7_sumcheck_claims: opening_state.sumcheck_claims.clone(),
            joint_opening_proof,
            #[cfg(test)]
            joint_commitment_for_test: self.joint_commitment_for_test.clone(),
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            bytecode_K: self.one_hot_params.bytecode_k,
            log_k_chunk: self.one_hot_params.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: self.one_hot_params.lookups_ra_virtual_log_k_chunk,
        };

        let prove_duration = start.elapsed();

        tracing::info!(
            "Proved in {:.1}s ({:.1} kHz / padded {:.1} kHz)",
            prove_duration.as_secs_f64(),
            self.unpadded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
            self.padded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
        );

        (proof, debug_info)
    }

    #[tracing::instrument(skip_all, name = "generate_and_commit_witness_polynomials")]
    fn generate_and_commit_witness_polynomials(
        &mut self,
    ) -> (
        Vec<PCS::Commitment>,
        HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) {
        let _guard =
            DoryGlobals::initialize(1 << self.one_hot_params.log_k_chunk, self.padded_trace_len);
        // Generate and commit to all witness polynomials using streaming tier1/tier2 pattern
        let T = DoryGlobals::get_T();
        let polys = all_committed_polynomials(&self.one_hot_params);
        let row_len = DoryGlobals::get_num_columns();
        let num_rows = T / DoryGlobals::get_max_num_rows();

        tracing::debug!(
            "Generating and committing {} witness polynomials with T={}, row_len={}, num_rows={}",
            polys.len(),
            T,
            row_len,
            num_rows
        );

        // Tier 1: Compute row commitments for each polynomial
        let mut row_commitments: Vec<Vec<PCS::ChunkState>> = vec![vec![]; num_rows];

        self.lazy_trace
            .clone()
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(row_len)
            .zip(row_commitments.iter_mut())
            .par_bridge()
            .for_each(|(chunk, row_tier1_commitments)| {
                let res: Vec<_> = polys
                    .par_iter()
                    .map(|poly| {
                        poly.stream_witness_and_commit_rows::<_, PCS>(
                            &self.preprocessing.generators,
                            self.preprocessing,
                            &chunk,
                            &self.one_hot_params,
                        )
                    })
                    .collect();
                *row_tier1_commitments = res;
            });

        // Transpose: row_commitments[row][poly] -> tier1_per_poly[poly][row]
        let tier1_per_poly: Vec<Vec<PCS::ChunkState>> = (0..polys.len())
            .into_par_iter()
            .map(|poly_idx| {
                row_commitments
                    .iter()
                    .flat_map(|row| row.get(poly_idx).cloned())
                    .collect()
            })
            .collect();

        // Tier 2: Compute final commitments from tier1 commitments
        let (commitments, hints): (Vec<_>, Vec<_>) = tier1_per_poly
            .into_par_iter()
            .zip(&polys)
            .map(|(tier1_commitments, poly)| {
                let onehot_k = poly.get_onehot_k(&self.one_hot_params);
                PCS::aggregate_chunks(&self.preprocessing.generators, onehot_k, &tier1_commitments)
            })
            .unzip();

        let hint_map = HashMap::from_iter(zip_eq(polys, hints));

        // Append commitments to transcript
        for commitment in &commitments {
            self.transcript.append_serializable(commitment);
        }

        (commitments, hint_map)
    }

    fn generate_and_commit_untrusted_advice(&mut self) -> Option<PCS::Commitment> {
        if self.program_io.untrusted_advice.is_empty() {
            return None;
        }

        DoryGlobals::initialize_untrusted_advice(
            1,
            self.program_io.memory_layout.max_untrusted_advice_size as usize / 8,
        );
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let (commitment, _hint) = PCS::commit(&poly, &self.preprocessing.generators);
        self.transcript.append_serializable(&commitment);

        self.advice.untrusted_advice_polynomial = Some(poly);

        Some(commitment)
    }

    fn generate_and_commit_trusted_advice(&mut self) {
        if self.program_io.trusted_advice.is_empty() {
            return;
        }

        let mut trusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_trusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.trusted_advice,
            Some(&mut trusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(trusted_advice_vec);
        self.advice.trusted_advice_polynomial = Some(poly);
        self.transcript
            .append_serializable(self.advice.trusted_advice_commitment.as_ref().unwrap());
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.bytecode,
        );
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Every sum-check with num_rounds > 1 requires a schedule
        // which dictates the compute_message and bind methods.
        // Using LinearOnlySchedule to benchmark linear-only mode (no streaming).
        // Outer remaining sumcheck has degree 3 (multiquadratic)
        // Number of rounds = tau.len() - 1 (cycle variables only)
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.bytecode,
            &uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        let (sumcheck_proof, _r_stage1) = BatchedSumcheck::prove(
            vec![&mut spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage2(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");

        // Stage 2a: Prove univariate-skip first round for product virtualization
        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialization params
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.trace.len(),
        );
        let ram_output_check_params = OutputSumcheckParams::new(
            self.one_hot_params.ram_k,
            &self.program_io,
            &mut self.transcript,
        );
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
            );

        // Initialization
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.trace),
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.trace,
            &self.program_io.memory_layout,
        );
        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.program_io.memory_layout,
            &self.initial_ram_state,
        );
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.initial_ram_state,
            &self.final_ram_state,
            &self.program_io.memory_layout,
        );
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.trace),
            );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "ProductVirtualRemainderProver",
                &spartan_product_virtual_remainder,
            );
            print_data_structure_heap_usage("RamRafEvaluationSumcheckProver", &ram_raf_evaluation);
            print_data_structure_heap_usage("RamReadWriteCheckingProver", &ram_read_write_checking);
            print_data_structure_heap_usage("OutputSumcheckProver", &ram_output_check);
            print_data_structure_heap_usage(
                "InstructionLookupsClaimReductionSumcheckProver",
                &instruction_claim_reduction,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_product_virtual_remainder),
            Box::new(ram_raf_evaluation),
            Box::new(ram_read_write_checking),
            Box::new(ram_output_check),
            Box::new(instruction_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_start_flamechart.svg");
        tracing::info!("Stage 2 proving");
        let (sumcheck_proof, _r_stage2) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");
        drop_in_background_thread(instances);

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage3(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");

        // Initialization params
        let spartan_shift_params = ShiftSumcheckParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input_params =
            InstructionInputParams::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialize
        let spartan_shift = ShiftSumcheckProver::initialize(
            spartan_shift_params,
            Arc::clone(&self.trace),
            &self.preprocessing.bytecode,
        );
        let spartan_instruction_input = InstructionInputSumcheckProver::initialize(
            spartan_instruction_input_params,
            &self.trace,
            &self.opening_accumulator,
        );
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            spartan_registers_claim_reduction_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("ShiftSumcheckProver", &spartan_shift);
            print_data_structure_heap_usage(
                "InstructionInputSumcheckProver",
                &spartan_instruction_input,
            );
            print_data_structure_heap_usage(
                "RegistersClaimReductionSumcheckProver",
                &spartan_registers_claim_reduction,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_shift),
            Box::new(spartan_instruction_input),
            Box::new(spartan_registers_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_start_flamechart.svg");
        tracing::info!("Stage 3 proving");
        let (sumcheck_proof, _r_stage3) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage4(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial,
            &self.advice.trusted_advice_polynomial,
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            &mut self.transcript,
            ram::read_write_checking::needs_single_advice_opening(self.trace.len()),
        );
        let ram_ra_booleanity_params =
            ram::ra_booleanity_params(self.trace.len(), &self.one_hot_params, &mut self.transcript);
        let ram_val_evaluation_params = ValEvaluationSumcheckParams::new_from_prover(
            &self.one_hot_params,
            &self.opening_accumulator,
            &self.initial_ram_state,
            self.trace.len(),
        );
        let ram_val_final_params =
            ValFinalSumcheckParams::new_from_prover(self.trace.len(), &self.opening_accumulator);

        let registers_read_write_checking = RegistersReadWriteCheckingProver::initialize(
            registers_read_write_checking_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_ra_booleanity = ram::gen_ra_booleanity_prover(
            ram_ra_booleanity_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_final = ValFinalSumcheckProver::initialize(
            ram_val_final_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersReadWriteCheckingProver",
                &registers_read_write_checking,
            );
            print_data_structure_heap_usage("ram BooleanitySumcheckProver", &ram_ra_booleanity);
            print_data_structure_heap_usage("RamValEvaluationSumcheckProver", &ram_val_evaluation);
            print_data_structure_heap_usage("ValFinalSumcheckProver", &ram_val_final);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
            Box::new(ram_ra_booleanity),
            Box::new(ram_val_evaluation),
            Box::new(ram_val_final),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_start_flamechart.svg");
        tracing::info!("Stage 4 proving");
        let (sumcheck_proof, _r_stage4) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage5(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);
        let ram_hamming_booleanity_params = HammingBooleanityParams::new(&self.opening_accumulator);
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf_params = InstructionReadRafParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);
        let ram_ra_reduction = RamRaReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_read_raf =
            LookupsReadRafSumcheckProver::initialize(lookups_read_raf_params, &self.trace);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
            print_data_structure_heap_usage(
                "ram HammingBooleanitySumcheckProver",
                &ram_hamming_booleanity,
            );
            print_data_structure_heap_usage("RamRaReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("LookupsReadRafSumcheckProver", &lookups_read_raf);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
            Box::new(ram_hamming_booleanity),
            Box::new(ram_ra_reduction),
            Box::new(lookups_read_raf),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");
        tracing::info!("Stage 5 proving");
        let (sumcheck_proof, _r_stage5) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage6(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");

        let bytecode_read_raf_params = BytecodeReadRafParams::gen(
            &self.preprocessing.bytecode,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let bytecode_hamming_weight_params = bytecode::ra_hamming_weight_params(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let bytecode_booleanity_params = bytecode::ra_booleanity_params(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_hamming_weight_params = ram::ra_hamming_weight_params(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_ra_virtual_params = RamRaVirtualParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let lookups_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_hamming_weight_params = instruction_lookups::ra_hamming_weight_params(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_booleanity_params = instruction_lookups::ra_booleanity_params(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            &self.trace,
            &self.preprocessing.bytecode,
        );
        let (bytecode_hamming_weight, bytecode_booleanity) = bytecode::gen_ra_one_hot_provers(
            bytecode_hamming_weight_params,
            bytecode_booleanity_params,
            &self.trace,
            &self.preprocessing.bytecode,
            &self.one_hot_params,
        );
        let ram_hamming_weight = ram::gen_ra_hamming_weight_prover(
            ram_hamming_weight_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_ra_virtual =
            LookupsRaSumcheckProver::initialize(lookups_ra_virtual_params, &self.trace);
        let (lookups_ra_booleanity, lookups_ra_hamming_weight) =
            instruction_lookups::gen_ra_one_hot_provers(
                lookups_hamming_weight_params,
                lookups_booleanity_params,
                &self.trace,
                &self.one_hot_params,
            );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("BytecodeReadRafSumcheckProver", &bytecode_read_raf);
            print_data_structure_heap_usage(
                "bytecode HammingWeightSumcheckProver",
                &bytecode_hamming_weight,
            );
            print_data_structure_heap_usage(
                "bytecode BooleanitySumcheckProver",
                &bytecode_booleanity,
            );
            print_data_structure_heap_usage("ram HammingWeightSumcheckProver", &ram_hamming_weight);
            print_data_structure_heap_usage("RamRaVirtualSumcheckProver", &ram_ra_virtual);
            print_data_structure_heap_usage("LookupsRaSumcheckProver", &lookups_ra_virtual);
            print_data_structure_heap_usage(
                "lookups BooleanitySumcheckProver",
                &lookups_ra_booleanity,
            );
            print_data_structure_heap_usage(
                "lookups HammingWeightSumcheckProver",
                &lookups_ra_hamming_weight,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(bytecode_read_raf),
            Box::new(bytecode_hamming_weight),
            Box::new(bytecode_booleanity),
            Box::new(ram_hamming_weight),
            Box::new(ram_ra_virtual),
            Box::new(lookups_ra_virtual),
            Box::new(lookups_ra_booleanity),
            Box::new(lookups_ra_hamming_weight),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_start_flamechart.svg");
        tracing::info!("Stage 6 proving");
        let (sumcheck_proof, _r_stage6) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_trusted_advice(&mut self) -> (Option<PCS::Proof>, Option<PCS::Proof>) {
        use crate::poly::opening_proof::SumcheckId;
        let poly = match &self.advice.trusted_advice_polynomial {
            Some(p) => p,
            None => return (None, None),
        };
        let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);

        // Prove at RamValEvaluation point
        let (point_val, _) = self
            .opening_accumulator
            .get_trusted_advice_opening(SumcheckId::RamValEvaluation)
            .unwrap();
        let proof_val = PCS::prove(
            &self.preprocessing.generators,
            poly,
            &point_val.r,
            None,
            &mut self.transcript,
        );

        // Prove at RamValFinalEvaluation point - only if different from ValEvaluation
        let proof_val_final =
            if ram::read_write_checking::needs_single_advice_opening(self.trace.len()) {
                None
            } else {
                let (point_val_final, _) = self
                    .opening_accumulator
                    .get_trusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                    .unwrap();
                Some(PCS::prove(
                    &self.preprocessing.generators,
                    poly,
                    &point_val_final.r,
                    None,
                    &mut self.transcript,
                ))
            };

        (Some(proof_val), proof_val_final)
    }

    #[tracing::instrument(skip_all)]
    fn prove_untrusted_advice(&mut self) -> (Option<PCS::Proof>, Option<PCS::Proof>) {
        use crate::poly::opening_proof::SumcheckId;
        let poly = match &self.advice.untrusted_advice_polynomial {
            Some(p) => p,
            None => return (None, None),
        };
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);

        // Prove at RamValEvaluation point
        let (point_val, _) = self
            .opening_accumulator
            .get_untrusted_advice_opening(SumcheckId::RamValEvaluation)
            .unwrap();
        let proof_val = PCS::prove(
            &self.preprocessing.generators,
            poly,
            &point_val.r,
            None,
            &mut self.transcript,
        );

        // Prove at RamValFinalEvaluation point - only if different from ValEvaluation
        let proof_val_final =
            if ram::read_write_checking::needs_single_advice_opening(self.trace.len()) {
                None
            } else {
                let (point_val_final, _) = self
                    .opening_accumulator
                    .get_untrusted_advice_opening(SumcheckId::RamValFinalEvaluation)
                    .unwrap();
                Some(PCS::prove(
                    &self.preprocessing.generators,
                    poly,
                    &point_val_final.r,
                    None,
                    &mut self.transcript,
                ))
            };

        (Some(proof_val), proof_val_final)
    }

    /// Stage 7: Batch opening reduction sumcheck.
    /// Generates witness polynomials, runs the sumcheck, and stores state for Stage 8.
    #[tracing::instrument(skip_all)]
    fn prove_stage7(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        tracing::info!("Stage 7 proving (batch opening reduction sumcheck)");

        let _guard = DoryGlobals::initialize(self.one_hot_params.k_chunk, self.padded_trace_len);

        let polynomials_map = CommittedPolynomial::generate_witness_batch(
            &all_committed_polynomials(&self.one_hot_params),
            self.preprocessing,
            &self.trace,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("Committed polynomials map", &polynomials_map);

        // Prepare sumcheck
        self.opening_accumulator
            .prepare_for_sumcheck(&polynomials_map);

        // Run sumcheck
        let (sumcheck_proof, r_sumcheck) = self
            .opening_accumulator
            .prove_batch_opening_sumcheck(&mut self.transcript);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = self
            .opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck, &mut self.transcript);

        // Compute joint commitment for testing using streaming RLC
        #[cfg(test)]
        {
            let streaming_data = Arc::new(RLCStreamingData {
                bytecode: self.preprocessing.bytecode.clone(),
                memory_layout: self.preprocessing.memory_layout.clone(),
            });
            self.joint_commitment_for_test = Some(
                self.opening_accumulator
                    .compute_joint_commitment_for_test::<PCS>(
                        &polynomials_map,
                        &state,
                        &self.preprocessing.generators,
                        Some((
                            self.lazy_trace.clone(),
                            streaming_data,
                            self.one_hot_params.clone(),
                        )),
                    ),
            );
        }

        // Store state and polynomials in accumulator for Stage 8
        self.opening_accumulator.opening_reduction_state = Some(state);
        self.opening_accumulator.polynomials_for_opening = Some(polynomials_map);

        sumcheck_proof
    }

    /// Stage 8: Dory batch opening proof.
    /// Uses the state from Stage 7 (stored in accumulator) to build the RLC polynomial and prove the opening.
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        let _guard = DoryGlobals::initialize(self.one_hot_params.k_chunk, self.padded_trace_len);

        let state = self
            .opening_accumulator
            .opening_reduction_state
            .as_ref()
            .expect("Stage 7 must be called before Stage 8");

        let polynomials = self
            .opening_accumulator
            .polynomials_for_opening
            .take()
            .expect("Stage 7 must be called before Stage 8");

        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: self.preprocessing.bytecode.clone(),
            memory_layout: self.preprocessing.memory_layout.clone(),
        });

        // Build RLC polynomial and combined hint
        let (joint_poly, hint) = self.opening_accumulator.build_rlc_polynomial::<PCS>(
            polynomials,
            opening_proof_hints,
            state,
            Some((
                self.lazy_trace.clone(),
                streaming_data,
                self.one_hot_params.clone(),
            )),
        );

        // Dory opening proof
        PCS::prove(
            &self.preprocessing.generators,
            &joint_poly,
            &state.r_sumcheck,
            Some(hint),
            &mut self.transcript,
        )
    }
}

pub struct JoltAdvice<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub untrusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "JoltProverPreprocessing::gen")]
    pub fn gen(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        let max_T: usize = max_trace_length.next_power_of_two();
        let log_chunk = get_log_k_chunk(max_T);

        let bytecode = BytecodePreprocessing::preprocess(bytecode);
        let ram = RAMPreprocessing::preprocess(memory_init);

        let generators = PCS::setup_prover(log_chunk + max_T.log_2());

        JoltProverPreprocessing {
            generators,
            bytecode,
            ram,
            memory_layout,
        }
    }

    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        Self {
            generators,
            bytecode: preprocessing.bytecode.clone(),
            ram: preprocessing.ram.clone(),
            memory_layout: preprocessing.memory_layout.clone(),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> Serializable
    for JoltProverPreprocessing<F, PCS>
{
}

#[cfg(feature = "allocative")]
fn write_instance_flamegraph_svg(
    instances: &[Box<dyn SumcheckInstanceProver<impl JoltField, impl Transcript>>],
    path: impl AsRef<Path>,
) {
    let mut flamegraph = FlameGraphBuilder::default();
    for instance in instances {
        instance.update_flamegraph(&mut flamegraph)
    }
    write_flamegraph_svg(flamegraph, path);
}

#[cfg(test)]
mod tests {
    use crate::host;
    use crate::poly::commitment::dory::DoryCommitmentScheme;
    use crate::zkvm::prover::JoltProverPreprocessing;
    use crate::zkvm::verifier::{JoltVerifier, JoltVerifierPreprocessing};
    use crate::zkvm::{RV64IMACProver, RV64IMACVerifier};
    use ark_bn254::Fr;
    use serial_test::serial;

    #[test]
    #[serial]
    fn fib_e2e_dory() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&100u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &inputs, &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn small_trace_e2e_dory() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            256,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let log_chunk = 8; // Use default log_chunk for tests
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &inputs, &[], &[], None);

        assert!(
            prover.padded_trace_len <= (1 << log_chunk),
            "Test requires T <= chunk_size ({}), got T = {}",
            1 << log_chunk,
            prover.padded_trace_len
        );

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn sha3_e2e_dory() {
        // Ensure SHA3 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_keccak256 as _;
        // SHA3 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-keccak256 crate is linked (see lib.rs)

        let mut program = host::Program::new("sha3-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &inputs, &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
        assert_eq!(
            io_device.inputs, inputs,
            "Inputs mismatch: expected {:?}, got {:?}",
            inputs, io_device.inputs
        );
        let expected_output = &[
            0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88,
            0x33, 0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f,
            0x44, 0x72, 0x4, 0x66,
        ];
        assert_eq!(io_device.outputs, expected_output, "Outputs mismatch",);
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory() {
        // Ensure SHA2 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_sha2 as _;
        // SHA2 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-sha2 crate is linked (see lib.rs)
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &inputs, &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
        let expected_output = &[
            0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
            0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
            0x3b, 0x50, 0xd2, 0x57,
        ];
        assert_eq!(
            io_device.outputs, expected_output,
            "Outputs mismatch: expected {:?}, got {:?}",
            expected_output, io_device.outputs
        );
    }

    #[test]
    #[serial]
    fn advice_e2e_dory() {
        use crate::poly::{
            commitment::commitment_scheme::CommitmentScheme,
            multilinear_polynomial::MultilinearPolynomial,
        };
        use crate::zkvm::ram::populate_memory_states;

        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let leaf1: [u8; 32] = [5u8; 32];
        let leaf2: [u8; 32] = [6u8; 32];
        let leaf3: [u8; 32] = [7u8; 32];
        let leaf4: [u8; 32] = [8u8; 32];
        let inputs = postcard::to_stdvec(&leaf1.as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&leaf4).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&leaf2).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&leaf3).unwrap());

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        let max_trusted_advice_size = preprocessing.memory_layout.max_trusted_advice_size;
        let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
        populate_memory_states(0, &trusted_advice, Some(&mut trusted_advice_words), None);
        let trusted_advice_commitment = {
            crate::poly::commitment::dory::DoryGlobals::initialize_trusted_advice(
                1,
                (max_trusted_advice_size as usize) / 8,
            );
            let _ctx = crate::poly::commitment::dory::DoryGlobals::with_context(
                crate::poly::commitment::dory::DoryContext::TrustedAdvice,
            );
            let poly = MultilinearPolynomial::<ark_bn254::Fr>::from(trusted_advice_words);
            let (trusted_advice_commitment, _hint) =
                <crate::poly::commitment::dory::DoryCommitmentScheme as CommitmentScheme>::commit(
                    &poly,
                    &preprocessing.generators,
                );
            trusted_advice_commitment
        };

        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_advice_commitment),
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_advice_commitment),
            debug_info,
        )
        .unwrap();
        let verification_result = verifier.verify();
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
        assert_eq!(
            io_device.inputs, inputs,
            "Inputs mismatch: expected {:?}, got {:?}",
            inputs, io_device.inputs
        );
        let expected_output = &[
            0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
            0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32,
            0xbb, 0x16, 0xd7,
        ];
        assert_eq!(
            io_device.outputs, expected_output,
            "Outputs mismatch: expected {:?}, got {:?}",
            expected_output, io_device.outputs
        );
    }

    #[test]
    #[serial]
    fn memory_ops_e2e_dory() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&[], &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &[], &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn btreemap_e2e_dory() {
        let mut program = host::Program::new("btreemap-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&50u32).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &inputs, &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn muldiv_e2e_dory() {
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &[50], &[], &[], None);
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
        .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    #[should_panic]
    fn truncated_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();
        let (lazy_trace, mut trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);
        trace.truncate(100);
        program_io.outputs[0] = 0; // change the output to 0

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover = RV64IMACProver::gen_from_trace(
            &preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            final_memory_state,
        );

        let (proof, _) = prover.prove();

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
        verifier.verify().unwrap();
    }

    #[test]
    #[serial]
    #[should_panic]
    fn malicious_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        program_io.memory_layout.output_start = program_io.memory_layout.input_start;
        program_io.memory_layout.output_end = program_io.memory_layout.input_end;
        program_io.memory_layout.termination = program_io.memory_layout.input_start;

        let prover = RV64IMACProver::gen_from_trace(
            &preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            final_memory_state,
        );
        let (proof, _) = prover.prove();

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let verifier =
            JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
        verifier.verify().unwrap();
    }
}
