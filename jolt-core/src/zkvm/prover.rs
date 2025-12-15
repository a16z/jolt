use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use crate::poly::commitment::dory::DoryContext;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::{print_data_structure_heap_usage, write_flamegraph_svg};
use crate::zkvm::config::get_log_k_chunk as get_log_k_chunk_from_log_t;
use crate::{
    field::JoltField,
    guest,
    poly::{
        commitment::{commitment_scheme::StreamingCommitmentScheme, dory::DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            DoryOpeningState, OpeningAccumulator, ProverOpeningAccumulator, SumcheckId,
        },
        rlc_polynomial::{RLCStreamingData, TraceSource},
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckProver},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round, UniSkipFirstRoundProof},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
    zkvm::{
        bytecode::read_raf_checking::ReadRafSumcheckParams as BytecodeReadRafParams,
        claim_reductions::{
            AdviceClaimReductionParams, AdviceClaimReductionProver,
            HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
            IncReductionSumcheckParams, IncReductionSumcheckProver,
            InstructionLookupsClaimReductionSumcheckParams,
            InstructionLookupsClaimReductionSumcheckProver, RaReductionParams,
            RamRaReductionSumcheckProver,
        },
        config::{get_log_k_chunk, OneHotParams},
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckParams,
            read_raf_checking::ReadRafSumcheckParams as InstructionReadRafParams,
        },
        ram::{
            hamming_booleanity::HammingBooleanitySumcheckParams,
            output_check::OutputSumcheckParams,
            populate_memory_states,
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
            instruction_input::InstructionInputParams,
            outer::{OuterRemainingSumcheckParams, OuterUniSkipParams, OuterUniSkipProver},
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
            read_raf_checking::ReadRafSumcheckProver as BytecodeReadRafSumcheckProver,
            BytecodePreprocessing,
        },
        fiat_shamir_preamble,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver,
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
            instruction_input::InstructionInputSumcheckProver, outer::OuterRemainingSumcheckProver,
            product::ProductVirtualRemainderProver, shift::ShiftSumcheckProver,
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
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
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
            trusted_advice_hint,
            final_memory_state,
        )
    }

    pub fn gen_from_trace(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        mut trace: Vec<Cycle>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
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
        // We may need extra padding so the main Dory matrix has enough column variables
        // to embed advice commitments committed in their own 1-row contexts.
        let mut padded_trace_len = padded_trace_len;
        let has_trusted_advice = !program_io.trusted_advice.is_empty();
        let has_untrusted_advice = !program_io.untrusted_advice.is_empty();
        let mut max_advice_vars = 0usize;
        if has_trusted_advice {
            let words = (preprocessing.memory_layout.max_trusted_advice_size as usize) / 8;
            let cols = words.next_power_of_two().max(1);
            max_advice_vars = max_advice_vars.max(cols.log_2());
        }
        if has_untrusted_advice {
            let words = (preprocessing.memory_layout.max_untrusted_advice_size as usize) / 8;
            let cols = words.next_power_of_two().max(1);
            max_advice_vars = max_advice_vars.max(cols.log_2());
        }
        if max_advice_vars > 0 {
            // Require main sigma (columns exponent) >= max_advice_vars so advice fits in the
            // leftmost columns of the main matrix.
            while {
                let log_t = padded_trace_len.log_2();
                let log_k_chunk = get_log_k_chunk_from_log_t(log_t);
                let total_vars = log_k_chunk + log_t;
                let sigma_main = total_vars.div_ceil(2);
                sigma_main < max_advice_vars
            } {
                // Double T (keep power-of-two) until constraint holds.
                if padded_trace_len >= preprocessing.max_padded_trace_length {
                    panic!(
                        "Trace too small to embed advice into single Dory opening: need main sigma >= {max_advice_vars}, \
but reached max_padded_trace_length={} (increase max_trace_length in preprocessing or reduce max_*_advice_size)",
                        preprocessing.max_padded_trace_length
                    );
                }
                padded_trace_len =
                    (padded_trace_len * 2).min(preprocessing.max_padded_trace_length);
            }
        }

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
                untrusted_advice_hint: None,
                trusted_advice_hint,
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

        let (commitments, mut opening_proof_hints) = self.generate_and_commit_witness_polynomials();
        let untrusted_advice_commitment = self.generate_and_commit_untrusted_advice();
        self.generate_and_commit_trusted_advice();

        // Add advice hints for batched Stage 8 opening
        if let Some(hint) = self.advice.trusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::TrustedAdvice, hint);
        }
        if let Some(hint) = self.advice.untrusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::UntrustedAdvice, hint);
        }

        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof) = self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof) = self.prove_stage2();
        let stage3_sumcheck_proof = self.prove_stage3();
        let stage4_sumcheck_proof = self.prove_stage4();
        let stage5_sumcheck_proof = self.prove_stage5();
        let stage6_sumcheck_proof = self.prove_stage6();
        // Advice claims are now reduced in Stage 6 and batched into Stage 8
        // The old separate advice proofs are no longer generated
        let stage7_sumcheck_proof = self.prove_stage7();

        #[cfg(test)]
        {
            // Compute and store the joint commitment for cross-checking in the verifier.
            // This helps catch commitment/RLC mismatches early when changing batching logic.
            if let Some(ref state) = self.opening_accumulator.dory_opening_state {
                let mut commitments_map: HashMap<CommittedPolynomial, PCS::Commitment> =
                    HashMap::new();
                for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
                    .into_iter()
                    .zip_eq(&commitments)
                {
                    commitments_map.insert(polynomial, commitment.clone());
                }
                if let Some(ref commitment) = untrusted_advice_commitment {
                    if state
                        .polynomials
                        .contains(&CommittedPolynomial::UntrustedAdvice)
                    {
                        commitments_map
                            .insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
                    }
                }
                if let Some(ref commitment) = self.advice.trusted_advice_commitment {
                    if state
                        .polynomials
                        .contains(&CommittedPolynomial::TrustedAdvice)
                    {
                        commitments_map
                            .insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
                    }
                }

                // Accumulate gamma coefficients per polynomial (merge duplicates).
                let mut rlc_map = HashMap::new();
                for (gamma, poly) in state.gamma_powers.iter().zip(state.polynomials.iter()) {
                    *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
                }
                let (coeffs, comms): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                    .into_iter()
                    .map(|(k, v)| (v, commitments_map.remove(&k).unwrap()))
                    .unzip();
                self.joint_commitment_for_test = Some(PCS::combine_commitments(&comms, &coeffs));
            }
        }
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

        let dory_state = self
            .opening_accumulator
            .dory_opening_state
            .as_ref()
            .expect("Stage 7 must be called before finalizing proof");
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
            stage7_sumcheck_proof,
            // Note: verifier no longer uses this field, but kept for proof format compatibility
            stage7_sumcheck_claims: dory_state.claims.clone(),
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

        // Commit untrusted advice in its dedicated Dory context, using a fixed 1-row matrix.
        //
        // This makes the advice commitment independent of the trace length, while still allowing
        // Stage 8 to batch it into the single Dory opening proof by interpreting it as a
        // zero-padded submatrix of the main polynomial matrix.

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let advice_cols = poly.len();
        let _guard = DoryGlobals::initialize_untrusted_advice_1row(advice_cols);
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
        let (commitment, hint) = PCS::commit(&poly, &self.preprocessing.generators);
        self.transcript.append_serializable(&commitment);

        self.advice.untrusted_advice_polynomial = Some(poly);
        self.advice.untrusted_advice_hint = Some(hint);

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

        let spartan_outer_remaining_params = OuterRemainingSumcheckParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining = OuterRemainingSumcheckProver::initialize(
            spartan_outer_remaining_params,
            Arc::clone(&self.trace),
            &self.preprocessing.bytecode,
        );

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

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("ShiftSumcheckProver", &spartan_shift);
            print_data_structure_heap_usage(
                "InstructionInputSumcheckProver",
                &spartan_instruction_input,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(spartan_shift), Box::new(spartan_instruction_input)];

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
            print_data_structure_heap_usage("RamValEvaluationSumcheckProver", &ram_val_evaluation);
            print_data_structure_heap_usage("ValFinalSumcheckProver", &ram_val_final);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
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
        // Note: RamHammingBooleanity moved to Stage 6 so it shares r_cycle_stage6
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
            print_data_structure_heap_usage("RamRaReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("LookupsReadRafSumcheckProver", &lookups_read_raf);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
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

        // RamHammingBooleanity - uses r_cycle from Stage 5's RamRaReduction
        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

        // Booleanity: combines instruction, bytecode, and ram booleanity into one
        // (extracts r_address and r_cycle from Stage 5 internally)
        let booleanity_params = BooleanitySumcheckParams::new(
            self.trace.len().log_2(),
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
        let inc_reduction_params = IncReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Advice claim reduction - may be None if no advice present
        let advice_reduction_params = AdviceClaimReductionParams::new(
            &self.program_io.memory_layout,
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            &self.trace,
            &self.preprocessing.bytecode,
        );
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);

        // Booleanity prover - handles all three families
        let booleanity = BooleanitySumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.bytecode,
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
        let inc_reduction =
            IncReductionSumcheckProver::initialize(inc_reduction_params, self.trace.clone());

        // Initialize advice reduction prover if there's advice
        let advice_reduction = advice_reduction_params.map(|params| {
            AdviceClaimReductionProver::initialize(
                params,
                self.advice.trusted_advice_polynomial.clone(),
                self.advice.untrusted_advice_polynomial.clone(),
            )
        });

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("BytecodeReadRafSumcheckProver", &bytecode_read_raf);
            print_data_structure_heap_usage(
                "ram HammingBooleanitySumcheckProver",
                &ram_hamming_booleanity,
            );
            print_data_structure_heap_usage("BooleanitySumcheckProver", &booleanity);
            print_data_structure_heap_usage("RamRaSumcheckProver", &ram_ra_virtual);
            print_data_structure_heap_usage("LookupsRaSumcheckProver", &lookups_ra_virtual);
            print_data_structure_heap_usage("IncReductionSumcheckProver", &inc_reduction);
            if let Some(ref advice) = advice_reduction {
                print_data_structure_heap_usage("AdviceClaimReductionProver", advice);
            }
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(bytecode_read_raf),
            Box::new(ram_hamming_booleanity),
            Box::new(booleanity),
            Box::new(ram_ra_virtual),
            Box::new(lookups_ra_virtual),
            Box::new(inc_reduction),
        ];
        if let Some(advice) = advice_reduction {
            instances.push(Box::new(advice));
        }

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

    // Note: prove_trusted_advice and prove_untrusted_advice have been removed.
    // Advice claims are now reduced via AdviceClaimReduction in Stage 6 and proven
    // as part of the batched Stage 8 opening proof.

    /// Stage 7: HammingWeight + ClaimReduction sumcheck (only log_k_chunk rounds).
    /// Produces `DoryOpeningState` for Stage 8.
    #[tracing::instrument(skip_all)]
    fn prove_stage7(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        tracing::info!("Stage 7 proving (HammingWeight claim reduction)");

        // Create params and prover for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let mut hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            self.preprocessing,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        // 3. Run sumcheck (only log_k_chunk rounds!)
        let instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
            vec![&mut hw_prover];
        let (sumcheck_proof, r_address_stage7) = BatchedSumcheck::prove(
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // 4. Collect all claims for DoryOpeningState
        let mut claims = Vec::new();
        let mut polynomials = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncReduction in Stage 6)
        // These are at r_cycle_stage6 only (length log_T)
        let (_ram_inc_point, ram_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncReduction,
            );
        let (_rd_inc_point, rd_inc_claim) = self
            .opening_accumulator
            .get_committed_polynomial_opening(CommittedPolynomial::RdInc, SumcheckId::IncReduction);

        #[cfg(test)]
        {
            // Verify that Inc openings are at the same point as r_cycle from Booleanity
            let (unified_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::Booleanity,
            );
            let log_k_chunk = self.one_hot_params.log_k_chunk;
            let r_cycle_from_unified = &unified_point.r[log_k_chunk..];

            debug_assert_eq!(
                _ram_inc_point.r.as_slice(),
                r_cycle_from_unified,
                "RamInc opening point should match r_cycle from Booleanity"
            );
            debug_assert_eq!(
                _rd_inc_point.r.as_slice(),
                r_cycle_from_unified,
                "RdInc opening point should match r_cycle from Booleanity"
            );
        }

        // Apply Lagrange factor for dense polys: ∏_{i<log_k_chunk} (1 - r_address[i])
        // Because dense polys have fewer variables, we need to account for this
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        claims.push(ram_inc_claim * lagrange_factor);
        claims.push(rd_inc_claim * lagrange_factor);
        polynomials.push(CommittedPolynomial::RamInc);
        polynomials.push(CommittedPolynomial::RdInc);

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        // These are at (r_address_stage7, r_cycle_stage6)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            claims.push(claim);
            polynomials.push(CommittedPolynomial::RamRa(i));
        }

        // 5. Build unified opening point: (r_address_stage7 || r_cycle_stage6)
        // Note: r_address_stage7 is little-endian from sumcheck, convert to big-endian
        let mut r_address_be = r_address_stage7.clone();
        r_address_be.reverse();

        // Extract r_cycle from Booleanity (same source as HammingWeightClaimReduction uses)
        let (unified_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_cycle_stage6 = &unified_point.r[log_k_chunk..];

        let opening_point = [r_address_be.as_slice(), r_cycle_stage6].concat();

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with Main context dimensions so they can be batched.
        // They have fewer variables than main polynomials, so we apply Lagrange factors.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
        {
            let advice_vars = advice_point.len();
            // Dory uses little-endian variable order; the commitment layer reverses the opening point.
            // Compute the embedding selector for a 1-row advice matrix placed in the first 2^advice_vars columns:
            //   selector = eq(row_bits, 0) * eq(col_bits[advice_vars..], 0)
            let mut r_le = opening_point.clone();
            r_le.reverse();
            let sigma = DoryGlobals::get_num_columns().log_2();
            let nu = DoryGlobals::get_max_num_rows().log_2();
            debug_assert_eq!(sigma + nu, r_le.len());
            let (r_cols, r_rows) = r_le.split_at(sigma);

            let row_factor: F = r_rows.iter().map(|r| F::one() - (*r).into()).product();
            let col_prefix_factor: F = r_cols
                .iter()
                .skip(advice_vars)
                .map(|r| F::one() - (*r).into())
                .product();

            claims.push(advice_claim * row_factor * col_prefix_factor);
            polynomials.push(CommittedPolynomial::TrustedAdvice);
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
        {
            let advice_vars = advice_point.len();
            let mut r_le = opening_point.clone();
            r_le.reverse();
            let sigma = DoryGlobals::get_num_columns().log_2();
            let nu = DoryGlobals::get_max_num_rows().log_2();
            debug_assert_eq!(sigma + nu, r_le.len());
            let (r_cols, r_rows) = r_le.split_at(sigma);

            let row_factor: F = r_rows.iter().map(|r| F::one() - (*r).into()).product();
            let col_prefix_factor: F = r_cols
                .iter()
                .skip(advice_vars)
                .map(|r| F::one() - (*r).into())
                .product();

            claims.push(advice_claim * row_factor * col_prefix_factor);
            polynomials.push(CommittedPolynomial::UntrustedAdvice);
        }

        // 6. Sample gamma and compute powers for RLC
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // 7. Store DoryOpeningState for Stage 8
        self.opening_accumulator.dory_opening_state = Some(DoryOpeningState {
            opening_point,
            gamma_powers,
            claims,
            polynomials,
        });

        sumcheck_proof
    }

    /// Stage 8: Dory batch opening proof.
    /// Builds streaming RLC polynomial directly from trace (no witness regeneration needed).
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        let _guard = DoryGlobals::initialize(self.one_hot_params.k_chunk, self.padded_trace_len);

        let state = self
            .opening_accumulator
            .dory_opening_state
            .as_ref()
            .expect("Stage 7 must be called before Stage 8");

        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: self.preprocessing.bytecode.clone(),
            memory_layout: self.preprocessing.memory_layout.clone(),
        });

        // Build advice polynomials map for RLC
        let mut advice_polys = HashMap::new();
        if let Some(poly) = self.advice.trusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::TrustedAdvice, poly);
        }
        if let Some(poly) = self.advice.untrusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::UntrustedAdvice, poly);
        }

        // Build streaming RLC polynomial directly (no witness poly regeneration!)
        // Use materialized trace (default, single pass) instead of lazy trace
        let (joint_poly, hint) = state.build_streaming_rlc::<PCS>(
            self.one_hot_params.clone(),
            TraceSource::Materialized(Arc::clone(&self.trace)),
            streaming_data,
            opening_proof_hints,
            advice_polys,
        );

        #[cfg(test)]
        {
            // Sanity-check joint evaluation using Dory's **little-endian** basis convention.
            // Dory treats point[0] as the LSB; our commitment wrapper reverses the opening point
            // before passing it into Dory. Mirror that here.
            fn lagrange_basis_le<F: JoltField>(point: &[F]) -> Vec<F> {
                let n = point.len();
                if n == 0 {
                    return vec![F::one()];
                }
                let mut out = vec![F::zero(); 1 << n];
                out[0] = F::one() - point[0];
                out[1] = point[0];
                for (level, p) in point[1..].iter().enumerate() {
                    let mid = 1 << (level + 1);
                    let one_minus_p = F::one() - *p;
                    for i in 0..mid {
                        let l_val = out[i];
                        out[mid + i] = l_val * *p;
                        out[i] = l_val * one_minus_p;
                    }
                }
                out
            }

            let num_cols = DoryGlobals::get_num_columns();
            let num_rows = DoryGlobals::get_max_num_rows();
            let sigma = num_cols.log_2();
            let nu = num_rows.log_2();
            debug_assert_eq!(nu + sigma, state.opening_point.len());

            // Dory uses opposite endianness relative to Jolt: reverse the point.
            let mut r_le = state.opening_point.clone();
            r_le.reverse();

            let cols: Vec<F> = r_le[..sigma].iter().map(|c| (*c).into()).collect();
            let rows: Vec<F> = r_le[sigma..].iter().map(|c| (*c).into()).collect();
            let right_vec = lagrange_basis_le::<F>(&cols);
            let left_vec = lagrange_basis_le::<F>(&rows);
            debug_assert_eq!(right_vec.len(), num_cols);
            debug_assert_eq!(left_vec.len(), num_rows);

            let vmv: Vec<F> = match &joint_poly {
                MultilinearPolynomial::RLC(rlc) => rlc.vector_matrix_product(&left_vec),
                _ => panic!("Expected RLC joint polynomial in Stage 8"),
            };
            let eval_from_vmv: F = vmv.iter().zip(right_vec.iter()).map(|(a, b)| *a * *b).sum();

            let joint_claim: F = state
                .gamma_powers
                .iter()
                .zip(state.claims.iter())
                .map(|(gamma, claim)| *gamma * *claim)
                .sum();

            // Isolate advice contribution vs base contribution.
            let base_claim: F = state
                .gamma_powers
                .iter()
                .zip(state.claims.iter())
                .zip(state.polynomials.iter())
                .filter(|(_, poly)| {
                    !matches!(
                        poly,
                        CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice
                    )
                })
                .map(|((gamma, claim), _)| *gamma * *claim)
                .sum();

            let advice_claim = joint_claim - base_claim;

            let base_eval: F = match &joint_poly {
                MultilinearPolynomial::RLC(rlc) => {
                    if let Some(ctx) = &rlc.streaming_context {
                        let mut ctx_no_advice = ctx.as_ref().clone();
                        ctx_no_advice.advice_polys.clear();
                        let rlc_no_advice = crate::poly::rlc_polynomial::RLCPolynomial::<F> {
                            dense_rlc: vec![],
                            one_hot_rlc: vec![],
                            streaming_context: Some(Arc::new(ctx_no_advice)),
                        };
                        let vmv_no_advice = rlc_no_advice.vector_matrix_product(&left_vec);
                        vmv_no_advice
                            .iter()
                            .zip(right_vec.iter())
                            .map(|(a, b)| *a * *b)
                            .sum()
                    } else {
                        panic!("Expected streaming context for Stage 8 RLC")
                    }
                }
                _ => unreachable!(),
            };

            let advice_eval = eval_from_vmv - base_eval;

            // Per-advice polynomial diagnostics (Trusted vs Untrusted).
            let per_advice_eval: Vec<(CommittedPolynomial, F)> = match &joint_poly {
                MultilinearPolynomial::RLC(rlc) => {
                    let ctx = rlc
                        .streaming_context
                        .as_ref()
                        .expect("Expected streaming context for Stage 8 RLC");
                    ctx.advice_polys
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let mut ctx_one = ctx.as_ref().clone();
                            ctx_one.advice_polys = vec![ctx.as_ref().advice_polys[i].clone()];
                            let rlc_one = crate::poly::rlc_polynomial::RLCPolynomial::<F> {
                                dense_rlc: vec![],
                                one_hot_rlc: vec![],
                                streaming_context: Some(Arc::new(ctx_one)),
                            };
                            let vmv_one = rlc_one.vector_matrix_product(&left_vec);
                            let eval_one: F = vmv_one
                                .iter()
                                .zip(right_vec.iter())
                                .map(|(a, b)| *a * *b)
                                .sum();
                            // We don't have the polynomial ID here (ctx only stores coeff+poly),
                            // so we label based on position: TrustedAdvice then UntrustedAdvice.
                            let poly_id = if i == 0 {
                                CommittedPolynomial::TrustedAdvice
                            } else {
                                CommittedPolynomial::UntrustedAdvice
                            };
                            (poly_id, eval_one)
                        })
                        .collect()
                }
                _ => vec![],
            };

            debug_assert_eq!(
                base_eval, base_claim,
                "Stage 8 base claim mismatch (non-advice)"
            );

            // If we ever hit the mismatch, provide more context.
            if advice_eval != advice_claim {
                // Recompute common row selector (row index 0) from the point.
                let mut r_le2 = state.opening_point.clone();
                r_le2.reverse();
                let sigma2 = DoryGlobals::get_num_columns().log_2();
                let (r_cols2, r_rows2) = r_le2.split_at(sigma2);
                let row_factor: F = r_rows2.iter().map(|r| F::one() - (*r).into()).product();
                let log_k_chunk2 = self.one_hot_params.log_k_chunk;
                let total_vars2 = state.opening_point.len();
                let log_t2 = total_vars2 - log_k_chunk2;
                let cycle_be2 = &state.opening_point[log_k_chunk2..];
                // Reconstruct Stage 6 batched challenges (little-endian) from the Booleanity opening point.
                // Booleanity stores (addr_be || cycle_be) where each segment is reversed from the LE
                // binding order used in sumcheck.
                let (bool_point_be, _) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(0),
                    SumcheckId::Booleanity,
                );
                let addr_be6 = &bool_point_be.r[..log_k_chunk2];
                let cycle_be6 = &bool_point_be.r[log_k_chunk2..];
                let mut r_sumcheck_reconstructed_le: Vec<F::Challenge> =
                    Vec::with_capacity(addr_be6.len() + cycle_be6.len());
                // address_le then cycle_le
                r_sumcheck_reconstructed_le.extend(addr_be6.iter().copied().rev());
                r_sumcheck_reconstructed_le.extend(cycle_be6.iter().copied().rev());

                #[allow(dead_code)]
                #[derive(Debug)]
                struct AdvicePolyDebug<F: JoltField> {
                    poly_id: CommittedPolynomial,
                    eval_contrib: F,
                    claim_contrib: F,
                    coeff_sum: F,
                    claim_scaled: Option<F>,
                    raw_opening: Option<F>,
                    predicted_scaled: Option<F>,
                    embedded_eval: Option<F>,
                    row_factor: F,
                    col_prefix_factor: F,
                    point_matches: bool,
                    actual_point_le_pos_in_cols: Option<usize>,
                    actual_point_be_pos_in_cycle_be: Option<usize>,
                    actual_point_le_pos_in_r_sumcheck: Option<usize>,
                    expected_point_be: Vec<F::Challenge>,
                    actual_point_be: Option<Vec<F::Challenge>>,
                }

                let mut per_poly_details: Vec<AdvicePolyDebug<F>> = Vec::new();
                for (poly_id, eval_contrib) in &per_advice_eval {
                    // Find the claim+gamma for this polynomial in the state
                    let mut claim_contrib = F::zero();
                    let mut coeff_sum = F::zero();
                    let mut claim_scaled_opt = None;
                    for ((gamma, claim), poly) in state
                        .gamma_powers
                        .iter()
                        .zip(state.claims.iter())
                        .zip(state.polynomials.iter())
                    {
                        if poly == poly_id {
                            claim_contrib += *gamma * *claim;
                            coeff_sum += *gamma;
                            claim_scaled_opt = Some(*claim);
                        }
                    }

                    // Raw claim from AdviceClaimReduction opening
                    let raw_opening = match poly_id {
                        CommittedPolynomial::TrustedAdvice => self
                            .opening_accumulator
                            .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(_, c)| c),
                        CommittedPolynomial::UntrustedAdvice => self
                            .opening_accumulator
                            .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(_, c)| c),
                        _ => None,
                    };

                    // Column selector depends on advice_vars (opening point length for that advice)
                    let advice_vars = match poly_id {
                        CommittedPolynomial::TrustedAdvice => self
                            .opening_accumulator
                            .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(p, _)| p.len())
                            .unwrap_or(0),
                        CommittedPolynomial::UntrustedAdvice => self
                            .opening_accumulator
                            .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(p, _)| p.len())
                            .unwrap_or(0),
                        _ => 0,
                    };
                    let col_prefix_factor: F = r_cols2
                        .iter()
                        .skip(advice_vars)
                        .map(|r| F::one() - (*r).into())
                        .product();

                    let predicted_scaled =
                        raw_opening.map(|raw| raw * row_factor * col_prefix_factor);

                    // Recover embedded evaluation (divide out coeff_sum if nonzero)
                    let embedded_eval = if coeff_sum.is_zero() {
                        None
                    } else {
                        Some(*eval_contrib * coeff_sum.inverse().unwrap())
                    };

                    // Expected advice opening point (big-endian) derived from the Stage 8 opening point:
                    // take the low `advice_vars` column challenges (little-endian), then reverse to big-endian.
                    let expected_point_be: Vec<F::Challenge> =
                        r_cols2.iter().take(advice_vars).copied().rev().collect();

                    // Actual advice opening point from the accumulator (big-endian).
                    let actual_point_be: Option<Vec<F::Challenge>> = match poly_id {
                        CommittedPolynomial::TrustedAdvice => self
                            .opening_accumulator
                            .get_trusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(p, _)| p.r.clone()),
                        CommittedPolynomial::UntrustedAdvice => self
                            .opening_accumulator
                            .get_untrusted_advice_opening(SumcheckId::AdviceClaimReduction)
                            .map(|(p, _)| p.r.clone()),
                        _ => None,
                    };
                    let point_matches = actual_point_be
                        .as_ref()
                        .map(|p| p.as_slice() == expected_point_be.as_slice())
                        .unwrap_or(false);

                    // If it doesn't match, try to locate where the actual point sits within the
                    // Stage-8 column challenges (in little-endian order).
                    let actual_point_le_pos_in_cols: Option<usize> =
                        actual_point_be.as_ref().and_then(|p_be| {
                            let p_le: Vec<F::Challenge> =
                                p_be.iter().copied().rev().collect::<Vec<_>>();
                            r_cols2
                                .windows(advice_vars)
                                .position(|w| w == p_le.as_slice())
                        });

                    // Also locate where the big-endian point appears inside the big-endian
                    // cycle segment of the Stage 8 opening point.
                    let actual_point_be_pos_in_cycle_be: Option<usize> =
                        actual_point_be.as_ref().and_then(|p_be| {
                            cycle_be2
                                .windows(advice_vars)
                                .position(|w| w == p_be.as_slice())
                        });

                    // Locate the instance-local LE challenges inside the reconstructed global LE r_sumcheck.
                    let actual_point_le_pos_in_r_sumcheck: Option<usize> =
                        actual_point_be.as_ref().and_then(|p_be| {
                            let p_le: Vec<F::Challenge> =
                                p_be.iter().copied().rev().collect::<Vec<_>>();
                            r_sumcheck_reconstructed_le
                                .windows(advice_vars)
                                .position(|w| w == p_le.as_slice())
                        });

                    per_poly_details.push(AdvicePolyDebug {
                        poly_id: *poly_id,
                        eval_contrib: *eval_contrib,
                        claim_contrib,
                        coeff_sum,
                        claim_scaled: claim_scaled_opt,
                        raw_opening,
                        predicted_scaled,
                        embedded_eval,
                        row_factor,
                        col_prefix_factor,
                        point_matches,
                        actual_point_le_pos_in_cols,
                        actual_point_be_pos_in_cycle_be,
                        actual_point_le_pos_in_r_sumcheck,
                        expected_point_be,
                        actual_point_be,
                    });
                }

                panic!(
                    "Stage 8 advice mismatch details:\n  log_k_chunk={log_k_chunk2} log_t={log_t2} sigma={sigma2} nu={} \n  per_advice_eval={per_advice_eval:?}\n  per_poly_details={per_poly_details:?}\n  advice_eval={advice_eval}\n  advice_claim={advice_claim}\n  base_eval={base_eval}\n  base_claim={base_claim}\n  joint_eval={eval_from_vmv}\n  joint_claim={joint_claim}",
                    r_rows2.len().log_2()
                );
            }
        }

        // Dory opening proof at the unified point
        PCS::prove(
            &self.preprocessing.generators,
            &joint_poly,
            &state.opening_point,
            Some(hint),
            &mut self.transcript,
        )
    }
}

pub struct JoltAdvice<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub untrusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    /// Hint for untrusted advice (for batched Dory opening)
    pub untrusted_advice_hint: Option<PCS::OpeningProofHint>,
    /// Hint for trusted advice (for batched Dory opening)
    pub trusted_advice_hint: Option<PCS::OpeningProofHint>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    /// Maximum padded trace length (power of 2)
    pub max_padded_trace_length: usize,
    /// log2 of chunk size for one-hot encoding
    pub log_k_chunk: usize,
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
            max_padded_trace_length: max_T,
            log_k_chunk: log_chunk,
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

    fn commit_trusted_advice_preprocessing_only(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trusted_advice_bytes: &[u8],
    ) -> (
        <DoryCommitmentScheme as crate::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment,
        <DoryCommitmentScheme as crate::poly::commitment::commitment_scheme::CommitmentScheme>::OpeningProofHint,
    ) {
        use crate::poly::{
            commitment::commitment_scheme::CommitmentScheme,
            multilinear_polynomial::MultilinearPolynomial,
        };
        use crate::zkvm::ram::populate_memory_states;

        let max_trusted_advice_size = preprocessing.memory_layout.max_trusted_advice_size;
        let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
        populate_memory_states(0, trusted_advice_bytes, Some(&mut trusted_advice_words), None);

        let poly = MultilinearPolynomial::<Fr>::from(trusted_advice_words);
        let advice_cols = poly.len();

        let _guard =
            crate::poly::commitment::dory::DoryGlobals::initialize_trusted_advice_1row(advice_cols);
        let (commitment, hint) = {
            let _ctx = crate::poly::commitment::dory::DoryGlobals::with_context(
                crate::poly::commitment::dory::DoryContext::TrustedAdvice,
            );
            <crate::poly::commitment::dory::DoryCommitmentScheme as CommitmentScheme>::commit(
                &poly,
                &preprocessing.generators,
            )
        };
        (commitment, hint)
    }

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
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
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
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );

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
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
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
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
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
    fn sha2_e2e_dory_with_untrusted_advice_noise() {
        // SHA2 guest does not consume advice, but providing untrusted advice should still:
        // - commit it in its dedicated 1-row Dory context,
        // - reduce its claims in Stage 6,
        // - batch it into the single Stage 8 Dory opening proof (streaming VMV path).
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();

        // Add some untrusted advice bytes (well below max_untrusted_advice_size).
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        // Trace once to obtain the program IO (memory layout).
        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &[]);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &untrusted_advice,
            &[],
            None,
            None,
        );
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
            "Outputs mismatch (untrusted advice noise should not affect sha2 output)"
        );
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory_with_trusted_advice_noise() {
        // SHA2 guest does not consume advice, but providing trusted advice should still:
        // - use preprocessing-only commit (TrustedAdvice 1-row context),
        // - reduce its claims in Stage 6,
        // - batch it into the single Stage 8 Dory opening proof (streaming VMV path).
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();

        // Add some trusted advice bytes (well below max_trusted_advice_size).
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();

        // Trace once to obtain the program IO (memory layout).
        let (_, _, _, io_device) = program.trace(&inputs, &[], &trusted_advice);

        let preprocessing = JoltProverPreprocessing::gen(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &trusted_advice,
            Some(trusted_commitment.clone()),
            Some(trusted_hint),
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_commitment),
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
            "Outputs mismatch (trusted advice noise should not affect sha2 output)"
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

        let (_, _trace, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

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

        let poly = MultilinearPolynomial::<ark_bn254::Fr>::from(trusted_advice_words);
        let advice_cols = poly.len();
        // Commit trusted advice in its dedicated Dory context, using a fixed 1-row matrix.
        // This makes the commitment preprocessing-only (independent of trace length) while still
        // allowing it to be batched into the single Stage 8 Dory opening proof.
        let _guard =
            crate::poly::commitment::dory::DoryGlobals::initialize_trusted_advice_1row(advice_cols);
        let (trusted_advice_commitment, trusted_advice_hint) = {
            let _ctx = crate::poly::commitment::dory::DoryGlobals::with_context(
                crate::poly::commitment::dory::DoryContext::TrustedAdvice,
            );
            <crate::poly::commitment::dory::DoryCommitmentScheme as CommitmentScheme>::commit(
                &poly,
                &preprocessing.generators,
            )
        };

        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_advice_commitment),
            Some(trusted_advice_hint), // Pass hint for batched Stage 8 opening
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
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &[], &[], &[], None, None);
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
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
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
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &[50], &[], &[], None, None);
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
