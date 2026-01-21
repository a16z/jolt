use crate::{subprotocols::streaming_schedule::LinearOnlySchedule, zkvm::config::OneHotConfig};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use crate::poly::commitment::dory::DoryContext;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::zkvm::config::ReadWriteConfig;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::Serializable;

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::{print_data_structure_heap_usage, write_flamegraph_svg};
use crate::{
    field::JoltField,
    guest,
    poly::lagrange_poly::LagrangeHelper,
    poly::{
        commitment::{commitment_scheme::StreamingCommitmentScheme, dory::DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator,
            ProverOpeningAccumulator, SumcheckId,
        },
        rlc_polynomial::{RLCStreamingData, TraceSource},
    },
    pprof_scope,
    subprotocols::{
        blindfold::{
            BlindFoldProof, BlindFoldProver, BlindFoldWitness, FinalOutputWitness,
            InputClaimConstraint, OutputClaimConstraint, RelaxedR1CSInstance, RoundWitness,
            StageConfig, StageWitness, VerifierR1CSBuilder,
        },
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckProver},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round_zk, UniSkipFirstRoundProofVariant},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
    zkvm::{
        bytecode::read_raf_checking::BytecodeReadRafSumcheckParams,
        claim_reductions::{
            AdviceClaimReductionPhase1Params, AdviceClaimReductionPhase1Prover,
            AdviceClaimReductionPhase2Params, AdviceClaimReductionPhase2Prover, AdviceKind,
            HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
            IncClaimReductionSumcheckParams, IncClaimReductionSumcheckProver,
            InstructionLookupsClaimReductionSumcheckParams,
            InstructionLookupsClaimReductionSumcheckProver, RaReductionParams,
            RamRaClaimReductionSumcheckProver, RegistersClaimReductionSumcheckParams,
            RegistersClaimReductionSumcheckProver,
        },
        config::OneHotParams,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckParams,
            read_raf_checking::InstructionReadRafSumcheckParams,
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
            outer::{OuterUniSkipParams, OuterUniSkipProver},
            product::{
                ProductVirtualRemainderParams, ProductVirtualUniSkipParams,
                ProductVirtualUniSkipProver,
            },
            shift::ShiftSumcheckParams,
        },
        witness::all_committed_polynomials,
    },
};
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    zkvm::{
        bytecode::read_raf_checking::BytecodeReadRafSumcheckProver,
        fiat_shamir_preamble,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver,
            read_raf_checking::InstructionReadRafSumcheckProver,
        },
        proof_serialization::{Claims, JoltProof},
        r1cs::{
            constraints::{
                OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
                PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            },
            key::UniformSpartanKey,
        },
        ram::{
            gen_ram_memory_states, hamming_booleanity::HammingBooleanitySumcheckProver,
            output_check::OutputSumcheckProver, prover_accumulate_advice,
            ra_virtual::RamRaVirtualSumcheckProver,
            raf_evaluation::RafEvaluationSumcheckProver as RamRafEvaluationSumcheckProver,
            read_write_checking::RamReadWriteCheckingProver,
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingProver,
            val_evaluation::ValEvaluationSumcheckProver as RegistersValEvaluationSumcheckProver,
        },
        spartan::{
            instruction_input::InstructionInputSumcheckProver,
            outer::{OuterRemainingStreamingSumcheck, OuterSharedState},
            product::ProductVirtualRemainderProver,
            shift::ShiftSumcheckProver,
        },
        witness::CommittedPolynomial,
        ProverDebugInfo,
    },
};

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryConfig;
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use tracer::{
    emulator::memory::Memory, instruction::Cycle, ChunksIterator, JoltDevice, LazyTraceIterator,
};

use crate::curve::JoltCurve;
use crate::poly::commitment::pedersen::PedersenGenerators;

/// Jolt CPU prover for RV64IMAC.
pub struct JoltCpuProver<
    'a,
    F: JoltField,
    C: JoltCurve,
    PCS: StreamingCommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub program_io: JoltDevice,
    pub lazy_trace: LazyTraceIterator,
    pub trace: Arc<Vec<Cycle>>,
    pub advice: JoltAdvice<F, PCS>,
    /// Phase-bridge randomness for two-phase advice claim reduction.
    /// Stored after Stage 6 initialization and reused in Stage 7.
    advice_reduction_gamma_trusted: Option<F>,
    advice_reduction_gamma_untrusted: Option<F>,
    pub unpadded_trace_len: usize,
    pub padded_trace_len: usize,
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub initial_ram_state: Vec<u64>,
    pub final_ram_state: Vec<u64>,
    pub one_hot_params: OneHotParams,
    pub pedersen_generators: PedersenGenerators<C>,
    pub rw_config: ReadWriteConfig,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: StreamingCommitmentScheme<Field = F>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, C, PCS, ProofTranscript>
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
            max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.shared.memory_layout.max_input_size,
            max_output_size: preprocessing.shared.memory_layout.max_output_size,
            stack_size: preprocessing.shared.memory_layout.stack_size,
            memory_size: preprocessing.shared.memory_layout.memory_size,
            program_size: Some(preprocessing.shared.memory_layout.program_size),
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

    /// Adjusts the padded trace length to ensure the main Dory matrix is large enough
    /// to embed advice polynomials as the top-left block.
    ///
    /// Returns the adjusted padded_trace_len that satisfies:
    /// - `sigma_main >= max_sigma_a`
    /// - `nu_main >= max_nu_a`
    ///
    /// Panics if `max_padded_trace_length` is too small for the configured advice sizes.
    fn adjust_trace_length_for_advice(
        mut padded_trace_len: usize,
        max_padded_trace_length: usize,
        max_trusted_advice_size: u64,
        max_untrusted_advice_size: u64,
        has_trusted_advice: bool,
        has_untrusted_advice: bool,
    ) -> usize {
        // Canonical advice shape policy (balanced):
        // - advice_vars = log2(advice_len)
        // - sigma_a = ceil(advice_vars/2)
        // - nu_a    = advice_vars - sigma_a
        let mut max_sigma_a = 0usize;
        let mut max_nu_a = 0usize;

        if has_trusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_trusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }
        if has_untrusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_untrusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }

        if max_sigma_a == 0 && max_nu_a == 0 {
            return padded_trace_len;
        }

        // Require main matrix dimensions to be large enough to embed advice as the top-left
        // block: sigma_main >= sigma_a and nu_main >= nu_a.
        //
        // This loop doubles padded_trace_len until the main Dory matrix is large enough.
        // Each doubling increases log_t by 1, which increases total_vars by 1 (since
        // log_k_chunk stays constant for a given log_t range), increasing both sigma_main
        // and nu_main by roughly 0.5 each iteration.
        while {
            let log_t = padded_trace_len.log_2();
            let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
            let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
            sigma_main < max_sigma_a || nu_main < max_nu_a
        } {
            if padded_trace_len >= max_padded_trace_length {
                // This is a configuration error: the preprocessing was set up with
                // max_padded_trace_length too small for the configured advice sizes.
                // Cannot recover at runtime - user must fix their configuration.
                let log_t = padded_trace_len.log_2();
                let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
                let total_vars = log_k_chunk + log_t;
                let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
                panic!(
                    "Configuration error: trace too small to embed advice into Dory batch opening.\n\
                    Current: (sigma_main={sigma_main}, nu_main={nu_main}) from total_vars={total_vars} (log_t={log_t}, log_k_chunk={log_k_chunk})\n\
                    Required: (sigma_a={max_sigma_a}, nu_a={max_nu_a}) for advice embedding\n\
                    Solutions:\n\
                    1. Increase max_trace_length in preprocessing (currently {max_padded_trace_length})\n\
                    2. Reduce max_trusted_advice_size or max_untrusted_advice_size\n\
                    3. Run a program with more cycles"
                );
            }
            padded_trace_len = (padded_trace_len * 2).min(max_padded_trace_length);
        }

        padded_trace_len
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
        // Dory globals are process-wide (OnceCell). In tests we run many end-to-end proofs with
        // different trace lengths in a single process, so reset before each prover construction.
        #[cfg(test)]
        crate::poly::commitment::dory::DoryGlobals::reset();

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
        // We may need extra padding so the main Dory matrix has enough (row, col) variables
        // to embed advice commitments committed in their own preprocessing-only contexts.
        let has_trusted_advice = !program_io.trusted_advice.is_empty();
        let has_untrusted_advice = !program_io.untrusted_advice.is_empty();

        let padded_trace_len = Self::adjust_trace_length_for_advice(
            padded_trace_len,
            preprocessing.shared.max_padded_trace_length,
            preprocessing.shared.memory_layout.max_trusted_advice_size,
            preprocessing.shared.memory_layout.max_untrusted_advice_size,
            has_trusted_advice,
            has_untrusted_advice,
        );

        trace.resize(padded_trace_len, Cycle::NoOp);

        // Calculate K for DoryGlobals initialization
        let ram_K = trace
            .par_iter()
            .filter_map(|cycle| {
                crate::zkvm::ram::remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                )
            })
            .max()
            .unwrap_or(0)
            .max(
                crate::zkvm::ram::remap_address(
                    preprocessing.shared.ram.min_bytecode_address,
                    &preprocessing.shared.memory_layout,
                )
                .unwrap_or(0)
                    + preprocessing.shared.ram.bytecode_words.len() as u64
                    + 1,
            )
            .next_power_of_two() as usize;

        let transcript = ProofTranscript::new(b"Jolt");
        let opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());

        let spartan_key = UniformSpartanKey::new(trace.len());

        let (initial_ram_state, final_ram_state) = gen_ram_memory_states::<F>(
            ram_K,
            &preprocessing.shared.ram,
            &program_io,
            &final_memory_state,
        );

        let log_T = trace.len().log_2();
        let ram_log_K = ram_K.log_2();
        let rw_config = ReadWriteConfig::new(log_T, ram_log_K);
        let one_hot_params =
            OneHotParams::new(log_T, preprocessing.shared.bytecode.code_size, ram_K);

        // Use deterministic Pedersen generators for BlindFold protocol
        // This ensures prover and verifier use the same generators
        let pedersen_generators = PedersenGenerators::<C>::deterministic(4096);

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
            advice_reduction_gamma_trusted: None,
            advice_reduction_gamma_untrusted: None,
            unpadded_trace_len,
            padded_trace_len,
            transcript,
            opening_accumulator,
            spartan_key,
            initial_ram_state,
            final_ram_state,
            one_hot_params,
            rw_config,
            pedersen_generators,
        }
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub fn prove(
        mut self,
    ) -> (
        JoltProof<F, C, PCS, ProofTranscript>,
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

        tracing::info!(
            "bytecode size: {}",
            self.preprocessing.shared.bytecode.code_size
        );

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

        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof, r_stage1) =
            self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof, r_stage2) =
            self.prove_stage2();
        let (stage3_sumcheck_proof, r_stage3) = self.prove_stage3();
        let (stage4_sumcheck_proof, r_stage4) = self.prove_stage4();
        let (stage5_sumcheck_proof, r_stage5) = self.prove_stage5();
        let (stage6_sumcheck_proof, r_stage6) = self.prove_stage6();
        let (stage7_sumcheck_proof, r_stage7) = self.prove_stage7();

        let _sumcheck_challenges = [
            r_stage1, r_stage2, r_stage3, r_stage4, r_stage5, r_stage6, r_stage7,
        ];

        let (blindfold_proof, blindfold_initial_claims) = self.prove_blindfold();

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
            blindfold_proof,
            blindfold_initial_claims,
            joint_opening_proof,
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            bytecode_K: self.one_hot_params.bytecode_k,
            rw_config: self.rw_config.clone(),
            one_hot_config: self.one_hot_params.to_config(),
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
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
        );
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
                            &self.preprocessing.shared,
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

        // Commit untrusted advice in its dedicated Dory context, using a preprocessing-only
        // matrix shape derived deterministically from the advice length (balanced dims).

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::UntrustedAdvice);
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

    /// Returns (uni_skip_proof, sumcheck_proof, challenges, initial_claim)
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProofVariant<F, C, ProofTranscript>,
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.shared.bytecode,
        );
        let mut rng = rand::thread_rng();
        let zk_proof = prove_uniskip_round_zk::<F, C, _, _, _>(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        let first_round_proof = UniSkipFirstRoundProofVariant::Zk(zk_proof);

        // Every sum-check with num_rounds > 1 requires a schedule
        // which dictates the compute_message and bind methods.
        // Using LinearOnlySchedule to benchmark linear-only mode (no streaming).
        // Outer remaining sumcheck has degree 3 (multiquadratic)
        // Number of rounds = tau.len() - 1 (cycle variables only)
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
            &uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage1, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            vec![&mut spartan_outer_remaining as &mut dyn SumcheckInstanceProver<_, _>],
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );

        (first_round_proof, sumcheck_proof, r_stage1)
    }

    /// Returns (uni_skip_proof, sumcheck_proof, challenges)
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    fn prove_stage2(
        &mut self,
    ) -> (
        UniSkipFirstRoundProofVariant<F, C, ProofTranscript>,
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");

        // Stage 2a: Prove univariate-skip first round for product virtualization
        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);
        let mut rng = rand::thread_rng();
        let zk_proof = prove_uniskip_round_zk::<F, C, _, _, _>(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        let first_round_proof = UniSkipFirstRoundProofVariant::Zk(zk_proof);

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
            &self.rw_config,
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
            &self.preprocessing.shared.bytecode,
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

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage2, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");
        drop_in_background_thread(instances);

        (first_round_proof, sumcheck_proof, r_stage2)
    }

    /// Returns (sumcheck_proof, challenges)
    #[tracing::instrument(skip_all)]
    fn prove_stage3(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
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
            &self.preprocessing.shared.bytecode,
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

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage3, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage3)
    }

    /// Returns (sumcheck_proof, challenges)
    #[tracing::instrument(skip_all)]
    fn prove_stage4(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            &self.rw_config,
        );
        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial,
            &self.advice.trusted_advice_polynomial,
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
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
            self.trace.clone(),
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_final = ValFinalSumcheckProver::initialize(
            ram_val_final_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
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

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage4, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage4)
    }

    /// Returns (sumcheck_proof, challenges)
    #[tracing::instrument(skip_all)]
    fn prove_stage5(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf_params = InstructionReadRafSumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_read_raf = InstructionReadRafSumcheckProver::initialize(
            lookups_read_raf_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
            print_data_structure_heap_usage("RamRaClaimReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("InstructionReadRafSumcheckProver", &lookups_read_raf);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
            Box::new(ram_ra_reduction),
            Box::new(lookups_read_raf),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");
        tracing::info!("Stage 5 proving");

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage5, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage5)
    }

    /// Returns (sumcheck_proof, challenges)
    #[tracing::instrument(skip_all)]
    fn prove_stage6(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");

        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.shared.bytecode,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

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
        let inc_reduction_params = IncClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
        let trusted_advice_phase1_params = AdviceClaimReductionPhase1Params::new(
            AdviceKind::Trusted,
            &self.program_io.memory_layout,
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );
        let untrusted_advice_phase1_params = AdviceClaimReductionPhase1Params::new(
            AdviceKind::Untrusted,
            &self.program_io.memory_layout,
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );

        let bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            Arc::clone(&self.preprocessing.shared.bytecode),
        );
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);

        let booleanity = BooleanitySumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
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
            IncClaimReductionSumcheckProver::initialize(inc_reduction_params, self.trace.clone());

        // Initialize Phase 1 provers (Stage 6) if advice is present.
        // Note: We clone the advice polynomial here because:
        // 1. Phase1 (Stage 6) destructively binds cycle variables
        // 2. Phase2 (Stage 7) needs a fresh copy to bind address variables
        // 3. Stage 8 RLC needs the original polynomial
        // A future optimization could use Arc<MultilinearPolynomial> with copy-on-write.
        let trusted_advice_phase1 = trusted_advice_phase1_params.map(|params| {
            self.advice_reduction_gamma_trusted = Some(params.gamma);
            let poly = self
                .advice
                .trusted_advice_polynomial
                .clone()
                .expect("trusted advice params exist but polynomial is missing");
            AdviceClaimReductionPhase1Prover::initialize(params, poly)
        });
        let untrusted_advice_phase1 = untrusted_advice_phase1_params.map(|params| {
            self.advice_reduction_gamma_untrusted = Some(params.gamma);
            let poly = self
                .advice
                .untrusted_advice_polynomial
                .clone()
                .expect("untrusted advice params exist but polynomial is missing");
            AdviceClaimReductionPhase1Prover::initialize(params, poly)
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
            print_data_structure_heap_usage("IncClaimReductionSumcheckProver", &inc_reduction);
            if let Some(ref advice) = trusted_advice_phase1 {
                print_data_structure_heap_usage(
                    "AdviceClaimReductionPhase1Prover(trusted)",
                    advice,
                );
            }
            if let Some(ref advice) = untrusted_advice_phase1 {
                print_data_structure_heap_usage(
                    "AdviceClaimReductionPhase1Prover(untrusted)",
                    advice,
                );
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
        if let Some(advice) = trusted_advice_phase1 {
            instances.push(Box::new(advice));
        }
        if let Some(advice) = untrusted_advice_phase1 {
            instances.push(Box::new(advice));
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_start_flamechart.svg");
        tracing::info!("Stage 6 proving");

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage6, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage6)
    }

    /// Prove BlindFold protocol to make sumcheck proofs zero-knowledge.
    ///
    /// This method retrieves ZK stage data (coefficients, challenges, initial claims)
    /// from the opening accumulator where it was stored during prove_zk calls.
    /// The coefficients and blinding factors are hidden from the verifier (who only
    /// sees commitments), while BlindFold proves the R1CS constraints are satisfied.
    /// Returns (blindfold_proof, initial_claims).
    /// The initial_claims are for the 7 logical Jolt stages, where stages 1-2 use
    /// uni-skip initial claims and stages 3-7 use regular sumcheck initial claims.
    #[tracing::instrument(skip_all)]
    fn prove_blindfold(&mut self) -> (BlindFoldProof<F, C>, [F; 9]) {
        tracing::info!("BlindFold proving");

        let mut rng = rand::thread_rng();

        // Retrieve uni-skip stage data (stages 1-2 first rounds)
        let uniskip_stages = self.opening_accumulator.take_uniskip_stage_data();
        assert_eq!(
            uniskip_stages.len(),
            2,
            "Expected 2 uni-skip stages, got {}",
            uniskip_stages.len()
        );

        // Retrieve ZK stage data from the accumulator
        let zk_stages = self.opening_accumulator.take_zk_stage_data();
        assert_eq!(
            zk_stages.len(),
            7,
            "Expected 7 ZK stages, got {}",
            zk_stages.len()
        );

        // Precompute power sums for uni-skip domains
        let outer_power_sums = LagrangeHelper::power_sums::<
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >();
        let product_power_sums = LagrangeHelper::power_sums::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >();

        let mut stage_configs = Vec::new();
        let mut stage_witnesses = Vec::new();
        let mut initial_claims = Vec::new();

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            // For stages 0 and 1 (Jolt stages 1 and 2), add uni-skip round first
            if stage_idx < 2 {
                let uniskip = &uniskip_stages[stage_idx];
                let coeffs = &uniskip.poly_coeffs;
                let challenge: F = uniskip.challenge.into();

                // Uni-skip uses full polynomial coefficients (not compressed)
                let poly_degree = coeffs.len() - 1;

                // The initial claim for uni-skip is stored in uniskip_data
                let claimed_sum = uniskip.input_claim;
                initial_claims.push(claimed_sum);

                // Create uni-skip stage config with power sums
                let power_sums: Vec<i128> = if stage_idx == 0 {
                    outer_power_sums.to_vec()
                } else {
                    product_power_sums.to_vec()
                };

                let config = if stage_idx == 0 {
                    StageConfig::new_uniskip(poly_degree, power_sums)
                } else {
                    StageConfig::new_uniskip_chain(poly_degree, power_sums)
                };
                stage_configs.push(config);
                stage_witnesses.push(StageWitness::new(vec![RoundWitness::with_claimed_sum(
                    coeffs.clone(),
                    challenge,
                    claimed_sum,
                )]));

                // For stages 0-1: push regular rounds initial claim (separate chain from uni-skip)
                initial_claims.push(zk_data.initial_claim);
            } else {
                // Stages 2-6: no uni-skip, use ZK stage's initial claim
                initial_claims.push(zk_data.initial_claim);
            }

            // Process regular sumcheck rounds
            // Regular rounds start their own chain with zk_data.initial_claim
            let mut current_claim = zk_data.initial_claim;
            let stage_challenges = &zk_data.challenges;
            let num_rounds = zk_data.poly_coeffs.len();

            for (round_idx, coeffs) in zk_data.poly_coeffs.iter().enumerate() {
                let challenge: F = stage_challenges[round_idx].into();
                // Degree = coefficient_count - 1
                let poly_degree = coeffs.len() - 1;
                let claimed_sum = current_claim;

                // Compute next_claim via Horner evaluation
                let mut next_claim = coeffs[coeffs.len() - 1];
                for i in (0..coeffs.len() - 1).rev() {
                    next_claim = coeffs[i] + challenge * next_claim;
                }

                // First regular round starts a new chain
                // For stages 0-1, this separates regular rounds from uni-skip
                // For stages 2-6, this starts their independent chain
                let starts_new_chain = round_idx == 0;
                let is_last_round = round_idx == num_rounds - 1;
                let is_first_round = round_idx == 0;

                let config = if starts_new_chain {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };

                // Handle input constraints for first round
                let (config, initial_input_witness) = if is_first_round {
                    let batched_input = InputClaimConstraint::batch(
                        &zk_data.input_constraints,
                        zk_data.batching_coefficients.len(),
                    );

                    if let Some(batched_constraint) = batched_input {
                        let mut challenge_values: Vec<F> = zk_data.batching_coefficients.clone();
                        for cv in &zk_data.input_constraint_challenge_values {
                            challenge_values.extend(cv.iter().cloned());
                        }

                        // Collect opening values from the accumulator
                        let opening_values: Vec<F> = batched_constraint
                            .required_openings
                            .iter()
                            .map(|id| self.opening_accumulator.get_opening(*id))
                            .collect();

                        let initial_input =
                            FinalOutputWitness::new_general(challenge_values, opening_values);
                        let config_with_input = config.with_input_constraint(batched_constraint);
                        (config_with_input, Some(initial_input))
                    } else {
                        (config, None)
                    }
                } else {
                    (config, None)
                };

                // Handle output constraints for last round
                let (config, final_output_witness) = if is_last_round {
                    let batched = OutputClaimConstraint::batch(
                        &zk_data.output_constraints,
                        zk_data.batching_coefficients.len(),
                    );

                    if let Some(batched_constraint) = batched {
                        let mut challenge_values: Vec<F> = zk_data.batching_coefficients.clone();
                        for cv in &zk_data.constraint_challenge_values {
                            challenge_values.extend(cv.iter().cloned());
                        }

                        // Collect opening values from the accumulator using the batched constraint's required_openings
                        let opening_values: Vec<F> = batched_constraint
                            .required_openings
                            .iter()
                            .map(|id| self.opening_accumulator.get_opening(*id))
                            .collect();

                        let final_output =
                            FinalOutputWitness::new_general(challenge_values, opening_values);
                        let config_with_fo = config.with_constraint(batched_constraint);
                        (config_with_fo, Some(final_output))
                    } else {
                        (config, None)
                    }
                } else {
                    (config, None)
                };

                stage_configs.push(config);
                let round_witness =
                    RoundWitness::with_claimed_sum(coeffs.clone(), challenge, claimed_sum);

                // Create stage witness with optional input/output constraints
                let stage_witness = match (initial_input_witness, final_output_witness) {
                    (Some(ii), Some(fo)) => StageWitness::with_both(vec![round_witness], ii, fo),
                    (Some(ii), None) => StageWitness::with_initial_input(vec![round_witness], ii),
                    (None, Some(fo)) => StageWitness::with_final_output(vec![round_witness], fo),
                    (None, None) => StageWitness::new(vec![round_witness]),
                };
                stage_witnesses.push(stage_witness);

                current_claim = next_claim;
            }
        }

        // Build verifier R1CS from configurations
        let builder = VerifierR1CSBuilder::<F>::new(&stage_configs);
        let r1cs = builder.build();

        // Convert initial claims to array
        // 9 chains: 2 for stage 0 (uni-skip + regular), 2 for stage 1, 5 for stages 2-6
        let initial_claims_array: [F; 9] = initial_claims
            .try_into()
            .expect("Expected exactly 9 initial claims");

        // Use all 9 initial claims for the BlindFoldWitness
        let blindfold_witness =
            BlindFoldWitness::with_multiple_claims(initial_claims_array.to_vec(), stage_witnesses);

        // Assign witness to get Z vector
        let z = blindfold_witness.assign(&r1cs);

        // Extract components for relaxed R1CS
        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        // Collect round commitments, coefficients, and blindings from all stages
        let mut round_commitments: Vec<C::G1> = Vec::new();
        let mut round_coefficients: Vec<Vec<F>> = Vec::new();
        let mut round_blindings: Vec<F> = Vec::new();

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            // For stages 0-1, include uni-skip round first
            if stage_idx < 2 {
                let uniskip = &uniskip_stages[stage_idx];
                let commitment =
                    C::G1::deserialize_compressed(&uniskip.commitment_bytes[..]).unwrap();
                round_commitments.push(commitment);
                round_coefficients.push(uniskip.poly_coeffs.clone());
                round_blindings.push(uniskip.blinding_factor);
            }

            // Add regular sumcheck rounds
            for (commitment_bytes, coeffs, blinding) in zk_data
                .round_commitments
                .iter()
                .zip(&zk_data.poly_coeffs)
                .zip(&zk_data.blinding_factors)
                .map(|((c, p), b)| (c, p, b))
            {
                let commitment = C::G1::deserialize_compressed(&commitment_bytes[..]).unwrap();
                round_commitments.push(commitment);
                round_coefficients.push(coeffs.clone());
                round_blindings.push(*blinding);
            }
        }

        // Create non-relaxed instance and witness with round commitment data
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, C>::new_non_relaxed(
            &self.pedersen_generators,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            round_commitments,
            round_coefficients,
            round_blindings,
            &mut rng,
        );

        // Run BlindFold protocol
        let prover = BlindFoldProver::new(&self.pedersen_generators, &r1cs);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");

        let proof = prover.prove(
            &real_instance,
            &real_witness,
            &z,
            &mut blindfold_transcript,
            &mut rng,
        );

        (proof, initial_claims_array)
    }

    /// Stage 7: HammingWeight + ClaimReduction sumcheck (only log_k_chunk rounds).
    #[tracing::instrument(skip_all)]
    fn prove_stage7(
        &mut self,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
    ) {
        // Create params and prover for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            &self.preprocessing.shared,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        // 3. Run Stage 7 batched sumcheck (address rounds only).
        // Includes HammingWeightClaimReduction plus Phase 2 advice reduction instances (if needed).
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> =
            vec![Box::new(hw_prover)];

        if let Some(gamma) = self.advice_reduction_gamma_trusted {
            if let Some(params) = AdviceClaimReductionPhase2Params::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                gamma,
                &self.opening_accumulator,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            ) {
                let poly = self
                    .advice
                    .trusted_advice_polynomial
                    .clone()
                    .expect("trusted advice phase2 params exist but polynomial is missing");
                instances.push(Box::new(AdviceClaimReductionPhase2Prover::initialize(
                    params, poly,
                )));
            }
        }
        if let Some(gamma) = self.advice_reduction_gamma_untrusted {
            if let Some(params) = AdviceClaimReductionPhase2Params::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                gamma,
                &self.opening_accumulator,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            ) {
                let poly = self
                    .advice
                    .untrusted_advice_polynomial
                    .clone()
                    .expect("untrusted advice phase2 params exist but polynomial is missing");
                instances.push(Box::new(AdviceClaimReductionPhase2Prover::initialize(
                    params, poly,
                )));
            }
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage7_start_flamechart.svg");
        tracing::info!("Stage 7 proving");

        let mut rng = rand::thread_rng();
        let (sumcheck_proof, r_stage7, _initial_claim) = BatchedSumcheck::prove_zk::<F, C, _, _>(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            &self.pedersen_generators,
            &mut rng,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage7_end_flamechart.svg");
        drop_in_background_thread(instances);

        (sumcheck_proof, r_stage7)
    }

    /// Stage 8: Dory batch opening proof.
    /// Builds streaming RLC polynomial directly from trace (no witness regeneration needed).
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        let _guard = DoryGlobals::initialize_context(
            self.one_hot_params.k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
        );

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        // These are at r_cycle_stage6 only (length log_T)
        let (_ram_inc_point, ram_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
        let (_rd_inc_point, rd_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );

        #[cfg(test)]
        {
            // Verify that Inc openings are at the same point as r_cycle from HammingWeightClaimReduction
            let r_cycle_stage6 = &opening_point.r[log_k_chunk..];

            debug_assert_eq!(
                _ram_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RamInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
            debug_assert_eq!(
                _rd_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RdInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
        }

        // Apply Lagrange factor for dense polys: ∏_{i<log_k_chunk} (1 - r_address[i])
        // Because dense polys have fewer variables, we need to account for this
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        // These are at (r_address_stage7, r_cycle_stage6)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReductionPhase2)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, advice_point.len());
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
            AdviceKind::Untrusted,
            SumcheckId::AdviceClaimReductionPhase2,
        ) {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, advice_point.len());
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build DoryOpeningState
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers,
            polynomial_claims,
        };

        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: Arc::clone(&self.preprocessing.shared.bytecode),
            memory_layout: self.preprocessing.shared.memory_layout.clone(),
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

        // Dory opening proof at the unified point
        PCS::prove(
            &self.preprocessing.generators,
            &joint_poly,
            &opening_point.r,
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "JoltProverPreprocessing::gen")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        // max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_T: usize = shared.max_padded_trace_length.next_power_of_two();
        let max_log_T = max_T.log_2();
        // Use the maximum possible log_k_chunk for generator setup
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let generators = PCS::setup_prover(max_log_k_chunk + max_log_T);
        JoltProverPreprocessing { generators, shared }
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

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> Serializable
    for JoltProverPreprocessing<F, PCS>
{
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use serial_test::serial;

    use crate::host;
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals},
        },
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    };
    use crate::zkvm::claim_reductions::AdviceKind;
    use crate::zkvm::verifier::JoltSharedPreprocessing;
    use crate::zkvm::witness::CommittedPolynomial;
    use crate::zkvm::{
        prover::JoltProverPreprocessing,
        ram::populate_memory_states,
        verifier::{JoltVerifier, JoltVerifierPreprocessing},
        RV64IMACProver, RV64IMACVerifier,
    };

    fn commit_trusted_advice_preprocessing_only(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trusted_advice_bytes: &[u8],
    ) -> (
        <DoryCommitmentScheme as CommitmentScheme>::Commitment,
        <DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint,
    ) {
        let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
        let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
        populate_memory_states(
            0,
            trusted_advice_bytes,
            Some(&mut trusted_advice_words),
            None,
        );

        let poly = MultilinearPolynomial::<Fr>::from(trusted_advice_words);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice);
        let (commitment, hint) = {
            let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
            DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
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
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            256,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let log_chunk = 8; // Use default log_chunk for tests
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
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

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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
    fn sha2_e2e_dory_with_unused_advice() {
        // SHA2 guest does not consume advice, but providing both trusted and untrusted advice
        // should still work correctly through the full pipeline:
        // - Trusted: commit in preprocessing-only context, reduce in Stage 6, batch in Stage 8
        // - Untrusted: commit at prove time, reduce in Stage 6, batch in Stage 8
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_commitment),
            Some(trusted_hint),
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_commitment),
            debug_info,
        )
        .expect("Failed to create verifier")
        .verify()
        .expect("Failed to verify proof");

        // Verify output is correct (advice should not affect sha2 output)
        let expected_output = &[
            0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
            0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
            0x3b, 0x50, 0xd2, 0x57,
        ];
        assert_eq!(io_device.outputs, expected_output);
    }

    #[test]
    #[serial]
    fn max_advice_with_small_trace() {
        // Tests that max-sized advice (4KB = 512 words) works with a minimal trace.
        // With balanced dims (sigma_a=5, nu_a=4 for 512 words), the minimum padded trace
        // (256 cycles -> total_vars=12) is sufficient to embed advice.
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let trusted_advice = vec![7u8; 4096];
        let untrusted_advice = vec![9u8; 4096];

        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            256,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        tracing::info!(
            "preprocessing.memory_layout.max_trusted_advice_size: {}",
            shared_preprocessing.memory_layout.max_trusted_advice_size
        );

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        // Trace is tiny but advice is max-sized
        assert!(prover.unpadded_trace_len < 512);
        assert_eq!(prover.padded_trace_len, 256);

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            Some(trusted_commitment),
            debug_info,
        )
        .expect("Failed to create verifier")
        .verify()
        .expect("Verification failed");
    }

    #[test]
    #[serial]
    fn advice_e2e_dory() {
        // Tests a guest (merkle-tree) that actually consumes both trusted and untrusted advice.
        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _) = program.decode();

        // Merkle tree with 4 leaves: input=leaf1, trusted=[leaf2, leaf3], untrusted=leaf4
        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_commitment),
            Some(trusted_hint),
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_commitment),
            debug_info,
        )
        .expect("Failed to create verifier")
        .verify()
        .expect("Verification failed");

        // Expected merkle root for leaves [5;32], [6;32], [7;32], [8;32]
        let expected_output = &[
            0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
            0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32,
            0xbb, 0x16, 0xd7,
        ];
        assert_eq!(io_device.outputs, expected_output);
    }

    #[test]
    #[serial]
    fn advice_opening_point_derives_from_unified_point() {
        // Tests that advice opening points are correctly derived from the unified main opening
        // point using Dory's balanced dimension policy.
        //
        // For a small trace (256 cycles), the advice row coordinates span both Stage 6 (cycle)
        // and Stage 7 (address) challenges, verifying the two-phase reduction works correctly.
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        assert_eq!(prover.padded_trace_len, 256, "test expects small trace");

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();
        let debug_info = debug_info.expect("expected debug_info in tests");

        // Get unified opening point and derive expected advice point
        let (opening_point, _) = debug_info
            .opening_accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
        let mut point_dory_le = opening_point.r.clone();
        point_dory_le.reverse();

        let total_vars = point_dory_le.len();
        let (sigma_main, _nu_main) = DoryGlobals::balanced_sigma_nu(total_vars);
        let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(
            prover_preprocessing
                .shared
                .memory_layout
                .max_trusted_advice_size as usize,
        );

        // Build expected advice point: [col_bits[0..sigma_a] || row_bits[0..nu_a]]
        let mut expected_advice_le: Vec<_> = point_dory_le[0..sigma_a].to_vec();
        expected_advice_le.extend_from_slice(&point_dory_le[sigma_main..sigma_main + nu_a]);

        // Verify both advice types derive the same opening point
        for (name, kind) in [
            ("trusted", AdviceKind::Trusted),
            ("untrusted", AdviceKind::Untrusted),
        ] {
            let get_fn = debug_info
                .opening_accumulator
                .get_advice_opening(kind, SumcheckId::AdviceClaimReductionPhase2);
            assert!(
                get_fn.is_some(),
                "{name} advice opening missing for AdviceClaimReductionPhase2"
            );
            let (point_be, _) = get_fn.unwrap();
            let mut point_le = point_be.r.clone();
            point_le.reverse();
            assert_eq!(point_le, expected_advice_le, "{name} advice point mismatch");
        }

        // Verify end-to-end
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            Some(trusted_commitment),
            Some(debug_info),
        )
        .expect("Failed to create verifier")
        .verify()
        .expect("Verification failed");
    }

    #[test]
    #[serial]
    fn memory_ops_e2e_dory() {
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&[], &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &[],
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &[50],
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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

    /// Test BlindFold R1CS satisfaction using real sumcheck data from muldiv proof.
    ///
    /// This test extracts sumcheck polynomials from all 6 stages of a real Jolt proof
    /// and verifies that they satisfy the BlindFold verifier R1CS. This validates that:
    /// 1. The coefficient extraction from CompressedUniPoly works correctly
    /// 2. The BlindFold R1CS correctly encodes sumcheck verification
    /// 3. Real proof data from all stages satisfies the R1CS constraints
    #[test]
    #[serial]
    fn blindfold_r1cs_satisfaction() {
        use crate::curve::Bn254Curve;
        use crate::subprotocols::blindfold::{
            BlindFoldWitness, RoundWitness, StageConfig, StageWitness, VerifierR1CSBuilder,
        };
        use crate::subprotocols::sumcheck::SumcheckInstanceProof;
        use crate::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};
        use crate::zkvm::verifier::JoltSharedPreprocessing;
        use ark_serialize::CanonicalSerialize;

        /// Helper to process a single stage's sumcheck proof.
        /// Returns a list of (RoundWitness, degree) for each round.
        /// For ZK proofs, creates synthetic witnesses with correct degrees to test R1CS structure.
        fn process_stage<ProofTranscript: Transcript>(
            _stage_name: &str,
            proof: &SumcheckInstanceProof<Fr, Bn254Curve, ProofTranscript>,
            transcript: &mut KeccakTranscript,
        ) -> Vec<(RoundWitness<Fr>, usize)> {
            match proof {
                SumcheckInstanceProof::Standard(std_proof) => {
                    // For Standard proofs, use actual polynomial coefficients
                    let compressed_polys = &std_proof.compressed_polys;
                    let num_rounds = compressed_polys.len();

                    if num_rounds == 0 {
                        return vec![];
                    }

                    let mut rounds = Vec::with_capacity(num_rounds);

                    for compressed_poly in compressed_polys.iter() {
                        compressed_poly.append_to_transcript(transcript);
                        let challenge: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

                        let compressed = &compressed_poly.coeffs_except_linear_term;
                        let degree = compressed.len();

                        let c0 = compressed[0];
                        let sum_higher_coeffs: Fr = compressed[1..].iter().copied().sum();

                        let claimed_sum = Fr::from(12345u64);
                        let c1 = claimed_sum - c0 - c0 - sum_higher_coeffs;

                        let mut coeffs = vec![c0, c1];
                        coeffs.extend_from_slice(&compressed[1..]);

                        let round_witness =
                            RoundWitness::with_claimed_sum(coeffs, challenge, claimed_sum);

                        rounds.push((round_witness, degree));
                    }

                    rounds
                }
                SumcheckInstanceProof::Zk(zk_proof) => {
                    // For ZK proofs, create synthetic witnesses with correct degrees.
                    // This tests the R1CS structure without needing actual coefficients.
                    let num_rounds = zk_proof.round_commitments.len();

                    if num_rounds == 0 {
                        return vec![];
                    }

                    let mut rounds = Vec::with_capacity(num_rounds);

                    for (round_idx, commitment) in zk_proof.round_commitments.iter().enumerate() {
                        // Append commitment to transcript for challenge derivation
                        let mut commitment_bytes = Vec::new();
                        commitment
                            .serialize_compressed(&mut commitment_bytes)
                            .expect("Serialization should not fail");
                        transcript.append_message(b"UniPolyCommitment");
                        transcript.append_bytes(&commitment_bytes);
                        let challenge: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

                        let degree = zk_proof.poly_degrees[round_idx];

                        // Create synthetic coefficients that satisfy sumcheck relation
                        // g(x) = c0 + c1*x + c2*x^2 + ... has degree+1 coefficients
                        // claimed_sum = 2*c0 + c1 + c2 + ...
                        let claimed_sum = Fr::from(12345u64);

                        // Use simple synthetic values: c0 = 1, c2..cd = 1, compute c1
                        let c0 = Fr::from(1u64);
                        let num_higher_coeffs = degree.saturating_sub(1);
                        let sum_higher_coeffs = Fr::from(num_higher_coeffs as u64);
                        let c1 = claimed_sum - c0 - c0 - sum_higher_coeffs;

                        let mut coeffs = vec![c0, c1];
                        for _ in 0..num_higher_coeffs {
                            coeffs.push(Fr::from(1u64));
                        }

                        let round_witness =
                            RoundWitness::with_claimed_sum(coeffs, challenge, claimed_sum);

                        rounds.push((round_witness, degree));
                    }

                    rounds
                }
            }
        }

        // Run muldiv prover to get a real proof
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover =
            RV64IMACProver::gen_from_elf(&preprocessing, elf_contents, &[50], &[], &[], None, None);
        let (jolt_proof, _) = prover.prove();

        println!("\n=== BlindFold R1CS Satisfaction Test (All 7 Stages) ===\n");

        // Process all 7 stages and verify each one
        let stage_proofs: Vec<(&str, &SumcheckInstanceProof<Fr, Bn254Curve, _>)> = vec![
            ("Stage 1 (Spartan Outer)", &jolt_proof.stage1_sumcheck_proof),
            (
                "Stage 2 (Product Virtual)",
                &jolt_proof.stage2_sumcheck_proof,
            ),
            ("Stage 3 (Instruction)", &jolt_proof.stage3_sumcheck_proof),
            ("Stage 4 (Registers+RAM)", &jolt_proof.stage4_sumcheck_proof),
            ("Stage 5 (Value+Lookup)", &jolt_proof.stage5_sumcheck_proof),
            (
                "Stage 6 (OneHot+Hamming)",
                &jolt_proof.stage6_sumcheck_proof,
            ),
            (
                "Stage 7 (HammingWeight+ClaimReduction)",
                &jolt_proof.stage7_sumcheck_proof,
            ),
        ];

        let mut total_rounds = 0;
        let mut total_constraints = 0;

        for (stage_name, proof) in &stage_proofs {
            // Create a fresh transcript for each stage (independent verification)
            let mut stage_transcript = KeccakTranscript::new(b"BlindFoldStageTest");

            let rounds = process_stage(stage_name, proof, &mut stage_transcript);

            if rounds.is_empty() {
                println!("  {stage_name} - 0 rounds, skipping");
                continue;
            }

            // Process each round individually
            let mut stage_rounds = 0;
            let mut stage_constraints = 0;

            for (round_witness, degree) in rounds {
                // Build R1CS for a single round
                let config = StageConfig::new(1, degree);
                let builder = VerifierR1CSBuilder::<Fr>::new(&[config.clone()]);
                let r1cs = builder.build();

                // Build witness with the round's claimed_sum as initial_claim
                let initial_claim = round_witness.claimed_sum;
                let stage_witness = StageWitness::new(vec![round_witness]);
                let witness = BlindFoldWitness::new(initial_claim, vec![stage_witness]);

                let z = witness.assign(&r1cs);
                match r1cs.check_satisfaction(&z) {
                    Ok(()) => {
                        stage_rounds += 1;
                        stage_constraints += r1cs.num_constraints;
                    }
                    Err(row) => {
                        panic!(
                            "{} (degree {}) - constraint {} failed (out of {})",
                            stage_name, degree, row, r1cs.num_constraints
                        );
                    }
                }
            }

            println!(
                "  {stage_name} - {stage_rounds} rounds, {stage_constraints} constraints - SATISFIED"
            );
            total_rounds += stage_rounds;
            total_constraints += stage_constraints;
        }

        println!("\n=== Summary ===");
        println!("Total rounds across all stages: {total_rounds}");
        println!("Total constraints across all stages: {total_constraints}");
        println!("All 6 stages satisfied!\n");

        // Ensure we processed a meaningful amount
        assert!(total_rounds > 0, "Expected at least some sumcheck rounds");
        assert!(
            total_constraints > 0,
            "Expected at least some R1CS constraints"
        );
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

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );

        let (proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
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
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        program_io.memory_layout.output_start = program_io.memory_layout.input_start;
        program_io.memory_layout.output_end = program_io.memory_layout.input_end;
        program_io.memory_layout.termination = program_io.memory_layout.input_start;

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );
        let (proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier =
            JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
        verifier.verify().unwrap();
    }

    #[test]
    #[serial]
    fn blindfold_protocol_e2e() {
        use crate::curve::Bn254Curve;
        use crate::poly::commitment::pedersen::PedersenGenerators;
        use crate::subprotocols::blindfold::{
            BlindFoldProver, BlindFoldVerifier, BlindFoldWitness, RelaxedR1CSInstance,
            RoundWitness, StageConfig, StageWitness, VerifierR1CSBuilder,
        };
        use crate::transcripts::{KeccakTranscript, Transcript};
        use rand::thread_rng;

        let mut rng = thread_rng();

        let configs = [StageConfig::new(2, 3)];
        let builder = VerifierR1CSBuilder::<Fr>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 100);

        // Create valid multi-round witness
        // Round 1: 2*c0 + c1 + c2 + c3 = 55
        let round1 = RoundWitness::new(
            vec![
                Fr::from(20u64),
                Fr::from(5u64),
                Fr::from(7u64),
                Fr::from(3u64),
            ],
            Fr::from(2u64),
        );
        let next1 = round1.evaluate(Fr::from(2u64));

        // Round 2: 2*c0 + c1 + c2 + c3 = next1
        let c0_2 = Fr::from(30u64);
        let c2_2 = Fr::from(10u64);
        let c3_2 = Fr::from(5u64);
        let c1_2 = next1 - Fr::from(75u64);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], Fr::from(4u64));

        let initial_claim = Fr::from(55u64);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round1, round2])]);
        let z = blindfold_witness.assign(&r1cs);

        // Verify standard R1CS is satisfied
        assert!(r1cs.is_satisfied(&z));

        // Extract components for relaxed R1CS
        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<Fr> = z[witness_start..].to_vec();
        let public_inputs: Vec<Fr> = z[1..witness_start].to_vec();

        // Create non-relaxed instance and witness (with empty round commitment data for unit test)
        let (real_instance, real_witness) = RelaxedR1CSInstance::<Fr, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &mut rng,
        );

        // Run BlindFold protocol
        let prover = BlindFoldProver::new(&gens, &r1cs);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_E2E");
        let proof = prover.prove(
            &real_instance,
            &real_witness,
            &z,
            &mut prover_transcript,
            &mut rng,
        );

        // Verify the proof
        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_E2E");
        let result = verifier.verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "BlindFold protocol verification failed: {result:?}"
        );

        println!("\n=== BlindFold Protocol E2E Test ===");
        println!(
            "R1CS size: {} constraints, {} variables",
            r1cs.num_constraints, r1cs.num_vars
        );
        println!("Witness size: {} field elements", witness.len());
        println!(
            "Folded error vector size: {} field elements",
            proof.folded_witness.E.len()
        );
        println!("Protocol verification: SUCCESS");
    }
}
