#[cfg(feature = "zk")]
mod blindfold;
mod opening;
pub(crate) mod preprocessing;
mod stages;
#[cfg(test)]
mod tests;
mod witness;

pub use preprocessing::{JoltAdvice, JoltProverPreprocessing};

use std::sync::Arc;
use std::time::Instant;

use common::jolt_device::MemoryConfig;
use rayon::prelude::*;
use tracer::{emulator::memory::Memory, instruction::Cycle, JoltDevice, LazyTraceIterator};

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::guest;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::pprof_scope;
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstanceProof};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::claim_reductions::AdviceClaimReductionProver;
use crate::zkvm::config::{OneHotConfig, OneHotParams, ReadWriteConfig};
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::ram::gen_ram_memory_states;
use crate::zkvm::witness::CommittedPolynomial;
use crate::zkvm::{fiat_shamir_preamble, ProverDebugInfo};

#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(not(feature = "zk"))]
use crate::subprotocols::univariate_skip::prove_uniskip_round;
#[cfg(feature = "zk")]
use crate::subprotocols::univariate_skip::prove_uniskip_round_zk;

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
    advice_reduction_prover_trusted: Option<AdviceClaimReductionProver<F>>,
    advice_reduction_prover_untrusted: Option<AdviceClaimReductionProver<F>>,
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
    #[cfg(feature = "zk")]
    stage8_zk_data: Option<Stage8ZkData<F>>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
struct Stage8ZkData<F: JoltField> {
    opening_ids: Vec<OpeningId>,
    constraint_coeffs: Vec<F>,
    joint_claim: F,
    y_blinding: F,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, C, PCS, ProofTranscript>
{
    #[allow(clippy::too_many_arguments)]
    pub fn gen_from_elf(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
        advice_tape: Option<tracer::AdviceTape>,
    ) -> Self {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.shared.memory_layout.max_input_size,
            max_output_size: preprocessing.shared.memory_layout.max_output_size,
            stack_size: preprocessing.shared.memory_layout.stack_size,
            heap_size: preprocessing.shared.memory_layout.heap_size,
            program_size: Some(preprocessing.shared.memory_layout.program_size),
        };

        let (lazy_trace, trace, final_memory_state, program_io, _advice_tape_out) = {
            let _pprof_trace = pprof_scope!("trace");
            guest::program::trace(
                elf_contents,
                None,
                inputs,
                untrusted_advice,
                trusted_advice,
                &memory_config,
                advice_tape,
            )
        };

        let num_riscv_cycles: usize = trace
            .par_iter()
            .map(|cycle| {
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

    fn adjust_trace_length_for_advice(
        mut padded_trace_len: usize,
        max_padded_trace_length: usize,
        max_trusted_advice_size: u64,
        max_untrusted_advice_size: u64,
        has_trusted_advice: bool,
        has_untrusted_advice: bool,
    ) -> usize {
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

        while {
            let log_t = padded_trace_len.log_2();
            let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
            let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
            sigma_main < max_sigma_a || nu_main < max_nu_a
        } {
            if padded_trace_len >= max_padded_trace_length {
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
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let unpadded_trace_len = trace.len();
        let padded_trace_len = if unpadded_trace_len < 256 {
            256
        } else {
            (trace.len() + 1).next_power_of_two()
        };
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
            advice_reduction_prover_trusted: None,
            advice_reduction_prover_untrusted: None,
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
            #[cfg(feature = "zk")]
            stage8_zk_data: None,
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
        let start = Instant::now();
        let _pprof_prove = pprof_scope!("prove");

        fiat_shamir_preamble(
            &self.program_io,
            self.one_hot_params.ram_k,
            self.trace.len(),
            &mut self.transcript,
        );

        let (commitments, mut opening_proof_hints) = self.generate_and_commit_witness_polynomials();
        let untrusted_advice_commitment = self.generate_and_commit_untrusted_advice();
        self.generate_and_commit_trusted_advice();

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

        let joint_opening_proof = self.prove_stage8(opening_proof_hints);
        #[cfg(feature = "zk")]
        let blindfold_proof = self.prove_blindfold(&joint_opening_proof);

        #[cfg(not(feature = "zk"))]
        let opening_claims =
            crate::zkvm::proof_serialization::Claims(self.opening_accumulator.openings.clone());

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
            #[cfg(feature = "zk")]
            blindfold_proof,
            joint_opening_proof,
            #[cfg(not(feature = "zk"))]
            opening_claims,
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            bytecode_K: self.one_hot_params.bytecode_k,
            rw_config: self.rw_config.clone(),
            one_hot_config: self.one_hot_params.to_config(),
            dory_layout: DoryGlobals::get_layout(),
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

    fn prove_batched_sumcheck(
        &mut self,
        instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
        #[cfg(feature = "zk")]
        {
            let mut rng = rand::thread_rng();
            BatchedSumcheck::prove_zk::<F, C, _, _>(
                instances,
                &mut self.opening_accumulator,
                &mut self.transcript,
                &self.pedersen_generators,
                &mut rng,
            )
        }
        #[cfg(not(feature = "zk"))]
        {
            let (proof, r, claim) = BatchedSumcheck::prove(
                instances,
                &mut self.opening_accumulator,
                &mut self.transcript,
            );
            (SumcheckInstanceProof::Standard(proof), r, claim)
        }
    }

    fn prove_uniskip(
        &mut self,
        instance: &mut impl SumcheckInstanceProver<F, ProofTranscript>,
    ) -> UniSkipFirstRoundProofVariant<F, C, ProofTranscript> {
        #[cfg(feature = "zk")]
        {
            let mut rng = rand::thread_rng();
            let zk_proof = prove_uniskip_round_zk::<F, C, _, _, _>(
                instance,
                &mut self.opening_accumulator,
                &mut self.transcript,
                &self.pedersen_generators,
                &mut rng,
            );
            UniSkipFirstRoundProofVariant::Zk(zk_proof)
        }
        #[cfg(not(feature = "zk"))]
        {
            let proof = prove_uniskip_round(
                instance,
                &mut self.opening_accumulator,
                &mut self.transcript,
            );
            UniSkipFirstRoundProofVariant::Standard(proof)
        }
    }
}
