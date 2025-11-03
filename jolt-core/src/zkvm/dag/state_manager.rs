use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::compute_d_parameter;
use crate::zkvm::{JoltProverPreprocessing, JoltVerifierPreprocessing};
use rayon::prelude::*;
use tracer::emulator::memory::Memory;
use tracer::instruction::Cycle;
use tracer::{JoltDevice, LazyTraceIterator};

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub lazy_trace: Option<LazyTraceIterator>,
    pub trace: Arc<Vec<Cycle>>,
    pub final_memory_state: Memory,
    pub untrusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    pub trusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
}

pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub trace_length: usize,
}

pub struct StateManager<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub ram_d: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    pub verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F, PCS> StateManager<'a, F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn new_prover(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        trace: Vec<Cycle>,
        program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        final_memory_state: Memory,
    ) -> Self {
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

        let ram_d = compute_d_parameter(ram_K);
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;
        let twist_sumcheck_switch_index = chunk_size.log_2();

        Self {
            untrusted_advice_commitment: None,
            trusted_advice_commitment,
            program_io,
            ram_K,
            ram_d,
            twist_sumcheck_switch_index,
            prover_state: Some(ProverState {
                preprocessing,
                lazy_trace: Some(lazy_trace),
                trace: Arc::new(trace),
                final_memory_state,
                untrusted_advice_polynomial: None,
                trusted_advice_polynomial: None,
            }),
            verifier_state: None,
        }
    }

    pub fn get_prover_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS>,
        &Option<LazyTraceIterator>,
        &Vec<Cycle>,
        &JoltDevice,
        &Memory,
    ) {
        if let Some(ref prover_state) = self.prover_state {
            (
                prover_state.preprocessing,
                &prover_state.lazy_trace,
                &prover_state.trace,
                &self.program_io,
                &prover_state.final_memory_state,
            )
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_trace_arc(&self) -> Arc<Vec<Cycle>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.trace.clone()
        } else {
            panic!("Prover state not initialized");
        }
    }
}

// TODO: Perhaps better we have something like a JoltClaim with this stuff in
// it and have a method on that to append that to the transcript.
pub fn fiat_shamir_preamble(
    program_io: &JoltDevice,
    ram_K: usize,
    trace_length: usize,
    transcript: &mut impl Transcript,
) {
    transcript.append_u64(program_io.memory_layout.max_input_size);
    transcript.append_u64(program_io.memory_layout.max_output_size);
    transcript.append_u64(program_io.memory_layout.memory_size);
    transcript.append_bytes(&program_io.inputs);
    transcript.append_bytes(&program_io.outputs);
    transcript.append_u64(program_io.panic as u64);
    transcript.append_u64(ram_K as u64);
    transcript.append_u64(trace_length as u64);
}
