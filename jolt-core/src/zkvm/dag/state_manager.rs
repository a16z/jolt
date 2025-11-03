use std::collections::BTreeMap;
use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::ReducedOpeningProof;
use crate::subprotocols::sumcheck::{SumcheckInstanceProof, UniSkipFirstRoundProof};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::compute_d_parameter;
use crate::zkvm::{JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing};
use num_derive::FromPrimitive;
use rayon::prelude::*;
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::{JoltDevice, LazyTraceIterator};

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum ProofKeys {
    Stage1UniSkipFirstRound,
    Stage1Sumcheck,
    Stage2UniSkipFirstRound,
    Stage2Sumcheck,
    Stage3Sumcheck,
    Stage4Sumcheck,
    Stage5Sumcheck,
    Stage6Sumcheck,
    TrustedAdviceProof,
    UntrustedAdviceProof,
    ReducedOpeningProof, // Implicitly Stage 7
}

pub enum ProofData<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    SumcheckProof(SumcheckInstanceProof<F, ProofTranscript>),
    ReducedOpeningProof(ReducedOpeningProof<F, PCS, ProofTranscript>),
    OpeningProof(PCS::Proof),
    UniSkipFirstRoundProof(UniSkipFirstRoundProof<F, ProofTranscript>),
}

pub type Proofs<F, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

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

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub proofs: Proofs<F, PCS, ProofTranscript>,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub ram_d: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    pub verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F, ProofTranscript, PCS> StateManager<'a, F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
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
        let proofs = BTreeMap::new();
        let commitments = vec![];

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

        let ram_d = compute_d_parameter(ram_K);
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;
        let twist_sumcheck_switch_index = chunk_size.log_2();

        Self {
            proofs,
            commitments,
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

    /// Only used in tests; in practice, the verifier state manager is
    /// constructed using `JoltProof::to_verifier_state_manager`
    #[cfg(test)]
    pub fn new_verifier(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        trace_length: usize,
        ram_K: usize,
        twist_sumcheck_switch_index: usize,
    ) -> Self {
        let proofs = BTreeMap::new();
        let commitments = vec![];
        let ram_d = compute_d_parameter(ram_K);

        StateManager {
            proofs,
            commitments,
            untrusted_advice_commitment: None,
            trusted_advice_commitment: None,
            program_io,
            ram_K,
            ram_d,
            twist_sumcheck_switch_index,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length,
            }),
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

    pub fn get_verifier_data(&self) -> (&'a JoltVerifierPreprocessing<F, PCS>, &JoltDevice, usize) {
        if let Some(ref verifier_state) = self.verifier_state {
            (
                verifier_state.preprocessing,
                &self.program_io,
                verifier_state.trace_length,
            )
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_trace_len(&self) -> usize {
        if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.trace_length
        } else if let Some(ref prover_state) = self.prover_state {
            prover_state.trace.len()
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    pub fn get_bytecode(&self) -> &[Instruction] {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.bytecode.bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.bytecode.bytecode
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    pub fn get_shared_preprocessing(&self) -> &JoltSharedPreprocessing {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    pub fn fiat_shamir_preamble(&self, transcript: &mut ProofTranscript) {
        transcript.append_u64(self.program_io.memory_layout.max_input_size);
        transcript.append_u64(self.program_io.memory_layout.max_output_size);
        transcript.append_u64(self.program_io.memory_layout.memory_size);
        transcript.append_bytes(&self.program_io.inputs);
        transcript.append_bytes(&self.program_io.outputs);
        transcript.append_u64(self.program_io.panic as u64);
        transcript.append_u64(self.ram_K as u64);

        if let Some(ref verifier_state) = self.verifier_state {
            transcript.append_u64(verifier_state.trace_length as u64);
        } else if let Some(ref prover_state) = self.prover_state {
            transcript.append_u64(prover_state.trace.len() as u64);
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }
}
