use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::field::JoltField;
use crate::jolt::vm::{JoltCommitments, JoltProverPreprocessing, JoltVerifierPreprocessing};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use tracer::emulator::memory::Memory;
use tracer::instruction::RV32IMCycle;
use tracer::JoltDevice;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    SpartanOuterSumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
    RegistersSwitchIndex,
}

pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    SpartanOuterData(SumcheckInstanceProof<F, ProofTranscript>),
    BatchableSumcheckData(SumcheckInstanceProof<F, ProofTranscript>),
    RegisterSwitchIndex(usize),
}

pub type Proofs<F, ProofTranscript> = HashMap<ProofKeys, ProofData<F, ProofTranscript>>;

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: Option<&'a JoltProverPreprocessing<F, PCS>>,
    pub trace: Option<Vec<RV32IMCycle>>,
    pub program_io: Option<JoltDevice>,
    pub final_memory_state: Option<Memory>,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>,
}

pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: Option<&'a JoltVerifierPreprocessing<F, PCS>>,
    pub program_io: Option<JoltDevice>,
    pub trace_length: Option<usize>,
    pub accumulator: Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>,
}

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub transcript: Rc<RefCell<ProofTranscript>>,
    pub proofs: Rc<RefCell<Proofs<F, ProofTranscript>>>,
    pub commitments: Rc<RefCell<Option<JoltCommitments<F, PCS>>>>,
    prover_state: Option<ProverState<'a, F, PCS>>,
    verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new_prover(
        prover_accumulator: Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>,
        transcript: Rc<RefCell<ProofTranscript>>,
        proofs: Rc<RefCell<Proofs<F, ProofTranscript>>>,
        commitments: Rc<RefCell<Option<JoltCommitments<F, PCS>>>>,
    ) -> Self {
        Self {
            transcript,
            proofs,
            commitments,
            prover_state: Some(ProverState {
                preprocessing: None,
                trace: None,
                program_io: None,
                final_memory_state: None,
                accumulator: prover_accumulator,
            }),
            verifier_state: None,
        }
    }

    pub fn new_verifier(
        verifier_accumulator: Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>,
        transcript: Rc<RefCell<ProofTranscript>>,
        proofs: Rc<RefCell<Proofs<F, ProofTranscript>>>,
        commitments: Rc<RefCell<Option<JoltCommitments<F, PCS>>>>,
    ) -> Self {
        Self {
            transcript,
            proofs,
            commitments,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing: None,
                program_io: None,
                trace_length: None,
                accumulator: verifier_accumulator,
            }),
        }
    }

    pub fn set_prover_data(
        &mut self,
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<RV32IMCycle>,
        program_io: JoltDevice,
        final_memory_state: Memory,
    ) {
        if let Some(ref mut prover_state) = self.prover_state {
            prover_state.preprocessing = Some(preprocessing);
            prover_state.trace = Some(trace);
            prover_state.program_io = Some(program_io);
            prover_state.final_memory_state = Some(final_memory_state);
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn set_verifier_data(
        &mut self,
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        trace_length: usize,
    ) {
        if let Some(ref mut verifier_state) = self.verifier_state {
            verifier_state.preprocessing = Some(preprocessing);
            verifier_state.program_io = Some(program_io);
            verifier_state.trace_length = Some(trace_length);
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_prover_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS>,
        &Vec<RV32IMCycle>,
        &JoltDevice,
        &Memory,
    ) {
        if let Some(ref prover_state) = self.prover_state {
            (
                prover_state.preprocessing.expect("Preprocessing not set"),
                prover_state.trace.as_ref().expect("Trace not set"),
                prover_state
                    .program_io
                    .as_ref()
                    .expect("Program IO not set"),
                prover_state
                    .final_memory_state
                    .as_ref()
                    .expect("Final memory state not set"),
            )
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_verifier_data(&self) -> (&'a JoltVerifierPreprocessing<F, PCS>, &JoltDevice, usize) {
        if let Some(ref verifier_state) = self.verifier_state {
            (
                verifier_state.preprocessing.expect("Preprocessing not set"),
                verifier_state
                    .program_io
                    .as_ref()
                    .expect("Program IO not set"),
                verifier_state.trace_length.expect("Trace length not set"),
            )
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_prover_accumulator(&self) -> Rc<RefCell<ProverOpeningAccumulator<F, PCS>>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.clone()
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_transcript(&self) -> Rc<RefCell<ProofTranscript>> {
        self.transcript.clone()
    }

    pub fn get_verifier_accumulator(&self) -> Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>> {
        if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.clone()
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_commitments(&self) -> JoltCommitments<F, PCS> {
        self.commitments
            .borrow()
            .as_ref()
            .expect("Commitments not set")
            .clone()
    }

    pub fn set_commitments(&self, commitments: JoltCommitments<F, PCS>) {
        *self.commitments.borrow_mut() = Some(commitments);
    }
}
