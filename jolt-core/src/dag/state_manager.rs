use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::field::JoltField;
use crate::jolt::vm::{JoltCommitments, JoltProverPreprocessing, JoltVerifierPreprocessing};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningPoint, OpeningsExt, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
    LITTLE_ENDIAN,
};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use tracer::emulator::memory::Memory;
use tracer::instruction::RV32IMCycle;
use tracer::JoltDevice;

pub struct ProgramData<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<RV32IMCycle>,
    pub program_io: JoltDevice,
    pub final_memory_state: Memory,
}

pub struct VerifierProgramData<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub preprocessing: Arc<JoltVerifierPreprocessing<F, PCS>>,
    pub program_io: JoltDevice,
    pub trace_length: usize,
}

// Wrappers to avoid Orphan rules
#[derive(Debug, Clone, Default)]
pub struct Proofs<F: JoltField, ProofTranscript: Transcript>(
    pub HashMap<ProofKeys, ProofData<F, ProofTranscript>>,
);

#[derive(Debug, Clone, Default)]
pub struct Claims<F: JoltField>(pub HashMap<OpeningsKeys, F>);

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    SpartanOuterSumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
}

#[derive(Debug, Clone)]
pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    SpartanOuterData(SumcheckInstanceProof<F, ProofTranscript>),
    BatchableSumcheckData(SumcheckInstanceProof<F, ProofTranscript>),
}

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<RV32IMCycle>,
    pub program_io: JoltDevice,
    pub final_memory_state: Memory,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>,
}

pub struct VerifierState<F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: Arc<JoltVerifierPreprocessing<F, PCS>>,
    pub program_io: JoltDevice,
    pub trace_length: usize,
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
    verifier_state: Option<VerifierState<F, PCS>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new_prover(
        program_data: ProgramData<'a, F, PCS>,
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
                preprocessing: program_data.preprocessing,
                trace: program_data.trace,
                program_io: program_data.program_io,
                final_memory_state: program_data.final_memory_state,
                accumulator: prover_accumulator,
            }),
            verifier_state: None,
        }
    }

    pub fn new_verifier(
        program_data: VerifierProgramData<F, PCS>,
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
                preprocessing: program_data.preprocessing,
                program_io: program_data.program_io,
                trace_length: program_data.trace_length,
                accumulator: verifier_accumulator,
            }),
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
                prover_state.preprocessing,
                &prover_state.trace,
                &prover_state.program_io,
                &prover_state.final_memory_state,
            )
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_verifier_data(&self) -> (&JoltVerifierPreprocessing<F, PCS>, &JoltDevice, usize) {
        if let Some(ref verifier_state) = self.verifier_state {
            (
                &verifier_state.preprocessing,
                &verifier_state.program_io,
                verifier_state.trace_length,
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

    /// Gets the opening point for a given key from whichever accumulator is available.
    /// Returns the opening point from the prover accumulator if available, otherwise from the verifier accumulator.
    pub fn get_opening_point(&self, key: OpeningsKeys) -> Option<OpeningPoint<LITTLE_ENDIAN, F>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.borrow().get_opening_point(key)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.borrow().get_opening_point(key)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Gets the opening value for a given key from whichever accumulator is available.
    /// Returns the opening value from the prover accumulator if available, otherwise from the verifier accumulator.
    pub fn get_opening(&self, key: OpeningsKeys) -> F {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.borrow().get_opening(key)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.borrow().get_opening(key)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Gets a specific spartan z value from the evaluation openings from whichever accumulator is available.
    /// Returns the spartan z value from the prover accumulator if available, otherwise from the verifier accumulator.
    pub fn get_spartan_z(&self, index: JoltR1CSInputs) -> F {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .evaluation_openings()
                .get_spartan_z(index)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .evaluation_openings()
                .get_spartan_z(index)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Gets a specific evaluation opening value by key from whichever accumulator is available.
    /// Returns the evaluation opening value from the prover accumulator if available, otherwise from the verifier accumulator.
    pub fn get_evaluation_opening(
        &self,
        key: &OpeningsKeys,
    ) -> Option<(OpeningPoint<LITTLE_ENDIAN, F>, F)> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .evaluation_openings()
                .get(key)
                .cloned()
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .evaluation_openings()
                .get(key)
                .cloned()
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }
}
