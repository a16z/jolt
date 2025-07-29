use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, ReducedOpeningProof, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::{JoltProverPreprocessing, JoltVerifierPreprocessing};
use num_derive::FromPrimitive;
use tracer::emulator::memory::Memory;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};
use tracer::JoltDevice;

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum ProofKeys {
    Stage1Sumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
    Stage4Sumcheck,
    ReducedOpeningProof,
    TwistSumcheckSwitchIndex,
    RamK,
}

pub enum ProofData<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    SumcheckProof(SumcheckInstanceProof<F, ProofTranscript>),
    ReducedOpeningProof(ReducedOpeningProof<F, PCS, ProofTranscript>),
    SumcheckSwitchIndex(usize),
    RamK(usize),
}

pub type Proofs<F, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: Option<&'a JoltProverPreprocessing<F, PCS>>,
    pub trace: Option<Vec<RV32IMCycle>>,
    pub program_io: Option<JoltDevice>,
    pub final_memory_state: Option<Memory>,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
}

pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: Option<&'a JoltVerifierPreprocessing<F, PCS>>,
    pub program_io: Option<JoltDevice>,
    pub trace_length: Option<usize>,
    pub accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
}

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub transcript: Rc<RefCell<ProofTranscript>>,
    pub proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new_prover(
        prover_accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: Rc<RefCell<ProofTranscript>>,
        proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
        commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
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
        verifier_accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: Rc<RefCell<ProofTranscript>>,
        proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
        commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
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

    pub fn get_bytecode(&self) -> &[RV32IMInstruction] {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state
                .preprocessing
                .expect("Preprocessing not set")
                .shared
                .bytecode
                .bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state
                .preprocessing
                .expect("Preprocessing not set")
                .shared
                .bytecode
                .bytecode
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    pub fn get_prover_accumulator(&self) -> Rc<RefCell<ProverOpeningAccumulator<F>>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.clone()
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_transcript(&self) -> Rc<RefCell<ProofTranscript>> {
        self.transcript.clone()
    }

    pub fn get_verifier_accumulator(&self) -> Rc<RefCell<VerifierOpeningAccumulator<F>>> {
        if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.clone()
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_commitments(&self) -> Rc<RefCell<Vec<PCS::Commitment>>> {
        self.commitments.clone()
    }

    pub fn set_commitments(&self, commitments: Vec<PCS::Commitment>) {
        *self.commitments.borrow_mut() = commitments;
    }

    /// Gets the opening for a given virtual polynomial from whichever accumulator is available.
    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .get_virtual_polynomial_opening(polynomial, sumcheck)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .get_virtual_polynomial_opening(polynomial, sumcheck)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Gets the opening for a given committed polynomial from whichever accumulator is available.
    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .get_committed_polynomial_opening(polynomial, sumcheck)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .get_committed_polynomial_opening(polynomial, sumcheck)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }
}
