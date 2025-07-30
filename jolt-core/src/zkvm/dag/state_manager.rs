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
use rayon::prelude::*;
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
}

pub enum ProofData<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    SumcheckProof(SumcheckInstanceProof<F, ProofTranscript>),
    ReducedOpeningProof(ReducedOpeningProof<F, PCS, ProofTranscript>),
    SumcheckSwitchIndex(usize),
}

pub type Proofs<F, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<RV32IMCycle>,
    pub final_memory_state: Memory,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
}

pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub trace_length: usize,
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
    pub ram_K: usize,
    pub program_io: JoltDevice,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    pub verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new_prover(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<RV32IMCycle>,
        program_io: JoltDevice,
        final_memory_state: Memory,
    ) -> Self {
        let opening_accumulator = ProverOpeningAccumulator::new();
        let opening_accumulator = Rc::new(RefCell::new(opening_accumulator));
        let transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(BTreeMap::new()));
        let commitments = Rc::new(RefCell::new(vec![]));

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
            .next_power_of_two() as usize;

        Self {
            transcript,
            proofs,
            commitments,
            program_io,
            ram_K,
            prover_state: Some(ProverState {
                preprocessing,
                trace,
                final_memory_state,
                accumulator: opening_accumulator,
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
    ) -> Self {
        let opening_accumulator = VerifierOpeningAccumulator::new();
        let opening_accumulator = Rc::new(RefCell::new(opening_accumulator));
        let transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(BTreeMap::new()));
        let commitments = Rc::new(RefCell::new(vec![]));

        StateManager {
            transcript,
            proofs,
            commitments,
            program_io,
            ram_K,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length,
                accumulator: opening_accumulator,
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
                &self.program_io,
                &prover_state.final_memory_state,
            )
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

    pub fn get_bytecode(&self) -> &[RV32IMInstruction] {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.bytecode.bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.bytecode.bytecode
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

    pub fn fiat_shamir_preamble(&mut self) {
        let transcript = self.get_transcript();
        transcript
            .borrow_mut()
            .append_u64(self.program_io.memory_layout.max_input_size);
        transcript
            .borrow_mut()
            .append_u64(self.program_io.memory_layout.max_output_size);
        transcript
            .borrow_mut()
            .append_u64(self.program_io.memory_layout.memory_size);
        transcript
            .borrow_mut()
            .append_bytes(&self.program_io.inputs);
        transcript
            .borrow_mut()
            .append_bytes(&self.program_io.outputs);
        transcript
            .borrow_mut()
            .append_u64(self.program_io.panic as u64);
        transcript.borrow_mut().append_u64(self.ram_K as u64);

        if let Some(ref verifier_state) = self.verifier_state {
            transcript
                .borrow_mut()
                .append_u64(verifier_state.trace_length as u64);
        } else if let Some(ref prover_state) = self.prover_state {
            transcript
                .borrow_mut()
                .append_u64(prover_state.trace.len() as u64);
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }
}
