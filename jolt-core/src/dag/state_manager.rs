use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Index, RangeFull};

use std::sync::{Arc, Mutex};

use crate::field::JoltField;
use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use tracer::emulator::memory::Memory;
use tracer::instruction::RV32IMCycle;
use tracer::JoltDevice;

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

#[derive(Clone, Debug)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: JoltField> Index<usize> for OpeningPoint<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> Index<RangeFull> for OpeningPoint<E, F> {
    type Output = [F];

    fn index(&self, _index: RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F>) -> Self {
        Self { r }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    OuterSumcheckAz,    // Az claim from outer sumcheck
    OuterSumcheckBz,    // Bz claim from outer sumcheck
    OuterSumcheckCz,    // Cz claim from outer sumcheck
    OuterSumcheckRxVar, // rx_var from outer sumcheck -- TODO(markosg04)where is this used ?
}

pub type Openings<F> = HashMap<OpeningsKeys, (OpeningPoint<LITTLE_ENDIAN, F>, F)>;

pub trait OpeningsExt<F: JoltField> {
    fn get_spartan_z(&self, index: JoltR1CSInputs) -> F;
}

impl<F: JoltField> OpeningsExt<F> for Openings<F> {
    fn get_spartan_z(&self, index: JoltR1CSInputs) -> F {
        self.get(&OpeningsKeys::SpartanZ(index))
            .map(|(_, value)| *value)
            .unwrap_or(F::zero())
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    SpartanOuterSumcheck,
    SpartanInnerSumcheck,
    SpartanShiftSumcheck,
}

pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    Spartan(UniformSpartanProof<F, ProofTranscript>),
    SpartanSumcheck(SumcheckInstanceProof<F, ProofTranscript>),
}

pub type Proofs<F, ProofTranscript> = HashMap<ProofKeys, ProofData<F, ProofTranscript>>;

pub struct ProgramState<'a, F: JoltField, PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub preprocessing: Option<&'a JoltProverPreprocessing<F, PCS, ProofTranscript>>,
    pub trace: Option<Vec<RV32IMCycle>>,
    pub program_io: Option<JoltDevice>,
    pub final_memory_state: Option<Memory>,
}

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
> {
    pub openings: Arc<Mutex<Openings<F>>>,
    pub prover_accumulator: Option<Arc<Mutex<ProverOpeningAccumulator<F, PCS, ProofTranscript>>>>,
    pub verifier_accumulator:
        Option<Arc<Mutex<VerifierOpeningAccumulator<F, PCS, ProofTranscript>>>>,
    pub transcript: RefCell<&'a mut ProofTranscript>,
    pub proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
    program_state: ProgramState<'a, F, PCS, ProofTranscript>,
}

impl<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    > StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new(
        openings: Arc<Mutex<Openings<F>>>,
        prover_accumulator: Option<Arc<Mutex<ProverOpeningAccumulator<F, PCS, ProofTranscript>>>>,
        verifier_accumulator: Option<
            Arc<Mutex<VerifierOpeningAccumulator<F, PCS, ProofTranscript>>>,
        >,
        transcript: &'a mut ProofTranscript,
        proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
        program_state: ProgramState<'a, F, PCS, ProofTranscript>,
    ) -> Self {
        Self {
            openings,
            prover_accumulator,
            verifier_accumulator,
            transcript: RefCell::new(transcript),
            proofs,
            program_state,
        }
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        use OpeningsExt;
        self.openings.lock().unwrap().get_spartan_z(idx)
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings.lock().unwrap().get(&idx).unwrap().1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> OpeningPoint<LITTLE_ENDIAN, F> {
        self.openings.lock().unwrap().get(&idx).unwrap().0.clone()
    }

    pub fn set_program_data(
        &mut self,
        preprocessing: &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: Vec<RV32IMCycle>,
        program_io: JoltDevice,
        final_memory_state: Memory,
    ) {
        self.program_state.preprocessing = Some(preprocessing);
        self.program_state.trace = Some(trace);
        self.program_state.program_io = Some(program_io);
        self.program_state.final_memory_state = Some(final_memory_state);
    }

    pub fn get_program_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
        &Vec<RV32IMCycle>,
        &JoltDevice,
        &Memory,
    ) {
        (
            self.program_state
                .preprocessing
                .expect("Preprocessing not set"),
            self.program_state.trace.as_ref().expect("Trace not set"),
            self.program_state
                .program_io
                .as_ref()
                .expect("Program IO not set"),
            self.program_state
                .final_memory_state
                .as_ref()
                .expect("Final memory state not set"),
        )
    }
}
