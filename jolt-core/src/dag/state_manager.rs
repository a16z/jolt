use std::collections::HashMap;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

#[derive(Clone, Debug)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
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

pub type Openings<F> = HashMap<OpeningsKeys, (OpeningPoint<LITTLE_ENDIAN, F>, F)>;

impl<F: JoltField> Index<JoltR1CSInputs> for Openings<F> {
    type Output = F;

    fn index(&self, index: JoltR1CSInputs) -> &Self::Output {
        &self[&OpeningsKeys::SpartanZ(index)].1
    }
}


pub struct StateManager<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> {
    pub openings: Arc<Mutex<Openings<F>>>,
    pub prover_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
    pub verifier_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
    pub transcript: &'a mut ProofTranscript,
    pub proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    OuterSumcheckClaims, // (Az, Bz, Cz)
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> StateManager<'a, F, ProofTranscript, PCS> {
    pub fn new(
        openings: Arc<Mutex<Openings<F>>>,
        prover_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        verifier_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &'a mut ProofTranscript,
        proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
    ) -> Self {
        Self {
            openings,
            prover_accumulator,
            verifier_accumulator,
            transcript,
            proofs,
        }
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        self.openings(OpeningsKeys::SpartanZ(idx))
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings.lock().unwrap().get(&idx).unwrap().1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> OpeningPoint<LITTLE_ENDIAN, F> {
        self.openings.lock().unwrap().get(&idx).unwrap().0.clone()
    }

}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    SpartanOuterSumcheck,
    SpartanInnerSumcheck,
    SpartanShiftSumcheck,
    RegistersReadWrite(usize),
    RamReadWrite(usize),
    InstructionLookups(usize),
}

pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    Spartan(UniformSpartanProof<F, ProofTranscript>),
    Sumcheck(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3]),
}

pub type Proofs<F, ProofTranscript> = HashMap<ProofKeys, ProofData<F, ProofTranscript>>;
