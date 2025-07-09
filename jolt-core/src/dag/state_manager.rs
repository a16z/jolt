use std::collections::HashMap;
use std::ops::Index;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::r1cs::builder::CombinedUniformBuilder;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::key::UniformSpartanKey;
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
    pub transcript: RefCell<&'a mut ProofTranscript>,
    pub proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
    spartan_state: SpartanState<'a, F>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    OuterSumcheckClaims, // (Az, Bz, Cz)
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

pub struct SpartanState<'a, F: JoltField> {
    pub spartan_key: Option<&'a UniformSpartanKey<F>>,
    pub constraint_builder: Option<&'a CombinedUniformBuilder<F>>,
    pub input_polys: Option<Vec<MultilinearPolynomial<F>>>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> StateManager<'a, F, ProofTranscript, PCS> {
    pub fn new(
        openings: Arc<Mutex<Openings<F>>>,
        prover_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        verifier_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &'a mut ProofTranscript,
        proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
        spartan_state: SpartanState<'a, F>,
    ) -> Self {
        Self {
            openings,
            prover_accumulator,
            verifier_accumulator,
            transcript: RefCell::new(transcript),
            proofs,
            spartan_state
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

    pub fn set_spartan_data(
        &mut self,
        key: &'a UniformSpartanKey<F>,
        constraint_builder: &'a CombinedUniformBuilder<F>,
        input_polys: Vec<MultilinearPolynomial<F>>,
    ) {
        self.spartan_state.spartan_key = Some(key);
        self.spartan_state.constraint_builder = Some(constraint_builder);
        self.spartan_state.input_polys = Some(input_polys);
    }

    pub fn get_spartan_data(&self) -> (&'a UniformSpartanKey<F>, &'a CombinedUniformBuilder<F>, &Vec<MultilinearPolynomial<F>>) {
        (
            self.spartan_state.spartan_key.expect("Spartan key not set"),
            self.spartan_state.constraint_builder.expect("Constraint builder not set"),
            self.spartan_state.input_polys.as_ref().expect("Input polys not set"),
        )
    }
}