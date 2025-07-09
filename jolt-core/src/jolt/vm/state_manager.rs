use std::collections::HashMap;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::transcript::Transcript;

pub type Openings<F> = HashMap<OpeningsKeys, (Vec<F>, F)>;

impl<F> Index<JoltR1CSInputs> for Openings<F> {
    type Output = F;

    fn index(&self, index: JoltR1CSInputs) -> &Self::Output {
        &self[&OpeningsKeys::SpartanZ(index)].1
    }
}

pub struct StateManager<'a, F: JoltField, PCS: CommitmentScheme<T, Field = F>, T: Transcript> {
    pub transcript: &'a mut T,
    pub prover_accumulator: Option<Arc<Mutex<&'a mut ProverOpeningAccumulator<F, PCS, T>>>>,
    pub verifier_accumulator: Option<Arc<Mutex<&'a mut VerifierOpeningAccumulator<F, PCS, T>>>>,
    pub openings: Arc<Mutex<Openings<F>>>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
}

impl<'a, F: JoltField, PCS: CommitmentScheme<T, Field = F>, T: Transcript>
    StateManager<'a, F, PCS, T>
{
    pub fn prove() {
        todo!()
    }

    pub fn verify() {
        todo!()
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        self.openings(OpeningsKeys::SpartanZ(idx))
    }

    pub fn r_cycle(&self) -> Vec<F> {
        self.openings_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm))
    }

    pub fn r_address(&self) -> Vec<F> {
        todo!()
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings.lock().unwrap().get(&idx).unwrap().1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> Vec<F> {
        self.openings.lock().unwrap().get(&idx).unwrap().0.clone()
    }
}
