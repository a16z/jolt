use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tracer::instruction::RV32IMCycle;

use crate::field::JoltField;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::transcript::Transcript;

pub type Openings<F> = HashMap<OpeningsKeys, (Vec<F>, F)>;

pub struct StateManager<'a, F: JoltField, T: Transcript> {
    pub T: usize,
    pub log_T: usize,
    pub challenges: Challenges<F>,
    pub transcript: &'a mut T,
    prover_state: Option<ProverState<'a, F>>,
    pub verifier_openings: Option<Openings<F>>,
}

pub struct Challenges<F: JoltField> {
    pub instruction_booleanity: F,
    pub instruction_hamming: F,
    pub instruction_read_raf: F,
}

pub struct ProverState<'a, F: JoltField> {
    trace: &'a [RV32IMCycle],
    pub eq_r_cycle: Vec<F>,
    pub openings: Arc<Mutex<Openings<F>>>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
}

impl<'a, F: JoltField, T: Transcript> StateManager<'a, F, T> {
    pub fn prove() {
        todo!()
    }

    pub fn verify() {
        todo!()
    }

    pub fn prover_state(&self) -> &ProverState<F> {
        self.prover_state.as_ref().unwrap()
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
        match self.prover_state {
            Some(ref state) => state.openings.lock().unwrap()[&idx].1,
            None => self.verifier_openings.as_ref().unwrap()[&idx].1,
        }
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> Vec<F> {
        match self.prover_state {
            Some(ref state) => state.openings.lock().unwrap()[&idx].0.clone(),
            None => self.verifier_openings.as_ref().unwrap()[&idx].0.clone(),
        }
    }

    pub fn trace(&self) -> &'a [RV32IMCycle] {
        self.prover_state.as_ref().unwrap().trace
    }
}
