use std::sync::{Arc, Mutex};

use tracer::instruction::RV32IMCycle;

use crate::field::JoltField;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::transcript::Transcript;

pub struct StateManager<'a, F: JoltField, T: Transcript> {
    pub T: usize,
    pub log_T: usize,
    pub challenges: Challenges<F>,
    pub r_cycle: Option<Vec<F>>,
    pub r_address: Option<Vec<F>>,
    pub transcript: &'a mut T,
    prover_state: Option<ProverState<'a, F>>,
    verifier_openings: Option<VerifierOpenings<F>>,
}

pub struct Challenges<F: JoltField> {
    pub instruction_booleanity: F,
    pub instruction_hamming: F,
    pub instruction_read_raf: F,
}

pub struct ProverState<'a, F: JoltField> {
    trace: &'a [RV32IMCycle],
    pub eq_r_cycle: Vec<F>,
    pub openings: Arc<Mutex<ProverOpenings<F>>>,
}

pub struct ProverOpenings<F: JoltField> {
    pub z_claims: Option<Vec<F>>,
}

pub struct VerifierOpenings<F: JoltField> {
    pub z_claims: Vec<F>,
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
        match self.prover_state {
            Some(ref state) => {
                state.openings.lock().unwrap().z_claims.as_ref().unwrap()[idx.to_index()]
            }
            None => self.verifier_openings.as_ref().unwrap().z_claims[idx.to_index()],
        }
    }

    pub fn trace(&self) -> &[RV32IMCycle] {
        self.prover_state().trace
    }
}
