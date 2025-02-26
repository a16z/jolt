use crate::field::JoltField;
use r1csinstance::R1CSInstance;

extern crate core;
extern crate rand;
extern crate sha3;

/// `Instance` holds the description of R1CS matrices and a hash of the matrices
#[derive(Clone)]
pub struct Instance<F: JoltField> {
    inst: R1CSInstance<F>,
    digest: F,
}

impl<F: JoltField> Instance<F> {
    /// Constructs a new synthetic R1CS `Instance` and an associated satisfying assignment
    pub fn produce_synthetic_r1cs(
        num_cons: usize,
        num_vars: usize,
        num_inputs: usize,
    ) -> (Instance<F>, Vec<F>, Vec<F>) {
        let (inst, inputs, vars) =
            R1CSInstance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);

        let digest = inst.get_digest();

        (Instance { inst, digest }, vars, inputs)
    }
}

mod r1csinstance;
mod sparse_mlpoly;
pub mod spartan_memory_checking;
