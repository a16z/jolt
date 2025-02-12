use crate::field::JoltField;
use r1csinstance::R1CSInstance;

extern crate core;
extern crate rand;
extern crate sha3;

/// `Instance` holds the description of R1CS matrices and a hash of the matrices
#[derive(Clone)]
pub struct Instance<F: JoltField> {
    inst: R1CSInstance<F>,
    // digest: Vec<u8>,
}

impl<F: JoltField> Instance<F> {
    //     /// Constructs a new `Instance` and an associated satisfying assignment
    //     pub fn new(
    //         num_cons: usize,
    //         num_vars: usize,
    //         num_inputs: usize,
    //         A: &[(usize, usize, [u8; 32])],
    //         B: &[(usize, usize, [u8; 32])],
    //         C: &[(usize, usize, [u8; 32])],
    //     ) -> Result<Instance, R1CSError> {
    //         let (num_vars_padded, num_cons_padded) = {
    //             let num_vars_padded = {
    //                 let mut num_vars_padded = num_vars;

    //                 // ensure that num_inputs + 1 <= num_vars
    //                 num_vars_padded = max(num_vars_padded, num_inputs + 1);

    //                 // ensure that num_vars_padded a power of two
    //                 if num_vars_padded.next_power_of_two() != num_vars_padded {
    //                     num_vars_padded = num_vars_padded.next_power_of_two();
    //                 }
    //                 num_vars_padded
    //             };

    //             let num_cons_padded = {
    //                 let mut num_cons_padded = num_cons;

    //                 // ensure that num_cons_padded is at least 2
    //                 if num_cons_padded == 0 || num_cons_padded == 1 {
    //                     num_cons_padded = 2;
    //                 }

    //                 // ensure that num_cons_padded is power of 2
    //                 if num_cons.next_power_of_two() != num_cons {
    //                     num_cons_padded = num_cons.next_power_of_two();
    //                 }
    //                 num_cons_padded
    //             };

    //             (num_vars_padded, num_cons_padded)
    //         };

    //         let bytes_to_scalar =
    //             |tups: &[(usize, usize, [u8; 32])]| -> Result<Vec<(usize, usize, Scalar)>, R1CSError> {
    //                 let mut mat: Vec<(usize, usize, Scalar)> = Vec::new();
    //                 for &(row, col, val_bytes) in tups {
    //                     // row must be smaller than num_cons
    //                     if row >= num_cons {
    //                         return Err(R1CSError::InvalidIndex);
    //                     }

    //                     // col must be smaller than num_vars + 1 + num_inputs
    //                     if col >= num_vars + 1 + num_inputs {
    //                         return Err(R1CSError::InvalidIndex);
    //                     }

    //                     let val = Scalar::from_bytes(&val_bytes);
    //                     if val.is_some().unwrap_u8() == 1 {
    //                         // if col >= num_vars, it means that it is referencing a 1 or input in the satisfying
    //                         // assignment
    //                         if col >= num_vars {
    //                             mat.push((row, col + num_vars_padded - num_vars, val.unwrap()));
    //                         } else {
    //                             mat.push((row, col, val.unwrap()));
    //                         }
    //                     } else {
    //                         return Err(R1CSError::InvalidScalar);
    //                     }
    //                 }

    //                 // pad with additional constraints up until num_cons_padded if the original constraints were 0 or 1
    //                 // we do not need to pad otherwise because the dummy constraints are implicit in the sum-check protocol
    //                 if num_cons == 0 || num_cons == 1 {
    //                     for i in tups.len()..num_cons_padded {
    //                         mat.push((i, num_vars, Scalar::zero()));
    //                     }
    //                 }

    //                 Ok(mat)
    //             };

    //         let A_scalar = bytes_to_scalar(A);
    //         if A_scalar.is_err() {
    //             return Err(A_scalar.err().unwrap());
    //         }

    //         let B_scalar = bytes_to_scalar(B);
    //         if B_scalar.is_err() {
    //             return Err(B_scalar.err().unwrap());
    //         }

    //         let C_scalar = bytes_to_scalar(C);
    //         if C_scalar.is_err() {
    //             return Err(C_scalar.err().unwrap());
    //         }

    //         let inst = R1CSInstance::new(
    //             num_cons_padded,
    //             num_vars_padded,
    //             num_inputs,
    //             &A_scalar.unwrap(),
    //             &B_scalar.unwrap(),
    //             &C_scalar.unwrap(),
    //         );

    //         let digest = inst.get_digest();

    //         Ok(Instance { inst, digest })
    //     }

    //     /// Checks if a given R1CSInstance is satisfiable with a given variables and inputs assignments
    //     pub fn is_sat(
    //         &self,
    //         vars: &VarsAssignment,
    //         inputs: &InputsAssignment,
    //     ) -> Result<bool, R1CSError> {
    //         if vars.assignment.len() > self.inst.get_num_vars() {
    //             return Err(R1CSError::InvalidNumberOfInputs);
    //         }

    //         if inputs.assignment.len() != self.inst.get_num_inputs() {
    //             return Err(R1CSError::InvalidNumberOfInputs);
    //         }

    //         // we might need to pad variables
    //         let padded_vars = {
    //             let num_padded_vars = self.inst.get_num_vars();
    //             let num_vars = vars.assignment.len();
    //             if num_padded_vars > num_vars {
    //                 vars.pad(num_padded_vars)
    //             } else {
    //                 vars.clone()
    //             }
    //         };

    //         Ok(self
    //             .inst
    //             .is_sat(&padded_vars.assignment, &inputs.assignment))
    //     }

    /// Constructs a new synthetic R1CS `Instance` and an associated satisfying assignment
    pub fn produce_synthetic_r1cs(
        num_cons: usize,
        num_vars: usize,
        num_inputs: usize,
    ) -> (Instance<F>, Vec<F>, Vec<F>) {
        let (inst, inputs, vars) =
            R1CSInstance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);
        // let digest = inst.get_digest();
        (Instance { inst }, vars, inputs)
    }
}

mod errors;
mod r1csinstance;
mod sparse_mlpoly;
mod spartan_memory_checking;
