use jolt_field::Field;
use jolt_r1cs::ConstraintMatrices;

use crate::{CheckedInputs, R1csRelationStatement};

#[derive(Clone, Copy, Debug)]
pub struct R1csRelationInputs<'a, F: Field> {
    pub checked: &'a CheckedInputs,
    pub relation: &'a ConstraintMatrices<F>,
    pub public_inputs: &'a [F],
    pub proof_relation: R1csRelationStatement,
}
