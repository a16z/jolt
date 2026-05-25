use jolt_field::Field;
use jolt_r1cs::{ConstraintMatrices, Variable};

#[derive(Clone, Debug)]
pub struct WrapperProtocol<F: Field> {
    pub r1cs: ConstraintMatrices<F>,
    pub witness: Vec<F>,
    pub public_inputs: Vec<F>,
    pub layout: WrapperLayout,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WrapperLayout {
    pub public_inputs: Vec<Variable>,
}

impl<F: Field> WrapperProtocol<F> {
    pub fn new(
        r1cs: ConstraintMatrices<F>,
        witness: Vec<F>,
        public_inputs: Vec<F>,
        layout: WrapperLayout,
    ) -> Self {
        Self {
            r1cs,
            witness,
            public_inputs,
            layout,
        }
    }
}
