use jolt_field::Field;

/// Value read from rs1; 0 when the instruction has no rs1 operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs1Value(pub u64);

/// Value read from rs2; 0 when the instruction has no rs2 operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs2Value(pub u64);

/// Value written to rd; 0 when the instruction has no rd operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdWriteValue(pub u64);

impl Rs1Value {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Rs2Value {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RdWriteValue {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}
