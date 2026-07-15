use jolt_field::{signed::S128, Field};

/// Left lookup operand of the instruction's lookup query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeftLookupOperand(pub u64);

/// Right lookup operand of the instruction's lookup query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RightLookupOperand(pub u128);

/// Left instruction input (rs1 value or PC, per the instruction shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeftInstructionInput(pub u64);

/// Right instruction input (rs2 value or immediate, per the instruction
/// shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RightInstructionInput(pub i128);

/// Signed 128-bit truncated product of the instruction inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Product(pub S128);

/// The instruction's immediate operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Imm(pub i128);

impl LeftLookupOperand {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RightLookupOperand {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u128(self.0)
    }
}

impl LeftInstructionInput {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl RightInstructionInput {
    pub fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Product {
    /// The product may exceed `i128`: fall back to the sign/magnitude split
    /// when the truncated representation does not fit.
    pub fn to_field<F: Field>(self) -> F {
        if let Some(value) = self.0.to_i128() {
            F::from_i128(value)
        } else {
            let magnitude = self.0.magnitude_as_u128();
            if self.0.is_positive {
                F::from_u128(magnitude)
            } else {
                -F::from_u128(magnitude)
            }
        }
    }
}

impl Imm {
    pub fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}
