use jolt_field::{
    signed::{S128, S64},
    Field,
};
use jolt_lookup_tables::LookupQuery;
use jolt_program::execution::TraceRow;

use super::{lookup_query, Extract, WitnessEnv};
use crate::WitnessError;
use crate::RV64_XLEN;

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

impl Extract for LeftLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, _) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&lookup_query(row));
        Ok(Self(left))
    }
}

impl RightLookupOperand {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u128(self.0)
    }
}

impl Extract for RightLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (_, right) = LookupQuery::<RV64_XLEN>::to_lookup_operands(&lookup_query(row));
        Ok(Self(right))
    }
}

impl LeftInstructionInput {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for LeftInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, _) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(left))
    }
}

impl RightInstructionInput {
    pub fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RightInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (_, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(right))
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

impl Extract for Product {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let (left, right) = LookupQuery::<RV64_XLEN>::to_instruction_inputs(&lookup_query(row));
        Ok(Self(
            S64::from_u64(left).mul_trunc::<2, 2>(&S128::from_i128(right)),
        ))
    }
}

impl Imm {
    pub fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for Imm {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.instruction.operands.imm))
    }
}
