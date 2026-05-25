//! Typed clear-mode inputs consumed by stage 1.

use jolt_claims::protocols::jolt::{
    formulas::spartan::{outer_opening, SpartanOuterDimensions},
    JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage1Claims<F: Field> {
    pub uniskip_output_claim: F,
    pub outer: SpartanOuterClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanOuterClaims<F: Field> {
    pub left_instruction_input: F,
    pub right_instruction_input: F,
    pub product: F,
    pub should_branch: F,
    pub pc: F,
    pub unexpanded_pc: F,
    pub imm: F,
    pub ram_address: F,
    pub rs1_value: F,
    pub rs2_value: F,
    pub rd_write_value: F,
    pub ram_read_value: F,
    pub ram_write_value: F,
    pub left_lookup_operand: F,
    pub right_lookup_operand: F,
    pub next_unexpanded_pc: F,
    pub next_pc: F,
    pub next_is_virtual: F,
    pub next_is_first_in_sequence: F,
    pub lookup_output: F,
    pub should_jump: F,
    pub flags: SpartanOuterFlagClaims<F>,
}

impl<F: Field> SpartanOuterClaims<F> {
    pub(crate) fn r1cs_input_claims(
        &self,
        dimensions: &SpartanOuterDimensions,
    ) -> Result<Vec<F>, VerifierError> {
        dimensions
            .variables()
            .iter()
            .copied()
            .map(|variable| {
                self.claim(variable)
                    .ok_or_else(|| VerifierError::MissingOpeningClaim {
                        id: outer_opening(variable),
                    })
            })
            .collect()
    }

    pub(crate) fn claim(&self, variable: JoltVirtualPolynomial) -> Option<F> {
        match variable {
            JoltVirtualPolynomial::LeftInstructionInput => Some(self.left_instruction_input),
            JoltVirtualPolynomial::RightInstructionInput => Some(self.right_instruction_input),
            JoltVirtualPolynomial::Product => Some(self.product),
            JoltVirtualPolynomial::ShouldBranch => Some(self.should_branch),
            JoltVirtualPolynomial::PC => Some(self.pc),
            JoltVirtualPolynomial::UnexpandedPC => Some(self.unexpanded_pc),
            JoltVirtualPolynomial::Imm => Some(self.imm),
            JoltVirtualPolynomial::RamAddress => Some(self.ram_address),
            JoltVirtualPolynomial::Rs1Value => Some(self.rs1_value),
            JoltVirtualPolynomial::Rs2Value => Some(self.rs2_value),
            JoltVirtualPolynomial::RdWriteValue => Some(self.rd_write_value),
            JoltVirtualPolynomial::RamReadValue => Some(self.ram_read_value),
            JoltVirtualPolynomial::RamWriteValue => Some(self.ram_write_value),
            JoltVirtualPolynomial::LeftLookupOperand => Some(self.left_lookup_operand),
            JoltVirtualPolynomial::RightLookupOperand => Some(self.right_lookup_operand),
            JoltVirtualPolynomial::NextUnexpandedPC => Some(self.next_unexpanded_pc),
            JoltVirtualPolynomial::NextPC => Some(self.next_pc),
            JoltVirtualPolynomial::NextIsVirtual => Some(self.next_is_virtual),
            JoltVirtualPolynomial::NextIsFirstInSequence => Some(self.next_is_first_in_sequence),
            JoltVirtualPolynomial::LookupOutput => Some(self.lookup_output),
            JoltVirtualPolynomial::ShouldJump => Some(self.should_jump),
            JoltVirtualPolynomial::OpFlags(flag) => self.flags.claim(flag),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanOuterFlagClaims<F: Field> {
    pub add_operands: F,
    pub subtract_operands: F,
    pub multiply_operands: F,
    pub load: F,
    pub store: F,
    pub jump: F,
    pub write_lookup_output_to_rd: F,
    pub virtual_instruction: F,
    pub assert: F,
    pub do_not_update_unexpanded_pc: F,
    pub advice: F,
    pub is_compressed: F,
    pub is_first_in_sequence: F,
    pub is_last_in_sequence: F,
}

impl<F: Field> SpartanOuterFlagClaims<F> {
    fn claim(&self, flag: CircuitFlags) -> Option<F> {
        match flag {
            CircuitFlags::AddOperands => Some(self.add_operands),
            CircuitFlags::SubtractOperands => Some(self.subtract_operands),
            CircuitFlags::MultiplyOperands => Some(self.multiply_operands),
            CircuitFlags::Load => Some(self.load),
            CircuitFlags::Store => Some(self.store),
            CircuitFlags::Jump => Some(self.jump),
            CircuitFlags::WriteLookupOutputToRD => Some(self.write_lookup_output_to_rd),
            CircuitFlags::VirtualInstruction => Some(self.virtual_instruction),
            CircuitFlags::Assert => Some(self.assert),
            CircuitFlags::DoNotUpdateUnexpandedPC => Some(self.do_not_update_unexpanded_pc),
            CircuitFlags::Advice => Some(self.advice),
            CircuitFlags::IsCompressed => Some(self.is_compressed),
            CircuitFlags::IsFirstInSequence => Some(self.is_first_in_sequence),
            CircuitFlags::IsLastInSequence => Some(self.is_last_in_sequence),
        }
    }
}
