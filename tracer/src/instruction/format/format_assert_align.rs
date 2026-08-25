use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// Format for assert instructions that use `rs1` and `imm` but do not write to a destination register.
///
/// Used by:
/// - `VirtualAssert` - asserts that rs1 is non-zero
/// - `VirtualAssertHalfwordAlignment` - asserts halfword alignment
/// - `VirtualAssertWordAlignment` - asserts word alignment
///
/// Note: Some assert instructions (like `VirtualAssertEQ`, `VirtualAssertLTE`) use two source
/// registers and therefore use `FormatB` instead.
///
/// WARNING: `imm` is the sign-extended offset reinterpreted as `u64`, matching
/// `FormatI`/`FormatU`/`FormatJ`. The alignment asserts carry `AddOperands`, so
/// their lookup index is `rs1 + imm` and the R1CS row
/// `RightLookupOperand == LeftInstructionInput + RightInstructionInput` compares
/// it against the normalized `imm` as a field element. Storing a *signed* `imm`
/// here would make a negative effective address produce a lookup index of
/// `2^128 - |rs1 + imm|`, which is unequal to `rs1 + imm` in any field and so
/// unprovable — and whose only satisfying representative, `p - |rs1 + imm|`,
/// lies in the fp128 alias band (see `instruction_read_raf`'s canonical-address
/// term). Effective addresses are mod 2^64; the type now says so.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAssert {
    pub rs1: u8,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatAssert {
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateFormatAssert {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rs1: if operands.rs1.unwrap() == 0 {
                0
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }
}

impl InstructionFormat for FormatAssert {
    type RegisterState = RegisterStateFormatAssert;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu, self.rs1 as usize);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            imm: rng.next_u64(),
        }
    }
}

impl From<NormalizedOperands> for FormatAssert {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatAssert> for NormalizedOperands {
    fn from(format: FormatAssert) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: None,
            rd: None,
            imm: format.imm as i128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alignment asserts carry `AddOperands`, so their normalized `imm` is
    /// compared against a lookup index as a field element. A *negative*
    /// normalized `imm` would make that index `2^128 - |rs1 + imm|`, which is
    /// unequal to `rs1 + imm` in any field — and whose only satisfying
    /// representative, `p - |rs1 + imm|`, sits in the fp128 alias band.
    ///
    /// `emit_address` forwards the raw signed load offset, so the wrap has to
    /// happen here, exactly as `FormatI`/`FormatU`/`FormatJ` do by storing `u64`.
    #[test]
    fn normalized_immediate_is_wrapped_to_u64() {
        for imm in [
            0i128,
            1,
            -1,
            -8,
            i32::MIN as i128,
            i64::MIN as i128,
            u32::MAX as i128,
        ] {
            let operands = NormalizedOperands {
                rs1: Some(1),
                rs2: None,
                rd: None,
                imm,
            };
            let round_tripped: NormalizedOperands = FormatAssert::from(operands).into();
            assert_eq!(
                round_tripped.imm, imm as u64 as i128,
                "imm {imm} must round-trip through the u64 wrap"
            );
            assert!(
                round_tripped.imm >= 0,
                "normalized imm {} stayed negative",
                round_tripped.imm
            );
        }
    }
}
