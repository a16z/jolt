use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I ADDI: `rd = rs1 + imm` (wrapping). Immediate already decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Addi<T = ()>(pub T);
