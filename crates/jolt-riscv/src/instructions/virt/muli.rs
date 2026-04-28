use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MulI<T = ()>(pub T);
