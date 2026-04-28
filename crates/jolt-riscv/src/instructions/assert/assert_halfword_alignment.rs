use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_HALFWORD_ALIGNMENT: checks whether `rs1 + imm` is 2-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertHalfwordAlignment<T = ()>(pub T);
