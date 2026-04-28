use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ASSERT_WORD_ALIGNMENT: checks whether `rs1 + imm` is 4-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertWordAlignment<T = ()>(pub T);
