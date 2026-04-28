use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2IW<T = ()>(pub T);
