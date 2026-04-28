use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual POW2I: computes `2^imm` with immediate exponent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2I<T = ()>(pub T);
