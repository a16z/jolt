use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual POW2: computes `2^rs1` using the low 6 bits of `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Pow2<T = ()>(pub T);
