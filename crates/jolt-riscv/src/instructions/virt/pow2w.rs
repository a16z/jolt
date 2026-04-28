use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual POW2W: computes `2^(rs1 mod 32)` for 32-bit mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct Pow2W<T = ()>(pub T);
