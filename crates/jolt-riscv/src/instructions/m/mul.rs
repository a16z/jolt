use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64M MUL: signed multiply, lower 64 bits of the 128-bit product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Mul<T = ()>(pub T);
