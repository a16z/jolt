use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual ZERO_EXTEND_WORD: zero-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualZeroExtendWord<T = ()>(pub T);
