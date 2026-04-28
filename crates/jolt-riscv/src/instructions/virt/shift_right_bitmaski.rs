use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SHIFT_RIGHT_BITMASKI: bitmask for the shift amount stored in the immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct VirtualShiftRightBitmaski<T = ()>(pub T);
