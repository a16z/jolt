use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SHIFT_RIGHT_BITMASK: bitmask for the shift amount stored in `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualShiftRightBitmask<T = ()>(pub T);
