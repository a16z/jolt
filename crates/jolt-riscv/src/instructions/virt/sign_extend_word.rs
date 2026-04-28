use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual SIGN_EXTEND_WORD: sign-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualSignExtendWord<T = ()>(pub T);
