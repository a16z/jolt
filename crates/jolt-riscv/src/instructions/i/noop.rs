use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// No-operation pseudo-instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(IsNoop)]
pub struct Noop<T = ()>(pub T);
