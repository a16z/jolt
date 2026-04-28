use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I FENCE: memory ordering fence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct Fence<T = ()>(pub T);
