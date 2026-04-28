use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SB: store byte (lowest 8 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sb<T = ()>(pub T);
