use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SW: store word (lowest 32 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sw<T = ()>(pub T);
