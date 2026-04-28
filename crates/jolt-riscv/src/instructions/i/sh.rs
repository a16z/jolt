use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// RV64I SH: store halfword (lowest 16 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sh<T = ()>(pub T);
