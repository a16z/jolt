use serde::{Deserialize, Serialize};

/// RV32I MRET: machine-mode return from trap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mret<T = ()>(pub T);
