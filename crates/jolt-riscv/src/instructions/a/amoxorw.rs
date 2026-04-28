use serde::{Deserialize, Serialize};

/// RV32A AMOXOR.W: atomic XOR word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoXorW<T = ()>(pub T);
