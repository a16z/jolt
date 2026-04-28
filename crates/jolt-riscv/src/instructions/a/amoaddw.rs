use serde::{Deserialize, Serialize};

/// RV32A AMOADD.W: atomic add word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoAddW<T = ()>(pub T);
