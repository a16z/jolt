use serde::{Deserialize, Serialize};

/// RV32A AMOMAX.W: atomic signed max word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMaxW<T = ()>(pub T);
