use serde::{Deserialize, Serialize};

/// RV32A AMOOR.W: atomic OR word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoOrW<T = ()>(pub T);
