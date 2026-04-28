use serde::{Deserialize, Serialize};

/// RV32A AMOAND.W: atomic AND word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoAndW<T = ()>(pub T);
