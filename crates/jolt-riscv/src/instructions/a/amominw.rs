use serde::{Deserialize, Serialize};

/// RV32A AMOMIN.W: atomic signed min word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoMinW<T = ()>(pub T);
