use serde::{Deserialize, Serialize};

/// RV64A AMOAND.D: atomic AND doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoAndD<T = ()>(pub T);
