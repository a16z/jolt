use serde::{Deserialize, Serialize};

/// RV64A AMOXOR.D: atomic XOR doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoXorD<T = ()>(pub T);
