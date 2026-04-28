use serde::{Deserialize, Serialize};

/// RV64A AMOSWAP.D: atomic swap doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoSwapD<T = ()>(pub T);
