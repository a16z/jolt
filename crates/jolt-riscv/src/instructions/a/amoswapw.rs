use serde::{Deserialize, Serialize};

/// RV32A AMOSWAP.W: atomic swap word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmoSwapW<T = ()>(pub T);
