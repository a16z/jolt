use serde::{Deserialize, Serialize};

/// RV32A LR.W: load-reserved word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LrW<T = ()>(pub T);
