use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LW: advice tape value for load word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdviceLw<T = ()>(pub T);
