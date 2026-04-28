use serde::{Deserialize, Serialize};

/// Virtual ADVICE_LD: advice tape value for load doubleword.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdviceLd<T = ()>(pub T);
