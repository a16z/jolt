use serde::{Deserialize, Serialize};

/// Virtual SW: virtual store word used in tracer-emitted virtual sequences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VirtualSw<T = ()>(pub T);
