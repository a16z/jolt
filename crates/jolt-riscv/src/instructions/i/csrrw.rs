use serde::{Deserialize, Serialize};

/// RV32I (Zicsr) CSRRW: atomic CSR read+write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Csrrw<T = ()>(pub T);
