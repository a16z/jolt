use serde::{Deserialize, Serialize};

/// RV32I (Zicsr) CSRRS: atomic CSR read+set bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Csrrs<T = ()>(pub T);
