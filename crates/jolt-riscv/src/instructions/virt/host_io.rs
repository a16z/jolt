use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

/// Virtual HOST_IO: host I/O side-effect instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
pub struct VirtualHostIO<T = ()>(pub T);
