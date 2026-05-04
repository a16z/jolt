//! Marker trait for concrete instruction types that can convert through
//! Jolt's canonical normalized bytecode row.

use crate::NormalizedInstruction;

pub trait JoltInstruction:
    Copy + Into<NormalizedInstruction> + TryFrom<NormalizedInstruction>
{
}

impl JoltInstruction for NormalizedInstruction {}
