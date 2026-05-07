//! Marker trait for concrete instruction types that can convert through
//! Jolt's canonical normalized bytecode row.

use crate::NormalizedInstruction;

pub trait JoltInstruction:
    Copy + Into<NormalizedInstruction> + TryFrom<NormalizedInstruction>
{
    fn normalize(&self) -> NormalizedInstruction {
        (*self).into()
    }

    fn is_virtual(&self) -> bool {
        self.normalize().virtual_sequence_remaining.is_some()
    }
}

impl JoltInstruction for NormalizedInstruction {}
