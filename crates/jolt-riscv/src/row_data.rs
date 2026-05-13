//! Marker trait for concrete instruction payloads that can convert through
//! Jolt's canonical final bytecode row.

use crate::JoltInstructionRow;

pub trait JoltInstructionRowData:
    Copy + Into<JoltInstructionRow> + TryFrom<JoltInstructionRow>
{
    fn jolt_instruction_row(&self) -> JoltInstructionRow {
        (*self).into()
    }

    fn is_virtual(&self) -> bool {
        self.jolt_instruction_row()
            .virtual_sequence_remaining
            .is_some()
    }
}

impl JoltInstructionRowData for JoltInstructionRow {}
