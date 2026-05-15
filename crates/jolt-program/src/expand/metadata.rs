use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow};

use crate::expand::{materialize::MAX_FINAL_ROWS_PER_SOURCE, ExpansionError};

const MAX_METADATA_SEQUENCE_ROWS: usize = u16::MAX as usize + 1;

/// Stamps position metadata (`is_first_in_sequence`, `virtual_sequence_remaining`) on recipe output.
pub(super) fn stamp_instruction_sequence(
    rows: Vec<JoltInstructionRow>,
    is_compressed: bool,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
    stamp_sequence_metadata(rows, is_compressed, MAX_FINAL_ROWS_PER_SOURCE, profile)
}

/// Same as `stamp_instruction_sequence` but for inline provider output (higher capacity limit).
pub(super) fn stamp_inline_sequence(
    rows: Vec<JoltInstructionRow>,
    is_compressed: bool,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
    stamp_sequence_metadata(rows, is_compressed, MAX_METADATA_SEQUENCE_ROWS, profile)
}

fn stamp_sequence_metadata(
    mut rows: Vec<JoltInstructionRow>,
    is_compressed: bool,
    capacity: usize,
    profile: JoltInstructionProfile,
) -> Result<Vec<JoltInstructionRow>, ExpansionError> {
    if rows.is_empty() {
        return Err(ExpansionError::EmptySequence);
    }
    if rows.len() > capacity {
        return Err(ExpansionError::CapacityExceeded {
            actual: rows.len(),
            capacity,
        });
    }
    for row in &rows {
        if !profile.supports_jolt(row.instruction_kind) {
            return Err(ExpansionError::IllegalTargetInstruction(
                row.instruction_kind,
            ));
        }
    }

    let len = rows.len();
    for (index, row) in rows.iter_mut().enumerate() {
        row.is_first_in_sequence = index == 0;
        row.virtual_sequence_remaining = Some((len - index - 1) as u16);
        row.is_compressed = index == len - 1 && is_compressed;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow, NormalizedOperands,
        SourceExtension,
    };

    use super::*;

    #[test]
    fn rejects_profile_illegal_rows_before_stamping() {
        const RV64I_ONLY: JoltInstructionProfile = JoltInstructionProfile {
            source_extensions: &[SourceExtension::Rv64I],
            inline_extensions: &[],
        };

        let rows = vec![JoltInstructionRow {
            instruction_kind: JoltInstructionKind::MUL,
            address: 0x8000_0000,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }];

        assert!(matches!(
            stamp_instruction_sequence(rows, false, RV64I_ONLY),
            Err(ExpansionError::IllegalTargetInstruction(
                JoltInstructionKind::MUL
            ))
        ));
    }
}
