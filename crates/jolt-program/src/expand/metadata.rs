use jolt_riscv::{JoltRow, RV64IMAC_JOLT};

use crate::expand::{materialize::MAX_FINAL_ROWS_PER_SOURCE, ExpansionError};

const MAX_METADATA_SEQUENCE_ROWS: usize = u16::MAX as usize + 1;

/// Stamps position metadata (`is_first_in_sequence`, `virtual_sequence_remaining`) on recipe output.
pub(super) fn stamp_instruction_sequence(
    rows: Vec<JoltRow>,
    is_compressed: bool,
) -> Result<Vec<JoltRow>, ExpansionError> {
    stamp_sequence_metadata(rows, is_compressed, MAX_FINAL_ROWS_PER_SOURCE)
}

/// Same as `stamp_instruction_sequence` but for inline provider output (higher capacity limit).
pub(super) fn stamp_inline_sequence(
    rows: Vec<JoltRow>,
    is_compressed: bool,
) -> Result<Vec<JoltRow>, ExpansionError> {
    stamp_sequence_metadata(rows, is_compressed, MAX_METADATA_SEQUENCE_ROWS)
}

fn stamp_sequence_metadata(
    mut rows: Vec<JoltRow>,
    is_compressed: bool,
    capacity: usize,
) -> Result<Vec<JoltRow>, ExpansionError> {
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
        if !RV64IMAC_JOLT.supports_jolt(row.instruction_kind) {
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
    use jolt_riscv::{JoltInstructionKind, JoltRow, NormalizedOperands};

    use super::*;

    #[test]
    fn rejects_source_only_rows_before_stamping() {
        let rows = vec![JoltRow {
            instruction_kind: JoltInstructionKind::ADDIW,
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
            stamp_instruction_sequence(rows, false),
            Err(ExpansionError::IllegalTargetInstruction(
                JoltInstructionKind::ADDIW
            ))
        ));
    }
}
