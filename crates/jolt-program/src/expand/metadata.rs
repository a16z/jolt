use jolt_riscv::NormalizedInstruction;

use crate::expand::{grammar::is_target_legal, ExpansionError};

pub(super) fn stamp_sequence(
    rows: Vec<NormalizedInstruction>,
    is_compressed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    if rows.is_empty() {
        return Err(ExpansionError::EmptySequence);
    }
    for row in &rows {
        if !is_target_legal(row.instruction_kind) {
            return Err(ExpansionError::IllegalTargetInstruction(
                row.instruction_kind,
            ));
        }
    }

    let len = rows.len();
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(index, mut row)| {
            row.is_first_in_sequence = index == 0;
            row.virtual_sequence_remaining = Some((len - index - 1) as u16);
            row.is_compressed = index == len - 1 && is_compressed;
            row
        })
        .collect())
}
