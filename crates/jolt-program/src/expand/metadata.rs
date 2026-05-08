use jolt_riscv::NormalizedInstruction;

use crate::expand::ExpansionError;

pub(super) fn stamp_sequence(
    rows: Vec<NormalizedInstruction>,
    is_compressed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    if rows.is_empty() {
        return Err(ExpansionError::EmptySequence);
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
