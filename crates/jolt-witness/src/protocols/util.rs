//! Validation helpers shared by protocol witness providers.

use crate::{NamespaceId, WitnessError};

pub(crate) fn require_unique_ids<Id>(
    namespace: NamespaceId,
    ids: impl IntoIterator<Item = Id>,
    label: &'static str,
) -> Result<(), WitnessError>
where
    Id: Copy + Eq + core::fmt::Debug,
{
    let mut seen = Vec::new();
    for id in ids {
        if seen.contains(&id) {
            return Err(WitnessError::InvalidWitnessData {
                namespace: namespace.name,
                reason: format!("duplicate {label} id: {id:?}"),
            });
        }
        seen.push(id);
    }
    Ok(())
}

pub(crate) fn power_of_two_log_rows(
    namespace: NamespaceId,
    rows: usize,
) -> Result<usize, WitnessError> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(WitnessError::InvalidDimensions {
            namespace: namespace.name,
            reason: format!("row count must be a nonzero power of two, got {rows}"),
        });
    }
    Ok(rows.trailing_zeros() as usize)
}
