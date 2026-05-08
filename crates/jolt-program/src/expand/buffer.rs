use jolt_riscv::NormalizedInstruction;

use crate::expand::ExpansionError;

pub(super) const MAX_FINAL_ROWS_PER_SOURCE: usize = 64;

#[derive(Debug)]
pub(super) struct ExpansionBuffer {
    rows: Vec<NormalizedInstruction>,
}

impl ExpansionBuffer {
    pub(super) fn new() -> Self {
        Self {
            rows: Vec::with_capacity(MAX_FINAL_ROWS_PER_SOURCE),
        }
    }

    pub(super) fn push(&mut self, row: NormalizedInstruction) {
        self.rows.push(row);
    }

    pub(super) fn extend(
        &mut self,
        rows: impl IntoIterator<Item = NormalizedInstruction>,
    ) -> Result<(), ExpansionError> {
        for row in rows {
            if self.rows.len() == MAX_FINAL_ROWS_PER_SOURCE {
                return Err(ExpansionError::CapacityExceeded {
                    actual: self.rows.len() + 1,
                    capacity: MAX_FINAL_ROWS_PER_SOURCE,
                });
            }
            self.push(row);
        }
        Ok(())
    }

    pub(super) fn check_capacity(&self) -> Result<(), ExpansionError> {
        if self.rows.len() > MAX_FINAL_ROWS_PER_SOURCE {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len(),
                capacity: MAX_FINAL_ROWS_PER_SOURCE,
            });
        }
        Ok(())
    }

    pub(super) fn into_vec(self) -> Vec<NormalizedInstruction> {
        self.rows
    }
}
