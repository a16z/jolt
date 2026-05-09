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

    pub(super) fn push(&mut self, row: NormalizedInstruction) -> Result<(), ExpansionError> {
        if self.rows.len() == MAX_FINAL_ROWS_PER_SOURCE {
            return Err(ExpansionError::CapacityExceeded {
                actual: self.rows.len() + 1,
                capacity: MAX_FINAL_ROWS_PER_SOURCE,
            });
        }
        self.rows.push(row);
        Ok(())
    }

    pub(super) fn extend_vec(
        &mut self,
        rows: Vec<NormalizedInstruction>,
    ) -> Result<(), ExpansionError> {
        let mut index = 0;
        while index < rows.len() {
            self.push(rows[index])?;
            index += 1;
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
