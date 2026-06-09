use serde::{Deserialize, Serialize};

use crate::HyraxError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HyraxDimensions {
    pub num_vars: usize,
    pub row_vars: usize,
    pub col_vars: usize,
}

impl HyraxDimensions {
    pub fn new(num_vars: usize, row_vars: usize, col_vars: usize) -> Result<Self, HyraxError> {
        let dimensions = Self {
            num_vars,
            row_vars,
            col_vars,
        };
        dimensions.validate()?;
        Ok(dimensions)
    }

    pub fn validate(&self) -> Result<(), HyraxError> {
        if self.row_vars.checked_add(self.col_vars) != Some(self.num_vars) {
            return Err(HyraxError::InvalidDimensions {
                num_vars: self.num_vars,
                row_vars: self.row_vars,
                col_vars: self.col_vars,
            });
        }
        let _ = Self::dimension_len(self.num_vars)?;
        let _ = Self::dimension_len(self.row_vars)?;
        let _ = Self::dimension_len(self.col_vars)?;
        Ok(())
    }

    pub fn row_count(&self) -> Result<usize, HyraxError> {
        Self::dimension_len(self.row_vars)
    }

    pub fn row_len(&self) -> Result<usize, HyraxError> {
        Self::dimension_len(self.col_vars)
    }

    pub fn polynomial_len(&self) -> Result<usize, HyraxError> {
        Self::dimension_len(self.num_vars)
    }

    pub fn split_point<'a, F>(&self, point: &'a [F]) -> Result<(&'a [F], &'a [F]), HyraxError> {
        if point.len() != self.num_vars {
            return Err(HyraxError::PointLengthMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        Ok(point.split_at(self.row_vars))
    }

    fn dimension_len(dimension: usize) -> Result<usize, HyraxError> {
        if dimension >= usize::BITS as usize {
            return Err(HyraxError::DimensionTooLarge { dimension });
        }
        Ok(1usize << dimension)
    }
}
