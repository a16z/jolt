use crate::SumcheckError;
use jolt_field::Field;

/// Public transport widths, checked independently of any transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    rounds: usize,
    degree: usize,
}

impl SumcheckShape {
    pub fn new<F: Field>(rounds: usize, degree: usize) -> Result<Self, SumcheckError<F>> {
        if degree == 0 {
            return Err(SumcheckError::ZeroDegreeBound);
        }
        let _ = rounds
            .checked_mul(degree)
            .ok_or(SumcheckError::ShapeOverflow { rounds, degree })?;
        let _ = degree
            .checked_add(1)
            .ok_or(SumcheckError::DegreeOverflow { degree })?;
        Ok(Self { rounds, degree })
    }
    pub fn num_rounds(self) -> usize {
        self.rounds
    }
    pub fn degree_bound(self) -> usize {
        self.degree
    }
}
