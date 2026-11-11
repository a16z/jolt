use crate::SumcheckError;
use jolt_field::Field;

/// Public transport widths, checked independently of any field or transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    rounds: usize,
    degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SumcheckShapeError {
    #[error("sumcheck degree bound must be positive")]
    ZeroDegree,
    #[error("sumcheck shape of {rounds} rounds at degree {degree} is not representable")]
    Overflow { rounds: usize, degree: usize },
    #[error("sumcheck degree {degree} cannot be represented with its coefficient count")]
    DegreeOverflow { degree: usize },
    #[error("sumcheck round count {rounds} exceeds u32::MAX")]
    RoundCountOutOfRange { rounds: usize },
}

impl SumcheckShape {
    pub fn new(rounds: usize, degree: usize) -> Result<Self, SumcheckShapeError> {
        if degree == 0 {
            return Err(SumcheckShapeError::ZeroDegree);
        }
        let _ = rounds
            .checked_mul(degree)
            .ok_or(SumcheckShapeError::Overflow { rounds, degree })?;
        let _ = degree
            .checked_add(1)
            .ok_or(SumcheckShapeError::DegreeOverflow { degree })?;
        let _ = u32::try_from(rounds)
            .map_err(|_| SumcheckShapeError::RoundCountOutOfRange { rounds })?;
        Ok(Self { rounds, degree })
    }
    pub fn num_rounds(self) -> usize {
        self.rounds
    }
    pub fn degree_bound(self) -> usize {
        self.degree
    }
}

impl<E: Field> From<SumcheckShapeError> for SumcheckError<E> {
    fn from(error: SumcheckShapeError) -> Self {
        match error {
            SumcheckShapeError::ZeroDegree => Self::ZeroDegreeBound,
            SumcheckShapeError::Overflow { rounds, degree } => {
                Self::ShapeOverflow { rounds, degree }
            }
            SumcheckShapeError::DegreeOverflow { degree } => Self::DegreeOverflow { degree },
            SumcheckShapeError::RoundCountOutOfRange { rounds } => {
                Self::RoundCountOutOfRange { rounds }
            }
        }
    }
}
