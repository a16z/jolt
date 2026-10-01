//! Sumcheck round domains.

use crate::error::SumcheckError;
use jolt_field::Field;
use jolt_poly::lagrange::{centered_domain_start, centered_power_sums, CenteredIntegerDomainError};
use jolt_poly::UnivariatePoly;

pub trait SumcheckDomain<F: Field> {
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>>;

    /// Checks that `round`'s sum over this domain equals `running_sum`.
    fn check_round_sum(
        &self,
        round_index: usize,
        running_sum: F,
        round: &UnivariatePoly<F>,
    ) -> Result<(), SumcheckError<F>> {
        let expected = round.coefficients().len();
        let degree = expected
            .checked_sub(1)
            .ok_or(SumcheckError::EmptyRoundCoefficients)?;
        let coefficients = self.round_sum_coefficients(degree)?;
        if coefficients.len() != expected {
            return Err(SumcheckError::RoundSumCoefficientCountMismatch {
                round: round_index,
                expected,
                got: coefficients.len(),
            });
        }

        let actual = round
            .coefficients()
            .iter()
            .zip(&coefficients)
            .map(|(&coefficient, &scale)| coefficient * scale)
            .sum();
        if actual != running_sum {
            return Err(SumcheckError::RoundCheckFailed {
                round: round_index,
                expected: running_sum,
                actual,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckDomainSpec {
    BooleanHypercube,
    CenteredInteger { domain_size: usize },
}

impl SumcheckDomainSpec {
    pub const fn centered_integer(domain_size: usize) -> Self {
        Self::CenteredInteger { domain_size }
    }
}

impl<F> SumcheckDomain<F> for SumcheckDomainSpec
where
    F: Field,
{
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>> {
        match *self {
            Self::BooleanHypercube => BooleanHypercube.round_sum_coefficients(degree),
            Self::CenteredInteger { domain_size } => {
                CenteredIntegerDomain::new(domain_size).round_sum_coefficients(degree)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BooleanHypercube;

impl<F> SumcheckDomain<F> for BooleanHypercube
where
    F: Field,
{
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>> {
        Ok(core::iter::once(F::from_u64(2))
            .chain(core::iter::repeat_n(F::one(), degree))
            .collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenteredIntegerDomain {
    domain_size: usize,
}

impl CenteredIntegerDomain {
    pub const fn new(domain_size: usize) -> Self {
        Self { domain_size }
    }

    pub const fn domain_size(self) -> usize {
        self.domain_size
    }

    pub fn start(self) -> Result<i64, CenteredIntegerDomainError> {
        centered_domain_start(self.domain_size)
    }

    pub fn power_sums(self, num_powers: usize) -> Result<Vec<i128>, CenteredIntegerDomainError> {
        centered_power_sums(self.domain_size, num_powers)
    }
}

impl<F> SumcheckDomain<F> for CenteredIntegerDomain
where
    F: Field,
{
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>> {
        self.power_sums(degree + 1)
            .map_err(|_| SumcheckError::InvalidIntegerDomain {
                domain_size: self.domain_size,
            })
            .map(|power_sums| power_sums.into_iter().map(F::from_i128).collect())
    }
}
