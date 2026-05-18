//! Sumcheck round domains.

use crate::error::SumcheckError;
use crate::round_proof::ClearRound;
use crate::scalar::SumcheckScalar;
use jolt_poly::lagrange::{centered_domain_start, centered_power_sums, CenteredIntegerDomainError};

pub trait SumcheckDomain<F: SumcheckScalar> {
    fn check_round_sum<R>(
        &self,
        round_index: usize,
        running_sum: F,
        round: &R,
    ) -> Result<(), SumcheckError<F>>
    where
        R: ClearRound<F>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BooleanHypercube;

impl<F> SumcheckDomain<F> for BooleanHypercube
where
    F: SumcheckScalar,
{
    fn check_round_sum<R>(
        &self,
        round_index: usize,
        running_sum: F,
        round: &R,
    ) -> Result<(), SumcheckError<F>>
    where
        R: ClearRound<F>,
    {
        round.check_round_well_formed(round_index)?;
        let actual = round.evaluate(F::zero()) + round.evaluate(F::one());
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
    F: SumcheckScalar,
{
    fn check_round_sum<R>(
        &self,
        round_index: usize,
        running_sum: F,
        round: &R,
    ) -> Result<(), SumcheckError<F>>
    where
        R: ClearRound<F>,
    {
        round.check_round_well_formed(round_index)?;
        let start = self
            .start()
            .map_err(|_| SumcheckError::InvalidIntegerDomain {
                domain_size: self.domain_size,
            })?;

        let mut actual = F::zero();
        for offset in 0..self.domain_size {
            let offset =
                i64::try_from(offset).map_err(|_| SumcheckError::InvalidIntegerDomain {
                    domain_size: self.domain_size,
                })?;
            let point = start
                .checked_add(offset)
                .ok_or(SumcheckError::InvalidIntegerDomain {
                    domain_size: self.domain_size,
                })?;
            actual += round.evaluate(F::from_i64(point));
        }

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
