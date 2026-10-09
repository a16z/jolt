use crate::error::SumcheckError;
use crate::round_proof::ClearRound;
use core::ops::Range;
use jolt_field::Field;
#[cfg(feature = "binary")]
use jolt_field::F8;
#[cfg(feature = "binary")]
use jolt_poly::lagrange::f8_domain_nodes;
use jolt_poly::lagrange::{centered_domain_start, centered_power_sums, CenteredIntegerDomainError};

pub trait SumcheckDomain<F: Field> {
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>>;

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
        let coefficients = self.round_sum_coefficients(round.degree())?;
        let expected = round.degree() + 1;
        if coefficients.len() != expected {
            return Err(SumcheckError::RoundSumCoefficientCountMismatch {
                round: round_index,
                expected,
                got: coefficients.len(),
            });
        }

        let actual = round.coefficient_linear_combination(&coefficients);
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

/// The first `size` elements of `F8` in raw order, embedded in the proof field.
///
/// The field's `From<F8>` must be a field embedding, as it is for `F64`, `F128`,
/// and `F192`. Round-sum coefficients validate the size before the degree and
/// sum powers in the proof field. Valid sizes are 1 through 256.
#[cfg(feature = "binary")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F8Domain {
    size: usize,
}

#[cfg(feature = "binary")]
impl F8Domain {
    /// Records a size; validation is deferred to the round-sum computation.
    pub const fn new(size: usize) -> Self {
        Self { size }
    }

    /// Returns the requested number of points, even if the size is invalid.
    pub const fn size(self) -> usize {
        self.size
    }
}

#[cfg(feature = "binary")]
impl<F: Field + From<F8>> SumcheckDomain<F> for F8Domain {
    fn round_sum_coefficients(&self, degree: usize) -> Result<Vec<F>, SumcheckError<F>> {
        let nodes = f8_domain_nodes::<F>(self.size)
            .map_err(|error| SumcheckError::InvalidF8Domain { size: error.size })?;
        let count = degree
            .checked_add(1)
            .ok_or(SumcheckError::DegreeOverflow { degree })?;
        let mut sums = vec![F::zero(); count];
        for node in nodes {
            let mut power = F::one();
            for sum in &mut sums {
                *sum += power;
                power *= node;
            }
        }
        Ok(sums)
    }
}

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

/// Collisions among this many leading offsets are reported before the power
/// sums are computed, so a small characteristic is named even when the power
/// sums would overflow. The remaining offsets are scanned only once the power
/// sums exist, which bounds the scan by work already done.
const EAGER_COLLISION_SCAN: usize = 1 << 16;

/// Consecutive centered integers, whose images must be distinct in the field.
/// Round-sum checks reject collisions before mapping power sums into the field.
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
        let _ = self
            .start()
            .map_err(|_| SumcheckError::InvalidIntegerDomain {
                domain_size: self.domain_size,
            })?;
        let collides =
            |offsets: Range<usize>| offsets.into_iter().any(|k| F::from_u64(k as u64).is_zero());
        let not_distinct = SumcheckError::IntegerDomainNotDistinct {
            domain_size: self.domain_size,
        };
        let eager = self.domain_size.min(EAGER_COLLISION_SCAN);
        if collides(1..eager) {
            return Err(not_distinct);
        }
        let power_sums =
            self.power_sums(degree + 1)
                .map_err(|_| SumcheckError::InvalidIntegerDomain {
                    domain_size: self.domain_size,
                })?;
        if collides(eager..self.domain_size) {
            return Err(not_distinct);
        }
        Ok(power_sums.into_iter().map(F::from_i128).collect())
    }
}
