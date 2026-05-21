use jolt_field::Field;
use thiserror::Error;

use crate::eq_index_msb;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MleError {
    #[error("equality polynomial arity mismatch: left {left}, right {right}")]
    EqualityArityMismatch { left: usize, right: usize },
    #[error("invalid MLE range [{start}, {end})")]
    InvalidRange { start: u128, end: u128 },
    #[error("MLE arity {arity} exceeds u128 capacity")]
    DomainTooLarge { arity: usize },
    #[error("MLE range end {end} exceeds domain size {domain_size}")]
    RangeEndOutOfDomain { end: u128, domain_size: u128 },
    #[error("MLE block has {block_vars} variables but point has arity {arity}")]
    BlockVariablesExceedArity { block_vars: usize, arity: usize },
    #[error("MLE block start {start_index} is not aligned to block size {block_size}")]
    BlockStartUnaligned {
        start_index: usize,
        block_size: u128,
    },
    #[error("MLE block end {end} exceeds domain size {domain_size}")]
    BlockEndOutOfDomain { end: u128, domain_size: u128 },
}

pub fn sparse_mle_msb<F: Field>(start_index: usize, values: &[u64], point: &[F]) -> F {
    values
        .iter()
        .enumerate()
        .map(|(offset, value)| F::from_u64(*value) * eq_index_msb(point, start_index + offset))
        .sum()
}

pub fn sparse_segments_mle_msb<'a, F, I>(segments: I, point: &[F]) -> F
where
    F: Field,
    I: IntoIterator<Item = (usize, &'a [u64])>,
{
    segments
        .into_iter()
        .map(|(start_index, values)| sparse_mle_msb(start_index, values, point))
        .sum()
}

pub fn block_selector_mle_msb<F: Field>(
    start_index: usize,
    block_num_vars: usize,
    point: &[F],
) -> Result<F, MleError> {
    if block_num_vars > point.len() {
        return Err(MleError::BlockVariablesExceedArity {
            block_vars: block_num_vars,
            arity: point.len(),
        });
    }
    if block_num_vars >= usize::BITS as usize {
        return Err(MleError::DomainTooLarge {
            arity: block_num_vars,
        });
    }

    let block_size = 1u128
        .checked_shl(block_num_vars as u32)
        .ok_or(MleError::DomainTooLarge {
            arity: block_num_vars,
        })?;
    let domain_size = 1u128
        .checked_shl(point.len() as u32)
        .ok_or(MleError::DomainTooLarge { arity: point.len() })?;
    let start = start_index as u128;
    if !start.is_multiple_of(block_size) {
        return Err(MleError::BlockStartUnaligned {
            start_index,
            block_size,
        });
    }
    let end = start
        .checked_add(block_size)
        .ok_or(MleError::DomainTooLarge { arity: point.len() })?;
    if end > domain_size {
        return Err(MleError::BlockEndOutOfDomain { end, domain_size });
    }

    let selector_point_len = point.len() - block_num_vars;
    let block_index = usize::try_from(start / block_size)
        .map_err(|_| MleError::DomainTooLarge { arity: point.len() })?;
    Ok(eq_index_msb(&point[..selector_point_len], block_index))
}

pub fn range_mask_mle_msb<F: Field>(
    range_start: u128,
    range_end: u128,
    point: &[F],
) -> Result<F, MleError> {
    if range_start >= range_end {
        return Err(MleError::InvalidRange {
            start: range_start,
            end: range_end,
        });
    }
    let domain_size = 1u128
        .checked_shl(point.len() as u32)
        .ok_or(MleError::DomainTooLarge { arity: point.len() })?;
    if range_end > domain_size {
        return Err(MleError::RangeEndOutOfDomain {
            end: range_end,
            domain_size,
        });
    }

    Ok(less_than_mle_msb(range_end, point) - less_than_mle_msb(range_start, point))
}

fn less_than_mle_msb<F: Field>(bound: u128, point: &[F]) -> F {
    if Some(bound) == 1u128.checked_shl(point.len() as u32) {
        return F::one();
    }
    let mut lt_bound = F::zero();
    let mut eq_bound = F::one();
    for (index, challenge) in point.iter().enumerate() {
        if msb_bit(bound, point.len(), index) == 1 {
            lt_bound += eq_bound * (F::one() - *challenge);
            eq_bound *= *challenge;
        } else {
            eq_bound *= F::one() - *challenge;
        }
    }
    lt_bound
}

const fn msb_bit(value: u128, len: usize, position: usize) -> u8 {
    let shift = len - 1 - position;
    if shift < u128::BITS as usize {
        ((value >> shift) & 1) as u8
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use crate::{eq_index_msb, try_eq_mle};
    use jolt_field::{Fr, FromPrimitiveInt};
    use num_traits::{One, Zero};

    #[test]
    fn eq_index_uses_msb_order() {
        let point = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];

        assert_eq!(
            eq_index_msb(&point, 0b101),
            point[0] * (Fr::one() - point[1]) * point[2]
        );
        assert_eq!(
            eq_index_msb(&point, 0b010),
            (Fr::one() - point[0]) * point[1] * (Fr::one() - point[2])
        );
    }

    #[test]
    fn checked_eq_mle_rejects_arity_mismatch() {
        assert_eq!(
            try_eq_mle::<Fr>(&[Fr::from_u64(1)], &[Fr::from_u64(1), Fr::from_u64(0)]),
            Err(MleError::EqualityArityMismatch { left: 1, right: 2 })
        );
    }

    #[test]
    fn sparse_mle_matches_explicit_sum() {
        let point = [Fr::from_u64(2), Fr::from_u64(3)];
        let values = [7, 11];

        assert_eq!(
            sparse_mle_msb(1, &values, &point),
            Fr::from_u64(7) * eq_index_msb(&point, 1) + Fr::from_u64(11) * eq_index_msb(&point, 2)
        );
    }

    #[test]
    fn sparse_segments_mle_sums_segments() {
        let point = [Fr::from_u64(2), Fr::from_u64(3)];
        let left = [7];
        let right = [11];

        assert_eq!(
            sparse_segments_mle_msb([(1, left.as_slice()), (2, right.as_slice())], &point),
            sparse_mle_msb(1, &left, &point) + sparse_mle_msb(2, &right, &point)
        );
    }

    #[test]
    fn block_selector_evaluates_aligned_prefix() {
        let point = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];

        assert_eq!(
            block_selector_mle_msb(0b100, 1, &point)
                .unwrap_or_else(|error| panic!("selector should evaluate: {error}")),
            eq_index_msb(&point[..2], 0b10)
        );
        assert_eq!(
            block_selector_mle_msb(0, 3, &point)
                .unwrap_or_else(|error| panic!("whole-domain selector should evaluate: {error}")),
            Fr::one()
        );
    }

    #[test]
    fn block_selector_rejects_invalid_blocks() {
        assert_eq!(
            block_selector_mle_msb::<Fr>(0, 3, &[Fr::zero(), Fr::zero()]),
            Err(MleError::BlockVariablesExceedArity {
                block_vars: 3,
                arity: 2
            })
        );
        assert_eq!(
            block_selector_mle_msb::<Fr>(1, 1, &[Fr::zero(), Fr::zero()]),
            Err(MleError::BlockStartUnaligned {
                start_index: 1,
                block_size: 2
            })
        );
        assert_eq!(
            block_selector_mle_msb::<Fr>(4, 1, &[Fr::zero(), Fr::zero()]),
            Err(MleError::BlockEndOutOfDomain {
                end: 6,
                domain_size: 4
            })
        );
        assert_eq!(
            block_selector_mle_msb::<Fr>(
                0,
                usize::BITS as usize,
                &vec![Fr::zero(); usize::BITS as usize]
            ),
            Err(MleError::DomainTooLarge {
                arity: usize::BITS as usize
            })
        );
    }

    #[test]
    fn range_mask_matches_vertex_membership() {
        for index in 0..8 {
            let point = [
                Fr::from_u64(((index >> 2) & 1) as u64),
                Fr::from_u64(((index >> 1) & 1) as u64),
                Fr::from_u64((index & 1) as u64),
            ];
            let expected = if (2..5).contains(&index) {
                Fr::one()
            } else {
                Fr::zero()
            };
            assert_eq!(
                range_mask_mle_msb(2, 5, &point)
                    .unwrap_or_else(|error| panic!("range mask should evaluate: {error}")),
                expected
            );
        }
    }

    #[test]
    fn range_mask_rejects_invalid_ranges() {
        assert_eq!(
            range_mask_mle_msb::<Fr>(3, 3, &[Fr::zero(), Fr::zero()]),
            Err(MleError::InvalidRange { start: 3, end: 3 })
        );
        assert_eq!(
            range_mask_mle_msb(0, 4, &[Fr::from_u64(7), Fr::from_u64(11)])
                .unwrap_or_else(|error| panic!("full-domain range should evaluate: {error}")),
            Fr::one()
        );
        assert_eq!(
            range_mask_mle_msb::<Fr>(0, 5, &[Fr::zero(), Fr::zero()]),
            Err(MleError::RangeEndOutOfDomain {
                end: 5,
                domain_size: 4
            })
        );
    }
}
