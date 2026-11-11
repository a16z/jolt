use jolt_field::JoltField;
use thiserror::Error;

use crate::{eq_index_msb, EqPolynomial};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MleError {
    #[error("equality polynomial arity mismatch: left {left}, right {right}")]
    EqualityArityMismatch { left: usize, right: usize },
    #[error("invalid MLE range [{start}, {end})")]
    InvalidRange { start: u128, end: u128 },
    #[error("MLE intervals are not canonical: start {start} follows end {previous_end}")]
    NonCanonicalIntervals { previous_end: u128, start: u128 },
    #[error("MLE arity {arity} exceeds u128 capacity")]
    DomainTooLarge { arity: usize },
    #[error("MLE range end {end} exceeds domain size {domain_size}")]
    RangeEndOutOfDomain { end: u128, domain_size: u128 },
    #[error("MLE block has {block_vars} variables but point has arity {arity}")]
    BlockVariablesExceedArity { block_vars: usize, arity: usize },
    #[error("MLE block start {start_index} is not aligned to block size {block_size}")]
    BlockStartUnaligned { start_index: u128, block_size: u128 },
    #[error("MLE block end {end} exceeds domain size {domain_size}")]
    BlockEndOutOfDomain { end: u128, domain_size: u128 },
}

pub fn sparse_mle_msb<F: JoltField>(start_index: u128, values: &[u64], point: &[F]) -> F {
    sparse_segments_mle_msb(core::iter::once((start_index, values)), point)
}

pub fn sparse_segments_mle_msb<'a, F, I>(segments: I, point: &[F]) -> F
where
    F: JoltField,
    I: IntoIterator<Item = (u128, &'a [u64])>,
    I::IntoIter: Clone,
{
    let segments = segments.into_iter();
    let (entries, end_overflow) =
        segments
            .clone()
            .fold((0usize, false), |(entries, overflow), (start, values)| {
                (
                    entries.saturating_add(values.len()),
                    overflow || start.checked_add(values.len() as u128).is_none(),
                )
            });
    if entries == 0 {
        return F::zero();
    }
    let scalar = || {
        segments
            .clone()
            .map(|(start, values)| {
                values
                    .iter()
                    .enumerate()
                    .map(|(offset, value)| {
                        eq_index_msb(point, start + offset as u128).mul_u64(*value)
                    })
                    .sum::<F>()
            })
            .sum::<F>()
    };
    if point.is_empty() || end_overflow {
        return scalar();
    }
    // A bounded low table is shared by all contiguous public-memory segments.
    // Each high-address factor is constant for a whole table-width chunk.
    let entry_bits = (entries - 1).bit_width() as usize;
    let low_bits = point.len().min(10).min(entry_bits);
    let split = point.len() - low_bits;
    let width = 1usize << low_bits;
    let low = EqPolynomial::<F>::evals(&point[split..], None);
    let mut value = F::zero();
    for (mut address, mut words) in segments {
        while !words.is_empty() {
            let residue = (address & (width as u128 - 1)) as usize;
            let take = words.len().min(width - residue);
            let high = eq_index_msb(&point[..split], address >> low_bits);
            if !high.is_zero() {
                let dot: F = words[..take]
                    .iter()
                    .zip(&low[residue..residue + take])
                    .filter(|(word, _)| **word != 0)
                    .map(|(&word, &weight)| weight.mul_u64(word))
                    .sum();
                value += high * dot;
            }
            address += take as u128;
            words = &words[take..];
        }
    }
    value
}

pub fn block_selector_mle_msb<F: JoltField>(
    start_index: u128,
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
    let start = start_index;
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
    let block_index = start / block_size;
    Ok(eq_index_msb(&point[..selector_point_len], block_index))
}

pub fn range_mask_mle_msb<F: JoltField>(
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

/// Evaluate the Boolean mask of canonical disjoint, merged half-open intervals.
/// Each interval needs two O(log K) prefix evaluations and no domain-sized table.
pub fn interval_mask_mle_msb<F: JoltField>(
    intervals: &[(usize, usize)],
    point: &[F],
) -> Result<F, MleError> {
    let mut value = F::zero();
    let mut previous_end = None;
    for &(start, end) in intervals {
        if let Some(previous_end) = previous_end {
            if start <= previous_end {
                return Err(MleError::NonCanonicalIntervals {
                    previous_end: previous_end as u128,
                    start: start as u128,
                });
            }
        }
        value += range_mask_mle_msb(start as u128, end as u128, point)?;
        previous_end = Some(end);
    }
    Ok(value)
}

fn less_than_mle_msb<F: JoltField>(bound: u128, point: &[F]) -> F {
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
    use crate::{eq_index_msb, try_eq_mle, EqPolynomial};
    use jolt_field::{Fr, Ring};
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
    fn sparse_mle_accepts_a_run_ending_at_the_last_u128_index() {
        let point: Vec<Fr> = (0..128).map(|i| Fr::from_u64(i + 2)).collect();
        let all_ones = point.iter().fold(Fr::one(), |acc, x| acc * x);
        assert_eq!(
            sparse_mle_msb(u128::MAX, &[7], &point),
            all_ones * Fr::from_u64(7)
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
    fn sparse_mle_matches_dense_evaluation_across_blocks() {
        let point: Vec<Fr> = (2..8).map(Fr::from_u64).collect();
        let dense_eq = EqPolynomial::<Fr>::evals(&point, None);
        for (start, len) in [(0usize, 0usize), (0, 1), (5, 13), (3, 61), (60, 4)] {
            let values: Vec<u64> = (0..len as u64).map(|i| 3 * i + 1).collect();
            let mut dense = vec![0u64; dense_eq.len()];
            dense[start..start + len].copy_from_slice(&values);
            let expected = dense_eq
                .iter()
                .zip(&dense)
                .map(|(eq, value)| *eq * Fr::from_u64(*value))
                .sum::<Fr>();
            assert_eq!(
                sparse_mle_msb(start as u128, &values, &point),
                expected,
                "{start} {len}"
            );
        }
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
    fn interval_mask_matches_independent_vertex_sum() {
        for arity in 0..=5 {
            let size = 1usize << arity;
            let points: Vec<Vec<Fr>> = (0..3)
                .map(|kind| {
                    (0..arity)
                        .map(|i| match kind {
                            0 => Fr::from_u64(i as u64 + 2),
                            1 => Fr::zero(),
                            _ => Fr::one(),
                        })
                        .collect()
                })
                .collect();
            for (start, end) in
                (0..size).flat_map(|start| (start + 1..=size).map(move |end| (start, end)))
            {
                let mut intervals = vec![(start, end)];
                if end + 1 < size {
                    intervals.push((end + 1, size));
                }
                for point in &points {
                    let expected: Fr = (0..size)
                        .filter(|&word| intervals.iter().any(|&(lo, hi)| lo <= word && word < hi))
                        .map(|word| {
                            point
                                .iter()
                                .enumerate()
                                .fold(Fr::one(), |weight, (bit, r)| {
                                    weight
                                        * if word & (1 << (arity - bit - 1)) == 0 {
                                            Fr::one() - r
                                        } else {
                                            *r
                                        }
                                })
                        })
                        .sum();
                    assert_eq!(interval_mask_mle_msb(&intervals, point), Ok(expected));
                }
            }
            assert_eq!(interval_mask_mle_msb(&[], &points[0]), Ok(Fr::zero()));
        }
    }

    #[test]
    fn interval_mask_rejects_noncanonical_and_out_of_domain_metadata() {
        let point = [Fr::from_u64(7), Fr::from_u64(11)];
        for intervals in [
            vec![(0, 2), (1, 3)],
            vec![(0, 2), (2, 3)],
            vec![(2, 3), (0, 1)],
        ] {
            assert!(matches!(
                interval_mask_mle_msb(&intervals, &point),
                Err(MleError::NonCanonicalIntervals { .. })
            ));
        }
        assert_eq!(
            interval_mask_mle_msb(&[(0, 5)], &point),
            Err(MleError::RangeEndOutOfDomain {
                end: 5,
                domain_size: 4
            })
        );
        assert_eq!(
            interval_mask_mle_msb(&[(2, 2)], &point),
            Err(MleError::InvalidRange { start: 2, end: 2 })
        );
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

#[cfg(test)]
mod sparse_segment_table_tests {
    use super::*;
    use jolt_field::{Fr, Prime128Offset6A23801F};

    #[test]
    fn bounded_low_tables_match_scalar_mle_for_offsets_and_boolean_points() {
        check_sparse_segments::<Fr>();
        check_sparse_segments::<Prime128Offset6A23801F>();
    }

    fn check_sparse_segments<F: JoltField>() {
        for arity in [0, 1, 7, 11, 25, 129] {
            for kind in 0..3 {
                let point: Vec<F> = (0..arity)
                    .map(|i| {
                        F::from_u64(match kind {
                            0 => i as u64 + 7,
                            1 => 0,
                            _ => 1,
                        })
                    })
                    .collect();
                for len in [0, 1, 2, 127, 128, 1023, 2301] {
                    let words: Vec<u64> = (0..len)
                        .map(|i| {
                            if i % 7 == 6 {
                                0
                            } else {
                                (i as u64 + 1).wrapping_mul(0x000D_EADB_EEF1_2345)
                            }
                        })
                        .collect();
                    for base in [0, 1, 31, 1023, 2048, 8191] {
                        let expected: F = words
                            .iter()
                            .enumerate()
                            .map(|(i, &word)| {
                                F::from_u64(word) * eq_index_msb(&point, base + i as u128)
                            })
                            .sum();
                        assert_eq!(sparse_mle_msb(base, &words, &point), expected);
                        let split = len.min(1023);
                        assert_eq!(
                            sparse_segments_mle_msb(
                                [
                                    (base, &words[..split]),
                                    (base + split as u128, &words[split..])
                                ],
                                &point
                            ),
                            expected
                        );
                    }
                }
            }
        }
    }
}
