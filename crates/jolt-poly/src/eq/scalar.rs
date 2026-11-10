use super::EqPolynomial;
use crate::{MleError, PointLike, PointRef};
use jolt_field::Field;
pub fn try_eq_mle<F: Field>(left: &[F], right: &[F]) -> Result<F, MleError> {
    if left.len() != right.len() {
        return Err(MleError::EqualityArityMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    Ok(EqPolynomial::<F>::mle(left, right))
}

/// Equality weight at `index`, or zero when `index` is outside the domain.
#[inline]
pub fn eq_index<F: Field, P: PointLike<Coord = F> + ?Sized>(point: &P, index: u128) -> F {
    if point.num_vars() < u128::BITS as usize && index >> point.num_vars() != 0 {
        return F::zero();
    }
    eq_index_low_bits(point, index)
}

/// Equality weight using the low `point.len()` bits of `index` (higher bits alias).
#[inline]
pub fn eq_index_msb<F: Field>(point: &[F], index: u128) -> F {
    eq_index_low_bits(&PointRef::high_to_low(point), index)
}

#[inline]
fn eq_index_low_bits<F: Field, P: PointLike<Coord = F> + ?Sized>(point: &P, index: u128) -> F {
    let mut eq = F::one();
    for position in 0..point.num_vars() {
        let challenge = point.msb(position);
        let shift = point.num_vars() - 1 - position;
        let bit = if shift < u128::BITS as usize {
            (index >> shift) & 1
        } else {
            0
        };
        if bit == 1 {
            eq *= challenge;
        } else {
            eq *= F::one() - challenge;
        }
    }
    eq
}

pub fn boolean_bits_msb(num_vars: usize, index: usize) -> Vec<bool> {
    (0..num_vars)
        .map(|position| {
            let shift = num_vars - 1 - position;
            let bit = if shift < usize::BITS as usize {
                (index >> shift) & 1
            } else {
                0
            };
            bit == 1
        })
        .collect()
}

pub fn boolean_point_msb<F: Field>(num_vars: usize, index: usize) -> Vec<F> {
    boolean_bits_msb(num_vars, index)
        .into_iter()
        .map(|bit| F::from_u64(bit as u64))
        .collect()
}

pub fn boolean_index_msb<F: Field>(point: &[F]) -> Option<usize> {
    let mut index = 0usize;
    for value in point {
        index = index.checked_shl(1)?;
        if value.is_one() {
            index |= 1;
        } else if !value.is_zero() {
            return None;
        }
    }
    Some(index)
}

impl<F: Field> EqPolynomial<F> {
    /// Sum equality weights on `start..2^point.len()` in O(point.len()) time.
    /// Subcubes strictly above the index have remaining-coordinate mass one.
    pub fn suffix_sum<P: PointLike<Coord = F> + ?Sized>(point: &P, start: usize) -> F {
        let len = 1usize << point.num_vars();
        assert!(start <= len);
        if start == len {
            return F::zero();
        }
        let mut prefix = F::one();
        let mut greater = F::zero();
        for i in 0..point.num_vars() {
            let r = point.msb(i);
            if start & (1 << (point.num_vars() - i - 1)) == 0 {
                greater += prefix * r;
                prefix *= F::one() - r;
            } else {
                prefix *= r;
            }
        }
        greater + prefix
    }

    /// Sum equality weights at indices `0..len` in O(number of variables).
    pub fn prefix_sum<P: PointLike<Coord = F> + ?Sized>(point: &P, len: usize) -> F {
        F::one() - Self::suffix_sum(point, len)
    }
}
