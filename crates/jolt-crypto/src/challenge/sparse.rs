//! The [`SparseChallenge`] data type.
//!
//! A [`SparseChallenge`] is a sparse polynomial in `F[X] / (X^D + 1)` represented
//! by its non-zero terms. It is the unified output type for every sampling family
//! in [`crate::challenge::SparseChallengeConfig`], so downstream arithmetic can stay uniform
//! regardless of how a challenge was sampled.
//!
//! Production challenges are expected to come from the crate's fold-challenge
//! sampler, which constructs values satisfying the invariants below. Methods on this
//! type check cheap shape/range errors needed for memory safety, but they do
//! not re-validate every sampler invariant on the hot path.
//!
//! This module deliberately depends only on `jolt-algebra`; it does not pull in
//! the transcript layer or the sampler.

use crate::CryptoError;

use jolt_algebra::{ExtField, Field, Ring};
use smallvec::SmallVec;

use crate::challenge::{D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT};

/// Inline capacity chosen for the heaviest family, the D64 selective-L2 shell
/// `(31, 11)`. The D64 production family `(31, 10)` and every other ring
/// dimension's family have smaller weight and also fit.
pub const INLINE_SPARSE_WEIGHT: usize = D64_L2_OP_NORM_PM1_COUNT + D64_L2_OP_NORM_PM2_COUNT;

/// Public storage type for sparse challenge positions.
pub type SparseChallengePositions = SmallVec<[u32; INLINE_SPARSE_WEIGHT]>;
/// Public storage type for sparse challenge coefficients.
pub type SparseChallengeCoefficients = SmallVec<[i8; INLINE_SPARSE_WEIGHT]>;

#[inline]
pub(crate) fn accumulate_small_signed<F, E>(acc: &mut E, value: E, coeff: i64)
where
    F: Field + Ring,
    E: Field + ExtField<F>,
{
    match coeff {
        1 => *acc += value,
        -1 => *acc -= value,
        2 => {
            *acc += value;
            *acc += value;
        }
        -2 => {
            *acc -= value;
            *acc -= value;
        }
        _ => *acc += value.mul_base(F::from_i64(coeff)),
    }
}

/// Sparse polynomial in `F[X]/(X^D+1)` represented by its non-zero terms.
///
/// Sampler invariants:
/// - `positions.len() == coeffs.len()`
/// - all positions are `< D`
/// - positions are unique
/// - all coeffs are non-zero
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseChallenge {
    /// Coefficient indices (powers of `X`) where the polynomial is non-zero.
    /// Production challenges fit inline; custom heavier families spill to the heap.
    pub positions: SparseChallengePositions,
    /// Small integer coefficients at the corresponding positions. Stored
    /// as `i8` since every shipping sampling family caps `|coeff| <= 8`.
    pub coeffs: SparseChallengeCoefficients,
}

#[cfg(feature = "sample")]
impl SparseChallenge {
    /// The dense coefficients of a D = 64 challenge, built by a masked scan
    /// in which every output coefficient reads every support entry, so no
    /// address depends on the positions. Entries at a repeated position add.
    pub fn to_dense_ct(&self) -> Result<[i8; 64], CryptoError> {
        if self.positions.len() != self.coeffs.len() {
            return Err(CryptoError::InvalidInput(
                "sparse challenge positions and coefficients differ in length".into(),
            ));
        }
        // One aggregate range check: a position at or above 64 sets a high bit.
        if self
            .positions
            .iter()
            .fold(0u32, |acc, &position| acc | position)
            >= 64
        {
            return Err(CryptoError::InvalidInput(
                "sparse challenge position outside a D = 64 ring".into(),
            ));
        }
        Ok(crate::challenge::sampler::scatter_ct(
            &self.positions,
            &self.coeffs,
        ))
    }
}

impl SparseChallenge {
    /// Validate the sampler invariants for ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if positions/coefficients have mismatched lengths, if a
    /// position is outside `0..D`, if a coefficient is zero, or if a position is
    /// repeated.
    pub fn validate<const D: usize>(&self) -> Result<(), CryptoError> {
        self.validate_dyn(D)
    }

    /// Runtime-dimension form of [`Self::validate`].
    ///
    /// # Errors
    ///
    /// Returns an error if positions/coefficients have mismatched lengths, if a
    /// position is outside `0..ring_d`, if a coefficient is zero, or if a
    /// position is repeated.
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), CryptoError> {
        if self.positions.len() != self.coeffs.len() {
            return Err(CryptoError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        const INLINE_WORDS: usize = 32;
        let word_count = ring_d.div_ceil(u64::BITS as usize);
        let mut inline_seen = [0u64; INLINE_WORDS];
        let mut heap_seen = Vec::new();
        let seen = if let Some(inline) = inline_seen.get_mut(..word_count) {
            inline
        } else {
            heap_seen.try_reserve_exact(word_count).map_err(|_| {
                CryptoError::InvalidInput("challenge dimension is too large".into())
            })?;
            heap_seen.resize(word_count, 0);
            heap_seen.as_mut_slice()
        };
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= ring_d {
                return Err(CryptoError::InvalidInput(format!(
                    "sparse challenge position {pos} out of range for D={ring_d}"
                )));
            }
            if coeff == 0 {
                return Err(CryptoError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            let word = seen.get_mut(idx / u64::BITS as usize).ok_or_else(|| {
                CryptoError::Internal(
                    "validated sparse challenge position exceeds its bitmap".into(),
                )
            })?;
            let mask = 1u64 << (idx % u64::BITS as usize);
            if *word & mask != 0 {
                return Err(CryptoError::InvalidInput(
                    "sparse challenge positions must be unique".to_string(),
                ));
            }
            *word |= mask;
        }
        Ok(())
    }

    /// Evaluate this challenge against precomputed scalar powers.
    ///
    /// The small integer coefficients are first embedded into the base field
    /// `F`, then multiplied into `E` with a mixed base-field operation. The
    /// ordinary base-field case is `E = F`.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha_pows` does not have length `D`, or if a
    /// term would index outside the supplied powers. This method assumes the
    /// challenge came from the crate's fold-challenge sampler and therefore
    /// does not re-check uniqueness of positions on the hot path.
    pub fn eval_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<E, CryptoError>
    where
        F: Field + Ring,
        E: Field + ExtField<F>,
    {
        let ring_d = alpha_pows.len();
        if self.positions.len() != self.coeffs.len() {
            return Err(CryptoError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut acc = E::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            let Some(&alpha_pow) = alpha_pows.get(idx) else {
                return Err(CryptoError::InvalidInput(format!(
                    "sparse challenge position {idx} out of range for D={ring_d}"
                )));
            };
            if coeff == 0 {
                return Err(CryptoError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            accumulate_small_signed::<F, E>(&mut acc, alpha_pow, i64::from(coeff));
        }
        Ok(acc)
    }
}

#[expect(clippy::unwrap_used, reason = "test code")]
#[cfg(test)]
mod tests {
    use super::*;
    use jolt_algebra::Prime128Offset275;

    type F = Prime128Offset275;
    const D: usize = 4;

    fn alpha_pows() -> [F; D] {
        [
            F::from_u64(1),
            F::from_u64(3),
            F::from_u64(9),
            F::from_u64(27),
        ]
    }

    #[test]
    fn eval_at_pows_evaluates_sparse_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 2].into(),
            coeffs: vec![1, -2].into(),
        };

        let got = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap();
        let expected = F::from_u64(1) + F::from_i64(-2) * F::from_u64(9);

        assert_eq!(got, expected);
    }

    #[test]
    fn eval_at_pows_rejects_position_beyond_power_count() {
        // The ring dimension is `alpha_pows.len()`; a position that fits the
        // nominal D but not the supplied power table must be rejected.
        let challenge = SparseChallenge {
            positions: vec![D as u32 - 1].into(),
            coeffs: vec![1].into(),
        };

        let err = challenge
            .eval_at_pows::<F, F>(&alpha_pows()[..D - 1])
            .unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(_)));
    }

    #[test]
    fn eval_at_pows_rejects_out_of_range_position() {
        let challenge = SparseChallenge {
            positions: vec![D as u32].into(),
            coeffs: vec![1].into(),
        };

        let err = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(msg) if msg.contains("out of range")));
    }

    #[test]
    fn eval_at_pows_rejects_mismatched_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 1].into(),
            coeffs: vec![1].into(),
        };

        let err = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(msg) if msg.contains("length mismatch")));
    }

    #[test]
    fn validate_rejects_duplicate_positions() {
        let challenge = SparseChallenge {
            positions: vec![1, 1].into(),
            coeffs: vec![1, -1].into(),
        };

        let err = challenge.validate::<D>().unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(msg) if msg.contains("unique")));
    }

    #[test]
    fn validate_rejects_zero_coefficient() {
        let challenge = SparseChallenge {
            positions: vec![0, 1].into(),
            coeffs: vec![1, 0].into(),
        };

        let err = challenge.validate::<D>().unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(msg) if msg.contains("non-zero")));
    }

    #[test]
    fn validate_rejects_out_of_range_position() {
        let challenge = SparseChallenge {
            positions: vec![0, D as u32].into(),
            coeffs: vec![1, 1].into(),
        };

        let err = challenge.validate::<D>().unwrap_err();

        assert!(matches!(err, CryptoError::InvalidInput(msg) if msg.contains("out of range")));
    }
}
