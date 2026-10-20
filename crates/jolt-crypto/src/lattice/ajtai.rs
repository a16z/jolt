//! The Ajtai map `y = M · x` over `Z_q[X]/(X^D + 1)`, and its commitment
//! form `M · (G⁻¹(x) ‖ r)`.
//!
//! The matrix is public and is prepared once in CRT+NTT form. Each input
//! column is lifted into Montgomery residues by one branch-free product per
//! residue, transformed, and accumulated pointwise into every output row; each
//! row is reconstructed exactly over `Z` and reduced into `F` with a
//! constant-time Garner. No step reads an input-dependent table entry or
//! branches on an input value, so secret inputs (Whiteout's Gaussian mask and
//! its commitment digits) take the same path as public ones. `q ≡ 9 (mod 16)`
//! admits no NTT over `F_q` for `X^256 + 1`, which is why the arithmetic runs
//! over CRT primes.

use std::marker::PhantomData;

use jolt_field::parallel::*;
use jolt_field::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_field::tables::{q128_primes, validate_profile_crt_ring_degree, Q128_MAX_RING_D};
use jolt_field::{
    balanced_decompose_coefficients_pow2_i8_into, CanonicalEncoding, CrtNttParamSet,
    CyclotomicCrtNtt, CyclotomicRing, Field,
};

use super::view::{MatrixShape, MatrixView};
use crate::CryptoError;

/// Number of 32-bit CRT primes. Their product exceeds `2^179`, which holds
/// `2 · width · D · ⌊q/2⌋ · bound` for every shape used here.
const CRT_PRIMES: usize = 6;
/// Columns accumulated before one exact reconstruction, unless the CRT
/// capacity allows fewer. Also the unit of parallel work.
const COLUMN_CHUNK: usize = 64;

/// Balanced gadget digits: base `2^log_basis`, `levels` digits, digit 0 least
/// significant, decomposed by jolt-algebra's branch-free balanced
/// decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GadgetDigits {
    pub log_basis: u32,
    pub levels: usize,
}

impl GadgetDigits {
    /// Checks the parameters the `i8` decomposition accepts: base `2..=2^8`,
    /// at most one digit past 128 bits, and at most the field width in binary.
    pub fn validate<F: Field + CanonicalEncoding>(&self) -> Result<(), CryptoError> {
        let field_bits = field_bits::<F>()?;
        let levels = u32::try_from(self.levels)
            .map_err(|_| CryptoError::InvalidSetup("gadget level count overflows u32".into()))?;
        let valid = (1..=8).contains(&self.log_basis)
            && levels >= 1
            && levels.saturating_mul(self.log_basis) <= 128 + self.log_basis
            && (self.log_basis > 1 || levels <= field_bits);
        if valid {
            Ok(())
        } else {
            Err(CryptoError::InvalidSetup(format!(
                "gadget digits base 2^{} × {} are not i8 digits of a {field_bits}-bit field",
                self.log_basis, self.levels
            )))
        }
    }

    /// Whether the digits represent every element of `F`, as decomposing an
    /// arbitrary field element requires.
    pub fn covers_field<F: Field + CanonicalEncoding>(&self) -> Result<bool, CryptoError> {
        let levels = u32::try_from(self.levels)
            .map_err(|_| CryptoError::InvalidSetup("gadget level count overflows u32".into()))?;
        Ok(levels.saturating_mul(self.log_basis) >= field_bits::<F>()?)
    }

    fn params<F: Field + CanonicalEncoding>(&self) -> BalancedDecomposePow2Params<F> {
        BalancedDecomposePow2Params::new(self.levels, self.log_basis)
    }
}

/// One Ajtai level: `y = M · (G⁻¹(x) ‖ r)`.
///
/// The input columns are ring-major and digit-minor: column
/// `ring · levels + digit` is digit plane `digit` of message ring `ring`, as
/// in Akita's inner commitment. The `randomness_columns` hiding columns come
/// last. With none, the level is binding only and its bytes equal Akita's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AjtaiLevel {
    pub matrix: MatrixShape,
    pub digits: GadgetDigits,
    pub message_rings: usize,
    pub randomness_columns: usize,
}

impl AjtaiLevel {
    /// Checks that the matrix width is exactly the decomposed message plus
    /// the randomness columns, and that the digits cover `F`.
    pub fn validate<F: Field + CanonicalEncoding>(&self) -> Result<(), CryptoError> {
        self.digits.validate::<F>()?;
        let width = self
            .message_rings
            .checked_mul(self.digits.levels)
            .and_then(|columns| columns.checked_add(self.randomness_columns))
            .ok_or_else(|| CryptoError::InvalidSetup("Ajtai level width overflows".into()))?;
        if width == self.matrix.width {
            Ok(())
        } else {
            Err(CryptoError::InvalidSetup(format!(
                "Ajtai level needs width {width}, matrix has {}",
                self.matrix.width
            )))
        }
    }
}

/// A public matrix in CRT+NTT form, ready to apply.
pub struct PreparedMatrix<F, const D: usize> {
    shape: MatrixShape,
    /// Row-major entries.
    entries: Vec<CyclotomicCrtNtt<i32, CRT_PRIMES, D>>,
    params: CrtNttParamSet<i32, CRT_PRIMES, D>,
    field: PhantomData<F>,
}

impl<F: Field + CanonicalEncoding, const D: usize> PreparedMatrix<F, D> {
    /// Prepares `view`. Requires `view`'s ring dimension to be `D`, a power of
    /// two up to 1024.
    pub fn new(view: &MatrixView<'_, F>) -> Result<Self, CryptoError> {
        let shape = view.shape();
        if shape.ring_dim != D {
            return Err(CryptoError::InvalidSetup(format!(
                "matrix ring dimension {} differs from D = {D}",
                shape.ring_dim
            )));
        }
        validate_profile_crt_ring_degree(D, Q128_MAX_RING_D)?;
        // The CRT residues read each coefficient as a u128.
        field_bits::<F>().map(|_| ())?;
        let params = CrtNttParamSet::new(q128_primes());
        let entries = cfg_into_iter!(0..shape.rows * shape.width)
            .map(|index| {
                let coefficients = view.entry(index / shape.width, index % shape.width)?;
                Ok(CyclotomicCrtNtt::from_ring(
                    &CyclotomicRing::from_slice(coefficients),
                    &params,
                ))
            })
            .collect::<Result<Vec<_>, CryptoError>>()?;
        Ok(Self {
            shape,
            entries,
            params,
            field: PhantomData,
        })
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// `M · x` for centered integer columns `x`, laid out column by column
    /// with `D` coefficients each, every one at most `bound` in magnitude.
    ///
    /// Constant time in `x`: the bound is checked over all coefficients as
    /// one aggregate, which is the only value branched on, and every
    /// coefficient takes the same lift, transform, and accumulation.
    ///
    /// WARNING: `bound` must be a hard bound on every input the caller can
    /// produce, such as the Gaussian sampler's tail cut, never a statistical
    /// one. An input above `bound` returns an error, and for a secret input
    /// that error is an abort correlated with the secret.
    pub fn apply_centered(
        &self,
        input: &[i64],
        bound: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        let expected = self.shape.width.checked_mul(D).ok_or_else(|| {
            CryptoError::InvalidSetup("matrix input length overflows usize".into())
        })?;
        if input.len() != expected {
            return Err(CryptoError::InvalidSize {
                expected,
                actual: input.len(),
            });
        }
        if bound > i32::MAX as u32 {
            return Err(CryptoError::InvalidInput(format!(
                "input bound {bound} exceeds the i32 lift"
            )));
        }
        if !all_within(input, bound) {
            return Err(CryptoError::InvalidInput(format!(
                "an input coefficient exceeds its bound {bound}"
            )));
        }
        let chunk = self
            .params
            .crt_capacity()
            .max_safe_width::<F, D>(u64::from(bound))
            .ok_or_else(|| {
                CryptoError::InvalidSetup(format!(
                    "the CRT primes cannot hold one column at bound {bound}"
                ))
            })?
            .min(COLUMN_CHUNK);
        let partials = cfg_into_iter!(0..self.shape.width.div_ceil(chunk))
            .map(|index| self.apply_columns(input, index * chunk, chunk))
            .collect::<Result<Vec<_>, CryptoError>>()?;
        let mut output = vec![CyclotomicRing::zero(); self.shape.rows];
        for partial in partials {
            for (out, value) in output.iter_mut().zip(partial) {
                *out += value;
            }
        }
        Ok(output)
    }

    /// The exact contribution of columns `start..start + len` (clamped to the
    /// width) to every output row.
    fn apply_columns(
        &self,
        input: &[i64],
        start: usize,
        len: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        let width = self.shape.width;
        let end = (start + len).min(width);
        let mut accumulators = vec![CyclotomicCrtNtt::zero(); self.shape.rows];
        let mut column = CyclotomicCrtNtt::zero();
        let mut lifted = [0i32; D];
        let columns = input
            .get(start * D..end * D)
            .ok_or_else(|| CryptoError::Internal("column chunk outside the input".into()))?;
        for (offset, coefficients) in columns.chunks_exact(D).enumerate() {
            for (dst, &value) in lifted.iter_mut().zip(coefficients) {
                // In range: `apply_centered` checked `|value| <= bound <= i32::MAX`.
                *dst = value as i32;
            }
            column.assign_centered_i32_ct(&lifted, &self.params);
            for (accumulator, row) in accumulators
                .iter_mut()
                .zip(self.entries.chunks_exact(width))
            {
                let entry = row
                    .get(start + offset)
                    .ok_or_else(|| CryptoError::Internal("matrix row is too short".into()))?;
                accumulator.add_assign_pointwise_mul(entry, &column, &self.params);
            }
        }
        Ok(accumulators
            .iter()
            .map(|accumulator| accumulator.to_ring_ct(&self.params))
            .collect())
    }
}

/// The commitment `M · (G⁻¹(message) ‖ randomness)` of one Ajtai level.
///
/// `randomness` holds the `randomness_columns · D` hiding digits, column by
/// column; every digit is an `i8`. The decomposition and the map are both
/// constant time in the message and the randomness.
pub fn commit<F: Field + CanonicalEncoding, const D: usize>(
    level: &AjtaiLevel,
    matrix: &PreparedMatrix<F, D>,
    message: &[CyclotomicRing<F, D>],
    randomness: &[i8],
) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
    level.validate::<F>()?;
    if !level.digits.covers_field::<F>()? {
        return Err(CryptoError::InvalidSetup(
            "commitment digits do not cover the field".into(),
        ));
    }
    if matrix.shape() != level.matrix {
        return Err(CryptoError::InvalidSetup(
            "prepared matrix shape differs from the level's".into(),
        ));
    }
    if message.len() != level.message_rings {
        return Err(CryptoError::InvalidSize {
            expected: level.message_rings,
            actual: message.len(),
        });
    }
    let randomness_len = level.randomness_columns * D;
    if randomness.len() != randomness_len {
        return Err(CryptoError::InvalidSize {
            expected: randomness_len,
            actual: randomness.len(),
        });
    }
    let params = level.digits.params::<F>();
    let mut input = Vec::with_capacity(level.matrix.width * D);
    let mut ring_digits = vec![0i8; level.digits.levels * D];
    for ring in message {
        balanced_decompose_coefficients_pow2_i8_into(
            ring.coefficients(),
            &mut ring_digits,
            &params,
        );
        input.extend(ring_digits.iter().map(|&digit| i64::from(digit)));
    }
    input.extend(randomness.iter().map(|&digit| i64::from(digit)));
    matrix.apply_centered(&input, 1 << 7)
}

/// Whether every `|x| <= bound`, evaluated over all of `input` without an
/// early exit or a branch on any one value.
fn all_within(input: &[i64], bound: u32) -> bool {
    let bound = i128::from(bound);
    let outside = input.iter().fold(0i128, |acc, &value| {
        let value = i128::from(value);
        acc | (bound - value) | (value + bound)
    });
    outside >= 0
}

/// `⌈log₂ q⌉`, the bit width of `F`'s modulus.
pub(super) fn field_bits<F: Field + CanonicalEncoding>() -> Result<u32, CryptoError> {
    let top = (-F::one())
        .to_u128_checked()
        .ok_or_else(|| CryptoError::InvalidSetup("field modulus exceeds 128 bits".into()))?;
    Ok(u128::BITS - top.leading_zeros())
}
