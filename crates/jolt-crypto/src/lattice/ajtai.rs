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
use std::sync::OnceLock;

use jolt_algebra::parallel::*;
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::tables::{q128_primes, validate_profile_crt_ring_degree, Q128_MAX_RING_D};
use jolt_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, CanonicalEncoding, CrtNttParamSet,
    CyclotomicCrtNtt, CyclotomicRing, Field,
};
use zeroize::Zeroizing;

use super::view::{MatrixShape, MatrixView};
use crate::CryptoError;

/// Number of 32-bit CRT primes. Their product exceeds `2^179`, which holds
/// `2 · width · D · ⌊q/2⌋ · bound` for every shape used here.
const CRT_PRIMES: usize = 6;
/// The largest input bound of [`PreparedMatrix::apply_centered`].
pub const MAX_CENTERED_BOUND: u64 = 1 << 62;
/// Bits per limb when an input exceeds the `i32` lift.
const LIMB_BITS: u32 = 30;
/// Columns accumulated before one exact reconstruction, unless the CRT
/// capacity allows fewer. Also the unit of parallel work.
const COLUMN_CHUNK: usize = 64;
const LIMB_MASK: u64 = (1 << LIMB_BITS) - 1;

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
pub struct PreparedMatrix<F: Field, const D: usize> {
    shape: MatrixShape,
    /// Row-major entries.
    entries: Vec<CyclotomicCrtNtt<i32, CRT_PRIMES, D>>,
    /// The same entries over `F`, for products with public field scalars;
    /// built from `entries` on the first [`Self::combine_columns`], which
    /// only a verifier calls, so a prover holds no second copy.
    coefficients: OnceLock<Vec<CyclotomicRing<F, D>>>,
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
                Ok(CyclotomicCrtNtt::from_ring(
                    &CyclotomicRing::<F, D>::from_slice(
                        view.entry(index / shape.width, index % shape.width)?,
                    ),
                    &params,
                ))
            })
            .collect::<Result<Vec<_>, CryptoError>>()?;
        Ok(Self {
            shape,
            entries,
            coefficients: OnceLock::new(),
            params,
            field: PhantomData,
        })
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// `Σ_col scales[col] · M[row][col]` for every row, over `F`: the
    /// matrix applied to columns that are each a public scalar times one
    /// common ring element, before that element. Not constant time: the
    /// scales must be public.
    pub fn combine_columns(&self, scales: &[F]) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        if scales.len() != self.shape.width {
            return Err(CryptoError::InvalidSize {
                expected: self.shape.width,
                actual: scales.len(),
            });
        }
        let coefficients = self.field_entries();
        // Parallel over column chunks as well as rows: matrices are a few
        // rows by thousands of columns.
        const COLUMN_CHUNK: usize = 256;
        let width = self.shape.width;
        Ok((0..self.shape.rows)
            .map(|row| {
                let row_entries = coefficients
                    .get(row * width..(row + 1) * width)
                    .unwrap_or_default();
                cfg_fold_reduce!(
                    0..width.div_ceil(COLUMN_CHUNK),
                    CyclotomicRing::zero,
                    |mut sum, chunk| {
                        let range = chunk * COLUMN_CHUNK..((chunk + 1) * COLUMN_CHUNK).min(width);
                        for (entry, scale) in row_entries
                            .get(range.clone())
                            .unwrap_or_default()
                            .iter()
                            .zip(scales.get(range).unwrap_or_default())
                        {
                            entry.scale_accumulate_into(&mut sum, *scale);
                        }
                        sum
                    },
                    |a, b| a + b
                )
            })
            .collect())
    }

    /// The entries over `F`, built on first use: once per matrix, so a
    /// verifier that keeps its setup pays for it once, not per proof.
    fn field_entries(&self) -> &[CyclotomicRing<F, D>] {
        self.coefficients.get_or_init(|| {
            cfg_iter!(self.entries)
                .map(|entry| entry.to_ring(&self.params))
                .collect()
        })
    }

    /// `M · x` for centered integer columns `x`, laid out column by column
    /// with `D` coefficients each, every one at most `bound ≤`
    /// [`MAX_CENTERED_BOUND`] in magnitude.
    ///
    /// Constant time in `x`: the bound is checked over all coefficients as
    /// one aggregate, which is the only value branched on, and every
    /// coefficient takes the same lift, transform, and accumulation. A bound
    /// above `i32::MAX` splits every input into balanced 30-bit limbs by
    /// shifts, applies the map to each limb, and recombines in `F`; the limb
    /// count depends on `bound` alone.
    ///
    /// WARNING: `bound` must be a hard bound on every input the caller can
    /// produce, such as the Gaussian sampler's tail cut, never a statistical
    /// one. An input above `bound` returns an error, and for a secret input
    /// that error is an abort correlated with the secret.
    pub fn apply_centered(
        &self,
        input: &[i64],
        bound: u64,
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
        if bound > MAX_CENTERED_BOUND {
            return Err(CryptoError::InvalidInput(format!(
                "input bound {bound} exceeds 2^62"
            )));
        }
        if !all_within(input, bound) {
            return Err(CryptoError::InvalidInput(format!(
                "an input coefficient exceeds its bound {bound}"
            )));
        }
        if let Some(bound) = u32::try_from(bound).ok().filter(|&b| b <= i32::MAX as u32) {
            return self.apply_i32(input, bound);
        }
        // Balanced base-2^30 digits: with `h_j = 2^29·(2^{30j} − 1)/(2^30 − 1)`,
        // `rest_j = ⌊(x + h_j)/2^{30j}⌋`, limb `j` is `rest_j`'s low 30 bits
        // sign-extended, and the top limb is `rest_j` itself. Each pass
        // recomputes its limb from `input` into one scratch buffer.
        // The limbs are as secret as the input (Whiteout's mask).
        let mut limb = Zeroizing::new(vec![0i64; input.len()].into_boxed_slice());
        let mut output = vec![CyclotomicRing::zero(); self.shape.rows];
        let mut weight = F::one();
        let radix = F::from_u64(1 << LIMB_BITS);
        let (mut offset, mut shift) = (0i128, 0u32);
        let mut remaining = bound;
        while remaining > i32::MAX as u64 {
            for (dst, &value) in limb.iter_mut().zip(input) {
                let rest = ((i128::from(value) + offset) >> shift) as i64;
                *dst = (rest << (64 - LIMB_BITS)) >> (64 - LIMB_BITS);
            }
            self.accumulate(&limb, 1 << (LIMB_BITS - 1), weight, &mut output)?;
            weight *= radix;
            offset += 1 << (LIMB_BITS - 1 + shift);
            shift += LIMB_BITS;
            remaining = (remaining >> LIMB_BITS) + 1;
        }
        for (dst, &value) in limb.iter_mut().zip(input) {
            *dst = ((i128::from(value) + offset) >> shift) as i64;
        }
        self.accumulate(&limb, remaining as u32, weight, &mut output)?;
        Ok(output)
    }

    /// `output += weight · M·x` for inputs within the `i32` lift.
    fn accumulate(
        &self,
        input: &[i64],
        bound: u32,
        weight: F,
        output: &mut [CyclotomicRing<F, D>],
    ) -> Result<(), CryptoError> {
        for (out, value) in output.iter_mut().zip(self.apply_i32(input, bound)?) {
            value.scale_accumulate_into(out, weight);
        }
        Ok(())
    }

    /// `M · x` for inputs already known to lie within `bound ≤ i32::MAX`.
    fn apply_i32(
        &self,
        input: &[i64],
        bound: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
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

    /// `Mᵀ · w` over field weights: entry `col` is `Σ_row w_row · M[row][col]`.
    ///
    /// Each weight coefficient is split into unsigned `LIMB_BITS`-bit limbs,
    /// and every limb plane runs the CRT product exactly. Not constant time: the
    /// weights must be public.
    pub fn apply_transposed(
        &self,
        weights: &[CyclotomicRing<F, D>],
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        if weights.len() != self.shape.rows {
            return Err(CryptoError::InvalidSize {
                expected: self.shape.rows,
                actual: weights.len(),
            });
        }
        if !self
            .params
            .crt_capacity()
            .supports::<F, D>(self.shape.rows, LIMB_MASK)
        {
            return Err(CryptoError::InvalidSetup(
                "the CRT primes cannot hold one transposed column".into(),
            ));
        }
        let canonical: Vec<[u128; D]> = weights
            .iter()
            .map(|ring| {
                let mut values = [0u128; D];
                for (dst, coefficient) in values.iter_mut().zip(ring.coefficients()) {
                    *dst = coefficient.to_u128_checked().ok_or_else(|| {
                        CryptoError::InvalidInput("weight exceeds 128 bits".into())
                    })?;
                }
                Ok(values)
            })
            .collect::<Result<_, CryptoError>>()?;
        let limbs = field_bits::<F>()?.div_ceil(LIMB_BITS) as usize;
        let mut output = vec![CyclotomicRing::<F, D>::zero(); self.shape.width];
        let mut scale = F::one();
        let radix = F::from_u64(1 << LIMB_BITS);
        for limb in 0..limbs {
            let shift = LIMB_BITS * limb as u32;
            let rows: Vec<CyclotomicCrtNtt<i32, CRT_PRIMES, D>> = canonical
                .iter()
                .map(|values| {
                    // In range: the mask keeps `LIMB_BITS < 32` bits.
                    let plane: [i32; D] =
                        values.map(|value| ((value >> shift) as u64 & LIMB_MASK) as i32);
                    let mut ntt = CyclotomicCrtNtt::zero();
                    ntt.assign_centered_i32_ct(&plane, &self.params);
                    ntt
                })
                .collect();
            let columns: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..self.shape.width)
                .map(|col| {
                    let mut accumulator = CyclotomicCrtNtt::zero();
                    for (row, weight) in rows.iter().enumerate() {
                        let entry = self
                            .entries
                            .get(row * self.shape.width + col)
                            .ok_or_else(|| CryptoError::Internal("matrix entry missing".into()))?;
                        accumulator.add_assign_pointwise_mul(entry, weight, &self.params);
                    }
                    Ok(accumulator.to_ring_ct(&self.params))
                })
                .collect::<Result<_, CryptoError>>()?;
            for (out, column) in output.iter_mut().zip(columns) {
                column.scale_accumulate_into(out, scale);
            }
            scale *= radix;
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
                // In range: every caller bounds `|value| <= bound <= i32::MAX`.
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
    // `G⁻¹(message)` is as secret as the message (Whiteout's `V`).
    let mut input = Zeroizing::new(Vec::with_capacity(level.matrix.width * D));
    let mut ring_digits = Zeroizing::new(vec![0i8; level.digits.levels * D]);
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
fn all_within(input: &[i64], bound: u64) -> bool {
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
