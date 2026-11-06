//! The Ajtai map `y = M · x` over `Z_q[X]/(X^D + 1)`, and its commitment
//! form `M · (G⁻¹(x) ‖ r)`.
//!
//! The matrix is public. It is kept over `F`, and transformed once into
//! CRT+NTT form on its first constant-time product. Each input
//! column is lifted into Montgomery residues by one branch-free product per
//! residue, transformed, and accumulated pointwise into every output row; each
//! row is reconstructed exactly over `Z` and reduced into `F` with a
//! constant-time Garner. No step reads an input-dependent table entry or
//! branches on an input value, so secret inputs (Whiteout's Gaussian mask and
//! its commitment digits) take the same path as public ones. `q ≡ 9 (mod 16)`
//! admits no NTT over `F_q` for `X^256 + 1`, which is why the arithmetic runs
//! over CRT primes.

use std::ops::Range;
use std::sync::OnceLock;

use jolt_algebra::parallel::*;
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::tables::{q128_primes, validate_profile_crt_ring_degree, Q128_MAX_RING_D};
use jolt_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, Accumulator, CanonicalEncoding, CrtNttParamSet,
    CyclotomicCrtNtt, CyclotomicRing, Field, WithAccumulator, RADIX_SUM_BITS,
};
use zeroize::Zeroizing;

use super::matrix::{derive_public_matrix_prefix, PublicMatrixSeed};
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
/// Seed of [`PreparedMatrix`]'s blind, a public matrix stream of one ring.
const BLIND_SEED: [u8; 32] = *b"jolt-crypto ajtai crt blind v1\0\0";

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

/// A public matrix, ready to apply.
///
/// It holds its entries over `F`, copied from the stream, which is all
/// [`Self::combine_columns`] reads: a verifier never transforms. The
/// CRT+NTT form the constant-time products read ([`Self::apply_centered`],
/// [`Self::apply_transposed`]) is built from them on the first such product
/// and kept.
pub struct PreparedMatrix<F: Field, const D: usize> {
    shape: MatrixShape,
    /// Row-major entries over `F`.
    coefficients: Vec<CyclotomicRing<F, D>>,
    /// The same entries in CRT+NTT form, row-major, and [`Self::blind`]'s
    /// image.
    ntt: OnceLock<NttForm<D>>,
    params: CrtNttParamSet<i32, CRT_PRIMES, D>,
    /// The public blind `K` every constant-time accumulator starts from
    /// ([`Self::apply_centered`]).
    blind: CyclotomicRing<F, D>,
}

/// [`PreparedMatrix`]'s CRT+NTT form.
struct NttForm<const D: usize> {
    entries: Vec<CyclotomicCrtNtt<i32, CRT_PRIMES, D>>,
    blind: CyclotomicCrtNtt<i32, CRT_PRIMES, D>,
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
        let coefficients = cfg_into_iter!(0..shape.rows * shape.width)
            .map(|index| {
                Ok(CyclotomicRing::<F, D>::from_slice(
                    view.entry(index / shape.width, index % shape.width)?,
                ))
            })
            .collect::<Result<Vec<_>, CryptoError>>()?;
        let blind = CyclotomicRing::from_slice(&derive_public_matrix_prefix::<F>(
            D,
            &PublicMatrixSeed::shake256_paged_v1(BLIND_SEED),
        )?);
        Ok(Self {
            shape,
            coefficients,
            ntt: OnceLock::new(),
            params: CrtNttParamSet::new(q128_primes()),
            blind,
        })
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// The CRT+NTT entries and blind, built on the first CRT product and
    /// kept. The transform reads only public values.
    ///
    /// WARNING: the build runs in parallel outside the cell's lock. Inside
    /// `get_or_init` it would deadlock: callers run on Rayon workers, and a
    /// worker that steals another caller's job while it waits on the build
    /// blocks on the cell it is initializing. Concurrent first callers may
    /// each build the form (one is kept).
    fn ntt_form(&self) -> &NttForm<D> {
        if let Some(form) = self.ntt.get() {
            return form;
        }
        let form = NttForm {
            entries: cfg_iter!(self.coefficients)
                .map(|entry| CyclotomicCrtNtt::from_ring(entry, &self.params))
                .collect(),
            blind: CyclotomicCrtNtt::from_ring(&self.blind, &self.params),
        };
        self.ntt.get_or_init(|| form)
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
            // One column's capacity holds the blind: `|K| ≤ q/2 ≤ D·(q/2)·b`
            // for any `b ≥ 1`.
            .max_safe_width::<F, D>(u64::from(bound.max(1)))
            .and_then(|width| width.checked_sub(1))
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                CryptoError::InvalidSetup(format!(
                    "the CRT primes cannot hold one column and the blind at bound {bound}"
                ))
            })?
            .min(COLUMN_CHUNK);
        let form = self.ntt_form();
        let partials = cfg_into_iter!(0..self.shape.width.div_ceil(chunk))
            .map(|index| self.apply_columns(form, input, index * chunk, chunk))
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
    /// Each weight coefficient is split into unsigned
    /// [`RADIX_SUM_BITS`]-bit limbs, and every limb plane runs the CRT
    /// product exactly. A column's planes leave CRT+NTT form together, with
    /// one field image per coefficient
    /// ([`CyclotomicCrtNtt::radix_sum_to_ring`]).
    ///
    /// Not constant time: the weights and the matrix must be public.
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
        // Four times the limb bound: the radix sum's rounding margin.
        if !self
            .params
            .crt_capacity()
            .supports::<F, D>(self.shape.rows, 1 << (RADIX_SUM_BITS + 2))
        {
            return Err(CryptoError::InvalidSetup(
                "the CRT primes cannot hold one transposed column".into(),
            ));
        }
        let (rows, width) = (self.shape.rows, self.shape.width);
        if rows == 0 {
            return Ok(vec![CyclotomicRing::zero(); width]);
        }
        let limbs = field_bits::<F>()?.div_ceil(RADIX_SUM_BITS) as usize;
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
        // Limb-major: plane `limb` of weight `row` at `limb * rows + row`.
        let planes: Vec<CyclotomicCrtNtt<i32, CRT_PRIMES, D>> = (0..limbs)
            .flat_map(|limb| {
                let shift = RADIX_SUM_BITS * limb as u32;
                canonical.iter().map(move |values| {
                    let plane: [i128; D] = values.map(|value| i128::from((value >> shift) as u32));
                    CyclotomicCrtNtt::from_centered_coefficients(&plane, &self.params)
                })
            })
            .collect();
        let entries = &self.ntt_form().entries;
        cfg_into_iter!(0..width)
            .map(|col| {
                let sums = planes
                    .chunks_exact(rows)
                    .map(|limb_planes| {
                        let mut accumulator = CyclotomicCrtNtt::zero();
                        for (row, plane) in limb_planes.iter().enumerate() {
                            let entry = entries.get(row * width + col).ok_or_else(|| {
                                CryptoError::Internal("matrix entry missing".into())
                            })?;
                            accumulator.add_assign_pointwise_mul(entry, plane, &self.params);
                        }
                        Ok(accumulator)
                    })
                    .collect::<Result<Vec<_>, CryptoError>>()?;
                Ok(CyclotomicCrtNtt::radix_sum_to_ring(&sums, &self.params)?)
            })
            .collect()
    }

    /// The exact contribution of columns `start..start + len` (clamped to the
    /// width) to every output row, over the CRT+NTT `form`.
    ///
    /// Each accumulator starts from the blind `K`'s image and `K` is
    /// subtracted in `F`, so the reconstruction never sees the degenerate
    /// all-zero product of an all-zero input: on Apple M4, `to_ring_ct` on a
    /// zero accumulator just computed takes measurably different time, even
    /// under DIT (`specs/whiteout-sigma.md` §2.8). `K` is a public
    /// pseudorandom ring element, so an input whose product is `−K` solves
    /// an inhomogeneous SIS instance.
    fn apply_columns(
        &self,
        form: &NttForm<D>,
        input: &[i64],
        start: usize,
        len: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        let width = self.shape.width;
        let end = (start + len).min(width);
        let mut accumulators = vec![form.blind.clone(); self.shape.rows];
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
                .zip(form.entries.chunks_exact(width))
            {
                let entry = row
                    .get(start + offset)
                    .ok_or_else(|| CryptoError::Internal("matrix row is too short".into()))?;
                accumulator.add_assign_pointwise_mul(entry, &column, &self.params);
            }
        }
        Ok(accumulators
            .iter()
            .map(|accumulator| accumulator.to_ring_ct(&self.params) - self.blind)
            .collect())
    }
}

impl<F: Field + CanonicalEncoding + WithAccumulator, const D: usize> PreparedMatrix<F, D> {
    /// `Σ_col scales[col] · M[row][col]` for every row, over `F`: the
    /// matrix applied to columns that are each a public scalar times one
    /// common ring element, before that element. Not constant time: the
    /// scales must be public.
    pub fn combine_columns(&self, scales: &[F]) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        self.combine_column_range(0..self.shape.width, scales)
    }

    /// [`Self::combine_columns`] over the columns in `columns` alone:
    /// `Σ_{col ∈ columns} scales[col − columns.start] · M[row][col]`. Each
    /// output coefficient sums its products unreduced and reduces once.
    pub fn combine_column_range(
        &self,
        columns: Range<usize>,
        scales: &[F],
    ) -> Result<Vec<CyclotomicRing<F, D>>, CryptoError> {
        if columns.start > columns.end
            || columns.end > self.shape.width
            || scales.len() != columns.len()
        {
            return Err(CryptoError::InvalidSize {
                expected: columns.len(),
                actual: scales.len(),
            });
        }
        let coefficients = &self.coefficients;
        // Parallel over column chunks as well as rows: matrices are a few
        // rows by thousands of columns.
        const COLUMN_CHUNK: usize = 256;
        let width = self.shape.width;
        let len = columns.len();
        Ok((0..self.shape.rows)
            .map(|row| {
                let row_entries = coefficients
                    .get(row * width + columns.start..row * width + columns.end)
                    .unwrap_or_default();
                let sums = cfg_fold_reduce!(
                    0..len.div_ceil(COLUMN_CHUNK),
                    || [F::Accumulator::default(); D],
                    |mut sums, chunk| {
                        let range = chunk * COLUMN_CHUNK..((chunk + 1) * COLUMN_CHUNK).min(len);
                        for (entry, scale) in row_entries
                            .get(range.clone())
                            .unwrap_or_default()
                            .iter()
                            .zip(scales.get(range).unwrap_or_default())
                        {
                            // Scales are public; callers pass sparse ones
                            // (one nonzero quarter per V plane).
                            if scale.is_zero() {
                                continue;
                            }
                            for (sum, coefficient) in sums.iter_mut().zip(entry.coefficients()) {
                                sum.fmadd(*coefficient, *scale);
                            }
                        }
                        sums
                    },
                    |mut sums, other| {
                        for (sum, other) in sums.iter_mut().zip(other) {
                            sum.merge(other);
                        }
                        sums
                    }
                );
                CyclotomicRing::from_coefficients(sums.map(Accumulator::reduce))
            })
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
