//! Limb-major prepared matrices over canonical 50-bit IFMA residues.

use crate::ntt::crt::{modular_inverse, GarnerData};
use crate::ntt::ifma52::{
    forward, forward_i16, ifma52_enabled, inverse, Ifma52Accumulator, Ifma52Prime, Ifma52Residues,
    Ifma52Twiddles, IFMA52_ACCUMULATOR_TERMS,
};
use crate::{
    cfg_into_iter, CanonicalEncoding, CrtCapacity, CrtNttParamSet, CyclotomicCrtNtt,
    CyclotomicRing, Field, NttPrime, PrimeWidth,
};
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Matrix entries transformed per parallel preparation task.
const PREPARE_TILE_ENTRIES: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Ifma52Tail<const K: usize> {
    modulus: i64,
    residue_weight: i64,
    digit_weights: [i64; K],
    /// Byte width of the tail residues.
    width: usize,
}

/// Parameters for one fixed-size IFMA52 CRT profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ifma52Params<const K: usize, const D: usize> {
    primes: [Ifma52Prime; K],
    /// Boxed for the same reason as `CrtNttParamSet::twiddles`.
    twiddles: Box<[Ifma52Twiddles<D>; K]>,
    garner: GarnerData<K>,
    use_ifma: bool,
    tail: Option<Ifma52Tail<K>>,
}

impl<const K: usize, const D: usize> Ifma52Params<K, D> {
    /// Validate prime moduli and compute transform and Garner tables.
    pub fn new(moduli: [u64; K]) -> Result<Self, AkitaError> {
        if K == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT profile must contain a prime".into(),
            ));
        }
        let primes: [Ifma52Prime; K] = moduli
            .map(Ifma52Prime::new)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("IFMA52 prime count mismatch".into()))?;
        let twiddles: Box<[Ifma52Twiddles<D>; K]> = primes
            .iter()
            .copied()
            .map(Ifma52Twiddles::compute)
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice()
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("IFMA52 twiddle count mismatch".into()))?;
        let garner = GarnerData::try_from_moduli(moduli)?;
        Ok(Self {
            primes,
            twiddles,
            garner,
            use_ifma: ifma52_enabled(),
            tail: None,
        })
    }

    /// Extend this profile with the exactness-only prime `tail`, whose
    /// transforms run over width `W`.
    pub fn with_tail<W: PrimeWidth>(mut self, tail: NttPrime<W>) -> Result<Self, AkitaError> {
        let modulus = tail.p.to_i64();
        if modulus <= 2 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 tail modulus must exceed 2".into(),
            ));
        }
        let mut residue_weight = 1i64;
        let mut digit_weights = [0; K];
        for index in (0..K).rev() {
            let inverse =
                modular_inverse(self.primes[index].modulus % modulus as u64, modulus as u64)?
                    as i64;
            residue_weight = (residue_weight * inverse) % modulus;
            digit_weights[index] = (-residue_weight).rem_euclid(modulus);
        }
        self.tail = Some(Ifma52Tail {
            modulus,
            residue_weight,
            digit_weights,
            width: core::mem::size_of::<W>(),
        });
        Ok(self)
    }

    /// Exact CRT capacity of this profile.
    pub fn crt_capacity(&self) -> CrtCapacity {
        let capacity =
            CrtCapacity::from_prime_moduli(self.primes.iter().map(|prime| prime.modulus as u128));
        self.tail.as_ref().map_or(capacity.clone(), |tail| {
            capacity.with_prime_modulus(tail.modulus as u128)
        })
    }

    /// Whether this profile's exactness-only tail is `prime`.
    #[must_use]
    pub fn has_tail<W: PrimeWidth>(&self, prime: NttPrime<W>) -> bool {
        matches!(
            &self.tail,
            Some(tail) if tail.width == core::mem::size_of::<W>() && tail.modulus == prime.p.to_i64()
        )
    }

    #[inline]
    fn reconstruct<F: Field + CanonicalEncoding, W: PrimeWidth>(
        &self,
        canonical: &[Ifma52Residues<D>; K],
        tail_canonical: Option<&[W; D]>,
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        if K == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT profile must contain a prime".into(),
            ));
        }
        if self.tail.as_ref().map(|tail| tail.width)
            != tail_canonical.map(|_| core::mem::size_of::<W>())
        {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 reconstruction tail does not match its parameters".into(),
            ));
        }
        let (field_weights, tail_field_weight) = self.garner.field_weights::<F>();
        // IFMA moduli are below 2^52, so canonical residues fit in i64.
        let mut mixed_radix = canonical.map(|limb| limb.0.map(|residue| residue as i64));
        self.garner.centered_mixed_radix(&mut mixed_radix);
        let coefficients = std::array::from_fn(|coefficient| {
            let digits = std::array::from_fn(|limb| mixed_radix[limb][coefficient]);

            let mut result = self.garner.digits_to_field(&digits, &field_weights);
            if let (Some(tail), Some(tail_canonical)) = (&self.tail, tail_canonical) {
                let tail_digit = i128::from(tail_canonical[coefficient].to_i64())
                    * i128::from(tail.residue_weight)
                    + digits
                        .iter()
                        .zip(tail.digit_weights)
                        .map(|(digit, weight)| i128::from(*digit) * i128::from(weight))
                        .sum::<i128>();
                let tail_modulus = i128::from(tail.modulus);
                let mut tail_digit = tail_digit.rem_euclid(tail_modulus);
                if tail_digit > tail_modulus / 2 {
                    tail_digit -= tail_modulus;
                }
                result += F::from_i64(tail_digit as i64) * tail_field_weight;
            }
            result
        });
        Ok(CyclotomicRing::from_coefficients(coefficients))
    }
}

/// A prepared row-major matrix within each contiguous CRT limb.
///
/// `limbs[k][row * width + column]` is one transformed ring. Keeping the CRT
/// limb outside the flat matrix makes the limb-major mat-vec loop contiguous.
#[derive(Debug)]
pub struct Ifma52NttMatrix<const K: usize, const D: usize> {
    limbs: [Vec<Ifma52Residues<D>>; K],
    params: Ifma52Params<K, D>,
    /// Largest centered coefficient magnitude of any prepared entry.
    entry_abs_bound: u128,
}

impl<const K: usize, const D: usize> Ifma52NttMatrix<K, D> {
    /// Prepare a flat row-major coefficient matrix in negacyclic NTT form.
    pub fn prepare<F: Field + CanonicalEncoding>(
        rings: &[CyclotomicRing<F, D>],
        params: &Ifma52Params<K, D>,
    ) -> Self {
        Self::prepare_centered(
            rings.len(),
            |index| rings[index].centered_coefficients_i128(),
            params,
        )
    }

    /// Prepare `len` flat row-major entries, given by their centered integer
    /// coefficients, in negacyclic NTT form.
    pub fn prepare_centered(
        len: usize,
        entry: impl Fn(usize) -> [i128; D] + Sync,
        params: &Ifma52Params<K, D>,
    ) -> Self {
        let mut limbs: [Vec<Ifma52Residues<D>>; K] =
            std::array::from_fn(|_| vec![Ifma52Residues([0; D]); len]);
        // Tile `t` owns entries `t * T..` of every CRT limb.
        let mut tiles: Vec<Vec<&mut [Ifma52Residues<D>]>> = (0..len.div_ceil(PREPARE_TILE_ENTRIES))
            .map(|_| Vec::with_capacity(K))
            .collect();
        for limb in &mut limbs {
            for (tile, entries) in tiles.iter_mut().zip(limb.chunks_mut(PREPARE_TILE_ENTRIES)) {
                tile.push(entries);
            }
        }
        let entry_abs_bound = cfg_into_iter!(tiles)
            .enumerate()
            .map(|(tile_index, mut tile)| {
                let tile_len = tile.first().map_or(0, |entries| entries.len());
                let mut tile_bound = 0;
                for offset in 0..tile_len {
                    let centered = entry(tile_index * PREPARE_TILE_ENTRIES + offset);
                    for value in centered {
                        tile_bound = tile_bound.max(value.unsigned_abs());
                    }
                    for (entries, (prime, twiddles)) in tile
                        .iter_mut()
                        .zip(params.primes.iter().zip(params.twiddles.iter()))
                    {
                        let transformed = &mut entries[offset];
                        *transformed = Ifma52Residues(
                            centered
                                .map(|value| value.rem_euclid(i128::from(prime.modulus)) as u64),
                        );
                        forward(transformed, *prime, twiddles, params.use_ifma);
                    }
                }
                tile_bound
            })
            .max()
            .unwrap_or(0);
        Self {
            limbs,
            params: params.clone(),
            entry_abs_bound,
        }
    }

    /// In-memory byte footprint of the prepared matrix.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.limbs.iter().map(Vec::len).sum::<usize>() * D * core::mem::size_of::<u64>()
    }

    /// Number of flat matrix entries in each CRT limb.
    #[must_use]
    pub fn len(&self) -> usize {
        self.limbs.first().map_or(0, Vec::len)
    }

    /// Whether the prepared matrix has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exact CRT capacity of the parameters bound to this matrix.
    #[must_use]
    pub fn crt_capacity(&self) -> CrtCapacity {
        self.params.crt_capacity()
    }

    /// Whether the bound parameters' exactness-only tail is `prime`.
    #[must_use]
    pub fn has_tail<W: PrimeWidth>(&self, prime: NttPrime<W>) -> bool {
        self.params.has_tail(prime)
    }

    /// Multiply by one exact signed-i16 vector.
    #[inline]
    pub fn mat_vec_i16<F: Field + CanonicalEncoding>(
        &self,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        if self.params.tail.is_some() {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 tail does not match its parameters".into(),
            ));
        }
        if num_rows == 0 || rhs.is_empty() {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }
        self.mat_vec_i16_canonical(num_rows, rhs)?
            .iter()
            .map(|canonical| self.params.reconstruct::<F, i16>(canonical, None))
            .collect()
    }

    /// Multiply by one exact signed-i16 vector with a CRT tail of width `W`.
    #[inline]
    pub fn mat_vec_i16_with_tail<F: Field + CanonicalEncoding, W: PrimeWidth>(
        &self,
        tail_matrix: &[CyclotomicCrtNtt<W, 1, D>],
        num_rows: usize,
        rhs: &[[i16; D]],
        tail_params: &CrtNttParamSet<W, 1, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let num_cols = rhs.len();
        let required = num_rows.checked_mul(num_cols).ok_or_else(|| {
            AkitaError::InvalidInput("prepared IFMA52 tail matrix extent overflow".into())
        })?;
        if !self.params.has_tail(tail_params.primes[0]) {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 tail does not match its parameters".into(),
            ));
        }
        if tail_matrix.len() < required {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 tail matrix prefix is undersized".into(),
            ));
        }
        if num_rows == 0 || num_cols == 0 {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }

        let accumulators = self.mat_vec_i16_canonical(num_rows, rhs)?;
        let tail_canonical =
            CyclotomicCrtNtt::mat_vec_i16_ntt(tail_matrix, num_rows, num_cols, rhs, tail_params)?
                .iter()
                .map(|accumulator| accumulator.centered_coefficients_with_params(tail_params)[0])
                .collect::<Vec<_>>();

        accumulators
            .iter()
            .zip(&tail_canonical)
            .map(|(canonical, tail)| self.params.reconstruct(canonical, Some(tail)))
            .collect()
    }

    #[inline(always)]
    fn mat_vec_i16_canonical(
        &self,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<[Ifma52Residues<D>; K]>, AkitaError> {
        let num_cols = rhs.len();
        let required = num_rows.checked_mul(num_cols).ok_or_else(|| {
            AkitaError::InvalidInput("prepared IFMA52 matrix extent overflow".into())
        })?;
        if self.limbs.iter().any(|limb| limb.len() < required) {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 matrix prefix is undersized".into(),
            ));
        }
        let rhs_abs_bound = rhs.iter().flatten().map(|value| value.unsigned_abs()).max();
        // Centered residues of the odd modulus `2A + 1` are exactly `[-A, A]`.
        let entry_modulus = self.entry_abs_bound.saturating_mul(2).saturating_add(1);
        if !self.params.crt_capacity().supports_modulus(
            num_cols,
            D,
            entry_modulus,
            rhs_abs_bound.map_or(0, u64::from),
        ) {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT capacity does not cover this mat-vec".into(),
            ));
        }

        let mut canonical = vec![[Ifma52Residues([0; D]); K]; num_rows];
        if num_rows == 0 || num_cols == 0 {
            return Ok(canonical);
        }
        let use_ifma = self.params.use_ifma;
        let tile_width = (64 * 1024 / (D * core::mem::size_of::<u64>()))
            .clamp(1, IFMA52_ACCUMULATOR_TERMS)
            .min(num_cols);
        let mut transformed_rhs = vec![Ifma52Residues([0; D]); tile_width];
        let mut accumulators = vec![Ifma52Accumulator::ZERO; num_rows];
        for (prime_index, ((matrix_limb, &prime), twiddles)) in self
            .limbs
            .iter()
            .zip(self.params.primes.iter())
            .zip(self.params.twiddles.iter())
            .enumerate()
        {
            accumulators.fill(Ifma52Accumulator::ZERO);
            let mut terms = 0;
            for tile_start in (0..num_cols).step_by(tile_width) {
                let tile_end = (tile_start + tile_width).min(num_cols);
                let tile = &mut transformed_rhs[..tile_end - tile_start];
                for (transformed, digits) in tile.iter_mut().zip(&rhs[tile_start..tile_end]) {
                    forward_i16(transformed, digits, prime, twiddles, use_ifma);
                }
                if terms + tile.len() > IFMA52_ACCUMULATOR_TERMS {
                    for accumulator in &mut accumulators {
                        accumulator.fold(prime, use_ifma);
                    }
                    terms = 0;
                }
                terms += tile.len();
                debug_assert!(terms <= IFMA52_ACCUMULATOR_TERMS);
                for (row, accumulator) in accumulators.iter_mut().enumerate() {
                    let row_start = row * num_cols;
                    accumulator.accumulate(
                        &matrix_limb[row_start + tile_start..row_start + tile_end],
                        tile,
                        use_ifma,
                    );
                }
            }
            for (accumulator, canonical) in accumulators.iter().zip(&mut canonical) {
                let limb = &mut canonical[prime_index];
                *limb = accumulator.reduce(prime, use_ifma);
                inverse(limb, prime, twiddles, use_ifma);
            }
        }
        Ok(canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::ifma52::IFMA52_PRIMES;
    use crate::ntt::tables::{q128_primes, I16_TAIL_PRIME};
    use jolt_field::{One, Prime128OffsetA7F7, Prime64Offset59, Ring, Zero};

    fn assert_limb_major_i16_matvec<const D: usize>() {
        type F = Prime64Offset59;
        let params =
            Ifma52Params::<2, D>::new([IFMA52_PRIMES[0], IFMA52_PRIMES[1]]).expect("params");
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64((entry * 13 + coefficient * 7) as i64 - 40)
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..3)
            .map(|column| std::array::from_fn(|index| (column * 9 + index) as i16 - 20))
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let actual = prepared.mat_vec_i16::<F>(2, &rhs).expect("matvec");
        let expected = matrix
            .chunks_exact(3)
            .map(|row| {
                row.iter()
                    .zip(&rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + *lhs
                            * CyclotomicRing::from_coefficients(
                                rhs.map(|value| F::from_i64(value.into())),
                            )
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn limb_major_i16_matvec_rejects_accumulations_past_crt_capacity() {
        const D: usize = 64;
        type F = Prime64Offset59;
        let params = Ifma52Params::<1, D>::new([IFMA52_PRIMES[0]]).expect("params");
        // One 50-bit prime covers 2 * 64 * 2^30 * 1 but not 2 * 64 * 2^30 * 2^15.
        let prepared = Ifma52NttMatrix::prepare_centered(1, |_| [1_i128 << 30; D], &params);
        let small = [[1_i16; D]];
        assert!(prepared.mat_vec_i16::<F>(1, &small).is_ok());
        let large = [[i16::MIN; D]];
        assert!(prepared.mat_vec_i16::<F>(1, &large).is_err());
    }

    #[test]
    fn limb_major_i16_matvec_matches_ring_arithmetic_at_all_ifma_dimensions() {
        assert_limb_major_i16_matvec::<64>();
        assert_limb_major_i16_matvec::<128>();
        assert_limb_major_i16_matvec::<256>();
        assert_limb_major_i16_matvec::<512>();
        assert_limb_major_i16_matvec::<1024>();
        assert_limb_major_i16_matvec::<2048>();
    }

    fn assert_mixed_ifma_i16_tail_matvec<const D: usize>() {
        type F = Prime64Offset59;
        let params = Ifma52Params::<1, D>::new([IFMA52_PRIMES[0]])
            .expect("params")
            .with_tail(I16_TAIL_PRIME)
            .expect("tail params");
        let tail_params = CrtNttParamSet::new([I16_TAIL_PRIME]);
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_u64(
                        IFMA52_PRIMES[0] + (entry as u64 + 1) * 65_537 + coefficient as u64 * 4_099,
                    )
                }))
            })
            .collect::<Vec<_>>();
        // Width 2 and digits in [-1, 1] keep the worst case `2 * 2 * D * A` below
        // `p * 12289` at D = 2048, while the sums still pass `p / 2`.
        let rhs = (0..2)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column * 3 + coefficient * 2) % 3) as i16 - 1)
            })
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let tail_matrix = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
            .collect::<Vec<_>>();
        let actual = prepared
            .mat_vec_i16_with_tail::<F, _>(&tail_matrix, 3, &rhs, &tail_params)
            .expect("mixed matvec");
        let expected = matrix
            .chunks_exact(2)
            .map(|row| {
                row.iter()
                    .zip(&rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + *lhs
                            * CyclotomicRing::from_coefficients(
                                rhs.map(|value| F::from_i64(value.into())),
                            )
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn mixed_ifma_matvec_rejects_a_tail_over_another_prime() {
        const D: usize = 64;
        let params = Ifma52Params::<1, D>::new([IFMA52_PRIMES[0]])
            .expect("params")
            .with_tail(I16_TAIL_PRIME)
            .expect("tail params");
        let other = NttPrime::new(13_313_i16);
        assert!(params.has_tail(I16_TAIL_PRIME));
        assert!(!params.has_tail(other));
        assert!(!params.has_tail(q128_primes()[0]));

        let prepared = Ifma52NttMatrix::<1, D>::prepare(
            &[CyclotomicRing::<Prime64Offset59, D>::zero()],
            &params,
        );
        let tail_params = CrtNttParamSet::new([other]);
        let tail_matrix = [CyclotomicCrtNtt::zero()];
        let result = prepared.mat_vec_i16_with_tail::<Prime64Offset59, _>(
            &tail_matrix,
            1,
            &[[0; D]],
            &tail_params,
        );
        assert!(matches!(result, Err(AkitaError::InvalidSetup(_))));
    }

    #[test]
    fn mixed_ifma_i16_tail_matvec_matches_ring_arithmetic_at_all_dimensions() {
        assert_mixed_ifma_i16_tail_matvec::<64>();
        assert_mixed_ifma_i16_tail_matvec::<128>();
        assert_mixed_ifma_i16_tail_matvec::<256>();
        assert_mixed_ifma_i16_tail_matvec::<512>();
        assert_mixed_ifma_i16_tail_matvec::<1024>();
        assert_mixed_ifma_i16_tail_matvec::<2048>();
    }

    fn assert_q128_ifma_tail_matvec<W: PrimeWidth, const D: usize>(tail_prime: NttPrime<W>) {
        type F = Prime128OffsetA7F7;
        let params = Ifma52Params::<3, D>::new(IFMA52_PRIMES)
            .expect("params")
            .with_tail(tail_prime)
            .expect("tail params");
        let tail_params = CrtNttParamSet::new([tail_prime]);
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    let low = (entry as u64 + 1) * 65_537 + coefficient as u64 * 4_099;
                    F::from_i128((u128::from(low) << 80) as i128) - F::from_u64(low.rotate_left(17))
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..3)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    if (column + coefficient) % 2 == 0 {
                        i16::MAX
                    } else {
                        i16::MIN
                    }
                })
            })
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let tail_matrix = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
            .collect::<Vec<_>>();
        let actual = prepared
            .mat_vec_i16_with_tail::<F, _>(&tail_matrix, 2, &rhs, &tail_params)
            .expect("Q128 mixed matvec");
        let expected = matrix
            .chunks_exact(3)
            .map(|row| {
                row.iter()
                    .zip(&rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + *lhs
                            * CyclotomicRing::from_coefficients(
                                rhs.map(|value| F::from_i64(value.into())),
                            )
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn q128_ifma_tail_matvec_matches_ring_arithmetic_at_all_dimensions() {
        let i32_tail = q128_primes()[0];
        assert_q128_ifma_tail_matvec::<_, 64>(I16_TAIL_PRIME);
        assert_q128_ifma_tail_matvec::<_, 64>(i32_tail);
        assert_q128_ifma_tail_matvec::<_, 128>(I16_TAIL_PRIME);
        assert_q128_ifma_tail_matvec::<_, 128>(i32_tail);
        assert_q128_ifma_tail_matvec::<_, 256>(I16_TAIL_PRIME);
        assert_q128_ifma_tail_matvec::<_, 256>(i32_tail);
        assert_q128_ifma_tail_matvec::<_, 512>(I16_TAIL_PRIME);
        assert_q128_ifma_tail_matvec::<_, 512>(i32_tail);
        assert_q128_ifma_tail_matvec::<_, 1024>(I16_TAIL_PRIME);
        assert_q128_ifma_tail_matvec::<_, 1024>(i32_tail);
        // The i32 tail prime supports negacyclic degrees through 1024.
        assert_q128_ifma_tail_matvec::<_, 2048>(I16_TAIL_PRIME);
    }

    fn assert_q128_tail_reconstruction<W: PrimeWidth>(tail_prime: NttPrime<W>) {
        const D: usize = 64;
        type F = Prime128OffsetA7F7;
        let params = Ifma52Params::<3, D>::new(IFMA52_PRIMES)
            .expect("params")
            .with_tail(tail_prime)
            .expect("tail params");
        let digits = IFMA52_PRIMES.map(|prime| (prime / 2) as i64);
        let residue = |modulus: u64| {
            let modulus = u128::from(modulus);
            let mut residue = 0u128;
            let mut weight = 1u128;
            for (digit, prime) in digits.iter().zip(IFMA52_PRIMES) {
                residue = (residue + (*digit as u128 * weight) % modulus) % modulus;
                weight = (weight * u128::from(prime)) % modulus;
            }
            residue as u64
        };
        let canonical = IFMA52_PRIMES.map(|prime| Ifma52Residues([residue(prime); D]));
        let tail_modulus = tail_prime.p.to_i64();
        let tail_residue = residue(tail_modulus as u64) as i64;
        let tail_centered = if tail_residue > tail_modulus / 2 {
            tail_residue - tail_modulus
        } else {
            tail_residue
        };
        let tail = [W::from_i64(tail_centered); D];

        let mut field_weight = F::one();
        let mut expected = F::zero();
        for (digit, prime) in digits.into_iter().zip(IFMA52_PRIMES) {
            expected += F::from_i64(digit) * field_weight;
            field_weight *= F::from_u64(prime);
        }
        assert_eq!(
            params
                .reconstruct::<F, _>(&canonical, Some(&tail))
                .expect("reconstruction"),
            CyclotomicRing::from_coefficients([expected; D])
        );
    }

    #[test]
    fn q128_tail_reconstruction_handles_maximum_centered_digits() {
        assert_q128_tail_reconstruction(I16_TAIL_PRIME);
        assert_q128_tail_reconstruction(q128_primes()[0]);
    }
}
