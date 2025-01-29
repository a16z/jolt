use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{prelude::*, PrimeField};
use ark_std::cmp::Ordering;
use ark_std::vec::Vec;
#[cfg(feature = "icicle")]
use icicle_core::curve::Affine;
use num_integer::Integer;
use rayon::prelude::*;

pub(crate) mod icicle;
use crate::field::JoltField;
#[cfg(feature = "icicle")]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
pub use icicle::*;

impl<F: JoltField, G: CurveGroup<ScalarField = F> + Icicle> VariableBaseMSM for G {}

#[cfg(feature = "icicle")]
pub type GpuBaseType<G: Icicle> = Affine<G::C>;
#[cfg(not(feature = "icicle"))]
pub type GpuBaseType<G: ScalarMul> = G::MulBase;

use itertools::Either;

/// Copy of ark_ec::VariableBaseMSM with minor modifications to speed up
/// known small element sized MSMs.
pub trait VariableBaseMSM: ScalarMul + Icicle
where
    Self::ScalarField: JoltField,
{
    fn msm_u8(
        bases: &[Self::MulBase],
        scalars: &[u8],
        max_num_bits: Option<usize>,
    ) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits =
                    max_num_bits.unwrap_or((*scalars.iter().max().unwrap() as usize).num_bits());
                match max_num_bits {
                    0 => Self::zero(),
                    1 => msm_binary(bases, scalars),
                    _ => msm_small(bases, scalars, max_num_bits),
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn msm_u16(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[u16],
        max_num_bits: Option<usize>,
        use_icicle: bool,
    ) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits =
                    max_num_bits.unwrap_or((*scalars.iter().max().unwrap() as usize).num_bits());
                match max_num_bits {
                    0 => Self::zero(),
                    1 => msm_binary(bases, scalars),
                    2..=10 => msm_small(bases, scalars, max_num_bits),
                    _ => msm_medium(bases, gpu_bases, scalars, max_num_bits, use_icicle),
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn msm_u32(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[u32],
        max_num_bits: Option<usize>,
        use_icicle: bool,
    ) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits =
                    max_num_bits.unwrap_or((*scalars.iter().max().unwrap() as usize).num_bits());
                match max_num_bits {
                    0 => Self::zero(),
                    1 => msm_binary(bases, scalars),
                    2..=10 => msm_small(bases, scalars, max_num_bits),
                    _ => msm_medium(bases, gpu_bases, scalars, max_num_bits, use_icicle),
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn msm_u64(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[u64],
        max_num_bits: Option<usize>,
        use_icicle: bool,
    ) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits =
                    max_num_bits.unwrap_or((*scalars.iter().max().unwrap() as usize).num_bits());
                match max_num_bits {
                    0 => Self::zero(),
                    1 => msm_binary(bases, scalars),
                    2..=10 => msm_small(bases, scalars, max_num_bits),
                    _ => msm_medium(bases, gpu_bases, scalars, max_num_bits, use_icicle),
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn msm_field_elements(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[Self::ScalarField],
        max_num_bits: Option<usize>,
        use_icicle: bool,
    ) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits =
                    max_num_bits.unwrap_or((*scalars.iter().max().unwrap()).num_bits() as usize);
                match max_num_bits {
                    0 => Self::zero(),
                    1 => scalars
                        .iter()
                        .zip(bases)
                        .filter(|(scalar, _base)| !scalar.is_zero())
                        .map(|(_scalar, base)| base)
                        .fold(Self::zero(), |sum, base| sum + base),
                    2..=10 => {
                        let scalars_u16 = &map_field_elements_to_u16(scalars);
                        msm_small(bases, scalars_u16, max_num_bits)
                    }
                    11..=64 => {
                        let scalars_u64 = &map_field_elements_to_u64(scalars);
                        msm_medium(bases, gpu_bases, scalars_u64, max_num_bits, use_icicle)
                    }
                    _ => {
                        if use_icicle {
                            #[cfg(feature = "icicle")]
                            {
                                let mut backup = vec![];
                                let gpu_bases = gpu_bases.unwrap_or_else(|| {
                                    backup = Self::get_gpu_bases(bases);
                                    &backup
                                });
                                return icicle_msm::<Self>(gpu_bases, scalars, max_num_bits);
                            }
                            #[cfg(not(feature = "icicle"))]
                            {
                                unreachable!(
                                    "icicle_init must not return true without the icicle feature"
                                );
                            }
                        }

                        let scalars = scalars
                            .par_iter()
                            .map(|s| s.into_bigint())
                            .collect::<Vec<_>>();
                        if Self::NEGATION_IS_CHEAP {
                            msm_bigint_wnaf(bases, &scalars, max_num_bits)
                        } else {
                            msm_bigint(bases, &scalars, max_num_bits)
                        }
                    }
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        poly: &MultilinearPolynomial<Self::ScalarField>,
        max_num_bits: Option<usize>,
    ) -> Result<Self, ProofVerifyError> {
        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());
        assert_eq!(bases.len(), gpu_bases.map_or(bases.len(), |b| b.len()));

        let use_icicle = use_icicle();

        match poly {
            MultilinearPolynomial::LargeScalars(poly) => Self::msm_field_elements(
                bases,
                gpu_bases,
                poly.evals_ref(),
                max_num_bits,
                use_icicle,
            ),
            MultilinearPolynomial::U8Scalars(poly) => {
                Self::msm_u8(bases, &poly.coeffs, max_num_bits)
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                Self::msm_u16(bases, gpu_bases, &poly.coeffs, max_num_bits, use_icicle)
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                Self::msm_u32(bases, gpu_bases, &poly.coeffs, max_num_bits, use_icicle)
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                Self::msm_u64(bases, gpu_bases, &poly.coeffs, max_num_bits, use_icicle)
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                // TODO(moodlezoup): This can be optimized
                let scalars: Vec<_> = poly
                    .coeffs
                    .par_iter()
                    .map(|x| Self::ScalarField::from_i64(*x))
                    .collect();
                Self::msm_field_elements(bases, gpu_bases, &scalars, max_num_bits, use_icicle)
            }
        }
    }

    #[tracing::instrument(skip_all)]
    fn batch_msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        polys: &[&MultilinearPolynomial<Self::ScalarField>],
    ) -> Vec<Self> {
        assert!(polys.par_iter().all(|s| s.len() == bases.len()));
        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());
        assert_eq!(bases.len(), gpu_bases.map_or(bases.len(), |b| b.len()));

        let use_icicle = use_icicle();

        if !use_icicle {
            let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu_only");
            let _guard = span.enter();
            return polys
                .into_par_iter()
                .map(|poly| Self::msm(bases, None, poly, None).unwrap())
                .collect();
        }

        // Split scalar batches into CPU and GPU workloads
        let span = tracing::span!(tracing::Level::INFO, "group_scalar_indices_parallel");
        let _guard = span.enter();
        let (cpu_batch, gpu_batch): (Vec<_>, Vec<_>) =
            polys
                .par_iter()
                .enumerate()
                .partition_map(|(i, poly)| match poly {
                    MultilinearPolynomial::LargeScalars(_) => {
                        let max_num_bits = poly.max_num_bits();
                        // Use GPU for large-scalar polynomials
                        Either::Right((i, max_num_bits, *poly))
                    }
                    _ => {
                        let max_num_bits = poly.max_num_bits();
                        Either::Left((i, max_num_bits, *poly))
                    }
                });
        drop(_guard);
        drop(span);
        let mut results = vec![Self::zero(); polys.len()];

        // Handle CPU computations in parallel
        let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu");
        let _guard = span.enter();
        let cpu_results: Vec<(usize, Self)> = cpu_batch
            .into_par_iter()
            .map(|(i, max_num_bits, poly)| {
                (
                    i,
                    Self::msm(bases, None, poly, Some(max_num_bits as usize)).unwrap(),
                )
            })
            .collect();
        drop(_guard);
        drop(span);

        // Store CPU results
        for (i, result) in cpu_results {
            results[i] = result;
        }

        // Handle GPU computations if available
        if !gpu_batch.is_empty() && use_icicle {
            #[cfg(feature = "icicle")]
            {
                let span = tracing::span!(tracing::Level::INFO, "batch_msms_gpu");
                let _guard = span.enter();
                let mut backup = vec![];
                let gpu_bases = gpu_bases.unwrap_or_else(|| {
                    backup = Self::get_gpu_bases(bases);
                    &backup
                });

                // includes putting the scalars and bases on device
                let slice_bit_size = 256 * gpu_batch[0].2.len() * 2;
                let slices_at_a_time = total_memory_bits() / slice_bit_size;

                // Process GPU batches with memory constraints
                for work_chunk in gpu_batch.chunks(slices_at_a_time) {
                    let (max_num_bits, chunk_polys): (Vec<_>, Vec<_>) = work_chunk
                        .par_iter()
                        .map(|(_, max_num_bits, poly)| (*max_num_bits, *poly))
                        .unzip();

                    let max_num_bits = max_num_bits.iter().max().unwrap();
                    let scalars: Vec<_> = chunk_polys
                        .into_iter()
                        .map(|poly| {
                            let poly: &DensePolynomial<Self::ScalarField> =
                                poly.try_into().unwrap();
                            poly.evals_ref()
                        })
                        .collect();
                    let batch_results =
                        icicle_batch_msm(gpu_bases, &scalars, *max_num_bits as usize);

                    // Store GPU results using original indices
                    for ((original_idx, _, _), result) in work_chunk.iter().zip(batch_results) {
                        results[*original_idx] = result;
                    }
                }
            }
            #[cfg(not(feature = "icicle"))]
            {
                unreachable!("icicle_init must not return true without the icicle feature");
            }
        }
        results
    }

    #[cfg(feature = "icicle")]
    #[tracing::instrument(skip_all)]
    fn get_gpu_bases(bases: &[Self::MulBase]) -> Vec<GpuBaseType<Self>> {
        bases
            .par_iter()
            .map(|base| <Self as Icicle>::from_ark_affine(base))
            .collect()
    }
}

pub fn use_icicle() -> bool {
    #[cfg(feature = "icicle")]
    return icicle_init();
    #[cfg(not(feature = "icicle"))]
    false
}

fn map_field_elements_to_u16<F: PrimeField>(field_elements: &[F]) -> Vec<u16> {
    field_elements
        .par_iter()
        .map(|s| {
            let bigint = s.into_bigint();
            let limbs: &[u64] = bigint.as_ref();
            limbs[0] as u16
        })
        .collect::<Vec<_>>()
}

fn map_field_elements_to_u64<F: PrimeField>(field_elements: &[F]) -> Vec<u64> {
    field_elements
        .par_iter()
        .map(|s| {
            let bigint = s.into_bigint();
            let limbs: &[u64] = bigint.as_ref();
            limbs[0]
        })
        .collect::<Vec<_>>()
}

// Compute msm using windowed non-adjacent form
#[tracing::instrument(skip_all)]
fn msm_bigint_wnaf<F: JoltField + PrimeField, V: VariableBaseMSM<ScalarField = F>>(
    bases: &[V::MulBase],
    scalars: &[<F as PrimeField>::BigInt],
    max_num_bits: usize,
) -> V {
    let c = if bases.len() < 32 {
        3
    } else {
        ln_without_floats(bases.len()) + 2
    };

    let num_bits = max_num_bits;
    let digits_count = num_bits.div_ceil(c);
    let scalar_digits = scalars
        .into_par_iter()
        .flat_map_iter(|s| make_digits_bigint(s, c, num_bits))
        .collect::<Vec<_>>();
    let zero = V::zero();
    let window_sums: Vec<_> = (0..digits_count)
        .into_par_iter()
        .map(|i| {
            let mut buckets = vec![zero; 1 << c];
            for (digits, base) in scalar_digits.chunks(digits_count).zip(bases) {
                // digits is the digits thing of the first scalar?
                let scalar = digits[i];
                match 0.cmp(&scalar) {
                    Ordering::Less => buckets[(scalar - 1) as usize] += base,
                    Ordering::Greater => buckets[(-scalar - 1) as usize] -= base,
                    Ordering::Equal => (),
                }
            }

            let mut running_sum = V::zero();
            let mut res = V::zero();
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
        + window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_bigint<F: JoltField + PrimeField, V: VariableBaseMSM<ScalarField = F>>(
    bases: &[V::MulBase],
    scalars: &[<F as PrimeField>::BigInt],
    max_num_bits: usize,
) -> V {
    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

    let c = if bases.len() < 32 {
        3
    } else {
        ln_without_floats(bases.len()) + 2
    };

    let one = V::ScalarField::one().into_bigint();

    let zero = V::zero();
    let window_starts = (0..max_num_bits).step_by(c);

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
        .map(|w_start| {
            let mut res = zero;
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
            let mut buckets = vec![zero; (1 << c) - 1];
            // This clone is cheap, because the iterator contains just a
            // pointer and an index into the original vectors.
            scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                if scalar == one {
                    // We only process unit scalars once in the first window.
                    if w_start == 0 {
                        res += base;
                    }
                } else {
                    let mut scalar = scalar;

                    // We right-shift by w_start, thus getting rid of the
                    // lower bits.
                    scalar.divn(w_start as u32);

                    // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                    let scalar = scalar.as_ref()[0] % (1 << c);

                    // If the scalar is non-zero, we update the corresponding
                    // bucket.
                    // (Recall that `buckets` doesn't have a zero bucket.)
                    if scalar != 0 {
                        buckets[(scalar - 1) as usize] += base;
                    }
                }
            });

            // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
            // This is computed below for b buckets, using 2b curve additions.
            //
            // We could first normalize `buckets` and then use mixed-addition
            // here, but that's slower for the kinds of groups we care about
            // (Short Weierstrass curves and Twisted Edwards curves).
            // In the case of Short Weierstrass curves,
            // mixed addition saves ~4 field multiplications per addition.
            // However normalization (with the inversion batched) takes ~6
            // field multiplications per element,
            // hence batch normalization is a slowdown.

            // `running_sum` = sum_{j in i..num_buckets} bucket[j],
            // where we iterate backward from i = num_buckets to 0.
            let mut running_sum = V::zero();
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
        + window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

// From: https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
fn make_digits_bigint(
    a: &impl BigInteger,
    w: usize,
    num_bits: usize,
) -> impl Iterator<Item = i64> + '_ {
    let scalar = a.as_ref();
    let radix: u64 = 1 << w;
    let window_mask: u64 = radix - 1;

    let mut carry = 0u64;
    let num_bits = if num_bits == 0 {
        a.num_bits() as usize
    } else {
        num_bits
    };
    let digits_count = num_bits.div_ceil(w);
    (0..digits_count).map(move |i| {
        // Construct a buffer of bits of the scalar, starting at `bit_offset`.
        let bit_offset = i * w;
        let u64_idx = bit_offset / 64;
        let bit_idx = bit_offset % 64;
        // Read the bits from the scalar
        let bit_buf = if bit_idx < 64 - w || u64_idx == scalar.len() - 1 {
            // This window's bits are contained in a single u64,
            // or it's the last u64 anyway.
            scalar[u64_idx] >> bit_idx
        } else {
            // Combine the current u64's bits with the bits from the next u64
            (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
        };
        // Read the actual coefficient value from the window
        let coef = carry + (bit_buf & window_mask); // coef = [0, 2^r)

        // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
        carry = (coef + radix / 2) >> w;
        let mut digit = (coef as i64) - (carry << w) as i64;

        if i == digits_count - 1 {
            digit += (carry << w) as i64;
        }
        digit
    })
}

/// Optimized implementation of multi-scalar multiplication.
#[tracing::instrument(skip_all)]
fn msm_medium<F, V, T>(
    bases: &[V::MulBase],
    _gpu_bases: Option<&[GpuBaseType<V>]>,
    scalars: &[T],
    max_num_bits: usize,
    _use_icicle: bool,
) -> V
where
    F: JoltField,
    V: VariableBaseMSM<ScalarField = F>,
    T: Into<u64> + Zero + Copy + Sync,
{
    let c = if bases.len() < 32 {
        3
    } else {
        ln_without_floats(bases.len()) + 2
    };

    let zero = V::zero();

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts = (0..max_num_bits).into_par_iter().step_by(c);

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
        .map(|w_start| {
            let mut res = zero;
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
            let mut buckets = vec![zero; (1 << c) - 1];
            // This clone is cheap, because the iterator contains just a
            // pointer and an index into the original vectors.
            scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                let scalar: u64 = scalar.into();
                if scalar == 1 {
                    // We only process unit scalars once in the first window.
                    if w_start == 0 {
                        res += base;
                    }
                } else {
                    let mut scalar = scalar;

                    // We right-shift by w_start, thus getting rid of the
                    // lower bits.
                    scalar >>= w_start;

                    // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                    scalar %= 1 << c;

                    // If the scalar is non-zero, we update the corresponding
                    // bucket.
                    // (Recall that `buckets` doesn't have a zero bucket.)
                    if scalar != 0 {
                        buckets[(scalar - 1) as usize] += base;
                    }
                }
            });

            // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
            // This is computed below for b buckets, using 2b curve additions.
            //
            // We could first normalize `buckets` and then use mixed-addition
            // here, but that's slower for the kinds of groups we care about
            // (Short Weierstrass curves and Twisted Edwards curves).
            // In the case of Short Weierstrass curves,
            // mixed addition saves ~4 field multiplications per addition.
            // However normalization (with the inversion batched) takes ~6
            // field multiplications per element,
            // hence batch normalization is a slowdown.

            // `running_sum` = sum_{j in i..num_buckets} bucket[j],
            // where we iterate backward from i = num_buckets to 0.
            let mut running_sum = V::zero();
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
        + window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

#[tracing::instrument(skip_all)]
fn msm_binary<F: JoltField, V: VariableBaseMSM<ScalarField = F>, T: Integer>(
    bases: &[V::MulBase],
    scalars: &[T],
) -> V {
    scalars
        .iter()
        .zip(bases)
        .filter(|(scalar, _base)| !scalar.is_zero())
        .map(|(_scalar, base)| base)
        .fold(V::zero(), |sum, base| sum + base)
}

#[tracing::instrument(skip_all)]
fn msm_small<F, V, T>(bases: &[V::MulBase], scalars: &[T], max_num_bits: usize) -> V
where
    F: JoltField,
    V: VariableBaseMSM<ScalarField = F>,
    T: Into<u64> + Zero + Copy,
{
    let num_buckets: usize = 1 << max_num_bits;
    // Assign things to buckets based on the scalar
    let mut buckets: Vec<V> = vec![V::zero(); num_buckets];
    scalars
        .iter()
        .zip(bases)
        .filter(|(scalar, _base)| !scalar.is_zero())
        .for_each(|(scalar, base)| {
            let bucket_index: u64 = (*scalar).into();
            buckets[bucket_index as usize] += base;
        });

    let mut result = V::zero();
    let mut running_sum = V::zero();
    buckets.iter().skip(1).rev().for_each(|bucket| {
        running_sum += bucket;
        result += running_sum;
    });
    result
}

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]: https://github.com/scipr-lab/zexe/issues/79#issue-556220473
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}
