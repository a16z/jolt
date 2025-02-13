use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{prelude::*, PrimeField};
use ark_std::cmp::Ordering;
use ark_std::vec::Vec;
#[cfg(feature = "icicle")]
use icicle_core::curve::Affine;
use rayon::prelude::*;

pub(crate) mod icicle;
use crate::utils::errors::ProofVerifyError;
pub use icicle::*;

impl<G: CurveGroup + Icicle> VariableBaseMSM for G {}

#[cfg(feature = "icicle")]
pub type GpuBaseType<G: Icicle> = Affine<G::C>;
#[cfg(not(feature = "icicle"))]
pub type GpuBaseType<G: ScalarMul> = G::MulBase;

use itertools::Either;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum MsmType {
    Zero,
    One,
    Small(usize),
    Medium(usize),
    Large(usize),
}

impl MsmType {
    fn from_u32(i: u32) -> MsmType {
        match i {
            0 => MsmType::Zero,
            1 => MsmType::One,
            2..=10 => MsmType::Small(i as usize),
            11..=64 => MsmType::Medium(i as usize),
            _ => MsmType::Large(i as usize),
        }
    }

    #[tracing::instrument(skip_all)]
    fn from_scalars<S: ScalarMul>(scalars: &[S::ScalarField]) -> MsmType {
        let max_num_bits = scalars
            .par_iter()
            .map(|s| s.into_bigint().num_bits())
            .max()
            .unwrap();
        MsmType::from_u32(max_num_bits)
    }

    #[allow(dead_code)]
    fn num_bits(&self) -> usize {
        match self {
            MsmType::Zero => 0,
            MsmType::One => 1,
            MsmType::Small(i) => *i,
            MsmType::Medium(i) => *i,
            MsmType::Large(i) => *i,
        }
    }

    fn prefers_icicle(&self) -> bool {
        match self {
            MsmType::Zero | MsmType::One | MsmType::Small(_) => false,
            #[cfg(feature = "icicle")]
            MsmType::Medium(_) | MsmType::Large(_) => true,
            #[cfg(not(feature = "icicle"))]
            _ => false,
        }
    }
}

type TrackedScalar<'a, P: Pairing> = (usize, &'a [P::ScalarField]);
pub type ScalarGroups<'a, P: Pairing> = (MsmType, Vec<TrackedScalar<'a, P>>);

/// Copy of ark_ec::VariableBaseMSM with minor modifications to speed up
/// known small element sized MSMs.
pub trait VariableBaseMSM: ScalarMul + Icicle {
    #[tracing::instrument(skip_all)]
    fn msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[Self::ScalarField],
    ) -> Result<Self, ProofVerifyError> {
        Self::inner_msm(bases, gpu_bases, scalars, true, None)
    }

    #[tracing::instrument(skip_all)]
    fn msm_with_type(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[Self::ScalarField],
        msm_type: MsmType,
    ) -> Result<Self, ProofVerifyError> {
        Self::inner_msm(bases, gpu_bases, scalars, true, Some(msm_type))
    }

    #[tracing::instrument(skip_all)]
    fn inner_msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalars: &[Self::ScalarField],
        allow_icicle: bool,
        msm_type: Option<MsmType>,
    ) -> Result<Self, ProofVerifyError> {
        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());
        assert_eq!(bases.len(), gpu_bases.map_or(bases.len(), |b| b.len()));

        (bases.len() == scalars.len())
            .then(|| {
                let msm_type = msm_type.unwrap_or_else(|| MsmType::from_scalars::<Self>(scalars));

                match msm_type {
                    MsmType::Zero => Self::zero(),
                    MsmType::One => {
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        msm_binary(bases, scalars_u64)
                    }
                    MsmType::Small(max_num_bits) => {
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        msm_small(bases, scalars_u64, max_num_bits)
                    }
                    MsmType::Medium(max_num_bits) => {
                        // TODO(sagar) caching this as "use_icicle = use_icicle" seems to cause a massive slowdown
                        if use_icicle(Some(msm_type.prefers_icicle() && allow_icicle)) {
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

                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        if Self::NEGATION_IS_CHEAP {
                            msm_u64_wnaf(bases, scalars_u64, max_num_bits)
                        } else {
                            msm_u64(bases, scalars_u64, max_num_bits)
                        }
                    }
                    MsmType::Large(max_num_bits) => {
                        if use_icicle(Some(msm_type.prefers_icicle() && allow_icicle)) {
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
                            msm_bigint_wnaf(bases, &scalars[..], max_num_bits)
                        } else {
                            msm_bigint(bases, &scalars[..], max_num_bits)
                        }
                    }
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn batch_msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalar_batches: &[&[Self::ScalarField]],
    ) -> Vec<Self> {
        Self::batch_msm_inner(bases, gpu_bases, scalar_batches, true, false)
    }

    #[tracing::instrument(skip_all)]
    fn variable_batch_msm(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalar_batches: &[&[Self::ScalarField]],
    ) -> Vec<Self> {
        Self::batch_msm_inner(bases, gpu_bases, scalar_batches, true, true)
    }

    #[tracing::instrument(skip_all)]
    fn batch_msm_inner(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        scalar_batches: &[&[Self::ScalarField]],
        allow_icicle: bool,
        _variable_batches: bool,
    ) -> Vec<Self> {
        assert!(scalar_batches.par_iter().all(|s| s.len() == bases.len()));
        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());
        assert_eq!(bases.len(), gpu_bases.map_or(bases.len(), |b| b.len()));

        if !use_icicle(Some(allow_icicle)) {
            let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu_only");
            let _guard = span.enter();
            return scalar_batches
                .into_par_iter()
                .map(|scalars| Self::inner_msm(bases, None, scalars, false, None).unwrap())
                .collect();
        }

        // Split scalar batches into CPU and GPU workloads
        let span = tracing::span!(tracing::Level::INFO, "group_scalar_indices_parallel");
        let _guard = span.enter();
        let (cpu_slices, gpu_slices): (Vec<_>, Vec<_>) = scalar_batches
            .par_iter()
            .enumerate()
            .partition_map(|(i, scalar_slice)| {
                let msm_type = MsmType::from_scalars::<Self>(scalar_slice);
                if use_icicle(Some(allow_icicle && msm_type.prefers_icicle())) {
                    Either::Right((i, msm_type, *scalar_slice))
                } else {
                    Either::Left((i, msm_type, *scalar_slice))
                }
            });
        drop(_guard);
        drop(span);
        let mut results = vec![Self::zero(); scalar_batches.len()];

        // Handle CPU computations in parallel
        let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu");
        let _guard = span.enter();
        let cpu_results: Vec<(usize, Self)> = cpu_slices
            .into_par_iter()
            .map(|(i, msm_type, scalars)| {
                (
                    i,
                    Self::msm_with_type(bases, None, scalars, msm_type).unwrap(),
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
        if !gpu_slices.is_empty() && use_icicle(Some(allow_icicle)) {
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
                let slice_bit_size = 256 * gpu_slices[0].2.len() * 2;
                let slices_at_a_time = total_memory_bits() / slice_bit_size;

                // Process GPU batches with memory constraints
                for work_chunk in gpu_slices.chunks(slices_at_a_time) {
                    let (scalar_types, chunk_scalars): (Vec<_>, Vec<&[Self::ScalarField]>) =
                        work_chunk
                            .par_iter()
                            .map(|(_, msm_type, scalars)| (*msm_type, *scalars))
                            .unzip();

                    let max_scalar_type = scalar_types.par_iter().max().unwrap();
                    let batch_results =
                        icicle_batch_msm::<Self>(gpu_bases, &chunk_scalars, *max_scalar_type);

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

fn use_icicle(additional_conditions: Option<bool>) -> bool {
    let additional = additional_conditions.unwrap_or(true);
    icicle_init() && additional
}

fn map_field_elements_to_u64<V: VariableBaseMSM>(field_elements: &[V::ScalarField]) -> Vec<u64> {
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
#[tracing::instrument(skip_all, name = "msm_bigint_wnaf")]
fn msm_bigint_wnaf<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[<V::ScalarField as PrimeField>::BigInt],
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
fn msm_bigint<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[<V::ScalarField as PrimeField>::BigInt],
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

// Compute msm using windowed non-adjacent form
#[tracing::instrument(skip_all, name = "msm_u64_wnaf")]
fn msm_u64_wnaf<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[u64],
    max_num_bits: usize,
) -> V {
    let c = if bases.len() < 32 {
        3
    } else {
        ln_without_floats(bases.len()) + 2
    };

    let digits_count = max_num_bits.div_ceil(c);
    let scalar_digits = scalars
        .into_par_iter()
        .flat_map_iter(|s| make_digits_u64(*s, c, max_num_bits))
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
            buckets.iter().rev().for_each(|b| {
                running_sum += b;
                res += &running_sum;
            });
            res
        })
        .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    let result = lowest
        + window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            });
    result
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_u64<V: VariableBaseMSM>(bases: &[V::MulBase], scalars: &[u64], max_num_bits: usize) -> V {
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

#[tracing::instrument(skip_all, name = "msm_binary")]
fn msm_binary<V: VariableBaseMSM>(bases: &[V::MulBase], scalars: &[u64]) -> V {
    scalars
        .iter()
        .zip(bases)
        .filter(|(&scalar, _base)| scalar != 0)
        .map(|(_scalar, base)| base)
        .fold(V::zero(), |sum, base| sum + base)
}

#[tracing::instrument(skip_all, name = "msm_small")]
fn msm_small<V: VariableBaseMSM>(bases: &[V::MulBase], scalars: &[u64], max_num_bits: usize) -> V {
    let num_buckets: usize = 1 << max_num_bits;
    // Assign things to buckets based on the scalar
    let mut buckets: Vec<V> = vec![V::zero(); num_buckets];
    scalars
        .iter()
        .zip(bases)
        .filter(|(&scalar, _base)| scalar != 0)
        .for_each(|(&scalar, base)| {
            buckets[scalar as usize] += base;
        });

    let mut result = V::zero();
    let mut running_sum = V::zero();
    buckets.iter().skip(1).rev().for_each(|bucket| {
        running_sum += bucket;
        result += running_sum;
    });
    result
}

fn make_digits_u64(scalar: u64, w: usize, num_bits: usize) -> impl Iterator<Item = i64> {
    let radix: u64 = 1 << w;
    let window_mask: u64 = radix - 1;
    let mut carry = 0u64;

    let digits_count = num_bits.div_ceil(w);
    (0..digits_count).map(move |i| {
        // Construct a buffer of bits of the scalar, starting at `bit_offset`.
        let bit_offset = i * w;
        let bit_idx = bit_offset % 64;
        // Read the bits from the scalar
        let bit_buf = scalar >> bit_idx;
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

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]: https://github.com/scipr-lab/zexe/issues/79#issue-556220473
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

#[cfg(test)]
mod tests {
    use crate::msm::MsmType;

    #[test]
    fn test_msm_type_conversion() {
        let msm_type = MsmType::from_u32(0);
        assert_eq!(msm_type, MsmType::Zero);
        assert_eq!(msm_type.num_bits(), 0);

        let msm_type = MsmType::from_u32(1);
        assert_eq!(msm_type, MsmType::One);
        assert_eq!(msm_type.num_bits(), 1);

        let msm_type = MsmType::from_u32(2);
        assert_eq!(msm_type, MsmType::Small(2));
        assert_eq!(msm_type.num_bits(), 2);

        let msm_type = MsmType::from_u32(11);
        assert_eq!(msm_type, MsmType::Medium(11));
        assert_eq!(msm_type.num_bits(), 11);

        let msm_type = MsmType::from_u32(65);
        assert_eq!(msm_type, MsmType::Large(65));
        assert_eq!(msm_type.num_bits(), 65);
    }
}
