use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{prelude::*, PrimeField};
use ark_std::cmp::Ordering;
use ark_std::vec::Vec;
#[cfg(feature = "icicle")]
use icicle_core::curve::Affine;
use num_integer::Integer;
use rayon::prelude::*;
use std::borrow::Borrow;

pub(crate) mod icicle;
pub(crate) mod arkmsm;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
pub use icicle::*;
pub use arkmsm::*;

impl<F: JoltField, G: CurveGroup<ScalarField = F> + Icicle> VariableBaseMSM for G {}

#[cfg(feature = "icicle")]
pub type GpuBaseType<G: Icicle> = Affine<G::C>;
#[cfg(not(feature = "icicle"))]
pub type GpuBaseType<G: ScalarMul> = G::MulBase;

use crate::poly::unipoly::UniPoly;
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
                        // Check if we should use arkmsm optimizations
                        #[cfg(feature = "arkmsm")]
                        if use_arkmsm() {
                            let scalars = scalars
                                .par_iter()
                                .map(|s| s.into_bigint())
                                .collect::<Vec<_>>();
                            
                            // Use our arkmsm optimized implementation
                            return arkmsm_msm::<Self::ScalarField, Self>(bases, &scalars, max_num_bits);
                        } 
                        
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
    fn batch_msm_common<P>(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        polys: &[P],
        variable_batches: bool,
    ) -> Vec<Self>
    where
        P: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        // Validate input lengths
        if variable_batches {
            assert!(polys.par_iter().all(|s| s.borrow().len() <= bases.len()));
        } else {
            assert!(polys.par_iter().all(|s| s.borrow().len() == bases.len()));
            assert_eq!(bases.len(), gpu_bases.map_or(bases.len(), |b| b.len()));
        }

        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());

        let use_icicle = use_icicle();

        // Handle CPU-only case
        if !use_icicle {
            let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu_only");
            let _guard = span.enter();
            return polys
                .into_par_iter()
                .map(|poly| {
                    let poly = poly.borrow();
                    let bases_slice = &bases[..poly.len()];
                    Self::msm(bases_slice, None, poly, None).unwrap()
                })
                .collect();
        }

        // Split scalar batches into CPU and GPU workloads
        let span = tracing::span!(tracing::Level::INFO, "group_scalar_indices_parallel");
        let _guard = span.enter();
        let (cpu_batch, gpu_batch): (Vec<_>, Vec<_>) =
            polys.par_iter().enumerate().partition_map(|(i, poly)| {
                let poly = poly.borrow();
                let max_num_bits = poly.max_num_bits();

                if max_num_bits > 10 {
                    match poly {
                        MultilinearPolynomial::LargeScalars(poly) => {
                            Either::Right((i, max_num_bits, poly.evals()))
                        }
                        MultilinearPolynomial::U16Scalars(poly) => {
                            Either::Right((i, max_num_bits, poly.coeffs_as_field_elements()))
                        }
                        MultilinearPolynomial::U32Scalars(poly) => {
                            Either::Right((i, max_num_bits, poly.coeffs_as_field_elements()))
                        }
                        MultilinearPolynomial::U64Scalars(poly) => {
                            Either::Right((i, max_num_bits, poly.coeffs_as_field_elements()))
                        }
                        MultilinearPolynomial::I64Scalars(poly) => {
                            Either::Right((i, max_num_bits, poly.coeffs_as_field_elements()))
                        }
                        MultilinearPolynomial::U8Scalars(_) => unreachable!(
                            "MultilinearPolynomial::U8Scalars cannot have more than 10 bits"
                        ),
                    }
                } else {
                    Either::Left((i, max_num_bits, poly))
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
                let bases_slice = &bases[..poly.len()];
                (
                    i,
                    Self::msm(bases_slice, None, poly, Some(max_num_bits)).unwrap(),
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

                if variable_batches {
                    // Variable-length batch processing
                    let batch = gpu_batch
                        .iter()
                        .map(|(i, max_num_bits, poly)| (*i, *max_num_bits, poly.as_slice()))
                        .collect::<Vec<_>>();
                    let batched_results = icicle_variable_batch_msm(gpu_bases, &batch);
                    for (index, result) in batched_results {
                        results[index] = result;
                    }
                } else {
                    // Fixed-length batch processing
                    let slice_bit_size = 256 * gpu_batch[0].2.len() * 2;
                    let slices_at_a_time = total_memory_bits() / slice_bit_size;

                    for work_chunk in gpu_batch.chunks(slices_at_a_time) {
                        let (max_num_bits, chunk_polys): (Vec<_>, Vec<_>) = work_chunk
                            .par_iter()
                            .map(|(_, max_num_bits, poly)| (*max_num_bits, poly.as_slice()))
                            .unzip();

                        let max_num_bits = max_num_bits.iter().max().unwrap();
                        let batch_results =
                            icicle_batch_msm(gpu_bases, &chunk_polys, *max_num_bits);

                        for ((index, _, _), result) in work_chunk.iter().zip(batch_results) {
                            results[*index] = result;
                        }
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

    #[tracing::instrument(skip_all)]
    fn batch_msm<P>(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        polys: &[P],
    ) -> Vec<Self>
    where
        P: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        Self::batch_msm_common(bases, gpu_bases, polys, false)
    }

    // a "batch" msm that can handle scalars of different sizes
    // it mostly amortizes copy costs of sending the generators to the GPU
    #[tracing::instrument(skip_all)]
    fn variable_batch_msm<P>(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        polys: &[P],
    ) -> Vec<Self>
    where
        P: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        Self::batch_msm_common(bases, gpu_bases, polys, true)
    }

    #[tracing::instrument(skip_all)]
    fn variable_batch_msm_univariate<P>(
        bases: &[Self::MulBase],
        gpu_bases: Option<&[GpuBaseType<Self>]>,
        polys: &[P],
    ) -> Vec<Self>
    where
        P: Borrow<UniPoly<Self::ScalarField>> + Sync,
    {
        assert!(polys
            .par_iter()
            .all(|s| s.borrow().coeffs.len() <= bases.len()));
        #[cfg(not(feature = "icicle"))]
        assert!(gpu_bases.is_none());

        let use_icicle = use_icicle();

        if !use_icicle {
            let span = tracing::span!(tracing::Level::INFO, "batch_msm_cpu_only");
            let _guard = span.enter();
            return polys
                .into_par_iter()
                .map(|poly| {
                    Self::msm_field_elements(
                        &bases[..poly.borrow().coeffs.len()],
                        None,
                        &poly.borrow().coeffs,
                        None,
                        false,
                    )
                    .unwrap()
                })
                .collect();
        }

        // Split scalar batches into CPU and GPU workloads
        let span = tracing::span!(tracing::Level::INFO, "group_scalar_indices_parallel");
        let _guard = span.enter();
        let (cpu_batch, gpu_batch): (Vec<_>, Vec<_>) =
            polys.par_iter().enumerate().partition_map(|(i, poly)| {
                let poly = poly.borrow();
                let max_num_bits = (*poly.coeffs.iter().max().unwrap()).num_bits() as usize;
                if use_icicle && max_num_bits > 10 {
                    Either::Right((i, max_num_bits, poly.coeffs.as_slice()))
                } else {
                    Either::Left((i, max_num_bits, poly))
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
                    Self::msm_field_elements(
                        &bases[..poly.borrow().coeffs.len()],
                        None,
                        &poly.borrow().coeffs,
                        Some(max_num_bits),
                        false,
                    )
                    .unwrap(),
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

                let batched_results = icicle_variable_batch_msm(gpu_bases, &gpu_batch);
                for (index, result) in batched_results {
                    results[index] = result;
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
    // Check if we should use arkmsm optimizations directly
    #[cfg(feature = "arkmsm")]
    if use_arkmsm() {
        return arkmsm_msm::<F, V>(bases, scalars, max_num_bits);
    }

    // Determine the window size
    let c = if scalars.len() < 32 {
        3
    } else {
        ln_without_floats(scalars.len()) + 2
    };

    let num_bits = max_num_bits;
    let mut max_bits = 0usize;
    let mut min_bits = max_num_bits;
    // Find the maximum and minimum number of bits in the scalars
    for s in scalars {
        let leading_bits = s.num_bits() as usize;
        let bits = if leading_bits < num_bits { leading_bits } else { num_bits };
        max_bits = std::cmp::max(max_bits, bits);
        min_bits = std::cmp::min(min_bits, bits);
    }

    // Optimization: Use special case for small window sizes
    if max_bits <= 10 {
        let scalars_u16 = scalars
            .par_iter()
            .map(|s| {
                let limbs: &[u64] = s.as_ref();
                limbs[0] as u16
            })
            .collect::<Vec<_>>();
        return msm_small(bases, &scalars_u16, max_bits);
    }

    let mut window_sums = Vec::new();
    window_sums.push(V::zero());

    // Actually the wnaf means all scalars are guaranteed to use the same bits
    for m in 0..=max_bits {
        let mut running_sum = V::zero();
        for (s, g) in scalars.iter().zip(bases.iter()) {
            if s.get_bit(m) {
                running_sum += g;
            }
        }
        window_sums.push(running_sum);
    }

    // Compute the actual window size based on the bit length
    let window_size = std::cmp::min(c, ln_without_floats(max_bits));
    
    // Use the signed bucket indexes optimization from arkmsm
    let bucket_max = (1 << window_size) / 2;
    let w = window_size as usize;
    let scalar_digits = scalars
        .par_iter()
        .map(|s| {
            let mut digits = Vec::with_capacity((max_bits + w - 1) / w);
            
            // Process each window
            for i in 0..((max_bits + w - 1) / w) {
                let mut digit = 0i64;
                let window_start = i * w;
                let window_end = std::cmp::min((i + 1) * w, max_bits);
                
                // Extract the digit in this window
                for j in window_start..window_end {
                    if s.get_bit(j) {
                        digit |= 1i64 << (j - window_start);
                    }
                }
                
                // Convert to signed digits if the digit is too large
                if digit > bucket_max as i64 {
                    digit = digit - (1i64 << w);
                }
                
                digits.push(digit);
            }
            
            digits
        })
        .collect::<Vec<_>>();

    // Rest of the implementation remains the same but uses the signed digits
    let mut res = V::zero();
    let mut running_sum = V::zero();

    for j in (0..((max_bits + w - 1) / w)).rev() {
        let mut tmp = V::zero();
        for _ in 0..w {
            tmp = tmp.double();
        }

        for (idx, s_digits) in scalar_digits.iter().enumerate() {
            if j < s_digits.len() {
                let digit = s_digits[j];
                if digit != 0 {
                    let base = &bases[idx];
                    if digit > 0 {
                        running_sum = running_sum + base;
                    } else {
                        running_sum = running_sum - base;
                    }
                }
            }
        }

        res = res + tmp;
        res = res + running_sum;
    }

    res
}

/// Optimized implementation of multi-scalar multiplication.
fn msm_bigint<F: JoltField + PrimeField, V: VariableBaseMSM<ScalarField = F>>(
    bases: &[V::MulBase],
    scalars: &[<F as PrimeField>::BigInt],
    max_num_bits: usize,
) -> V {
    // Check if we should use arkmsm optimizations directly
    #[cfg(feature = "arkmsm")]
    if use_arkmsm() {
        return arkmsm_msm::<F, V>(bases, scalars, max_num_bits);
    }

    let c = if scalars.len() < 32 {
        3
    } else {
        ln_without_floats(scalars.len()) + 2
    };

    let num_bits = max_num_bits;
    let mut max_bits = 0usize;
    for s in scalars {
        max_bits = std::cmp::max(max_bits, s.num_bits() as usize);
    }
    
    // Optimization: Use special case for small window sizes
    if max_bits <= 10 {
        let scalars_u16 = scalars
            .par_iter()
            .map(|s| {
                let limbs: &[u64] = s.as_ref();
                limbs[0] as u16
            })
            .collect::<Vec<_>>();
        return msm_small(bases, &scalars_u16, max_bits);
    }

    let window_size = c;
    let bucket_max = 1 << window_size;
    
    // Compute the scalar digits for all scalars
    let num_windows = (max_bits + window_size - 1) / window_size;
    let scalar_digits = scalars
        .par_iter()
        .map(|s| {
            let mut digits = Vec::with_capacity(num_windows);
            
            for i in 0..num_windows {
                let mut digit = 0u64;
                let window_start = i * window_size;
                let window_end = std::cmp::min((i + 1) * window_size, max_bits);
                
                for j in window_start..window_end {
                    if s.get_bit(j) {
                        digit |= 1u64 << (j - window_start);
                    }
                }
                
                digits.push(digit);
            }
            
            digits
        })
        .collect::<Vec<_>>();
    
    let mut result = V::zero();
    
    // Process windows from highest to lowest
    for w in (0..num_windows).rev() {
        // Double result 'window_size' times for each window except the highest
        if w != num_windows - 1 {
            for _ in 0..window_size {
                result = result.double();
            }
        }
        
        // Batch accumulation optimization: group points by bucket
        let mut buckets = vec![V::zero(); bucket_max];
        
        // Group scalar digits by bucket
        for (scalar_idx, s_digits) in scalar_digits.iter().enumerate() {
            if w < s_digits.len() {
                let digit = s_digits[w] as usize;
                if digit > 0 {
                    let base = &bases[scalar_idx];
                    buckets[digit] = buckets[digit] + base;
                }
            }
        }
        
        // Batch reduction optimization: use an efficient reduction pattern
        let mut running_sum = V::zero();
        
        // Process buckets in reverse order for efficiency
        for i in (1..bucket_max).rev() {
            running_sum = running_sum + buckets[i];
            result = result + running_sum;
        }
    }
    
    result
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
