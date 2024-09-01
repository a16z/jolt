use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{prelude::*, PrimeField};
use ark_std::cmp::Ordering;
use ark_std::vec::Vec;
use rayon::prelude::*;

pub(crate) mod icicle;
pub use icicle::*;

impl<G: CurveGroup + Icicle> VariableBaseMSM for G {}

/// Copy of ark_ec::VariableBaseMSM with minor modifications to speed up
/// known small element sized MSMs.
pub trait VariableBaseMSM: ScalarMul + Icicle {
    fn msm(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Result<Self, usize> {
        (bases.len() == scalars.len())
            .then(|| {
                let max_num_bits = scalars
                    .par_iter()
                    .map(|s| s.into_bigint().num_bits())
                    .max()
                    .unwrap();

                match max_num_bits {
                    0 => Self::zero(),
                    1 => {
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        msm_binary(bases, scalars_u64)
                    }
                    2..=10 => {
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        msm_small(bases, scalars_u64, max_num_bits as usize)
                    }
                    #[cfg(not(feature = "icicle"))]
                    11..=64 => {
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        if Self::NEGATION_IS_CHEAP {
                            msm_u64_wnaf(bases, scalars_u64, max_num_bits as usize)
                        } else {
                            msm_u64(bases, scalars_u64, max_num_bits as usize)
                        }
                    }
                    _ => {
                        #[cfg(feature = "icicle")]
                        {
                            let gpu_bases = bases.par_iter().map(|base| <Self as Icicle>::from_ark_affine(base)).collect::<Vec<_>>();
                            icicle_msm::<Self>(&gpu_bases, scalars)
                        }

                        #[cfg(not(feature = "icicle"))]
                        {
                            let scalars = scalars
                                .par_iter()
                                .map(|s| s.into_bigint())
                                .collect::<Vec<_>>();
                            if Self::NEGATION_IS_CHEAP {
                                msm_bigint_wnaf(bases, &scalars, max_num_bits as usize)
                            } else {
                                msm_bigint(bases, &scalars, max_num_bits as usize)
                            }
                        }
                    }
                }
            })
            .ok_or_else(|| bases.len().min(scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn batch_msm(bases: &[Self::MulBase], scalars: &[&[Self::ScalarField]]) -> Vec<Self> {
        assert!(scalars.iter().all(|s| s.len() == scalars[0].len()));
        assert_eq!(bases.len(), scalars[0].len());

        #[cfg(feature = "icicle")]
        let gpu_bases = bases.par_iter().map(|base| <Self as Icicle>::from_ark_affine(base)).collect::<Vec<_>>();

        let slice_bit_size = 256 * scalars[0].len() * 3; 
        let max_gpu_memory_gb = 5;
        let max_gpu_memory_bits = max_gpu_memory_gb * 1024 * 1024 * 1024 * 8; 
        let slices_at_a_time = max_gpu_memory_bits / slice_bit_size;

        #[derive(Debug, Clone, Copy)]
        enum MsmType {
            Zero,
            One,
            Small,
            Medium,
            Large,
        }

        let mut telemetry = Vec::new();

        for (i, scalar_slice) in scalars.iter().enumerate() {
            let max_num_bits = scalar_slice
                .par_iter()
                .map(|s| s.into_bigint().num_bits())
                .max()
                .unwrap();

            let msm_type = match max_num_bits {
                0 => MsmType::Zero,
                1 => MsmType::One,
                2..=10 => MsmType::Small,
                11..=64 => MsmType::Medium,
                _ => MsmType::Large,
            };

            telemetry.push((i, msm_type));
        }

        let mut results = vec![Self::zero(); scalars.len()];

        let run_msm = |indices: Vec<usize>, msm_type: MsmType, results: &mut Vec<Self>| {
            let partial_results: Vec<(usize, Self)> = match msm_type {
                MsmType::Zero => indices.into_par_iter().map(|i| (i, Self::zero())).collect(),
                MsmType::One => {
                    indices.into_par_iter().map(|i| {
                        let scalars = scalars[i];
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        (i, msm_binary(bases, scalars_u64))
                    }).collect()
                }
                MsmType::Small => {
                    indices.into_par_iter().map(|i| {
                        let scalars = scalars[i];
                        let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                        (i, msm_small(bases, scalars_u64, 10))
                    }).collect()
                }
                MsmType::Medium => {
                    #[cfg(feature = "icicle")]
                    {
                        
                        let scalar_batches: Vec<&[Self::ScalarField]> = indices.iter().map(|i| {
                            scalars[*i]
                        }).collect();

                        let batch_results = icicle_batch_msm::<Self>(&gpu_bases, &scalar_batches, 64);
                        batch_results.into_iter().enumerate().map(|(batch_index, result)| (indices[batch_index], result)).collect()
                    }

                    #[cfg(not(feature = "icicle"))]
                    {
                        indices.into_par_iter().map(|i| {
                            let scalars = scalars[i];
                            let scalars_u64 = &map_field_elements_to_u64::<Self>(scalars);
                            let result = if Self::NEGATION_IS_CHEAP {
                                msm_u64_wnaf(bases, scalars_u64, 64)
                            } else {
                                msm_u64(bases, scalars_u64, 64)
                            };
                            (i, result)
                        }).collect()
                    }
                }
                MsmType::Large => {
                    #[cfg(feature = "icicle")]
                    {
                        
                        let scalar_batches: Vec<&[Self::ScalarField]> = indices.iter().map(|i| {
                            scalars[*i]
                        }).collect();

                        let batch_results = icicle_batch_msm::<Self>(&gpu_bases, &scalar_batches, 256);
                        batch_results.into_iter().enumerate().map(|(batch_index, result)| (indices[batch_index], result)).collect()
                    }

                    #[cfg(not(feature = "icicle"))]
                    {
                        indices.into_par_iter().map(|i| {
                            let scalars = scalars[i];
                            let scalars = scalars
                                .par_iter()
                                .map(|s| s.into_bigint())
                                .collect::<Vec<_>>();
                            let result = if Self::NEGATION_IS_CHEAP {
                                msm_bigint_wnaf(bases, &scalars, 256)
                            } else {
                                msm_bigint(bases, &scalars, 256)
                            };
                            (i, result)
                        }).collect()
                    }
                }
            };

            for (i, result) in partial_results {
                results[i] = result;
            }
        };

        let mut zero_indices = Vec::new();
        let mut one_indices = Vec::new();
        let mut small_indices = Vec::new();
        let mut medium_indices = Vec::new();
        let mut large_indices = Vec::new();

        for (i, msm_type) in telemetry {
            match msm_type {
                MsmType::Zero => zero_indices.push(i),
                MsmType::One => one_indices.push(i),
                MsmType::Small => small_indices.push(i),
                MsmType::Medium => medium_indices.push(i),
                MsmType::Large => large_indices.push(i),
            }
        }

        run_msm(zero_indices, MsmType::Zero, &mut results);
        run_msm(one_indices, MsmType::One, &mut results);
        run_msm(small_indices, MsmType::Small, &mut results);

        {
            let span = tracing::span!(tracing::Level::INFO, "process_medium_indices");
            let _guard = span.enter();
            medium_indices.chunks(slices_at_a_time).for_each(|chunk| {
                run_msm(chunk.to_vec(), MsmType::Medium, &mut results);
            });
            drop(_guard);
        }

        {
            let span = tracing::span!(tracing::Level::INFO, "process_large_indices");
            let _guard = span.enter();
            large_indices.chunks(slices_at_a_time).for_each(|chunk| {
                run_msm(chunk.to_vec(), MsmType::Large, &mut results);
            });
            drop(_guard);
        }

        results
    }
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
// #[cfg(not(feature = "icicle"))]
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
    let digits_count = (num_bits + c - 1) / c;
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
// #[cfg(not(feature = "icicle"))]
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
// #[cfg(not(feature = "icicle"))]
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
    let digits_count = (num_bits + w - 1) / w;
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

    let digits_count = (max_num_bits + c - 1) / c;
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

    let digits_count = (num_bits + w - 1) / w;
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
