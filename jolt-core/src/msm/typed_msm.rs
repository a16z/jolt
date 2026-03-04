// Ported from a16z/arkworks-algebra fork (dev/twist-shout branch).
// Typed MSM: windowed multi-scalar multiplication specialized for small scalar types.
//
// Original code dual-licensed under Apache-2.0 and MIT.

use super::bucket::Bucket;
use ark_ec::models::short_weierstrass::{Projective, SWCurveConfig};
use ark_ec::ScalarMul;
use ark_ff::biginteger::BigInteger;
use ark_ff::PrimeField;
use ark_std::{cfg_chunks, cfg_iter, cfg_into_iter, iterable::Iterable, vec, vec::Vec};
use itertools::{Either, Itertools};
use jolt_field::signed::{S128, S64};
use std::ops::AddAssign;

/// Trait providing the `Bucket` accumulator type for typed MSM.
///
/// Implemented for `Projective<P>` using extended Jacobian [`Bucket<P>`].
/// The typed MSM free functions in this module are generic over this trait.
pub trait BucketMSM: ScalarMul + for<'a> AddAssign<&'a <Self as BucketMSM>::Bucket> {
    type Bucket: Default
        + Copy
        + Clone
        + for<'a> AddAssign<&'a Self::Bucket>
        + for<'a> std::ops::SubAssign<&'a Self::Bucket>
        + AddAssign<Self::MulBase>
        + std::ops::SubAssign<Self::MulBase>
        + for<'a> AddAssign<&'a Self::MulBase>
        + for<'a> std::ops::SubAssign<&'a Self::MulBase>
        + Send
        + Sync
        + Into<Self>;

    const ZERO_BUCKET: Self::Bucket;
}

impl<P: SWCurveConfig> BucketMSM for Projective<P> {
    type Bucket = Bucket<P>;
    const ZERO_BUCKET: Bucket<P> = Bucket::ZERO;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2) ≈ log2(a) * 0.69
    (ark_std::log2(a) * 69 / 100) as usize
}

fn preamble<A, B>(bases: &mut &[A], scalars: &mut &[B], _serial: bool) -> Option<usize> {
    let size = bases.len().min(scalars.len());
    if size == 0 {
        return None;
    }
    #[cfg(feature = "rayon")]
    let chunk_size = {
        let chunk_size = size / rayon::current_num_threads();
        if _serial || chunk_size == 0 {
            size
        } else {
            chunk_size
        }
    };
    #[cfg(not(feature = "rayon"))]
    let chunk_size = size;

    *bases = &bases[..size];
    *scalars = &scalars[..size];
    Some(chunk_size)
}

// ---------------------------------------------------------------------------
// Core windowed MSM for typed (Into<u128>) scalars
// ---------------------------------------------------------------------------

fn msm_serial_inner<'a, V: BucketMSM, S: Into<u128> + Copy + Send + Sync + 'a>(
    bases: impl Iterable<Item = &'a V::MulBase>,
    scalars: impl Iterable<Item = &'a S>,
) -> V {
    let c = if bases.len() < 32 {
        3
    } else {
        ln_without_floats(bases.len()) + 2
    };

    let zero = V::ZERO_BUCKET;
    let two_to_c = 1 << c;

    let window_sums: Vec<_> = (0..(core::mem::size_of::<S>() * 8))
        .step_by(c)
        .map(|w_start| {
            let mut res = zero;
            let mut buckets = vec![zero; two_to_c - 1];

            scalars
                .iter()
                .zip(bases.iter())
                .filter_map(|(&s, b)| {
                    let s = s.into();
                    (s != 0).then_some((s, b))
                })
                .for_each(|(scalar, base)| {
                    if scalar == 1 {
                        if w_start == 0 {
                            res += base;
                        }
                    } else {
                        let mut scalar = scalar;
                        scalar >>= w_start as u32;
                        scalar %= two_to_c as u128;

                        if scalar != 0 {
                            buckets[(scalar - 1) as usize] += base;
                        }
                    }
                });

            let mut running_sum = V::ZERO_BUCKET;
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    let lowest: V = window_sums
        .first()
        .copied()
        .map_or_else(V::zero, Into::into);

    lowest
        + &window_sums[1..]
            .iter()
            .rev()
            .fold(V::zero(), |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

// ---------------------------------------------------------------------------
// Public typed MSM functions
// ---------------------------------------------------------------------------

pub fn msm_binary<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[bool],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| {
            let mut res = V::ZERO_BUCKET;
            for (base, _) in bases.iter().zip(scalars).filter(|(_, &s)| s) {
                res += base;
            }
            res.into()
        })
        .sum()
}

pub fn msm_u8<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u8],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u16<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u16],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u32<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u32],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u64<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u64],
    serial: bool,
) -> V {
    if serial {
        return msm_serial_inner::<V, _>(bases, scalars);
    }
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u128<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u128],
    serial: bool,
) -> V {
    if serial {
        return msm_serial_inner::<V, _>(bases, scalars);
    }
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

// ---------------------------------------------------------------------------
// Signed scalar MSMs
// ---------------------------------------------------------------------------

pub fn msm_i64<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[i64],
    serial: bool,
) -> V {
    let (negative_bases, non_negative_bases): (Vec<V::MulBase>, Vec<V::MulBase>) =
        bases.iter().enumerate().partition_map(|(i, b)| {
            if scalars[i].is_negative() {
                Either::Left(*b)
            } else {
                Either::Right(*b)
            }
        });
    let (negative_scalars, non_negative_scalars): (Vec<u64>, Vec<u64>) =
        scalars.iter().partition_map(|s| {
            if s.is_negative() {
                Either::Left(s.unsigned_abs())
            } else {
                Either::Right(s.unsigned_abs())
            }
        });

    if serial {
        return msm_serial_inner::<V, _>(&non_negative_bases, &non_negative_scalars)
            - msm_serial_inner::<V, _>(&negative_bases, &negative_scalars);
    }

    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    let non_negative_msm: V = cfg_chunks!(non_negative_bases, chunk_size)
        .zip(cfg_chunks!(non_negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = cfg_chunks!(negative_bases, chunk_size)
        .zip(cfg_chunks!(negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

pub fn msm_i128<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[i128],
    serial: bool,
) -> V {
    let (negative_bases, non_negative_bases): (Vec<V::MulBase>, Vec<V::MulBase>) =
        bases.iter().enumerate().partition_map(|(i, b)| {
            if scalars[i].is_negative() {
                Either::Left(*b)
            } else {
                Either::Right(*b)
            }
        });
    let (negative_scalars, non_negative_scalars): (Vec<u64>, Vec<u64>) =
        scalars.iter().partition_map(|s| {
            let absolute_val = s.unsigned_abs();
            debug_assert!(
                absolute_val <= u64::MAX as u128,
                "msm_i128 only supports scalars in the range [-u64::MAX, u64::MAX]"
            );
            if s.is_negative() {
                Either::Left(absolute_val as u64)
            } else {
                Either::Right(absolute_val as u64)
            }
        });

    if serial {
        return msm_serial_inner::<V, _>(&non_negative_bases, &non_negative_scalars)
            - msm_serial_inner::<V, _>(&negative_bases, &negative_scalars);
    }

    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    let non_negative_msm: V = cfg_chunks!(non_negative_bases, chunk_size)
        .zip(cfg_chunks!(non_negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = cfg_chunks!(negative_bases, chunk_size)
        .zip(cfg_chunks!(negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

// ---------------------------------------------------------------------------
// SignedBigInt MSMs
// ---------------------------------------------------------------------------

pub fn msm_s64<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[S64],
    serial: bool,
) -> V {
    let (negative_bases, non_negative_bases): (Vec<V::MulBase>, Vec<V::MulBase>) =
        bases.iter().enumerate().partition_map(|(i, b)| {
            if !scalars[i].sign() {
                Either::Left(*b)
            } else {
                Either::Right(*b)
            }
        });
    let (negative_scalars, non_negative_scalars): (Vec<u64>, Vec<u64>) =
        scalars.iter().partition_map(|s| {
            let mag = s.magnitude_as_u64();
            if !s.sign() {
                Either::Left(mag)
            } else {
                Either::Right(mag)
            }
        });

    if serial {
        return msm_serial_inner::<V, _>(&non_negative_bases, &non_negative_scalars)
            - msm_serial_inner::<V, _>(&negative_bases, &negative_scalars);
    }

    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    let non_negative_msm: V = cfg_chunks!(non_negative_bases, chunk_size)
        .zip(cfg_chunks!(non_negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = cfg_chunks!(negative_bases, chunk_size)
        .zip(cfg_chunks!(negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

pub fn msm_s128<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[S128],
    serial: bool,
) -> V {
    let (negative_bases, non_negative_bases): (Vec<V::MulBase>, Vec<V::MulBase>) =
        bases.iter().enumerate().partition_map(|(i, b)| {
            if !scalars[i].sign() {
                Either::Left(*b)
            } else {
                Either::Right(*b)
            }
        });
    let (negative_scalars, non_negative_scalars): (Vec<u128>, Vec<u128>) =
        scalars.iter().partition_map(|s| {
            let mag = s.magnitude_as_u128();
            if !s.sign() {
                Either::Left(mag)
            } else {
                Either::Right(mag)
            }
        });

    if serial {
        return msm_serial_inner::<V, _>(&non_negative_bases, &non_negative_scalars)
            - msm_serial_inner::<V, _>(&negative_bases, &negative_scalars);
    }

    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    let non_negative_msm: V = cfg_chunks!(non_negative_bases, chunk_size)
        .zip(cfg_chunks!(non_negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = cfg_chunks!(negative_bases, chunk_size)
        .zip(cfg_chunks!(negative_scalars, chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

// ---------------------------------------------------------------------------
// Full BigInt MSM helpers (used by VariableBaseMSM::msm_bigint override)
// ---------------------------------------------------------------------------

#[inline]
fn get_group<const N: usize, A: Send + Sync, B: Send + Sync>(
    grouped: &[u64],
    f: impl Fn(usize) -> (A, B) + Send + Sync,
) -> (Vec<A>, Vec<B>) {
    let extract_index = |i| ((i << N) >> N) as usize;
    cfg_iter!(grouped)
        .map(|&i| f(extract_index(i)))
        .unzip::<_, _, Vec<_>, Vec<_>>()
}

#[inline]
fn uget_group<A: Send + Sync, B: Send + Sync>(
    grouped: &[u64],
    f: impl Fn(usize) -> (A, B) + Send + Sync,
) -> (Vec<A>, Vec<B>) {
    get_group::<3, _, _>(grouped, f)
}

#[inline]
fn iget_group<A: Send + Sync, B: Send + Sync>(
    grouped: &[u64],
    f: impl Fn(usize) -> (A, B) + Send + Sync,
) -> (Vec<A>, Vec<B>) {
    get_group::<4, _, _>(grouped, f)
}

#[inline(always)]
fn sub_scalar<B: BigInteger>(m: &B, scalar: &B) -> u64 {
    let mut negated = *m;
    negated.sub_with_borrow(scalar);
    negated.as_ref()[0]
}

/// Full-BigInt MSM that partitions scalars by magnitude and dispatches to typed MSM.
/// Used when negation is expensive (`NEGATION_IS_CHEAP == false`).
pub fn msm_unsigned<V: BucketMSM>(
    bases: &[V::MulBase],
    scalars: &[<V::ScalarField as PrimeField>::BigInt],
    serial: bool,
) -> V {
    let size = bases.len().min(scalars.len());
    let bases = &bases[..size];
    let scalars = &scalars[..size];

    let mut grouped = cfg_iter!(scalars)
        .enumerate()
        .filter(|(_, scalar)| !scalar.is_zero())
        .map(|(i, scalar)| {
            let num_bits = scalar.num_bits();
            let group = if num_bits <= 1 {
                1u8
            } else if num_bits <= 8 {
                3u8
            } else if num_bits <= 16 {
                4u8
            } else if num_bits <= 32 {
                5u8
            } else if num_bits <= 64 {
                6u8
            } else {
                7u8
            };
            (i as u64) ^ ((group as u64) << 61)
        })
        .collect::<Vec<_>>();
    let extract_group = |i: u64| (i >> 61) as u8;

    #[cfg(feature = "rayon")]
    grouped.par_sort_unstable_by_key(|i| extract_group(*i));
    #[cfg(not(feature = "rayon"))]
    grouped.sort_unstable_by_key(|i| extract_group(*i));

    let s1 = 0;
    let s3 = s1 + grouped[s1..].partition_point(|i| extract_group(*i) < 3);
    let s4 = s3 + grouped[s3..].partition_point(|i| extract_group(*i) < 4);
    let s5 = s4 + grouped[s4..].partition_point(|i| extract_group(*i) < 5);
    let s6 = s5 + grouped[s5..].partition_point(|i| extract_group(*i) < 6);
    let s7 = s6 + grouped[s6..].partition_point(|i| extract_group(*i) < 7);

    let (b1, sc1) = uget_group(&grouped[s1..s3], |i| {
        (bases[i], scalars[i].as_ref()[0] == 1)
    });
    let (b3, sc3) = uget_group(&grouped[s3..s4], |i| {
        (bases[i], scalars[i].as_ref()[0] as u8)
    });
    let (b4, sc4) = uget_group(&grouped[s4..s5], |i| {
        (bases[i], scalars[i].as_ref()[0] as u16)
    });
    let (b5, sc5) = uget_group(&grouped[s5..s6], |i| {
        (bases[i], scalars[i].as_ref()[0] as u32)
    });
    let (b6, sc6) = uget_group(&grouped[s6..s7], |i| {
        (bases[i], scalars[i].as_ref()[0] as u64)
    });
    let (b7, sc7) = uget_group(&grouped[s7..], |i| (bases[i], scalars[i]));

    let result: V = msm_binary::<V>(&b1, &sc1, serial)
        + msm_u8::<V>(&b3, &sc3, serial)
        + msm_u16::<V>(&b4, &sc4, serial)
        + msm_u32::<V>(&b5, &sc5, serial)
        + msm_u64::<V>(&b6, &sc6, serial)
        + msm_bigint::<V>(&b7, &sc7, V::ScalarField::MODULUS_BIT_SIZE as usize, serial);
    result.into()
}

/// Full-BigInt MSM that partitions scalars by sign and magnitude.
/// Used when negation is cheap (`NEGATION_IS_CHEAP == true`).
pub fn msm_signed<V: BucketMSM>(
    bases: &[V::MulBase],
    scalars: &[<V::ScalarField as PrimeField>::BigInt],
    serial: bool,
) -> V {
    let size = bases.len().min(scalars.len());
    let bases = &bases[..size];
    let scalars = &scalars[..size];

    let mut grouped = cfg_iter!(scalars)
        .enumerate()
        .filter(|(_, scalar)| !scalar.is_zero())
        .map(|(i, scalar)| {
            let mut p_minus_scalar = V::ScalarField::MODULUS;
            p_minus_scalar.sub_with_borrow(scalar);
            let num_bits = scalar.num_bits();
            let neg_num_bits = p_minus_scalar.num_bits();
            let group = if num_bits <= 1 {
                0u8
            } else if neg_num_bits <= 1 {
                1u8
            } else if num_bits <= 8 {
                2u8
            } else if neg_num_bits <= 8 {
                3u8
            } else if num_bits <= 16 {
                4u8
            } else if neg_num_bits <= 16 {
                5u8
            } else if num_bits <= 32 {
                6u8
            } else if neg_num_bits <= 32 {
                7u8
            } else if num_bits <= 64 {
                8u8
            } else if neg_num_bits <= 64 {
                9u8
            } else {
                10u8
            };
            (((i as u64) << 4) >> 4) ^ ((group as u64) << 60)
        })
        .collect::<Vec<_>>();
    let extract_group = |i: u64| (i >> 60) as u8;

    #[cfg(feature = "rayon")]
    grouped.par_sort_unstable_by_key(|i| extract_group(*i));
    #[cfg(not(feature = "rayon"))]
    grouped.sort_unstable_by_key(|i| extract_group(*i));

    let su1 = 0;
    let si1 = su1 + grouped[su1..].partition_point(|i| extract_group(*i) < 1);
    let su8 = si1 + grouped[si1..].partition_point(|i| extract_group(*i) < 2);
    let si8 = su8 + grouped[su8..].partition_point(|i| extract_group(*i) < 3);
    let su16 = si8 + grouped[si8..].partition_point(|i| extract_group(*i) < 4);
    let si16 = su16 + grouped[su16..].partition_point(|i| extract_group(*i) < 5);
    let su32 = si16 + grouped[si16..].partition_point(|i| extract_group(*i) < 6);
    let si32 = su32 + grouped[su32..].partition_point(|i| extract_group(*i) < 7);
    let su64 = si32 + grouped[si32..].partition_point(|i| extract_group(*i) < 8);
    let si64 = su64 + grouped[su64..].partition_point(|i| extract_group(*i) < 9);
    let sf = si64 + grouped[si64..].partition_point(|i| extract_group(*i) < 10);

    let m = V::ScalarField::MODULUS;
    let mut result: V;

    let (ub, us) = iget_group(&grouped[su1..si1], |i| {
        (bases[i], scalars[i].as_ref()[0] == 1)
    });
    let (ib, is) = iget_group(&grouped[si1..su8], |i| {
        (bases[i], sub_scalar(&m, &scalars[i]) == 1)
    });
    result = msm_binary::<V>(&ub, &us, serial) - msm_binary::<V>(&ib, &is, serial);

    let (ub, us) = iget_group(&grouped[su8..si8], |i| {
        (bases[i], scalars[i].as_ref()[0] as u8)
    });
    let (ib, is) = iget_group(&grouped[si8..su16], |i| {
        (bases[i], sub_scalar(&m, &scalars[i]) as u8)
    });
    result += msm_u8::<V>(&ub, &us, serial) - msm_u8::<V>(&ib, &is, serial);

    let (ub, us) = iget_group(&grouped[su16..si16], |i| {
        (bases[i], scalars[i].as_ref()[0] as u16)
    });
    let (ib, is) = iget_group(&grouped[si16..su32], |i| {
        (bases[i], sub_scalar(&m, &scalars[i]) as u16)
    });
    result += msm_u16::<V>(&ub, &us, serial) - msm_u16::<V>(&ib, &is, serial);

    let (ub, us) = iget_group(&grouped[su32..si32], |i| {
        (bases[i], scalars[i].as_ref()[0] as u32)
    });
    let (ib, is) = iget_group(&grouped[si32..su64], |i| {
        (bases[i], sub_scalar(&m, &scalars[i]) as u32)
    });
    result += msm_u32::<V>(&ub, &us, serial) - msm_u32::<V>(&ib, &is, serial);

    let (ub, us) = iget_group(&grouped[su64..si64], |i| {
        (bases[i], scalars[i].as_ref()[0])
    });
    let (ib, is) = iget_group(&grouped[si64..sf], |i| {
        (bases[i], sub_scalar(&m, &scalars[i]))
    });
    result += msm_u64::<V>(&ub, &us, serial) - msm_u64::<V>(&ib, &is, serial);

    let (bf, sf) = iget_group(&grouped[sf..], |i| (bases[i], scalars[i]));
    result += msm_bigint_wnaf::<V>(&bf, &sf);

    result.into()
}

// ---------------------------------------------------------------------------
// Windowed BigInt MSM (non-WNAF)
// ---------------------------------------------------------------------------

fn msm_bigint<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[<V::ScalarField as PrimeField>::BigInt],
    num_bits: usize,
    serial: bool,
) -> V {
    if preamble(&mut bases, &mut scalars, serial).is_none() {
        return V::zero();
    }
    let size = scalars.len();
    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

    let c = if size < 32 {
        3
    } else {
        ln_without_floats(size) + 2
    };

    let one = V::ScalarField::one().into_bigint();
    let zero = V::ZERO_BUCKET;

    let window_sums: Vec<_> = ark_std::cfg_into_iter!(0..num_bits)
        .step_by(c)
        .map(|w_start| {
            let mut res = zero;
            let mut buckets = vec![zero; (1 << c) - 1];

            scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                if scalar == one {
                    if w_start == 0 {
                        res += base;
                    }
                } else {
                    let mut scalar = scalar;
                    scalar >>= w_start as u32;
                    let scalar = scalar.as_ref()[0] % (1 << c);

                    if scalar != 0 {
                        buckets[(scalar - 1) as usize] += base;
                    }
                }
            });

            let mut running_sum = V::ZERO_BUCKET;
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    let lowest: V = window_sums
        .first()
        .copied()
        .map_or_else(V::zero, Into::into);

    lowest
        + &window_sums[1..]
            .iter()
            .rev()
            .fold(V::zero(), |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

// ---------------------------------------------------------------------------
// WNAF BigInt MSM
// ---------------------------------------------------------------------------

fn make_digits(
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
        let bit_offset = i * w;
        let u64_idx = bit_offset / 64;
        let bit_idx = bit_offset % 64;

        let bit_buf = if bit_idx < 64 - w || u64_idx == scalar.len() - 1 {
            scalar[u64_idx] >> bit_idx
        } else {
            (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
        };

        let coef = carry + (bit_buf & window_mask);
        carry = (coef + radix / 2) >> w;
        let mut digit = (coef as i64) - (carry << w) as i64;

        if i == digits_count - 1 {
            digit += (carry << w) as i64;
        }
        digit
    })
}

fn msm_bigint_wnaf_parallel<V: BucketMSM>(
    bases: &[V::MulBase],
    bigints: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = bases.len().min(bigints.len());
    let scalars = &bigints[..size];
    let bases = &bases[..size];

    let c = if size < 32 {
        3
    } else {
        ln_without_floats(size) + 2
    };

    let num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
    let digits_count = num_bits.div_ceil(c);

    #[cfg(feature = "rayon")]
    let scalar_digits = scalars
        .into_par_iter()
        .flat_map_iter(|s| make_digits(s, c, num_bits))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "rayon"))]
    let scalar_digits = scalars
        .iter()
        .flat_map(|s| make_digits(s, c, num_bits))
        .collect::<Vec<_>>();

    let zero = V::ZERO_BUCKET;
    let window_sums: Vec<_> = ark_std::cfg_into_iter!(0..digits_count)
        .map(|i| {
            let mut buckets = vec![zero; 1 << c];
            for (digits, base) in scalar_digits.chunks(digits_count).zip(bases) {
                use std::cmp::Ordering;
                let scalar = digits[i];
                match 0.cmp(&scalar) {
                    Ordering::Less => buckets[(scalar - 1) as usize] += base,
                    Ordering::Greater => buckets[(-scalar - 1) as usize] -= base,
                    Ordering::Equal => (),
                }
            }

            let mut running_sum = V::ZERO_BUCKET;
            let mut res = V::ZERO_BUCKET;
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            res
        })
        .collect();

    let lowest: V = (*window_sums.first().unwrap()).into();

    lowest
        + &window_sums[1..]
            .iter()
            .rev()
            .fold(V::zero(), |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

#[cfg(feature = "rayon")]
const THREADS_PER_CHUNK: usize = 2;

fn msm_bigint_wnaf<V: BucketMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = bases.len().min(scalars.len());
    if size == 0 {
        return V::zero();
    }

    #[cfg(feature = "rayon")]
    let chunk_size = {
        let cur_num_threads = rayon::current_num_threads();
        let num_chunks = if cur_num_threads < THREADS_PER_CHUNK {
            1
        } else {
            cur_num_threads / THREADS_PER_CHUNK
        };
        let chunk_size = size / num_chunks;
        if chunk_size == 0 { size } else { chunk_size }
    };
    #[cfg(not(feature = "rayon"))]
    let chunk_size = size;

    bases = &bases[..size];
    scalars = &scalars[..size];

    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| {
            #[cfg(feature = "rayon")]
            let result = rayon::ThreadPoolBuilder::new()
                .num_threads(THREADS_PER_CHUNK.min(rayon::current_num_threads()))
                .build()
                .unwrap()
                .install(|| msm_bigint_wnaf_parallel::<V>(bases, scalars));

            #[cfg(not(feature = "rayon"))]
            let result = msm_bigint_wnaf_parallel::<V>(bases, scalars);

            result
        })
        .sum()
}
