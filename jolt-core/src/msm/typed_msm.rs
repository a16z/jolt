// Ported from a16z/arkworks-algebra fork (dev/twist-shout branch).
// Typed MSM: windowed multi-scalar multiplication specialized for small scalar types.
//
// Original code dual-licensed under Apache-2.0 and MIT.

use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
use ark_ff::biginteger::{S128, S64};
use ark_std::{iterable::Iterable, vec, vec::Vec};
use itertools::{Either, Itertools};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2) ≈ log2(a) * 0.69
    (ark_std::log2(a) * 69 / 100) as usize
}

fn preamble<A, B>(bases: &mut &[A], scalars: &mut &[B], serial: bool) -> Option<usize> {
    let size = bases.len().min(scalars.len());
    if size == 0 {
        return None;
    }
    let chunk_size = {
        let chunk_size = size / rayon::current_num_threads();
        if serial || chunk_size == 0 {
            size
        } else {
            chunk_size
        }
    };

    *bases = &bases[..size];
    *scalars = &scalars[..size];
    Some(chunk_size)
}

// ---------------------------------------------------------------------------
// Core windowed MSM for typed (Into<u128>) scalars
// ---------------------------------------------------------------------------

fn msm_serial_inner<'a, V: VariableBaseMSM, S: Into<u128> + Copy + Send + Sync + 'a>(
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
        + window_sums[1..]
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

pub fn msm_binary<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[bool],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| {
            let mut res = V::ZERO_BUCKET;
            for (base, _) in bases.iter().zip(scalars).filter(|(_, &s)| s) {
                res += base;
            }
            res.into()
        })
        .sum()
}

pub fn msm_u8<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u8],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u16<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u16],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u32<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[u32],
    serial: bool,
) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars, serial) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u64<V: VariableBaseMSM>(
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
    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

pub fn msm_u128<V: VariableBaseMSM>(
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
    bases
        .par_chunks(chunk_size)
        .zip(scalars.par_chunks(chunk_size))
        .map(|(bases, scalars)| msm_serial_inner::<V, _>(bases, scalars))
        .sum()
}

// ---------------------------------------------------------------------------
// Signed scalar MSMs
// ---------------------------------------------------------------------------

pub fn msm_i64<V: VariableBaseMSM>(
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

    let non_negative_msm: V = non_negative_bases
        .par_chunks(chunk_size)
        .zip(non_negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = negative_bases
        .par_chunks(chunk_size)
        .zip(negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

pub fn msm_i128<V: VariableBaseMSM>(
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

    let non_negative_msm: V = non_negative_bases
        .par_chunks(chunk_size)
        .zip(non_negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = negative_bases
        .par_chunks(chunk_size)
        .zip(negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

// ---------------------------------------------------------------------------
// SignedBigInt MSMs
// ---------------------------------------------------------------------------

pub fn msm_s64<V: VariableBaseMSM>(
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

    let non_negative_msm: V = non_negative_bases
        .par_chunks(chunk_size)
        .zip(non_negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = negative_bases
        .par_chunks(chunk_size)
        .zip(negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}

pub fn msm_s128<V: VariableBaseMSM>(
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

    let non_negative_msm: V = non_negative_bases
        .par_chunks(chunk_size)
        .zip(non_negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    let negative_msm: V = negative_bases
        .par_chunks(chunk_size)
        .zip(negative_scalars.par_chunks(chunk_size))
        .map(|(b, s)| msm_serial_inner::<V, _>(b, s))
        .sum();
    non_negative_msm - negative_msm
}
