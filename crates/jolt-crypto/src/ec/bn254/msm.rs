use std::mem::MaybeUninit;

use ark_bn254::{g1::Config as G1Config, Fq, Fr, G1Affine, G1Projective};
use ark_ec::{models::short_weierstrass::Bucket, AdditiveGroup};
use ark_ff::{BigInteger256, Field, One, PrimeField, Zero};
use jolt_field::Fr as JoltFr;
use rayon::prelude::*;

use super::Bn254G1Affine;

const MAX_WINDOW_BITS: usize = 15;
const LARGE_MSM_WINDOW_BITS: usize = 16;
const SMALL_MSM_16_BIT_THRESHOLD: usize = 1 << 22;
const SMALL_SCALAR_SKEW_THRESHOLD: u32 = 64;

pub(super) fn g1_msm(bases: &[Bn254G1Affine], scalars: &[JoltFr]) -> G1Projective {
    if bases.is_empty() {
        return G1Projective::zero();
    }

    pippenger(Bn254G1Affine::as_inner_slice(bases), scalars)
}

#[expect(
    clippy::indexing_slicing,
    reason = "flat task indices and chunk bounds are derived from the input length"
)]
fn pippenger(bases: &[G1Affine], scalars: &[JoltFr]) -> G1Projective {
    let scalars = scalars
        .par_iter()
        .map(|scalar| Fr::from(*scalar).into_bigint())
        .collect::<Vec<_>>();
    let sampled = scalars.len().min(4096);
    let sampled_small = scalars[..sampled]
        .iter()
        .filter(|scalar| as_u16_scalar(scalar).is_some())
        .count();
    let small_count = if sampled_small * 8 >= sampled {
        scalars
            .iter()
            .filter(|scalar| as_u16_scalar(scalar).is_some())
            .count()
    } else {
        0
    };
    if small_count == scalars.len() {
        let scalars = scalars.iter().filter_map(as_u16_scalar).collect::<Vec<_>>();
        return small_msm(bases, &scalars);
    }
    if small_count != 0 && small_count >= scalars.len() / 8 {
        let mut full_bases = Vec::with_capacity(scalars.len() - small_count);
        let mut full_scalars = Vec::with_capacity(scalars.len() - small_count);
        let mut small_bases = Vec::with_capacity(small_count);
        let mut small_scalars = Vec::with_capacity(small_count);
        for (&base, scalar) in bases.iter().zip(&scalars) {
            if let Some(scalar) = as_u16_scalar(scalar) {
                if scalar != 0 {
                    small_bases.push(base);
                    small_scalars.push(scalar);
                }
            } else {
                full_bases.push(base);
                full_scalars.push(*scalar);
            }
        }
        let (full, small) = rayon::join(
            || pippenger_bigints(&full_bases, &full_scalars),
            || small_msm(&small_bases, &small_scalars),
        );
        return full + small;
    }
    pippenger_bigints(bases, &scalars)
}

#[inline]
fn as_u16_scalar(scalar: &BigInteger256) -> Option<u16> {
    (scalar.0[0] <= u16::MAX.into() && scalar.0[1..].iter().all(|&limb| limb == 0))
        .then_some(scalar.0[0] as u16)
}

#[expect(
    clippy::indexing_slicing,
    reason = "flat task indices and chunk bounds are derived from the input length"
)]
fn pippenger_bigints(bases: &[G1Affine], scalars: &[BigInteger256]) -> G1Projective {
    debug_assert!(!bases.is_empty());
    debug_assert_eq!(bases.len(), scalars.len());
    let max_window_bits = if bases.len() >= 1 << 20 {
        LARGE_MSM_WINDOW_BITS
    } else {
        MAX_WINDOW_BITS
    };
    let window_bits = if bases.len() < 32 {
        3
    } else {
        (bases.len().ilog2() as usize * 69 / 100 + 3).min(max_window_bits)
    };
    let window_count = Fr::MODULUS_BIT_SIZE as usize / window_bits + 1;
    let point_chunks = if bases.len() < 1 << 16 {
        1
    } else {
        (4 * rayon::current_num_threads())
            .div_ceil(window_count)
            .max(1)
    };
    let chunk_len = bases.len().div_ceil(point_chunks);
    let mut partials = (0..window_count * point_chunks)
        .into_par_iter()
        .map(|task| {
            let window = task / point_chunks;
            let chunk = task % point_chunks;
            let start = chunk * chunk_len;
            let end = (start + chunk_len).min(bases.len());
            hybrid_buckets(
                &bases[start..end],
                1 << (window_bits - 1),
                scalars[start..end]
                    .iter()
                    .map(|scalar| booth_digit(scalar, window, window_bits)),
            )
        })
        .collect::<Vec<_>>();
    let window_sums = partials
        .par_chunks_mut(point_chunks)
        .map(sum_buckets)
        .collect::<Vec<_>>();

    combine_window_sums(&window_sums, window_bits)
}

fn combine_window_sums(window_sums: &[G1Projective], window_bits: usize) -> G1Projective {
    window_sums
        .iter()
        .rev()
        .fold(G1Projective::zero(), |mut total, window| {
            for _ in 0..window_bits {
                let _doubled = total.double_in_place();
            }
            total += window;
            total
        })
}

#[expect(
    clippy::indexing_slicing,
    reason = "the last partial is removed before the remaining prefix is merged"
)]
fn sum_buckets(partials: &mut [Vec<G1Affine>]) -> G1Projective {
    let mut buckets: Vec<Bucket<G1Config>> = partials
        .last_mut()
        .map(std::mem::take)
        .unwrap_or_default()
        .into_iter()
        .map(Bucket::from)
        .collect();
    let prefix_len = partials.len().saturating_sub(1);
    for partial in &partials[..prefix_len] {
        for (bucket, other) in buckets.iter_mut().zip(partial) {
            *bucket += other;
        }
    }

    let mut running_sum = Bucket::<G1Config>::ZERO;
    let mut sum = Bucket::<G1Config>::ZERO;
    for bucket in buckets.iter().rev() {
        running_sum += bucket;
        sum += &running_sum;
    }
    G1Projective::from(sum)
}

#[expect(
    clippy::indexing_slicing,
    reason = "bucket indices are signed-digit magnitudes and linked indices originate in the input slices"
)]
fn hybrid_buckets(
    bases: &[G1Affine],
    bucket_count: usize,
    digits: impl Iterator<Item = i32>,
) -> Vec<G1Affine> {
    const NEGATIVE: usize = 1 << (usize::BITS - 1);
    const INDEX_MASK: usize = !NEGATIVE;

    let mut heads = vec![0; bucket_count];
    let mut counts = vec![0usize; bucket_count];
    let mut next = Vec::with_capacity(bases.len());
    next.resize_with(bases.len(), MaybeUninit::uninit);
    let mut remaining = 0;
    for (index, digit) in digits.enumerate() {
        if digit != 0 {
            let bucket = digit.unsigned_abs() as usize - 1;
            next[index] = MaybeUninit::new(heads[bucket]);
            heads[bucket] = (index + 1) | (usize::from(digit < 0) * NEGATIVE);
            counts[bucket] += 1;
            remaining += 1;
        }
    }

    let mut buckets = vec![G1Affine::identity(); heads.len()];
    // Affine passes consume one point per bucket; long chains would serialize the batch.
    let max_depth = (remaining / bucket_count * 8).max(64);
    for (bucket, head) in heads.iter_mut().enumerate() {
        if counts[bucket] <= max_depth {
            continue;
        }
        let mut sum = Bucket::<G1Config>::ZERO;
        while *head != 0 {
            let node = *head;
            let index = (node & INDEX_MASK) - 1;
            if node & NEGATIVE != 0 {
                sum -= &bases[index];
            } else {
                sum += &bases[index];
            }
            // SAFETY: each reachable node initialized its next slot above.
            *head = unsafe { next[index].assume_init() };
            remaining -= 1;
        }
        buckets[bucket] = G1Affine::from(G1Projective::from(sum));
    }
    let mut scheduled = Vec::with_capacity(heads.len());
    let mut denominators = Vec::with_capacity(heads.len());
    let mut prefixes = Vec::with_capacity(heads.len());
    while remaining != 0 {
        scheduled.clear();
        denominators.clear();
        for (bucket, head) in heads.iter_mut().enumerate() {
            let node = *head;
            if node == 0 {
                continue;
            }
            let index = (node & INDEX_MASK) - 1;
            // SAFETY: a node becomes reachable from `heads` only after its
            // corresponding `next` slot is initialized above.
            *head = unsafe { next[index].assume_init() };
            remaining -= 1;

            let point = if node & NEGATIVE != 0 {
                -bases[index]
            } else {
                bases[index]
            };
            if buckets[bucket].infinity {
                buckets[bucket] = point;
            } else if !point.infinity {
                let doubling = buckets[bucket].x == point.x;
                if doubling && buckets[bucket].y != point.y {
                    buckets[bucket] = G1Affine::identity();
                    continue;
                }
                denominators.push(if doubling {
                    point.y + point.y
                } else {
                    point.x - buckets[bucket].x
                });
                scheduled.push((bucket, point, doubling));
            }
        }
        batch_invert(&mut denominators, &mut prefixes);
        for ((bucket, point, doubling), inverse) in scheduled.iter().zip(&denominators) {
            buckets[*bucket] = add_affine(buckets[*bucket], *point, *inverse, *doubling);
        }
    }
    buckets
}

#[expect(
    clippy::expect_used,
    reason = "affine scheduling replaces every exceptional zero denominator"
)]
fn batch_invert(values: &mut [Fq], prefixes: &mut Vec<Fq>) {
    if values.is_empty() {
        return;
    }
    prefixes.clear();
    let mut product = Fq::one();
    for value in values.iter() {
        prefixes.push(product);
        product *= value;
    }
    let mut inverse = product.inverse().expect("denominator product is nonzero");
    for (value, prefix) in values.iter_mut().zip(prefixes.iter()).rev() {
        let original = *value;
        *value = inverse * prefix;
        inverse *= original;
    }
}

#[inline]
fn add_affine(p: G1Affine, q: G1Affine, denominator_inverse: Fq, doubling: bool) -> G1Affine {
    let numerator = if doubling {
        let x_squared = p.x.square();
        x_squared + x_squared + x_squared
    } else {
        q.y - p.y
    };
    let lambda = numerator * denominator_inverse;
    let x = lambda.square() - p.x - q.x;
    let y = lambda * (p.x - x) - p.y;
    G1Affine::new_unchecked(x, y)
}

#[inline(always)]
fn booth_digit(scalar: &BigInteger256, window: usize, width: usize) -> i32 {
    let skip = (window * width).saturating_sub(1);
    let limb_index = skip / 64;
    let bit_index = skip % 64;
    let mut bits = scalar.0.get(limb_index).copied().unwrap_or(0) >> bit_index;
    if bit_index + width + 1 > 64 {
        bits |= scalar.0.get(limb_index + 1).copied().unwrap_or(0) << (64 - bit_index);
    }
    if window == 0 {
        bits <<= 1;
    }
    bits &= (1 << (width + 1)) - 1;

    let positive = bits & (1 << width) == 0;
    let magnitude = (bits + 1) >> 1;
    if positive {
        magnitude as i32
    } else {
        -((!(magnitude - 1) & ((1 << width) - 1)) as i32)
    }
}

/// Scalar type for [`g1_msm_small`]: an unsigned integer read as little-endian
/// 8-bit Pippenger digits.
pub trait SmallScalar: Copy + Send + Sync {
    /// Number of 8-bit windows.
    const WINDOWS: usize;
    fn byte(self, window: usize) -> u8;
}

macro_rules! impl_small_scalar {
    ($($ty:ty),*) => {
        $(impl SmallScalar for $ty {
            const WINDOWS: usize = std::mem::size_of::<$ty>();

            #[inline(always)]
            fn byte(self, window: usize) -> u8 {
                (self >> (8 * window)) as u8
            }
        })*
    };
}

impl_small_scalar!(u8, u16, u32);

/// MSM for small unsigned scalars, using 16-bit buckets for large, unskewed
/// inputs and L1-resident 8-bit buckets otherwise.
pub(super) fn g1_msm_small<S: SmallScalar>(bases: &[Bn254G1Affine], scalars: &[S]) -> G1Projective {
    if bases.is_empty() {
        return G1Projective::zero();
    }
    if bases.len() >= SMALL_MSM_16_BIT_THRESHOLD
        && S::WINDOWS >= 2
        && !small_scalars_are_skewed(scalars)
    {
        return small_msm_16_bit(Bn254G1Affine::as_inner_slice(bases), scalars);
    }
    small_msm(Bn254G1Affine::as_inner_slice(bases), scalars)
}

#[expect(
    clippy::indexing_slicing,
    reason = "bucket index is a nonzero byte minus one within a 255-entry window"
)]
fn small_msm<S: SmallScalar>(bases: &[G1Affine], scalars: &[S]) -> G1Projective {
    let chunk_len = bases
        .len()
        .div_ceil(2 * rayon::current_num_threads())
        .max(2048);
    bases
        .par_chunks(chunk_len)
        .zip(scalars.par_chunks(chunk_len))
        .map(|(bases, scalars)| {
            let mut buckets = vec![Bucket::<G1Config>::ZERO; S::WINDOWS * 255];
            for (base, scalar) in bases.iter().zip(scalars) {
                for window in 0..S::WINDOWS {
                    let digit = scalar.byte(window) as usize;
                    if digit != 0 {
                        buckets[window * 255 + digit - 1] += base;
                    }
                }
            }
            buckets
                .chunks_exact(255)
                .rev()
                .fold(G1Projective::zero(), |mut total, window| {
                    for _ in 0..8 {
                        let _doubled = total.double_in_place();
                    }
                    let mut running_sum = Bucket::<G1Config>::ZERO;
                    let mut sum = Bucket::<G1Config>::ZERO;
                    for bucket in window.iter().rev() {
                        running_sum += bucket;
                        sum += &running_sum;
                    }
                    total += G1Projective::from(sum);
                    total
                })
        })
        .reduce(G1Projective::zero, |a, b| a + b)
}

/// Sixteen-bit buckets halve point insertions; at smaller sizes their fixed
/// reduction cost exceeds the saved mixed additions.
fn small_msm_16_bit<S: SmallScalar>(bases: &[G1Affine], scalars: &[S]) -> G1Projective {
    let chunk_len = bases.len().div_ceil(rayon::current_num_threads()).max(2048);
    bases
        .par_chunks(chunk_len)
        .zip(scalars.par_chunks(chunk_len))
        .map(|(bases, scalars)| {
            (0..S::WINDOWS.div_ceil(2))
                .rev()
                .fold(G1Projective::zero(), |mut total, window| {
                    for _ in 0..16 {
                        let _doubled = total.double_in_place();
                    }
                    let buckets = hybrid_buckets(
                        bases,
                        usize::from(u16::MAX),
                        scalars
                            .iter()
                            .map(|scalar| small_digit(*scalar, window) as i32),
                    );
                    let mut running_sum = Bucket::<G1Config>::ZERO;
                    let mut sum = Bucket::<G1Config>::ZERO;
                    for bucket in buckets.iter().rev() {
                        running_sum += bucket;
                        sum += &running_sum;
                    }
                    total += G1Projective::from(sum);
                    total
                })
        })
        .reduce(G1Projective::zero, |a, b| a + b)
}

#[inline]
fn small_digit<S: SmallScalar>(scalar: S, window: usize) -> usize {
    let byte = 2 * window;
    let low = usize::from(scalar.byte(byte));
    let high = if byte + 1 < S::WINDOWS {
        usize::from(scalar.byte(byte + 1))
    } else {
        0
    };
    low | high << 8
}

#[expect(
    clippy::indexing_slicing,
    reason = "two-byte digits index a 65536-entry histogram"
)]
fn small_scalars_are_skewed<S: SmallScalar>(scalars: &[S]) -> bool {
    // An odd stride avoids repeatedly sampling the same packed-column slot.
    let stride = scalars.len().div_ceil(4096) | 1;
    let mut counts = vec![0u32; 1 << 16];
    for window in 0..S::WINDOWS.div_ceil(2) {
        counts.fill(0);
        for scalar in scalars.iter().step_by(stride) {
            let digit = small_digit(*scalar, window);
            if digit != 0 {
                counts[digit] += 1;
                if counts[digit] > SMALL_SCALAR_SKEW_THRESHOLD {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use jolt_field::Field as _;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use crate::{Bn254, Bn254G1, JoltGroup, PairingGroup};

    fn bases(n: usize, rng: &mut ChaCha20Rng) -> Vec<Bn254G1Affine> {
        (0..n).map(|_| Bn254G1Affine(G1Affine::rand(rng))).collect()
    }

    fn check<S: SmallScalar + Into<u64>>(n: usize, scalars: Vec<S>) {
        let mut rng = ChaCha20Rng::seed_from_u64(n as u64);
        let bases = bases(n, &mut rng);
        let full: Vec<JoltFr> = scalars.iter().map(|s| JoltFr::from((*s).into())).collect();
        let expected = g1_msm(&bases, &full);
        assert_eq!(g1_msm_small(&bases, &scalars), expected);
        if S::WINDOWS >= 2 {
            assert_eq!(
                small_msm_16_bit(Bn254G1Affine::as_inner_slice(&bases), &scalars),
                expected
            );
        }
    }

    fn naive(bases: &[Bn254G1Affine], scalars: &[JoltFr]) -> G1Projective {
        bases
            .iter()
            .zip(scalars)
            .fold(Bn254G1::identity(), |sum, (base, scalar)| {
                sum + Bn254::g1_from_affine(base).scalar_mul(scalar)
            })
            .into_inner()
    }

    #[test]
    fn full_width_msm_matches_naive() {
        let mut rng = ChaCha20Rng::seed_from_u64(23);
        for n in [0, 1, 2, 7, 19, 32, 47] {
            let bases = bases(n, &mut rng);
            let scalars = (0..n).map(|_| JoltFr::random(&mut rng)).collect::<Vec<_>>();
            assert_eq!(g1_msm(&bases, &scalars), naive(&bases, &scalars));
        }
    }

    #[test]
    fn full_width_msm_handles_zero_identity_and_duplicate_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(29);
        let p = G1Affine::rand(&mut rng);
        let q = G1Affine::rand(&mut rng);
        let bases = vec![
            Bn254G1Affine(G1Affine::identity()),
            Bn254G1Affine(p),
            Bn254G1Affine(p),
            Bn254G1Affine(-p),
            Bn254G1Affine(q),
            Bn254G1Affine(q),
            Bn254G1Affine(G1Affine::identity()),
        ];
        let zero = vec![JoltFr::zero(); bases.len()];
        assert_eq!(g1_msm(&bases, &zero), G1Projective::zero());

        let scalars = (0..bases.len())
            .map(|_| JoltFr::random(&mut rng))
            .collect::<Vec<_>>();
        assert_eq!(g1_msm(&bases, &scalars), naive(&bases, &scalars));
    }

    #[test]
    fn full_width_msm_handles_mixed_scalar_widths() {
        let mut rng = ChaCha20Rng::seed_from_u64(31);
        let bases = bases(47, &mut rng);
        let scalars = (0..bases.len())
            .map(|index| {
                if index % 3 == 0 {
                    JoltFr::from((index as u16).wrapping_mul(257))
                } else {
                    JoltFr::random(&mut rng)
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(g1_msm(&bases, &scalars), naive(&bases, &scalars));
    }

    #[test]
    fn full_width_msm_handles_skewed_digits() {
        let mut rng = ChaCha20Rng::seed_from_u64(37);
        let bases = bases(257, &mut rng);
        let scalar = loop {
            let scalar = JoltFr::random(&mut rng);
            let bigint = Fr::from(scalar).into_bigint();
            if as_u16_scalar(&bigint).is_none() && booth_digit(&bigint, 0, 16) != 0 {
                break scalar;
            }
        };
        let mut scalars = vec![scalar; bases.len()];
        let expected = naive(&bases, &scalars);
        assert_eq!(g1_msm(&bases, &scalars), expected);
        for scalar in scalars.iter_mut().step_by(4) {
            *scalar = JoltFr::random(&mut rng);
        }
        assert_eq!(g1_msm(&bases, &scalars), naive(&bases, &scalars));
    }

    #[test]
    fn small_scalar_msm_matches_full_width() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        for n in [1, 2, 31, 1000, 4097] {
            check(n, (0..n).map(|_| rng.next_u32() as u8).collect());
            check(n, (0..n).map(|_| rng.next_u32() as u16).collect());
            check(n, (0..n).map(|_| rng.next_u32()).collect());
        }
    }

    #[test]
    fn small_scalar_msm_edge_values() {
        check(64, vec![0u16; 64]);
        check(64, vec![u16::MAX; 64]);
        check(64, vec![u32::MAX; 64]);
        check(5, vec![0u16, 1, 255, 256, u16::MAX]);
        assert_eq!(g1_msm_small::<u16>(&[], &[]), G1Projective::zero());
        assert!(!small_scalars_are_skewed(&[0u16; 128]));
        assert!(small_scalars_are_skewed(&[u16::MAX; 128]));
        assert!(small_scalars_are_skewed(&[1u32 << 31; 128]));
    }
}
