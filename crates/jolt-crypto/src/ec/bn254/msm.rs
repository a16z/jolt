use ark_bn254::{g1::Config as G1Config, Fr, G1Affine, G1Projective};
use ark_ec::{models::short_weierstrass::Bucket, AdditiveGroup};
use ark_ff::{BigInteger256, PrimeField, Zero};
use jolt_field::Fr as JoltFr;
use rayon::prelude::*;

use super::Bn254G1Affine;

const MAX_WINDOW_BITS: usize = 15;

pub(super) fn g1_msm(bases: &[Bn254G1Affine], scalars: &[JoltFr]) -> G1Projective {
    if bases.is_empty() {
        return G1Projective::zero();
    }

    pippenger(Bn254G1Affine::as_inner_slice(bases), scalars)
}

#[expect(
    clippy::indexing_slicing,
    reason = "digits have one entry per configured window and signed digits fit the configured buckets"
)]
fn pippenger(bases: &[G1Affine], scalars: &[JoltFr]) -> G1Projective {
    let window_bits = if bases.len() < 32 {
        3
    } else {
        (bases.len().ilog2() as usize * 69 / 100 + 3).min(MAX_WINDOW_BITS)
    };
    let window_count = Fr::MODULUS_BIT_SIZE as usize / window_bits + 1;
    let scalars = scalars
        .par_iter()
        .map(|scalar| Fr::from(*scalar).into_bigint())
        .collect::<Vec<_>>();
    // (window × point-chunk) tasks keep every core busy on the tail; the
    // chunk bucket arrays of one window are merged before its running sum.
    let point_chunks = if bases.len() < 1 << 16 {
        1
    } else {
        (4 * rayon::current_num_threads())
            .div_ceil(window_count)
            .max(1)
    };
    let chunk_len = bases.len().div_ceil(point_chunks);
    let window_sums = (0..window_count)
        .into_par_iter()
        .map(|window| {
            let mut partials: Vec<Vec<Bucket<G1Config>>> = bases
                .par_chunks(chunk_len)
                .zip(scalars.par_chunks(chunk_len))
                .map(|(bases, scalars)| {
                    let mut buckets = vec![Bucket::<G1Config>::ZERO; 1 << (window_bits - 1)];
                    for (base, scalar) in bases.iter().zip(scalars) {
                        let digit = booth_digit(scalar, window, window_bits);
                        if digit > 0 {
                            buckets[digit as usize - 1] += base;
                        } else if digit < 0 {
                            buckets[(-digit) as usize - 1] -= base;
                        }
                    }
                    buckets
                })
                .collect();
            let mut buckets = partials.pop().unwrap_or_default();
            for partial in &partials {
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
        })
        .collect::<Vec<_>>();

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

/// MSM for small unsigned scalars: one 8-bit bucket window per scalar byte
/// (`S::WINDOWS` mixed additions per point, buckets L1-resident), base chunks
/// processed in parallel and their partial sums combined at the end.
#[expect(
    clippy::indexing_slicing,
    reason = "bucket index is a nonzero byte minus one within a 255-entry window"
)]
pub(super) fn g1_msm_small<S: SmallScalar>(bases: &[Bn254G1Affine], scalars: &[S]) -> G1Projective {
    if bases.is_empty() {
        return G1Projective::zero();
    }
    let bases = Bn254G1Affine::as_inner_slice(bases);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn bases(n: usize, rng: &mut ChaCha20Rng) -> Vec<Bn254G1Affine> {
        (0..n).map(|_| Bn254G1Affine(G1Affine::rand(rng))).collect()
    }

    fn check<S: SmallScalar + Into<u64>>(n: usize, scalars: Vec<S>) {
        let mut rng = ChaCha20Rng::seed_from_u64(n as u64);
        let bases = bases(n, &mut rng);
        let full: Vec<JoltFr> = scalars.iter().map(|s| JoltFr::from((*s).into())).collect();
        assert_eq!(g1_msm_small(&bases, &scalars), g1_msm(&bases, &full));
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
    }
}
