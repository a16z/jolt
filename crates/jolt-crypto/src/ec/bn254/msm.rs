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
    let window_sums = (0..window_count)
        .into_par_iter()
        .map(|window| {
            let mut buckets = vec![Bucket::<G1Config>::ZERO; 1 << (window_bits - 1)];
            for (base, scalar) in bases.iter().zip(&scalars) {
                let digit = booth_digit(scalar, window, window_bits);
                if digit > 0 {
                    buckets[digit as usize - 1] += base;
                } else if digit < 0 {
                    buckets[(-digit) as usize - 1] -= base;
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
