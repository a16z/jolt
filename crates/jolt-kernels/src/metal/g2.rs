//! Host↔device representation for BN254 G2 points (the Fq2 twin of
//! [`super::g1`]).
//!
//! - Affine bases are `ark_bn254::G2Affine` host memory viewed in place as a
//!   `uint` stream (stride [`G2_AFFINE_U32_STRIDE`](super::field) u32s;
//!   layout pinned by const asserts in [`super::field`] and the
//!   `g2_affine_layout_matches_u32_view` test). Identity is the `(0, 0)`
//!   coordinate sentinel — the shader never reads the `infinity` flag.
//! - Results are Jacobian `(X, Y, Z)` Fq2-coordinate Montgomery limb runs
//!   (c0 then c1 per coordinate); `Z = 0` encodes the identity.

use ark_bn254::{Fq2 as ArkFq2, G2Affine, G2Projective};
use ark_ff::{BigInt, Zero};

use super::field::{FR_U32_LIMBS, G2_AFFINE_U32_STRIDE};

/// u32 words per Jacobian G2 result (X, Y, Z × 2 Fq × 8 limbs).
pub const G2_JAC_U32S: usize = 6 * FR_U32_LIMBS;

const _: () = assert!(std::mem::size_of::<G2Projective>() == G2_JAC_U32S * 4);

/// View affine G2 bases as the device `uint` stream. Identities must
/// already be lowered to the `(0, 0)` sentinel (the flag byte is dead to
/// the shader).
pub fn g2_bases_as_u32s(bases: &[G2Affine]) -> &[u32] {
    // SAFETY: the const asserts in `super::field` pin G2Affine's layout —
    // x at 0, y at 2·FR_U32_LIMBS·4, Fq2 c0 before c1, size a u32 multiple,
    // align ≥ u32 — and every bit pattern is a valid u32, for the same
    // lifetime as `bases`.
    unsafe {
        std::slice::from_raw_parts(
            bases.as_ptr().cast::<u32>(),
            bases.len() * G2_AFFINE_U32_STRIDE,
        )
    }
}

fn fq_from_mont_limbs(limbs: &[u32]) -> ark_bn254::Fq {
    let mut words = [0u64; 4];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from(limbs[2 * i]) | (u64::from(limbs[2 * i + 1]) << 32);
    }
    ark_bn254::Fq::new_unchecked(BigInt::new(words))
}

fn fq2_from_mont_limbs(limbs: &[u32]) -> ArkFq2 {
    ArkFq2::new(
        fq_from_mont_limbs(&limbs[..FR_U32_LIMBS]),
        fq_from_mont_limbs(&limbs[FR_U32_LIMBS..2 * FR_U32_LIMBS]),
    )
}

/// Rebuild one kernel result (Montgomery Jacobian limb run) as an arkworks
/// point. `Z = 0` is the device identity encoding.
pub fn g2_jac_from_device_limbs(limbs: &[u32]) -> G2Projective {
    debug_assert_eq!(limbs.len(), G2_JAC_U32S);
    let z_limbs = &limbs[4 * FR_U32_LIMBS..];
    if z_limbs.iter().all(|&w| w == 0) {
        return G2Projective::zero();
    }
    G2Projective {
        x: fq2_from_mont_limbs(&limbs[..2 * FR_U32_LIMBS]),
        y: fq2_from_mont_limbs(&limbs[2 * FR_U32_LIMBS..4 * FR_U32_LIMBS]),
        z: fq2_from_mont_limbs(z_limbs),
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;

    /// The zero-copy contract: an affine G2 point's device words are exactly
    /// its x/y Fq2 Montgomery limbs (c0 then c1) at the pinned offsets.
    #[test]
    fn g2_affine_layout_matches_u32_view() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let bases: Vec<G2Affine> = (0..3).map(|_| G2Affine::rand(&mut rng)).collect();
        let words = g2_bases_as_u32s(&bases);
        for (i, base) in bases.iter().enumerate() {
            let x = fq2_from_mont_limbs(&words[i * G2_AFFINE_U32_STRIDE..][..2 * FR_U32_LIMBS]);
            let y = fq2_from_mont_limbs(
                &words[i * G2_AFFINE_U32_STRIDE + 2 * FR_U32_LIMBS..][..2 * FR_U32_LIMBS],
            );
            assert_eq!(x, base.x);
            assert_eq!(y, base.y);
        }
    }
}
