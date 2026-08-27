//! Pure packed-mode geometry of the field-register extension: the u64 limb
//! decomposition of `FieldRdInc`'s canonical representative and the single
//! linear recomposition identity over `F`
//! (`specs/field-inline-portability.md`, Axis 1).

use jolt_field::{CanonicalBytes, Ring};

use crate::lattice::BALANCED_INC_BITS;

/// Bit width of one limb of the canonical representative: the shared
/// balanced-digit window, so every limb rides the digit machinery unchanged.
pub const FIELD_INC_LIMB_BITS: usize = BALANCED_INC_BITS;

/// Number of u64 limbs of `F`'s canonical representative (4 for a 254-bit
/// field, 2 for a 128-bit field).
pub fn field_inc_limb_count<F: CanonicalBytes>() -> usize {
    F::NUM_BYTES.div_ceil(FIELD_INC_LIMB_BITS / 8)
}

/// The little-endian u64 limbs of `value`'s canonical representative. Exactly
/// [`field_inc_limb_count`] limbs; a short final byte chunk zero-extends.
pub fn canonical_limbs<F: CanonicalBytes>(value: &F) -> Vec<u64> {
    let bytes = value.to_bytes_le_vec();
    bytes
        .chunks(FIELD_INC_LIMB_BITS / 8)
        .map(|chunk| {
            let mut limb = [0u8; 8];
            // `chunks(8)` yields 1..=8 bytes, so the prefix slice is in range.
            limb[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(limb)
        })
        .collect()
}

/// The radix weight of limb `limb` in the recomposition identity
/// `FieldRdInc = Σ_i 2^(64·i) · limb_i` — exact over `F` because the
/// canonical representative is `< p`, so the limbs recompose with no carries
/// and no modular wraparound.
pub fn limb_place_value<F: Ring>(limb: usize) -> F {
    F::pow2(FIELD_INC_LIMB_BITS * limb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr, Ring};

    #[test]
    fn limb_count_covers_the_canonical_byte_width() {
        // BN254 Fr: 32 canonical bytes -> 4 u64 limbs. (The 128-bit packed
        // field's count of 2 is pinned where that field is visible.)
        assert_eq!(field_inc_limb_count::<Fr>(), 4);
    }

    /// The spec's Axis 1 identity: `FieldRdInc = Σ_i 2^(64·i)·limb_i` is
    /// exact for every canonical representative.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn recomposition_is_exact_over_the_canonical_representative() {
        let one = Fr::from_u64(1);
        let values = [
            Fr::from_u64(0),
            one,
            Fr::from_u64(0) - one,
            Fr::from_u64(u64::MAX),
            Fr::pow2(64),
            Fr::pow2(127) - one,
            Fr::pow2(200) + Fr::from_u64(0x0123_4567_89ab_cdef),
            Fr::from_u64(2).inverse().unwrap(),
        ];
        for value in &values {
            let limbs = canonical_limbs(value);
            assert_eq!(limbs.len(), field_inc_limb_count::<Fr>());
            let recomposed: Fr = limbs
                .iter()
                .enumerate()
                .map(|(index, limb)| limb_place_value::<Fr>(index) * Fr::from_u64(*limb))
                .sum();
            assert_eq!(recomposed, *value);
        }
    }
}
