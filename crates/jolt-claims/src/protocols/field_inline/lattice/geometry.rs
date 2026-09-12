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
    let mut limbs = vec![0u64; field_inc_limb_count::<F>()];
    canonical_limbs_into(value, &mut limbs);
    limbs
}

/// Largest canonical encoding the limb decomposition handles without a heap
/// buffer: the 254-bit BN254 scalar field.
const MAX_CANONICAL_BYTES: usize = 32;

/// [`canonical_limbs`] written into a caller-owned buffer of exactly
/// [`field_inc_limb_count`] limbs — the per-cycle form the packed commit
/// loop uses (no allocation per cycle). Panics on a buffer of the wrong
/// length or a field wider than [`MAX_CANONICAL_BYTES`]; both are
/// compile-time facts of the instantiation.
pub fn canonical_limbs_into<F: CanonicalBytes>(value: &F, limbs: &mut [u64]) {
    assert!(
        F::NUM_BYTES <= MAX_CANONICAL_BYTES,
        "field encoding wider than the limb decomposition's byte buffer"
    );
    assert_eq!(
        limbs.len(),
        field_inc_limb_count::<F>(),
        "limb buffer must hold exactly the field's limb count"
    );
    let mut bytes = [0u8; MAX_CANONICAL_BYTES];
    value.to_bytes_le(&mut bytes[..F::NUM_BYTES]);
    for (limb, chunk) in limbs
        .iter_mut()
        .zip(bytes[..F::NUM_BYTES].chunks(FIELD_INC_LIMB_BITS / 8))
    {
        let mut word = [0u8; 8];
        // `chunks(8)` yields 1..=8 bytes, so the prefix slice is in range.
        word[..chunk.len()].copy_from_slice(chunk);
        *limb = u64::from_le_bytes(word);
    }
}

/// The radix weight of limb `limb` in the recomposition identity
/// `FieldRdInc = Σ_i 2^(64·i) · limb_i` — exact over `F` because the
/// canonical representative is `< p`, so the limbs recompose with no carries
/// and no modular wraparound.
pub fn limb_place_value<F: Ring>(limb: usize) -> F {
    F::pow2(FIELD_INC_LIMB_BITS * limb)
}

/// Evaluates the recomposition identity `Σ_i 2^(64·i) · limb_i` over `F`.
/// The identity holds pointwise on the boolean cube for the honest limb
/// columns, so it holds for their multilinear extensions at any point — the
/// stage-8 linear check the reduced `FieldRdInc` claim binds through.
pub fn recompose_limbs<F: Ring>(limbs: &[F]) -> F {
    limbs
        .iter()
        .enumerate()
        .map(|(limb, value)| limb_place_value::<F>(limb) * *value)
        .sum()
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
