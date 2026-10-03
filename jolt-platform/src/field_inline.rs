//! Proof fields that field-inline instructions execute over.
//!
//! The tracer names the field it executes over, jolt-host forwards that name to
//! field-inline guest builds through [`FIELD_INLINE_MODULUS_ENV`], and every
//! jolt-sdk limb conversion proves in-guest that the field the guest was
//! compiled for is the executing field.

/// Environment variable through which jolt-host tells a guest build which
/// proof field executes its field instructions.
pub const FIELD_INLINE_MODULUS_ENV: &str = "JOLT_FIELD_INLINE_MODULUS";

/// A prime field that field-inline instructions can execute over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldInlineModulus {
    /// The BN254 scalar field, proven with Dory.
    Bn254,
    /// The 128-bit prime `2^128 - 2^32 + 22537`, proven with Akita.
    Fp128,
}

const BN254_LIMBS: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];
const FP128_LIMBS: [u64; 2] = [0xffff_ffff_0000_5809, 0xffff_ffff_ffff_ffff];

impl FieldInlineModulus {
    /// The modulus as little-endian u64 limbs; the most significant limb is
    /// nonzero.
    pub const fn limbs(self) -> &'static [u64] {
        match self {
            Self::Bn254 => &BN254_LIMBS,
            Self::Fp128 => &FP128_LIMBS,
        }
    }

    /// The value [`FIELD_INLINE_MODULUS_ENV`] carries for this field.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bn254 => "bn254",
            Self::Fp128 => "fp128",
        }
    }

    /// Inverse of [`Self::name`].
    pub const fn from_name(name: &str) -> Option<Self> {
        if bytes_eq(name.as_bytes(), Self::Bn254.name().as_bytes()) {
            Some(Self::Bn254)
        } else if bytes_eq(name.as_bytes(), Self::Fp128.name().as_bytes()) {
            Some(Self::Fp128)
        } else {
            None
        }
    }

    /// Whether the little-endian `limbs` encode an integer below the modulus.
    /// Limbs past the modulus width are part of the integer, so they must be
    /// zero.
    pub fn is_canonical(self, limbs: &[u64]) -> bool {
        let modulus = self.limbs();
        for index in (0..limbs.len().max(modulus.len())).rev() {
            let limb = limbs.get(index).copied().unwrap_or(0);
            let bound = modulus.get(index).copied().unwrap_or(0);
            if limb != bound {
                return limb < bound;
            }
        }
        false
    }
}

const fn bytes_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULI: [FieldInlineModulus; 2] = [FieldInlineModulus::Bn254, FieldInlineModulus::Fp128];

    /// `p + delta` as little-endian limbs one wider than the modulus, for
    /// small `delta`.
    fn offset_from_modulus(modulus: FieldInlineModulus, delta: i64) -> [u64; 5] {
        let mut limbs = [0u64; 5];
        limbs[..modulus.limbs().len()].copy_from_slice(modulus.limbs());
        let mut carry = i128::from(delta);
        for limb in &mut limbs {
            let sum = i128::from(*limb) + carry;
            *limb = sum as u64;
            carry = sum >> 64;
        }
        limbs
    }

    #[test]
    fn names_round_trip() {
        for modulus in MODULI {
            assert_eq!(FieldInlineModulus::from_name(modulus.name()), Some(modulus));
        }
        assert_eq!(FieldInlineModulus::from_name("bn25"), None);
        assert_eq!(FieldInlineModulus::from_name(""), None);
    }

    #[test]
    fn canonical_limbs_are_exactly_the_integers_below_the_modulus() {
        for modulus in MODULI {
            let width = modulus.limbs().len();
            assert!(modulus.is_canonical(&[]));
            assert!(modulus.is_canonical(&[0]));
            assert!(modulus.is_canonical(&[1]));
            assert!(modulus.is_canonical(&[u64::MAX]));
            assert!(modulus.is_canonical(&[0, 1]));
            assert!(modulus.is_canonical(&offset_from_modulus(modulus, -1)));
            assert!(modulus.is_canonical(&offset_from_modulus(modulus, -1)[..width]));
            assert!(!modulus.is_canonical(modulus.limbs()));
            assert!(!modulus.is_canonical(&offset_from_modulus(modulus, 0)));
            assert!(!modulus.is_canonical(&offset_from_modulus(modulus, 1)));
            // A nonzero limb at or past the modulus width is at least 2^(64 * width) > p.
            let mut past_width = [0u64; 6];
            past_width[width] = 1;
            assert!(!modulus.is_canonical(&past_width));
            past_width[width] = 0;
            past_width[5] = 1;
            assert!(!modulus.is_canonical(&past_width));
        }
        // 2^128 - 1 fits two limbs but exceeds the 128-bit prime.
        assert!(FieldInlineModulus::Bn254.is_canonical(&[u64::MAX, u64::MAX]));
        assert!(!FieldInlineModulus::Fp128.is_canonical(&[u64::MAX, u64::MAX]));
    }
}
