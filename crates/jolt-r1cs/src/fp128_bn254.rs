//! Canonical `Prime128OffsetA7F7` arithmetic constrained over BN254 scalars.
//!
//! For q = 2^128 - 2^32 + 22537 and BN254 scalar modulus r, q² < 8r.
//! Canonical a,b,c,k give |ab-c-qk| < q². Checking this residual modulo
//! r and modulo 8 therefore proves integer equality. Every low bit used
//! below belongs to the same canonical decomposition as its full value.
//!
//! Witness generation uses native field arithmetic and is not constant-time.
//! It is not part of the soundness argument: ranges and both congruences are
//! enforced independently by constraints. Build a fresh circuit with `None`
//! for layout or `Some` for assignment; both walks emit identical matrices.

use jolt_field::{CanonicalBytes, Field, Fr, Prime128OffsetA7F7, Ring};
use thiserror::Error;

use crate::{LinearCombination, R1csBuilder, Variable};

/// The source modulus, derived from the native field's canonical definition.
pub const MODULUS: u128 = u128::MAX - (Prime128OffsetA7F7::C - 1);

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Fp128Error {
    #[error("source field value {value} is not canonical")]
    NonCanonical { value: u128 },
    #[error("variable {variable:?} is outside the supplied constraint builder")]
    UnknownVariable { variable: Variable },
    #[error("the source modulus has no inverse in the constraint field")]
    NonInvertibleModulus,
    #[error("the generated quotient exceeds the source-field range")]
    QuotientOutOfRange,
}

/// A canonical q-field element and its linked Boolean decomposition.
///
/// Values may only be composed in the builder that allocated them. The
/// decomposition is reusable across arithmetic operations in that builder.
#[derive(Clone, Debug)]
pub struct Fp128Var {
    variable: Variable,
    bits: Vec<Variable>,
    witness: Option<u128>,
}

impl Fp128Var {
    /// Allocate x and enforce 0 <= x < q, including for adversarial assignments.
    pub fn allocate(
        builder: &mut R1csBuilder<Fr>,
        witness: Option<u128>,
    ) -> Result<Self, Fp128Error> {
        if let Some(value) = witness {
            if value >= MODULUS {
                return Err(Fp128Error::NonCanonical { value });
            }
        }
        let variable = builder.alloc_witness(witness.map(Fr::from_u128));
        let bits = Self::allocate_bits(builder, witness, 128);
        builder.assert_equal(variable, Self::bits_lc(&bits));
        let complement = Self::allocate_bits(builder, witness.map(|x| MODULUS - 1 - x), 128);
        // Both summands are 128-bit integers, so this equality cannot wrap in Fr.
        builder.assert_equal(
            LinearCombination::variable(variable) + Self::bits_lc(&complement),
            LinearCombination::constant(Fr::from_u128(MODULUS - 1)),
        );
        Ok(Self {
            variable,
            bits,
            witness,
        })
    }

    /// The BN254 variable holding the canonical integer representative.
    pub fn variable(&self) -> Variable {
        self.variable
    }

    /// Allocate and constrain the canonical product modulo q.
    pub fn multiply(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Result<Self, Fp128Error> {
        self.validate_indices(builder)?;
        rhs.validate_indices(builder)?;
        let witnesses = self
            .witness
            .zip(rhs.witness)
            .map(|(a, b)| Self::product_witness(a, b))
            .transpose()?;
        let output = Self::allocate(builder, witnesses.map(|(c, _)| c))?;
        let quotient = Self::allocate(builder, witnesses.map(|(_, k)| k))?;
        self.enforce_product(builder, rhs, &output, &quotient);
        Ok(output)
    }

    /// Allocate and constrain the canonical sum modulo q.
    pub fn add(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Result<Self, Fp128Error> {
        self.validate_indices(builder)?;
        rhs.validate_indices(builder)?;
        let pair = self.witness.zip(rhs.witness);
        let value = pair.map(|(a, b)| {
            (Prime128OffsetA7F7::from_u128(a) + Prime128OffsetA7F7::from_u128(b))
                .to_canonical_u128()
        });
        let output = Self::allocate(builder, value)?;
        let carry =
            Self::allocate_bits(builder, pair.map(|(a, b)| u128::from(a >= MODULUS - b)), 1);
        // The entire residual has magnitude less than 2q, hence less than r.
        builder.assert_equal(
            LinearCombination::variable(self.variable) + rhs.variable.into(),
            LinearCombination::variable(output.variable)
                + Self::bits_lc(&carry).scale(Fr::from_u128(MODULUS)),
        );
        Ok(output)
    }

    fn enforce_product(
        &self,
        builder: &mut R1csBuilder<Fr>,
        rhs: &Self,
        output: &Self,
        quotient: &Self,
    ) {
        builder.assert_product(
            self.variable,
            rhs.variable,
            LinearCombination::variable(output.variable)
                + LinearCombination::variable(quotient.variable).scale(Fr::from_u128(MODULUS)),
        );
        // a0*b0-c0-k0 is in [-14,49]. If divisible by 8 its quotient
        // lies in [-1,6], so encode s = quotient+1 with exactly three bits.
        let low_quotient = self
            .witness
            .zip(rhs.witness)
            .zip(output.witness.zip(quotient.witness))
            .map(|((a, b), (c, k))| {
                let low = |x: u128| {
                    let [byte, ..] = x.to_le_bytes();
                    i16::from(byte & 7)
                };
                let residual = low(a) * low(b) - low(c) - low(k);
                // The offset is always in 0..=7, even for a nonsatisfying
                // witness supplied by an internal adversarial test.
                (residual / 8 + 1).unsigned_abs().into()
            });
        let shifted = Self::allocate_bits(builder, low_quotient, 3);
        builder.assert_product(
            self.low_bits_lc(),
            rhs.low_bits_lc(),
            output.low_bits_lc()
                + quotient.low_bits_lc()
                + Self::bits_lc(&shifted).scale(Fr::from_u64(8))
                - LinearCombination::constant(Fr::from_u64(8)),
        );
    }

    fn product_witness(a: u128, b: u128) -> Result<(u128, u128), Fp128Error> {
        let c = (Prime128OffsetA7F7::from_u128(a) * Prime128OffsetA7F7::from_u128(b))
            .to_canonical_u128();
        // The integer quotient is <q<r, so its Fr residue recovers it uniquely.
        let inverse = Fr::from_u128(MODULUS)
            .inverse()
            .ok_or(Fp128Error::NonInvertibleModulus)?;
        let k = (Fr::from_u128(a) * Fr::from_u128(b) - Fr::from_u128(c)) * inverse;
        let mut bytes = [0; 32];
        k.to_bytes_le(&mut bytes);
        let mut low = [0; 16];
        for (out, byte) in low.iter_mut().zip(bytes.iter()) {
            *out = *byte;
        }
        let quotient = u128::from_le_bytes(low);
        if bytes.iter().skip(16).any(|byte| *byte != 0) || quotient >= MODULUS {
            return Err(Fp128Error::QuotientOutOfRange);
        }
        Ok((c, quotient))
    }

    /// Reject value or decomposition indices outside `builder`.
    ///
    /// This does not establish builder identity: callers must still use each
    /// value in the builder that allocated it, even when indices coincide.
    pub fn validate_indices(&self, builder: &R1csBuilder<Fr>) -> Result<(), Fp128Error> {
        for &variable in std::iter::once(&self.variable).chain(&self.bits) {
            if variable.index() >= builder.num_vars() {
                return Err(Fp128Error::UnknownVariable { variable });
            }
        }
        Ok(())
    }

    fn allocate_bits(
        builder: &mut R1csBuilder<Fr>,
        witness: Option<u128>,
        width: usize,
    ) -> Vec<Variable> {
        (0..width)
            .map(|i| {
                let bit = builder.alloc_witness(witness.map(|x| Fr::from_u128((x >> i) & 1)));
                builder.assert_product(
                    bit,
                    LinearCombination::variable(bit) - LinearCombination::one(),
                    LinearCombination::zero(),
                );
                bit
            })
            .collect()
    }

    fn bits_lc<'a>(bits: impl IntoIterator<Item = &'a Variable>) -> LinearCombination<Fr> {
        let mut coefficient = Fr::from_u64(1);
        let mut result = LinearCombination::zero();
        for &bit in bits {
            result = result + LinearCombination::variable(bit).scale(coefficient);
            coefficient += coefficient;
        }
        result
    }

    fn low_bits_lc(&self) -> LinearCombination<Fr> {
        Self::bits_lc(self.bits.iter().take(3))
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect and mutate complete circuit witnesses"
)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn q() -> BigUint {
        (BigUint::from(1u8) << 128) - (BigUint::from(1u8) << 32) + BigUint::from(22537u32)
    }

    #[test]
    fn arithmetic_matches_integer_oracle_at_boundaries() {
        let cases = [
            0,
            1,
            7,
            8,
            (1u128 << 64) - 1,
            1u128 << 127,
            MODULUS - 2,
            MODULUS - 1,
        ];
        assert_eq!(BigUint::from(MODULUS), q());
        for a in cases {
            for b in cases {
                let mut builder = R1csBuilder::new();
                let av = Fp128Var::allocate(&mut builder, Some(a)).unwrap();
                let bv = Fp128Var::allocate(&mut builder, Some(b)).unwrap();
                let c = av.multiply(&mut builder, &bv).unwrap();
                let sum = av.add(&mut builder, &bv).unwrap();
                let expected_product = (BigUint::from(a) * BigUint::from(b)) % q();
                let expected_sum = (BigUint::from(a) + BigUint::from(b)) % q();
                assert_eq!(BigUint::from(c.witness.unwrap()), expected_product);
                assert_eq!(BigUint::from(sum.witness.unwrap()), expected_sum);
                let witness = builder.witness().unwrap();
                assert!(builder.into_matrices().check_witness(&witness).is_ok());
            }
        }
    }

    #[test]
    fn crt_bound_and_native_only_counterexample() {
        let r = BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap();
        let mut r_minus_one = [0; 32];
        (-Fr::from_u64(1)).to_bytes_le(&mut r_minus_one);
        assert_eq!(BigUint::from_bytes_le(&r_minus_one) + BigUint::from(1u8), r);
        assert_eq!(&r % BigUint::from(8u8), BigUint::from(1u8));
        assert_eq!(MODULUS % 8, 1);
        assert!(&r * BigUint::from(4u8) < q() * q());
        assert!(q() * q() < &r * BigUint::from(8u8));
        let a = MODULUS - 1;
        let c = 49_309_762_711_427_763_147_861_143_018_196_264_844;
        let k = 275_958_602_307_755_286_421_512_549_941_435_351_515;
        assert_eq!(
            BigUint::from(a) * BigUint::from(a) - BigUint::from(c) - q() * BigUint::from(k),
            r
        );
        assert_eq!(
            Fr::from_u128(a) * Fr::from_u128(a),
            Fr::from_u128(c) + Fr::from_u128(MODULUS) * Fr::from_u128(k)
        );
        let mut builder = R1csBuilder::new();
        let av = Fp128Var::allocate(&mut builder, Some(a)).unwrap();
        let cv = Fp128Var::allocate(&mut builder, Some(c)).unwrap();
        let kv = Fp128Var::allocate(&mut builder, Some(k)).unwrap();
        av.enforce_product(&mut builder, &av, &cv, &kv);
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn completed_witness_rejects_value_bit_and_complement_tampering() {
        let mut builder = R1csBuilder::new();
        let value = Fp128Var::allocate(&mut builder, Some(MODULUS - 1)).unwrap();
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        // q has a valid 128-bit decomposition but is not a canonical source value.
        let mut noncanonical = witness.clone();
        noncanonical[value.variable.index()] = Fr::from_u128(MODULUS);
        for (i, bit) in value.bits.iter().enumerate() {
            noncanonical[bit.index()] = Fr::from_u128((MODULUS >> i) & 1);
        }
        assert!(matrices.check_witness(&noncanonical).is_err());

        let mut unlinked_low_bit = witness.clone();
        let bit = value.bits[0];
        unlinked_low_bit[bit.index()] = Fr::from_u64(1) - unlinked_low_bit[bit.index()];
        assert!(matrices.check_witness(&unlinked_low_bit).is_err());

        let mut nonboolean = witness.clone();
        nonboolean[bit.index()] = Fr::from_u64(2);
        assert!(matrices.check_witness(&nonboolean).is_err());

        // Last allocation is the top complement bit; both ranges are enforced.
        let mut bad_complement = witness;
        let last = bad_complement.len() - 1;
        bad_complement[last] = Fr::from_u64(2);
        assert!(matrices.check_witness(&bad_complement).is_err());
    }

    #[test]
    fn completed_product_witness_rejects_output_and_quotient_tampering() {
        let mut builder = R1csBuilder::new();
        let a = Fp128Var::allocate(&mut builder, Some(MODULUS - 1)).unwrap();
        let c = Fp128Var::allocate(&mut builder, Some(1)).unwrap();
        let k = Fp128Var::allocate(&mut builder, Some(MODULUS - 2)).unwrap();
        a.enforce_product(&mut builder, &a, &c, &k);
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for index in [c.variable.index(), k.variable.index(), witness.len() - 1] {
            let mut tampered = witness.clone();
            tampered[index] += Fr::from_u64(1);
            assert!(matrices.check_witness(&tampered).is_err());
        }
        // Replace all range and complement bits consistently, so rejection
        // must come from arithmetic rather than a stale decomposition.
        for (value, replacement) in [(&c, 2), (&k, MODULUS - 3)] {
            let mut scratch = R1csBuilder::new();
            let _ = Fp128Var::allocate(&mut scratch, Some(replacement)).unwrap();
            let replacement = scratch.witness().unwrap();
            let mut tampered = witness.clone();
            let start = value.variable.index();
            tampered[start..start + replacement.len() - 1].copy_from_slice(&replacement[1..]);
            assert!(matrices.check_witness(&tampered).is_err());
        }
    }

    #[test]
    fn layout_is_independent_of_assignments() {
        let emit = |a, b| {
            let mut builder = R1csBuilder::new();
            let av = Fp128Var::allocate(&mut builder, a).unwrap();
            let bv = Fp128Var::allocate(&mut builder, b).unwrap();
            let product = av.multiply(&mut builder, &bv).unwrap();
            let _ = product.add(&mut builder, &av).unwrap();
            builder.into_matrices()
        };
        let empty = emit(None, None);
        for (a, b) in [(0, 1), (MODULUS - 1, MODULUS - 1)] {
            let known = emit(Some(a), Some(b));
            assert_eq!(empty.num_vars, known.num_vars);
            assert_eq!(empty.a, known.a);
            assert_eq!(empty.b, known.b);
            assert_eq!(empty.c, known.c);
        }
    }

    #[test]
    fn invalid_input_and_missing_layout_assignment_fail_explicitly() {
        let mut builder = R1csBuilder::new();
        assert!(matches!(
            Fp128Var::allocate(&mut builder, Some(MODULUS)),
            Err(Fp128Error::NonCanonical { .. })
        ));
        assert_eq!(builder.num_vars(), 1);
        let unknown = Fp128Var::allocate(&mut builder, None).unwrap();
        assert!(builder.witness().is_err());
        let mut other = R1csBuilder::new();
        assert!(matches!(
            unknown.multiply(&mut other, &unknown),
            Err(Fp128Error::UnknownVariable { .. })
        ));
    }
}
