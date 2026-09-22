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

use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, Fr, Prime128OffsetA7F7, Ring};
use thiserror::Error;

use crate::bn254_bits::{BitsError, ByteVar};
use crate::{LinearCombination, R1csBuilder, Variable};

/// The source modulus, derived from the native field's canonical definition.
pub const MODULUS: u128 = u128::MAX - (Prime128OffsetA7F7::C - 1);

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Fp128Error {
    #[error(transparent)]
    Bits(#[from] BitsError),
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
        let high_zero_count = witness.map(|x| 96 - (x >> 32).count_ones());
        let high_zeros = bits.iter().skip(32).fold(
            LinearCombination::constant(Fr::from_u64(96)),
            |sum, &bit| sum - LinearCombination::variable(bit),
        );
        let high_is_max =
            builder.alloc_witness(high_zero_count.map(|count| Fr::from_u64(u64::from(count == 0))));
        let inverse = builder.alloc_witness(high_zero_count.map(|count| {
            Fr::from_u64(u64::from(count))
                .inverse()
                .unwrap_or(Fr::from_u64(0))
        }));
        // S is in [0,96], so these rows force h=1 exactly when S=0.
        builder.assert_product(high_zeros.clone(), high_is_max, LinearCombination::zero());
        builder.assert_product(
            high_zeros,
            inverse,
            LinearCombination::one() - LinearCombination::variable(high_is_max),
        );
        let low_mask = (1u128 << 32) - 1;
        let max_low = (MODULUS & low_mask) - 1;
        let slack = Self::allocate_bits(
            builder,
            witness.zip(high_zero_count).map(|(x, count)| {
                if count == 0 {
                    max_low - (x & low_mask)
                } else {
                    0
                }
            }),
            15,
        );
        // q's high 96 bits are all one. Only that prefix needs L+d=22536;
        // L+d < 2^32+2^15 < r, and 15 bits cover every honest slack.
        builder.assert_product(
            high_is_max,
            Self::bits_lc(bits.iter().take(32)) + Self::bits_lc(&slack)
                - LinearCombination::constant(Fr::from_u128(max_low)),
            LinearCombination::zero(),
        );
        Ok(Self {
            variable,
            bits,
            witness,
        })
    }

    /// Allocate a canonical public constant and pin its field handle.
    pub fn constant(builder: &mut R1csBuilder<Fr>, value: u128) -> Result<Self, Fp128Error> {
        let result = Self::allocate(builder, Some(value))?;
        builder.assert_equal(
            result.variable,
            LinearCombination::constant(Fr::from_u128(value)),
        );
        Ok(result)
    }

    /// Bind sixteen existing little-endian Boolean bytes to a canonical q value.
    /// The caller authenticates the byte source; ONE and builder provenance remain external.
    pub fn from_le_bytes(
        builder: &mut R1csBuilder<Fr>,
        bytes: &[ByteVar; 16],
    ) -> Result<Self, Fp128Error> {
        let (encoded, witness) = Self::decode_le_bytes(builder, bytes)?;
        let value = Self::allocate(builder, witness)?;
        builder.assert_equal(value.variable, encoded);
        Ok(value)
    }

    /// Reduce sixteen Boolean bytes modulo q, including noncanonical encodings.
    /// Since h < 2^128 < 2q < r, h = x + bq with canonical x and Boolean b
    /// is an integer equality with a unique quotient, not merely a congruence.
    pub fn reduce_le_bytes(
        builder: &mut R1csBuilder<Fr>,
        bytes: &[ByteVar; 16],
    ) -> Result<Self, Fp128Error> {
        let (encoded, witness) = Self::decode_le_bytes(builder, bytes)?;
        let value = Self::allocate(builder, witness.map(|h| h % MODULUS))?;
        let quotient = Self::allocate_bits(builder, witness.map(|h| u128::from(h >= MODULUS)), 1);
        builder.assert_equal(
            encoded,
            LinearCombination::variable(value.variable)
                + Self::bits_lc(&quotient).scale(Fr::from_u128(MODULUS)),
        );
        Ok(value)
    }

    fn decode_le_bytes(
        builder: &R1csBuilder<Fr>,
        bytes: &[ByteVar; 16],
    ) -> Result<(LinearCombination<Fr>, Option<u128>), Fp128Error> {
        let mut encoded = LinearCombination::zero();
        let mut witness = Some(0u128);
        for (index, byte) in bytes.iter().enumerate() {
            byte.validate_indices(builder)?;
            let value = builder
                .evaluate(&byte.expression())
                .ok()
                .and_then(|x| x.to_u128_checked());
            witness = witness
                .zip(value)
                .map(|(sum, value)| sum | (value << (8 * index)));
            encoded = encoded + byte.expression().scale(Fr::from_u128(1u128 << (8 * index)));
        }
        Ok((encoded, witness))
    }

    /// Canonical little-endian bytes constrained to this same field handle.
    /// The bytes reuse this value's decomposition through equality constraints.
    pub fn to_le_bytes(&self, builder: &mut R1csBuilder<Fr>) -> Result<[ByteVar; 16], Fp128Error> {
        self.validate_indices(builder)?;
        Ok(std::array::from_fn(|index| {
            let expression = Self::bits_lc(self.bits.iter().skip(index * 8).take(8));
            let witness = builder
                .evaluate(&expression)
                .ok()
                .and_then(|value| value.to_u64_checked())
                .and_then(|value| u8::try_from(value).ok());
            let byte = ByteVar::allocate(builder, witness);
            builder.assert_equal(byte.expression(), expression);
            byte
        }))
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

    /// Allocate and constrain the canonical difference modulo q.
    pub fn subtract(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Result<Self, Fp128Error> {
        self.validate_indices(builder)?;
        rhs.validate_indices(builder)?;
        let pair = self.witness.zip(rhs.witness);
        let value = pair.map(|(a, b)| {
            (Prime128OffsetA7F7::from_u128(a) - Prime128OffsetA7F7::from_u128(b))
                .to_canonical_u128()
        });
        let output = Self::allocate(builder, value)?;
        let borrow = Self::allocate_bits(builder, pair.map(|(a, b)| u128::from(a < b)), 1);
        // Canonical operands/output and Boolean borrow bound the residual by 2q<r.
        builder.assert_equal(
            LinearCombination::variable(self.variable)
                + Self::bits_lc(&borrow).scale(Fr::from_u128(MODULUS)),
            LinearCombination::variable(rhs.variable)
                + LinearCombination::variable(output.variable),
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
    fn byte_reduction_tail_and_linkage() {
        for h in [0, MODULUS - 1, MODULUS, MODULUS + 1, u128::MAX] {
            let mut builder = R1csBuilder::new();
            let bytes = h
                .to_le_bytes()
                .map(|byte| ByteVar::allocate(&mut builder, Some(byte)));
            let value = Fp128Var::reduce_le_bytes(&mut builder, &bytes).unwrap();
            let mut witness = builder.witness().unwrap();
            assert_eq!(
                witness[value.variable().index()],
                Fr::from_u128(h % MODULUS)
            );
            let matrices = builder.into_matrices();
            assert!(matrices.check_witness(&witness).is_ok());
            // The quotient bit is the final allocation. Even with every canonical
            // value auxiliary unchanged, an incorrect quotient must be rejected.
            let quotient = witness.len() - 1;
            witness[quotient] = Fr::from_u64(1) - witness[quotient];
            assert!(matrices.check_witness(&witness).is_err());
            let mut canonical = R1csBuilder::new();
            let bytes = h.to_le_bytes().map(ByteVar::constant);
            assert_eq!(
                Fp128Var::from_le_bytes(&mut canonical, &bytes).is_ok(),
                h < MODULUS
            );
        }
        let build = |known: bool| {
            let mut builder = R1csBuilder::new();
            let bytes = u128::MAX
                .to_le_bytes()
                .map(|byte| ByteVar::allocate(&mut builder, known.then_some(byte)));
            let _ = Fp128Var::reduce_le_bytes(&mut builder, &bytes).unwrap();
            builder.into_matrices()
        };
        let known = build(true);
        let unknown = build(false);
        assert_eq!(known.num_vars, unknown.num_vars);
        assert_eq!(known.a, unknown.a);
        assert_eq!(known.b, unknown.b);
        assert_eq!(known.c, unknown.c);
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
            MODULUS - 22_538,
            MODULUS - 22_537,
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
    fn completed_witness_rejects_value_bit_and_slack_tampering() {
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

        // Last allocation is the top slack bit; its range is enforced.
        let mut bad_slack = witness;
        let last = bad_slack.len() - 1;
        bad_slack[last] = Fr::from_u64(2);
        assert!(matrices.check_witness(&bad_slack).is_err());
    }

    #[test]
    fn each_missing_high_bit_accepts_the_largest_low_word() {
        for missing in 32..128 {
            let x = u128::MAX ^ (1u128 << missing);
            assert!(BigUint::from(x) < q());
            let mut builder = R1csBuilder::new();
            let _ = Fp128Var::allocate(&mut builder, Some(x)).unwrap();
            let witness = builder.witness().unwrap();
            assert!(builder.into_matrices().check_witness(&witness).is_ok());
        }
    }

    #[test]
    fn forged_prefix_selector_and_inverse_are_rejected() {
        for x in [0, MODULUS - 1] {
            let mut builder = R1csBuilder::new();
            let value = Fp128Var::allocate(&mut builder, Some(x)).unwrap();
            let h = value.bits.last().unwrap().index() + 1;
            let inverse = h + 1;
            let mut witness = builder.witness().unwrap();
            let matrices = builder.into_matrices();
            if x == 0 {
                // S!=0: h=1 and inv=0 satisfy S*inv=1-h, but not S*h=0.
                witness[h] = Fr::from_u64(1);
                witness[inverse] = Fr::from_u64(0);
                for (i, bit) in witness.iter_mut().skip(inverse + 1).enumerate() {
                    *bit = Fr::from_u128((22_536u128 >> i) & 1);
                }
                assert!(matrices.check_witness(&witness).is_err());
                // Correct h=0 still requires the nonzero inverse.
                witness[h] = Fr::from_u64(0);
                assert!(matrices.check_witness(&witness).is_err());
            } else {
                // S=0: h=0 would disable the low bound, but violates the second row.
                witness[value.variable.index()] = Fr::from_u128(MODULUS);
                for (i, bit) in value.bits.iter().enumerate() {
                    witness[bit.index()] = Fr::from_u128((MODULUS >> i) & 1);
                }
                witness[h] = Fr::from_u64(0);
                assert!(matrices.check_witness(&witness).is_err());
            }
        }
    }

    #[test]
    fn negative_slack_cannot_admit_noncanonical_values() {
        let mut builder = R1csBuilder::new();
        let value = Fp128Var::allocate(&mut builder, Some(MODULUS - 1)).unwrap();
        let first_slack = value.bits.last().unwrap().index() + 3;
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        for x in [MODULUS, MODULUS + 1, u128::MAX] {
            let mut forged = witness.clone();
            forged[value.variable.index()] = Fr::from_u128(x);
            for (i, bit) in value.bits.iter().enumerate() {
                forged[bit.index()] = Fr::from_u128((x >> i) & 1);
            }
            // This makes the gated integer sum correct in Fr. Only slack
            // bitness prevents representing the necessary negative integer.
            forged[first_slack] = -Fr::from_u128(x - (MODULUS - 1));
            assert!(matrices.check_witness(&forged).is_err());
        }
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
        // Replace the entire canonical allocation consistently, so rejection
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect and corrupt complete field assignments"
)]
mod subtraction_tests {
    use super::*;
    use num_bigint::BigUint;

    #[test]
    fn subtraction_has_integer_ground_truth_and_constrained_borrow() {
        for (a, b) in [(0, MODULUS - 1), (MODULUS - 1, 0), (5, 7), (9, 9)] {
            let mut builder = R1csBuilder::new();
            let x = Fp128Var::allocate(&mut builder, Some(a)).unwrap();
            let y = Fp128Var::allocate(&mut builder, Some(b)).unwrap();
            let result = x.subtract(&mut builder, &y).unwrap();
            let expected: u128 = ((BigUint::from(a) + BigUint::from(MODULUS) - BigUint::from(b))
                % BigUint::from(MODULUS))
            .try_into()
            .unwrap();
            let mut witness = builder.witness().unwrap();
            assert_eq!(witness[result.variable().index()], Fr::from_u128(expected));
            let matrices = builder.into_matrices();
            assert!(matrices.check_witness(&witness).is_ok());
            let borrow = witness.last_mut().unwrap();
            *borrow = Fr::from_u64(1) - *borrow;
            assert!(matrices.check_witness(&witness).is_err());
        }
    }
}

#[cfg(all(test, feature = "integer-bn254"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fixed byte binding witness tests"
)]
mod byte_binding_tests {
    use super::*;
    use crate::integer_bn254::SignedVar;

    #[test]
    fn canonical_bytes_centering_and_unknown_shape_share_one_value() {
        let make = |known: bool| {
            let mut builder = R1csBuilder::new();
            let bytes = (MODULUS - 1)
                .to_le_bytes()
                .map(|byte| ByteVar::allocate(&mut builder, known.then_some(byte)));
            let value = Fp128Var::from_le_bytes(&mut builder, &bytes).unwrap();
            let encoded = value.to_le_bytes(&mut builder).unwrap();
            for (input, output) in bytes.iter().zip(&encoded) {
                builder.assert_equal(input.expression(), output.expression());
            }
            let centered = SignedVar::centered_from_handle(&mut builder, &value).unwrap();
            (builder, centered.variable())
        };
        let (known, centered) = make(true);
        let (unknown, _) = make(false);
        let witness = known.witness().unwrap();
        assert_eq!(witness[centered.index()], -Fr::from_u64(1));
        let matrix = known.into_matrices();
        let layout = unknown.into_matrices();
        assert_eq!(matrix.a, layout.a);
        assert_eq!(matrix.b, layout.b);
        assert_eq!(matrix.c, layout.c);
        assert_eq!(matrix.num_vars, layout.num_vars);
        assert!(matrix.check_witness(&witness).is_ok());
        let mut wrong = witness;
        wrong[1] = Fr::from_u64(1) - wrong[1];
        assert!(matrix.check_witness(&wrong).is_err());
    }
}
