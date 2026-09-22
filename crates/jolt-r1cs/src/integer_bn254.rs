//! Exact bounded integer relations over BN254. ONE must be fixed externally.
//! Handles retain the same-builder precondition; index checks do not prove provenance.
use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, Ring};
use num_bigint::BigUint;
use thiserror::Error;

use crate::bn254_bits::{BitVar, ByteVar};
use crate::fp128_bn254::{Fp128Error, Fp128Var, MODULUS};
use crate::{LinearCombination, R1csBuilder, Variable};

/// Invalid ranges, assignments, indices, or a missing integer no-wrap certificate.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum IntegerError {
    #[error("integer bound is unsupported or modulus is zero")]
    InvalidBound,
    #[error("witness is outside the prescribed exact range")]
    OutOfRange,
    #[error("integer residual may wrap the BN254 scalar modulus")]
    MayWrap,
    #[error("variable {variable:?} is outside this builder")]
    UnknownVariable { variable: Variable },
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error("shape or prescribed range mismatch")]
    Shape,
}

/// A materialized integer in the exact interval [-bound, bound].
#[derive(Clone, Debug)]
pub struct SignedVar {
    variable: Variable,
    bound: u128,
    witness: Option<i128>,
}

impl SignedVar {
    /// Allocate a signed integer with exact range, independently of witness generation.
    pub fn allocate(
        builder: &mut R1csBuilder<Fr>,
        bound: u128,
        witness: Option<i128>,
    ) -> Result<Self, IntegerError> {
        if bound > i128::MAX as u128 {
            return Err(IntegerError::InvalidBound);
        }
        if witness.is_some_and(|x| x.unsigned_abs() > bound) {
            return Err(IntegerError::OutOfRange);
        }
        let offset = witness.map(|x| {
            if x < 0 {
                bound - x.unsigned_abs()
            } else {
                bound + x.unsigned_abs()
            }
        });
        let encoded = Self::unsigned(builder, 2 * bound, offset)?;
        let variable = builder.alloc_witness(witness.map(Self::field));
        builder.assert_equal(
            variable,
            encoded - LinearCombination::constant(Fr::from_u128(bound)),
        );
        Ok(Self {
            variable,
            bound,
            witness,
        })
    }

    /// Link the unique centered q representative to an already canonical field handle.
    /// The optional signed assignment is checked and constrained, never trusted.
    pub fn centered(
        builder: &mut R1csBuilder<Fr>,
        canonical: &Fp128Var,
        witness: Option<i128>,
    ) -> Result<Self, IntegerError> {
        canonical.validate_indices(builder)?;
        let value = Self::allocate(builder, (MODULUS - 1) / 2, witness)?;
        let sign = builder.alloc_witness(witness.map(|x| Fr::from_u64(u64::from(x < 0))));
        let sign_lc = LinearCombination::variable(sign);
        builder.assert_product(
            sign_lc.clone(),
            sign_lc.clone() - LinearCombination::one(),
            LinearCombination::zero(),
        );
        builder.assert_equal(
            value.variable,
            LinearCombination::variable(canonical.variable())
                - sign_lc.scale(Fr::from_u128(MODULUS)),
        );
        Ok(value)
    }

    /// Center an existing canonical handle, deriving only the auxiliary assignment.
    /// The same range and linkage constraints as `centered` enforce the result.
    pub fn centered_from_handle(
        builder: &mut R1csBuilder<Fr>,
        canonical: &Fp128Var,
    ) -> Result<Self, IntegerError> {
        let witness = builder
            .evaluate(&canonical.variable().into())
            .ok()
            .and_then(|x| x.to_u128_checked())
            .map(|x| {
                if x >= MODULUS {
                    return Err(Fp128Error::NonCanonical { value: x });
                }
                Ok(if x > (MODULUS - 1) / 2 {
                    -((MODULUS - x) as i128)
                } else {
                    x as i128
                })
            })
            .transpose()?;
        Self::centered(builder, canonical, witness)
    }

    /// Constrain the absolute value in [0,bound], including its sign linkage.
    /// Since 2*bound<2^128<r, the signed equation cannot admit a wraparound alias.
    pub fn absolute_value(
        &self,
        builder: &mut R1csBuilder<Fr>,
    ) -> Result<LinearCombination<Fr>, IntegerError> {
        self.validate(builder)?;
        let magnitude = Self::unsigned(builder, self.bound, self.witness.map(i128::unsigned_abs))?;
        let sign = BitVar::allocate(builder, self.witness.map(|x| x < 0));
        builder.assert_product(
            magnitude.clone(),
            LinearCombination::one() - sign.expression().scale(Fr::from_u64(2)),
            self.variable,
        );
        Ok(magnitude)
    }

    /// The constrained signed field representative.
    pub fn variable(&self) -> Variable {
        self.variable
    }
    /// The exact absolute bound established by allocation.
    pub fn bound(&self) -> u128 {
        self.bound
    }

    fn validate(&self, builder: &R1csBuilder<Fr>) -> Result<(), IntegerError> {
        if self.variable.index() >= builder.num_vars() {
            return Err(IntegerError::UnknownVariable {
                variable: self.variable,
            });
        }
        Ok(())
    }

    fn field(x: i128) -> Fr {
        let magnitude = Fr::from_u128(x.unsigned_abs());
        if x < 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    // Both decompositions have at most 128 bits. Their sum is below 2^129<r,
    // so equality with bound enforces the exact interval, not merely a bit width.
    fn unsigned(
        builder: &mut R1csBuilder<Fr>,
        bound: u128,
        witness: Option<u128>,
    ) -> Result<LinearCombination<Fr>, IntegerError> {
        if witness.is_some_and(|x| x > bound) {
            return Err(IntegerError::OutOfRange);
        }
        let bytes = (128 - bound.leading_zeros()).div_ceil(8) as usize;
        let mut value = LinearCombination::zero();
        let mut slack = LinearCombination::zero();
        for i in 0..bytes {
            let shift = i * 8;
            let a = ByteVar::allocate(builder, witness.map(|x| (x >> shift) as u8));
            let b = ByteVar::allocate(builder, witness.map(|x| ((bound - x) >> shift) as u8));
            let scale = Fr::from_u128(1u128 << shift);
            value = value + a.expression().scale(scale);
            slack = slack + b.expression().scale(scale);
        }
        builder.assert_equal(
            value.clone() + slack,
            LinearCombination::constant(Fr::from_u128(bound)),
        );
        Ok(value)
    }

    /// Enforce the complete squared L2 norm of precisely the supplied list.
    /// Callers own list completeness; the Akita terminal owner checks its full shape.
    pub fn enforce_squared_l2(
        builder: &mut R1csBuilder<Fr>,
        values: &[Self],
        cap: u128,
    ) -> Result<(), IntegerError> {
        let mut bound = BigUint::from(cap);
        let mut norm = Some(0u128);
        for value in values {
            value.validate(builder)?;
            bound += BigUint::from(value.bound).pow(2);
            norm = match (norm, value.witness) {
                (Some(sum), Some(x)) => Some(
                    x.unsigned_abs()
                        .checked_mul(x.unsigned_abs())
                        .and_then(|v| sum.checked_add(v))
                        .ok_or(IntegerError::OutOfRange)?,
                ),
                _ => None,
            };
        }
        Self::check_no_wrap(&bound)?;
        if norm.is_some_and(|n| n > cap) {
            return Err(IntegerError::OutOfRange);
        }
        let slack = Self::unsigned(builder, cap, norm.map(|n| cap - n))?;
        let mut sum = slack;
        for value in values {
            sum = sum + builder.multiply(value.variable, value.variable);
        }
        builder.assert_equal(sum, LinearCombination::constant(Fr::from_u128(cap)));
        Ok(())
    }

    fn check_no_wrap(bound: &BigUint) -> Result<(), IntegerError> {
        let modulus =
            BigUint::from_bytes_le(&(-Fr::from_u64(1)).to_bytes_le_vec()) + BigUint::from(1u8);
        if bound >= &modulus {
            return Err(IntegerError::MayWrap);
        }
        Ok(())
    }
}

/// One fixed-coefficient linear or bilinear contribution to an integer row.
pub enum IntegerTerm<'a> {
    Linear {
        coefficient: i128,
        value: &'a SignedVar,
    },
    Product {
        coefficient: i128,
        lhs: &'a SignedVar,
        rhs: &'a SignedVar,
    },
}

impl IntegerTerm<'_> {
    /// Enforce sum(terms)=modulus*quotient as an integer equality.
    /// Checks the actual declared bounds and quotient range in release before emission.
    pub fn enforce_row(
        builder: &mut R1csBuilder<Fr>,
        terms: &[Self],
        modulus: u128,
        quotient: &SignedVar,
    ) -> Result<(), IntegerError> {
        if modulus == 0 {
            return Err(IntegerError::InvalidBound);
        }
        quotient.validate(builder)?;
        let mut bound = BigUint::from(modulus) * BigUint::from(quotient.bound);
        for term in terms {
            match term {
                Self::Linear { coefficient, value } => {
                    value.validate(builder)?;
                    bound += BigUint::from(coefficient.unsigned_abs()) * BigUint::from(value.bound);
                }
                Self::Product {
                    coefficient,
                    lhs,
                    rhs,
                } => {
                    lhs.validate(builder)?;
                    rhs.validate(builder)?;
                    bound += BigUint::from(coefficient.unsigned_abs())
                        * BigUint::from(lhs.bound)
                        * BigUint::from(rhs.bound);
                }
            }
        }
        SignedVar::check_no_wrap(&bound)?;
        let mut sum = LinearCombination::zero();
        for term in terms {
            sum = sum
                + match term {
                    Self::Linear { coefficient, value } => {
                        LinearCombination::variable(value.variable)
                            .scale(SignedVar::field(*coefficient))
                    }
                    Self::Product {
                        coefficient,
                        lhs,
                        rhs,
                    } => builder
                        .multiply(lhs.variable, rhs.variable)
                        .scale(SignedVar::field(*coefficient)),
                };
        }
        builder.assert_equal(
            sum,
            LinearCombination::variable(quotient.variable).scale(Fr::from_u128(modulus)),
        );
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests mutate complete assignments at known allocated coordinates"
)]
mod tests {
    use super::*;

    #[test]
    fn exact_signed_range_rejects_bitwidth_slack_attack() {
        let mut builder = R1csBuilder::new();
        let value = SignedVar::allocate(&mut builder, 3, Some(3)).unwrap();
        let mut witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        // Forge offset=7, slack=0, x=4: every bit and x binding is valid,
        // but offset+slack=6 must fail. Rounded byte bounds alone accept this.
        for i in 0..8 {
            witness[1 + i] = Fr::from_u64(u64::from(i < 3));
        }
        witness[value.variable().index()] = Fr::from_u64(4);
        assert_eq!(matrices.check_witness(&witness), Err(16));
    }

    #[test]
    fn centered_values_bind_negative_representatives_and_sign() {
        let mut builder = R1csBuilder::new();
        let canonical = Fp128Var::allocate(&mut builder, Some(MODULUS - 1)).unwrap();
        let value = SignedVar::centered(&mut builder, &canonical, Some(-1)).unwrap();
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        let mut wrong_sign = witness.clone();
        *wrong_sign.last_mut().unwrap() = Fr::from_u64(0);
        assert!(matrices.check_witness(&wrong_sign).is_err());
        let mut wrong_integer = witness;
        wrong_integer[value.variable().index()] = Fr::from_u128(MODULUS - 1);
        assert!(matrices.check_witness(&wrong_integer).is_err());
    }

    #[test]
    fn complete_norm_rejects_in_range_coordinates_with_excess_total() {
        let mut builder = R1csBuilder::new();
        let values: Vec<_> = [3, 3]
            .into_iter()
            .map(|x| SignedVar::allocate(&mut builder, 5, Some(x)).unwrap())
            .collect();
        let prefix = builder.num_vars();
        SignedVar::enforce_squared_l2(&mut builder, &values, 25).unwrap();
        let mut witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        let mut replacement = R1csBuilder::new();
        for _ in 0..2 {
            let _value = SignedVar::allocate(&mut replacement, 5, Some(4)).unwrap();
        }
        witness[..prefix].copy_from_slice(&replacement.witness().unwrap());
        let len = witness.len();
        witness[len - 2] = Fr::from_u64(16);
        witness[len - 1] = Fr::from_u64(16);
        assert_eq!(
            matrices.check_witness(&witness),
            Err(matrices.num_constraints - 1)
        );
    }

    #[test]
    fn integer_row_accepts_signed_products_and_rejects_bad_quotient() {
        // 7*(-4) - 2*3 = 17*(-2), over integers, not merely modulo r.
        let mut builder = R1csBuilder::new();
        let a = SignedVar::allocate(&mut builder, 7, Some(7)).unwrap();
        let b = SignedVar::allocate(&mut builder, 4, Some(-4)).unwrap();
        let c = SignedVar::allocate(&mut builder, 3, Some(3)).unwrap();
        let q = SignedVar::allocate(&mut builder, 2, Some(-2)).unwrap();
        IntegerTerm::enforce_row(
            &mut builder,
            &[
                IntegerTerm::Product {
                    coefficient: 1,
                    lhs: &a,
                    rhs: &b,
                },
                IntegerTerm::Linear {
                    coefficient: -2,
                    value: &c,
                },
            ],
            17,
            &q,
        )
        .unwrap();
        let mut witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        witness[q.variable().index()] = Fr::from_u64(2);
        assert!(matrices.check_witness(&witness).is_err());
    }

    #[test]
    fn release_guard_rejects_a_native_modulus_alias_before_product_emission() {
        let r = BigUint::from_bytes_le(&(-Fr::from_u64(1)).to_bytes_le_vec()) + BigUint::from(1u8);
        let a = i128::MAX;
        let b: i128 = (&r / BigUint::from(a as u128)).try_into().unwrap();
        let c: i128 = (&r % BigUint::from(a as u128)).try_into().unwrap();
        for sign in [-1, 1] {
            let mut builder = R1csBuilder::new();
            let x = SignedVar::allocate(&mut builder, a as u128, Some(a)).unwrap();
            let y = SignedVar::allocate(&mut builder, b as u128, Some(b)).unwrap();
            let z = SignedVar::allocate(&mut builder, c as u128, Some(c)).unwrap();
            let q = SignedVar::allocate(&mut builder, 0, Some(0)).unwrap();
            let variables = builder.num_vars();
            assert_eq!(
                IntegerTerm::enforce_row(
                    &mut builder,
                    &[
                        IntegerTerm::Product {
                            coefficient: sign,
                            lhs: &x,
                            rhs: &y
                        },
                        IntegerTerm::Linear {
                            coefficient: sign,
                            value: &z
                        },
                    ],
                    MODULUS,
                    &q
                ),
                Err(IntegerError::MayWrap)
            );
            assert_eq!(builder.num_vars(), variables);
        }
    }

    #[test]
    fn integer_layout_is_independent_of_assignments_and_checks_indices() {
        let mut known = R1csBuilder::new();
        let x = SignedVar::allocate(&mut known, 13, Some(-9)).unwrap();
        SignedVar::enforce_squared_l2(&mut known, std::slice::from_ref(&x), 100).unwrap();
        let mut unknown = R1csBuilder::new();
        let y = SignedVar::allocate(&mut unknown, 13, None).unwrap();
        SignedVar::enforce_squared_l2(&mut unknown, &[y], 100).unwrap();
        let known = known.into_matrices();
        let unknown = unknown.into_matrices();
        assert_eq!(known.num_vars, unknown.num_vars);
        assert_eq!(known.a, unknown.a);
        assert_eq!(known.b, unknown.b);
        assert_eq!(known.c, unknown.c);
        assert!(matches!(
            SignedVar::enforce_squared_l2(&mut R1csBuilder::new(), &[x], 100),
            Err(IntegerError::UnknownVariable { .. })
        ));
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests tamper the absolute-value sign")]
mod absolute_tests {
    use super::*;
    #[test]
    fn absolute_value_sign_is_constrained() {
        let mut builder = R1csBuilder::new();
        let value = SignedVar::allocate(&mut builder, 9, Some(-7)).unwrap();
        let absolute = value.absolute_value(&mut builder).unwrap();
        builder.assert_equal(absolute, LinearCombination::constant(Fr::from_u64(7)));
        let mut witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        *witness.last_mut().unwrap() = Fr::from_u64(0);
        assert!(matrices.check_witness(&witness).is_err());
    }
}
