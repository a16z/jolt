use arbitrary::{Arbitrary, Unstructured};
use jolt_field::arkworks::bn254::Fr;
use jolt_field::Field;

use crate::invariant::{CheckError, Invariant, InvariantViolation};

/// Input for the field scalar-multiplication invariant.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct FieldMulScalarInput {
    #[schemars(with = "[u8; 32]")]
    pub field: Fr,
    pub u64_scalar: u64,
    pub i64_scalar: i64,
    pub u128_scalar: u128,
    pub i128_scalar: i128,
}

impl<'a> Arbitrary<'a> for FieldMulScalarInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let bytes: [u8; 32] = u.arbitrary()?;
        Ok(Self {
            field: Fr::from_le_bytes_mod_order(&bytes),
            u64_scalar: u.arbitrary()?,
            i64_scalar: u.arbitrary()?,
            u128_scalar: u.arbitrary()?,
            i128_scalar: u.arbitrary()?,
        })
    }
}

/// `Field::mul_{u64,i64,u128,i128}` must agree with the reference
/// formula `self * Self::from_*(scalar)` for every input.
#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct FieldMulScalarInvariant;

impl Invariant for FieldMulScalarInvariant {
    type Setup = ();
    type Input = FieldMulScalarInput;

    fn name(&self) -> &str {
        "field_mul_scalar"
    }

    fn description(&self) -> String {
        "The optimized Field::mul_{u64,i64,u128,i128} methods on BN254 Fr \
         must produce the same result as the reference `self * Self::from_*(n)`."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: FieldMulScalarInput) -> Result<(), CheckError> {
        let f = input.field;

        // mul_u64
        let got = f.mul_u64(input.u64_scalar);
        let expected = f * Fr::from_u64(input.u64_scalar);
        if got != expected {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "mul_u64 mismatch",
                format!(
                    "field={f:?}, scalar={}, got={got:?}, expected={expected:?}",
                    input.u64_scalar
                ),
            )));
        }

        // mul_i64
        let got = f.mul_i64(input.i64_scalar);
        let expected = f * Fr::from_i64(input.i64_scalar);
        if got != expected {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "mul_i64 mismatch",
                format!(
                    "field={f:?}, scalar={}, got={got:?}, expected={expected:?}",
                    input.i64_scalar
                ),
            )));
        }

        // mul_u128
        let got = f.mul_u128(input.u128_scalar);
        let expected = f * Fr::from_u128(input.u128_scalar);
        if got != expected {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "mul_u128 mismatch",
                format!(
                    "field={f:?}, scalar={}, got={got:?}, expected={expected:?}",
                    input.u128_scalar
                ),
            )));
        }

        // mul_i128
        let got = f.mul_i128(input.i128_scalar);
        let expected = f * Fr::from_i128(input.i128_scalar);
        if got != expected {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "mul_i128 mismatch",
                format!(
                    "field={f:?}, scalar={}, got={got:?}, expected={expected:?}",
                    input.i128_scalar
                ),
            )));
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<FieldMulScalarInput> {
        vec![
            // Zero field element, zero scalars
            FieldMulScalarInput {
                field: Fr::from_u64(0),
                u64_scalar: 0,
                i64_scalar: 0,
                u128_scalar: 0,
                i128_scalar: 0,
            },
            // Identity
            FieldMulScalarInput {
                field: Fr::from_u64(1),
                u64_scalar: 1,
                i64_scalar: 1,
                u128_scalar: 1,
                i128_scalar: 1,
            },
            // Max unsigned scalars
            FieldMulScalarInput {
                field: Fr::from_u64(42),
                u64_scalar: u64::MAX,
                i64_scalar: i64::MAX,
                u128_scalar: u128::MAX,
                i128_scalar: i128::MAX,
            },
            // Min signed scalars
            FieldMulScalarInput {
                field: Fr::from_u64(7),
                u64_scalar: 0,
                i64_scalar: i64::MIN,
                u128_scalar: 0,
                i128_scalar: i128::MIN,
            },
            // -1 scalars (should yield field negation)
            FieldMulScalarInput {
                field: Fr::from_u64(123_456_789),
                u64_scalar: 0,
                i64_scalar: -1,
                u128_scalar: 0,
                i128_scalar: -1,
            },
            // Large field element (high bits set)
            FieldMulScalarInput {
                field: Fr::from_i64(-1),
                u64_scalar: u64::MAX,
                i64_scalar: i64::MIN,
                u128_scalar: u128::MAX,
                i128_scalar: i128::MIN,
            },
            // Zero field element, non-trivial scalars
            FieldMulScalarInput {
                field: Fr::from_u64(0),
                u64_scalar: 42,
                i64_scalar: -42,
                u128_scalar: 42,
                i128_scalar: -42,
            },
        ]
    }
}
