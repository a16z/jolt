//! Exact acceptance of one D64 mixed-shell candidate, independent of its sampler.
//!
//! The dense input must later be bound to sampled positions/signs. This module
//! establishes shell counts, all 32 frequency inequalities, and the actual native
//! table error; it does not establish byte consumption, retries or first acceptance.
use akita_challenges::{
    d64_selective_l2_op_norm_table, OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use jolt_field::{Fr, Ring};
use jolt_r1cs::bn254_bits::BitVar;
use jolt_r1cs::integer_bn254::{IntegerError, SignedVar};
use jolt_r1cs::{LinearCombination, R1csBuilder, Variable};
use thiserror::Error;

/// Invalid shell/handles, unsupported native parameters, or a rejected assignment.
#[derive(Debug, Error)]
pub enum OperatorNormR1csError {
    #[error("candidate is not the D64 (31,11) mixed shell")]
    InvalidShell,
    #[error("native operator-norm constants differ from the supported profile")]
    UnsupportedParameters,
    #[error("native operator-norm table initialization failed: {0}")]
    Native(String),
    #[error("candidate fails the strict upper inequality at frequency {frequency}")]
    Rejected { frequency: usize },
    #[error(transparent)]
    Integer(#[from] IntegerError),
}

/// Dense D64 coefficients constrained to exactly 31 magnitude-one and 11
/// magnitude-two entries. Zero signs may be either bit; the polynomial is unique.
pub struct D64ShellVar {
    coefficients: [Variable; 64],
    witness: Option<[i8; 64]>,
}

impl D64ShellVar {
    /// Bind existing coefficient handles to the exact mixed shell. Provided
    /// assignments generate auxiliaries but every coefficient/range/count is constrained.
    pub fn bind(
        builder: &mut R1csBuilder<Fr>,
        coefficients: [Variable; 64],
        witness: Option<[i8; 64]>,
    ) -> Result<Self, OperatorNormR1csError> {
        for &variable in &coefficients {
            if variable.index() >= builder.num_vars() {
                return Err(IntegerError::UnknownVariable { variable }.into());
            }
        }
        if witness.is_some_and(|values| {
            values.iter().any(|x| x.unsigned_abs() > 2)
                || values.iter().filter(|x| x.unsigned_abs() == 1).count() != 31
                || values.iter().filter(|x| x.unsigned_abs() == 2).count() != 11
        }) {
            return Err(OperatorNormR1csError::InvalidShell);
        }
        let mut units = LinearCombination::zero();
        let mut doubles = LinearCombination::zero();
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let value = witness
                .as_ref()
                .and_then(|values| values.get(index))
                .copied();
            let unit = BitVar::allocate(builder, value.map(|x| x.unsigned_abs() == 1));
            let double = BitVar::allocate(builder, value.map(|x| x.unsigned_abs() == 2));
            let sign = BitVar::allocate(builder, value.map(|x| x < 0));
            builder.assert_product(
                unit.expression(),
                double.expression(),
                LinearCombination::zero(),
            );
            builder.assert_product(
                unit.expression() + double.expression().scale(Fr::from_u64(2)),
                LinearCombination::one() - sign.expression().scale(Fr::from_u64(2)),
                coefficient,
            );
            units = units + unit.expression();
            doubles = doubles + double.expression();
        }
        builder.assert_equal(units, LinearCombination::constant(Fr::from_u64(31)));
        builder.assert_equal(doubles, LinearCombination::constant(Fr::from_u64(11)));
        Ok(Self {
            coefficients,
            witness,
        })
    }

    /// Enforce the AND of all 32 exact native upper inequalities. A known
    /// rejected assignment returns a typed error; unknown layout includes every
    /// frequency. Construction errors do not roll earlier builder allocations back.
    pub fn enforce_accepted(
        &self,
        builder: &mut R1csBuilder<Fr>,
    ) -> Result<(), OperatorNormR1csError> {
        for &variable in &self.coefficients {
            if variable.index() >= builder.num_vars() {
                return Err(IntegerError::UnknownVariable { variable }.into());
            }
        }
        let table = d64_selective_l2_op_norm_table()
            .map_err(|error| OperatorNormR1csError::Native(error.to_string()))?;
        let config = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
        let policy = OperatorNormRejection::D64_SELECTIVE_L2;
        let (cos, sin) = table.frequency_tables();
        if config.count_pm1 != 31
            || config.count_pm2 != 11
            || policy.threshold != 18
            || table.fractional_bits() != 48
            || table.root_coordinate_error() != 1
            || cos.len() != 2048
            || sin.len() != 2048
            || cos.iter().chain(sin).any(|x| x.unsigned_abs() > 1u64 << 48)
        {
            return Err(OperatorNormR1csError::UnsupportedParameters);
        }
        let radius = config.l1_norm() as u128 * table.root_coordinate_error() as u128;
        let bound = radius << table.fractional_bits();
        let threshold = u128::from(policy.threshold).pow(2) << (2 * table.fractional_bits());
        // |R|,|I|<=53*2^48; U and its comparison residual are below 2^111<r.
        // Table/profile validation above pins every factor in these bounds.
        for frequency in 0..32 {
            let mut accumulators = Vec::with_capacity(2);
            for entries in [cos, sin] {
                let mut expression = LinearCombination::zero();
                let mut witness = self.witness.as_ref().map(|_| 0i128);
                for (position, &variable) in self.coefficients.iter().enumerate() {
                    let entry = *entries
                        .get(position * 32 + frequency)
                        .ok_or(OperatorNormR1csError::UnsupportedParameters)?;
                    let scalar = Fr::from_u64(entry.unsigned_abs());
                    expression = expression
                        + LinearCombination::variable(variable).scale(if entry < 0 {
                            -scalar
                        } else {
                            scalar
                        });
                    witness = witness
                        .zip(
                            self.witness
                                .as_ref()
                                .and_then(|values| values.get(position)),
                        )
                        .map(|(sum, &coefficient)| {
                            sum + i128::from(coefficient) * i128::from(entry)
                        });
                }
                let sum = SignedVar::allocate(builder, bound, witness)?;
                builder.assert_equal(sum.variable(), expression);
                accumulators.push((sum, witness));
            }
            let [(real, real_value), (imaginary, imaginary_value)]: [(SignedVar, Option<i128>); 2] =
                accumulators
                    .try_into()
                    .map_err(|_| OperatorNormR1csError::UnsupportedParameters)?;
            Self::enforce_frequency(
                builder,
                &real,
                &imaginary,
                real_value.zip(imaginary_value),
                radius,
                threshold,
                frequency,
            )?;
        }
        Ok(())
    }
    fn enforce_frequency(
        builder: &mut R1csBuilder<Fr>,
        real: &SignedVar,
        imaginary: &SignedVar,
        witness: Option<(i128, i128)>,
        radius: u128,
        threshold: u128,
        frequency: usize,
    ) -> Result<(), OperatorNormR1csError> {
        let upper_witness = witness.map(|(r, i)| {
            let r = r.unsigned_abs();
            let i = i.unsigned_abs();
            r * r + i * i + 2 * radius * (r + i) + 2 * radius * radius
        });
        if upper_witness.is_some_and(|upper| upper > threshold) {
            return Err(OperatorNormR1csError::Rejected { frequency });
        }
        let abs_sum = real.absolute_value(builder)? + imaginary.absolute_value(builder)?;
        let upper = builder.multiply(real.variable(), real.variable())
            + builder.multiply(imaginary.variable(), imaginary.variable())
            + abs_sum.scale(Fr::from_u128(2 * radius))
            + LinearCombination::constant(Fr::from_u128(2 * radius * radius));
        // A signed interval suffices because U is nonnegative and U+threshold<r.
        let comparison = SignedVar::allocate(builder, threshold, upper_witness.map(|x| x as i128))?;
        builder.assert_equal(upper, comparison.variable());
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests inspect and corrupt constrained assignments"
)]
mod tests {
    use super::*;
    const ACCEPTED: [i8; 64] = [
        0, 0, 2, 1, 1, 0, 1, 2, 1, 2, -2, 0, 1, 2, 1, 1, 2, 1, -1, -1, 2, 1, 1, -1, -2, 1, -1, 0,
        -1, 0, -1, 0, 1, -1, 0, 0, -2, 0, -1, 0, 1, 0, 0, 0, -2, 0, -1, 0, 0, -1, 0, 0, -1, 1, 1,
        -1, 2, 0, 1, -1, 1, 0, 1, 0,
    ];

    fn allocate(builder: &mut R1csBuilder<Fr>, witness: Option<[i8; 64]>) -> D64ShellVar {
        let coefficients = std::array::from_fn(|i| {
            builder.alloc_witness(witness.map(|values| {
                let value = values[i];
                let magnitude = Fr::from_u64(u64::from(value.unsigned_abs()));
                if value < 0 {
                    -magnitude
                } else {
                    magnitude
                }
            }))
        });
        D64ShellVar::bind(builder, coefficients, witness).unwrap()
    }

    #[test]
    fn exact_predicate_matches_native_and_checks_later_frequencies() {
        let table = d64_selective_l2_op_norm_table().unwrap();
        let positions: Vec<_> = (0..64).collect();
        let mut accepted = R1csBuilder::new();
        let shell = allocate(&mut accepted, Some(ACCEPTED));
        assert!(table
            .accept_strict_parts(&positions, &ACCEPTED, 18)
            .unwrap());
        shell.enforce_accepted(&mut accepted).unwrap();
        let witness = accepted.witness().unwrap();
        let matrices = accepted.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        let mut wrong = witness.clone();
        wrong[shell.coefficients[0].index()] = Fr::from_u64(1);
        assert!(matrices.check_witness(&wrong).is_err());
        let mut bad_upper = witness;
        *bad_upper.last_mut().unwrap() += Fr::from_u64(1);
        assert!(matrices.check_witness(&bad_upper).is_err());
        let mut unknown = R1csBuilder::new();
        let shell = allocate(&mut unknown, None);
        shell.enforce_accepted(&mut unknown).unwrap();
        let unknown = unknown.into_matrices();
        assert_eq!(matrices.num_vars, unknown.num_vars);
        assert_eq!(matrices.a, unknown.a);
        assert_eq!(matrices.b, unknown.b);
        assert_eq!(matrices.c, unknown.c);
        for alternating in [false, true] {
            let rejected = std::array::from_fn(|i| {
                let magnitude = if i < 31 {
                    1
                } else if i < 42 {
                    2
                } else {
                    0
                };
                if alternating && i % 2 != 0 {
                    -magnitude
                } else {
                    magnitude
                }
            });
            assert!(!table
                .accept_strict_parts(&positions, &rejected, 18)
                .unwrap());
            let mut builder = R1csBuilder::new();
            let shell = allocate(&mut builder, Some(rejected));
            let error = shell.enforce_accepted(&mut builder).unwrap_err();
            assert!(
                matches!(error, OperatorNormR1csError::Rejected { frequency } if !alternating || frequency>0)
            );
        }
    }

    #[test]
    fn shell_count_is_enforced_after_coherent_zero_coordinate_mutation() {
        let mut builder = R1csBuilder::new();
        let shell = allocate(&mut builder, Some(ACCEPTED));
        let mut witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        // Position0 was zero. Change its coefficient and unit bit together;
        // bitness, exclusivity and signed coefficient relation remain valid.
        witness[shell.coefficients[0].index()] = Fr::from_u64(1);
        witness[65] = Fr::from_u64(1);
        assert_eq!(
            matrices.check_witness(&witness),
            Err(matrices.num_constraints - 2)
        );
    }

    #[test]
    fn exact_upper_boundary_and_forged_comparison_assignment() {
        let radius = 53;
        let scale = 1u128 << 48;
        let threshold = 324 * scale * scale;
        for sign in [-1, 1] {
            for (gap, accept) in [(54, true), (53, false)] {
                let value = sign * (18 * scale - gap) as i128;
                let mut builder = R1csBuilder::new();
                let real = SignedVar::allocate(&mut builder, 53 * scale, Some(value)).unwrap();
                let imaginary = SignedVar::allocate(&mut builder, 53 * scale, Some(0)).unwrap();
                let result = D64ShellVar::enforce_frequency(
                    &mut builder,
                    &real,
                    &imaginary,
                    Some((value, 0)),
                    radius,
                    threshold,
                    0,
                );
                assert_eq!(result.is_ok(), accept);
                if accept {
                    assert!(builder
                        .clone()
                        .into_matrices()
                        .check_witness(&builder.witness().unwrap())
                        .is_ok());
                } else {
                    // Bypass only the honest-assignment rejection. The emitted
                    // equation still uses the real accumulator and must reject.
                    D64ShellVar::enforce_frequency(
                        &mut builder,
                        &real,
                        &imaginary,
                        Some((0, 0)),
                        radius,
                        threshold,
                        0,
                    )
                    .unwrap();
                    let witness = builder.witness().unwrap();
                    let matrices = builder.into_matrices();
                    assert_eq!(
                        matrices.check_witness(&witness),
                        Err(matrices.num_constraints - 1)
                    );
                }
            }
        }
    }
}
