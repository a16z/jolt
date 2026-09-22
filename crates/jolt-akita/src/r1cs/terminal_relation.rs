//! Shared-handle D64 terminal equations. See `specs/akita-wrapper/terminal-relation.md`.
use akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG;
use akita_types::TerminalFoldParams;
use jolt_field::{Fr, Ring};
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var, MODULUS};
use jolt_r1cs::integer_bn254::{IntegerError, IntegerTerm, SignedVar};
use jolt_r1cs::{R1csBuilder, Variable};
use thiserror::Error;

use super::operator_norm::{D64ShellVar, OperatorNormR1csError};
use super::scalar::{ScalarR1csError, TerminalPointVar};
use super::terminal::{
    TerminalZ, A_QUOTIENT_BOUND, CAP, CONSISTENCY_QUOTIENT_BOUND, DEGREE, LENGTH, Q, WIDTH,
};

const RANK: usize = 3;
const BLOCKS: usize = 7;

#[derive(Debug, Error)]
pub enum TerminalRelationError {
    #[error("unsupported terminal geometry or handle shape")]
    Shape,
    #[error(transparent)]
    Integer(#[from] IntegerError),
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Scalar(#[from] ScalarR1csError),
    #[error(transparent)]
    Shell(#[from] OperatorNormR1csError),
}

/// Seven private dense challenge arrays, constrained to the native 31+11 shell.
/// Sampling, norm acceptance and the authenticated stream are caller obligations.
/// Passing actual retry output handles links this relation without witness copies.
pub struct TerminalChallengesVar {
    coefficients: Vec<Vec<SignedVar>>,
}

impl TerminalChallengesVar {
    pub fn bind(
        builder: &mut R1csBuilder<Fr>,
        coefficients: &[[Variable; DEGREE]],
    ) -> Result<Self, TerminalRelationError> {
        if coefficients.len() != BLOCKS {
            return Err(TerminalRelationError::Shape);
        }
        let mut result = Vec::with_capacity(BLOCKS);
        for block in coefficients {
            let mut values = Vec::with_capacity(DEGREE);
            for variable in block {
                if variable.index() >= builder.num_vars() {
                    return Err(IntegerError::UnknownVariable {
                        variable: *variable,
                    }
                    .into());
                }
                let value = match builder.evaluate(&(*variable).into()) {
                    Ok(value) => Some(
                        (-2i8..=2)
                            .find(|&x| {
                                let absolute = Fr::from_u64(u64::from(x.unsigned_abs()));
                                value == if x < 0 { -absolute } else { absolute }
                            })
                            .ok_or(TerminalRelationError::Shape)?,
                    ),
                    Err(_) => None,
                };
                values.push(value);
            }
            let witness: Option<Vec<_>> = values.iter().copied().collect();
            let witness = witness
                .map(|x| x.try_into().map_err(|_| TerminalRelationError::Shape))
                .transpose()?;
            let _ = D64ShellVar::bind(builder, *block, witness)?;
            let signed = block
                .iter()
                .zip(values)
                .map(|(&variable, value)| {
                    let signed = SignedVar::allocate(builder, 2, value.map(i128::from))?;
                    builder.assert_equal(signed.variable(), variable);
                    Ok(signed)
                })
                .collect::<Result<Vec<_>, IntegerError>>()?;
            result.push(signed);
        }
        Ok(Self {
            coefficients: result,
        })
    }
}

/// Logical inputs shared across every equation. ONE and same-builder provenance
/// are external; the profile authenticates neither the point nor the proof bytes.
pub struct TerminalRelationInputs<'a> {
    pub z: &'a TerminalZ,
    pub e: &'a [Fp128Var],
    pub t: &'a [Fp128Var],
    pub point: &'a [Fp128Var],
    pub claim: &'a Fp128Var,
    pub challenges: &'a TerminalChallengesVar,
}

/// Private quotient auxiliaries, in rank-major/coordinate-major row order.
/// None entries produce the same matrices with unknown assignments.
pub struct TerminalRelationWitness<'a> {
    pub a_quotients: &'a [Option<i128>],
    pub consistency_quotients: &'a [Option<i128>],
}

/// Checked public fixed-profile geometry and canonical setup matrix coefficients.
/// This shape check is not authentication of the schedule, descriptor or matrix.
pub struct TerminalRelationProfile {
    a: Vec<i128>,
}

impl TerminalRelationProfile {
    pub fn new(params: &TerminalFoldParams, a: &[u128]) -> Result<Self, TerminalRelationError> {
        let layout = &params.response_shape.layout;
        let group = match layout.groups.as_slice() {
            [group] => group,
            _ => return Err(TerminalRelationError::Shape),
        };
        if params.d_a() != DEGREE
            || params.inner_width() != WIDTH
            || params.inner.matrix.output_rank() != RANK
            || params.blocks.live_blocks != BLOCKS
            || params.blocks.positions_per_block != WIDTH
            || params.inner.digits.num_digits != 1
            || params.response_l2_sq_cap() != Some(CAP)
            || params.recursive_opening_num_vars().ok() != Some(17)
            || params.fold_challenge_config != D64_SELECTIVE_L2_CHALLENGE_CONFIG
            || layout.ring_dimension != DEGREE
            || group.z_coords != LENGTH
            || group.e_field_elems != BLOCKS * DEGREE
            || group.t_field_elems != BLOCKS * RANK * DEGREE
            || group.z_linf_cap.is_some()
            || a.len() != RANK * LENGTH
        {
            return Err(TerminalRelationError::Shape);
        }
        let a = a
            .iter()
            .map(|&value| {
                if value >= MODULUS {
                    return Err(Fp128Error::NonCanonical { value });
                }
                Ok(if value > Q {
                    -((MODULUS - value) as i128)
                } else {
                    value as i128
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { a })
    }

    /// Enforce all 192 A rows, 64 consistency rows and the scalar opening.
    /// Input canonical/range constraints already exist on the supplied handles;
    /// unique centered e/t and position weights are allocated once and shared.
    /// z's complete physical norm is established by TerminalZ, not reallocated.
    pub fn enforce(
        &self,
        builder: &mut R1csBuilder<Fr>,
        input: TerminalRelationInputs<'_>,
        witness: TerminalRelationWitness<'_>,
    ) -> Result<(), TerminalRelationError> {
        if input.e.len() != BLOCKS * DEGREE
            || input.t.len() != BLOCKS * RANK * DEGREE
            || input.point.len() != 17
            || witness.a_quotients.len() != RANK * DEGREE
            || witness.consistency_quotients.len() != DEGREE
        {
            return Err(TerminalRelationError::Shape);
        }
        for value in input
            .e
            .iter()
            .chain(input.t)
            .chain(input.point)
            .chain(std::iter::once(input.claim))
        {
            value.validate_indices(builder)?;
        }
        let e = input
            .e
            .iter()
            .map(|x| SignedVar::centered_from_handle(builder, x))
            .collect::<Result<Vec<_>, _>>()?;
        let t = input
            .t
            .iter()
            .map(|x| SignedVar::centered_from_handle(builder, x))
            .collect::<Result<Vec<_>, _>>()?;
        let point = TerminalPointVar::prepare(builder, input.point)?;
        let weights = point.centered_positions(builder)?;
        for row in 0..RANK {
            for coordinate in 0..DEGREE {
                let mut terms = Vec::with_capacity(LENGTH + BLOCKS * DEGREE);
                for (index, value) in input.z.coordinates().iter().enumerate() {
                    let position = index / DEGREE;
                    let source = index % DEGREE;
                    let a_index =
                        (row * WIDTH + position) * DEGREE + (coordinate + DEGREE - source) % DEGREE;
                    let coefficient = *self.a.get(a_index).ok_or(TerminalRelationError::Shape)?;
                    terms.push(IntegerTerm::Linear {
                        coefficient: if source <= coordinate {
                            coefficient
                        } else {
                            -coefficient
                        },
                        value,
                    });
                }
                Self::add_rhs(&mut terms, input.challenges, &t, RANK, row, coordinate)?;
                let quotient = SignedVar::allocate(
                    builder,
                    A_QUOTIENT_BOUND,
                    *witness
                        .a_quotients
                        .get(row * DEGREE + coordinate)
                        .ok_or(TerminalRelationError::Shape)?,
                )?;
                IntegerTerm::enforce_row(builder, &terms, MODULUS, &quotient)?;
            }
        }
        for coordinate in 0..DEGREE {
            let mut terms = Vec::with_capacity(WIDTH + BLOCKS * DEGREE);
            for (position, weight) in weights.iter().enumerate() {
                terms.push(IntegerTerm::Product {
                    coefficient: 1,
                    lhs: weight,
                    rhs: input
                        .z
                        .coordinates()
                        .get(position * DEGREE + coordinate)
                        .ok_or(TerminalRelationError::Shape)?,
                });
            }
            Self::add_rhs(&mut terms, input.challenges, &e, 1, 0, coordinate)?;
            let quotient = SignedVar::allocate(
                builder,
                CONSISTENCY_QUOTIENT_BOUND,
                *witness
                    .consistency_quotients
                    .get(coordinate)
                    .ok_or(TerminalRelationError::Shape)?,
            )?;
            IntegerTerm::enforce_row(builder, &terms, MODULUS, &quotient)?;
        }
        point.enforce_scalar_opening(builder, input.e, input.claim)?;
        Ok(())
    }

    fn add_rhs<'a>(
        terms: &mut Vec<IntegerTerm<'a>>,
        challenges: &'a TerminalChallengesVar,
        values: &'a [SignedVar],
        rank: usize,
        row: usize,
        coordinate: usize,
    ) -> Result<(), TerminalRelationError> {
        for (block, coefficients) in challenges.coefficients.iter().enumerate() {
            for (position, coefficient) in coefficients.iter().enumerate() {
                let index =
                    (block * rank + row) * DEGREE + (coordinate + DEGREE - position) % DEGREE;
                terms.push(IntegerTerm::Product {
                    coefficient: if position <= coordinate { -1 } else { 1 },
                    lhs: coefficient,
                    rhs: values.get(index).ok_or(TerminalRelationError::Shape)?,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fixed shell binding fixtures"
)]
mod tests {
    use super::*;

    #[test]
    fn shared_challenge_handles_enforce_shell_and_witness_independent_shape() {
        let make = |shift: usize, known: bool| {
            let mut builder = R1csBuilder::new();
            let blocks: Vec<_> = (0..BLOCKS)
                .map(|_| {
                    std::array::from_fn(|position| {
                        let position = (position + shift) % DEGREE;
                        let value = if position < 31 {
                            1
                        } else if position < 42 {
                            2
                        } else {
                            0
                        };
                        builder.alloc_witness(known.then_some(Fr::from_u64(value)))
                    })
                })
                .collect();
            let _ = TerminalChallengesVar::bind(&mut builder, &blocks).unwrap();
            builder
        };
        let first = make(0, true);
        let shifted = make(1, true);
        let unknown = make(0, false).into_matrices();
        let witness = first.witness().unwrap();
        let other = shifted.witness().unwrap();
        let matrix = first.into_matrices();
        let shifted = shifted.into_matrices();
        assert_eq!(matrix.a, shifted.a);
        assert_eq!(matrix.b, shifted.b);
        assert_eq!(matrix.c, shifted.c);
        assert_eq!(matrix.a, unknown.a);
        assert_eq!(matrix.b, unknown.b);
        assert_eq!(matrix.c, unknown.c);
        assert!(matrix.check_witness(&witness).is_ok());
        assert!(matrix.check_witness(&other).is_ok());
        let mut wrong = witness;
        wrong[1] = Fr::from_u64(2);
        assert!(matrix.check_witness(&wrong).is_err());
        let mut builder = R1csBuilder::new();
        assert!(matches!(
            TerminalChallengesVar::bind(&mut builder, &[]),
            Err(TerminalRelationError::Shape)
        ));
        assert_eq!(builder.num_vars(), 1);
    }
}
