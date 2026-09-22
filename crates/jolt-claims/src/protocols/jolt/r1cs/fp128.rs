//! Private q128 evaluation of the source-owned Spartan outer remainder formula.
use jolt_field::{Fr, Prime128OffsetA7F7};
use jolt_poly::r1cs::{eq_fp128_bn254, Fp128CenteredLagrangeShape, LagrangeR1csError};
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::R1csBuilder;
use thiserror::Error;

use super::{
    rv64, spartan_outer_constraints, spartan_outer_opening_columns, SPARTAN_OUTER_FIRST_GROUP_ROWS,
    SPARTAN_OUTER_SECOND_GROUP_ROWS, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
};
use crate::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use crate::protocols::jolt::relations::spartan::OuterRemainderOutputClaims;

#[derive(Debug, Error)]
pub enum OuterRemainderR1csError {
    #[error("invalid Spartan outer shape or opening column")]
    Shape,
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Polynomial(#[from] LagrangeR1csError),
}

enum MatrixSide {
    A,
    B,
}
enum Column {
    One,
    Opening(usize),
}
type Row = Vec<(Column, u128)>;

/// Public matrix geometry; all challenge-dependent row weights remain private.
pub struct SpartanOuterRemainderR1cs {
    rounds: usize,
    a: Vec<Row>,
    b: Vec<Row>,
    lagrange: Fp128CenteredLagrangeShape,
}
pub struct SpartanOuterRemainderVars {
    pub kernel: Fp128Var,
    pub az: Fp128Var,
    pub bz: Fp128Var,
    pub claim: Fp128Var,
}
impl SpartanOuterRemainderR1cs {
    pub fn new(log_t: usize) -> Result<Self, OuterRemainderR1csError> {
        if log_t >= 64 {
            return Err(OuterRemainderR1csError::Shape);
        }
        let shape = SpartanOuterDimensions::rv64(log_t);
        let columns = spartan_outer_opening_columns();
        if columns.len() != shape.variables().len() {
            return Err(OuterRemainderR1csError::Shape);
        }
        let matrices = spartan_outer_constraints::<Prime128OffsetA7F7>();
        if matrices.c.iter().any(|row| !row.is_empty()) {
            return Err(OuterRemainderR1csError::Shape);
        }
        let convert = |rows: Vec<Vec<(usize, Prime128OffsetA7F7)>>| {
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(column, value)| {
                            let column = if column == rv64::const_column() {
                                Column::One
                            } else {
                                Column::Opening(
                                    columns
                                        .iter()
                                        .position(|x| *x == column)
                                        .ok_or(OuterRemainderR1csError::Shape)?,
                                )
                            };
                            Ok((column, value.to_canonical_u128()))
                        })
                        .collect::<Result<Row, OuterRemainderR1csError>>()
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let a = convert(matrices.a)?;
        let b = convert(matrices.b)?;
        if a.len() != b.len()
            || SPARTAN_OUTER_FIRST_GROUP_ROWS
                .iter()
                .chain(&SPARTAN_OUTER_SECOND_GROUP_ROWS)
                .any(|row| *row >= a.len())
        {
            return Err(OuterRemainderR1csError::Shape);
        }
        Ok(Self {
            rounds: shape.remainder_rounds(),
            a,
            b,
            lagrange: Fp128CenteredLagrangeShape::new(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE)?,
        })
    }

    /// Validate every inherited handle before allocating polynomial intermediates.
    pub fn validate(
        &self,
        builder: &R1csBuilder<Fr>,
        tau: &[Fp128Var],
        uniskip: &Fp128Var,
        point: &[Fp128Var],
        openings: &OuterRemainderOutputClaims<Fp128Var>,
    ) -> Result<(), OuterRemainderR1csError> {
        if point.len() != self.rounds || tau.len() != self.rounds + 1 {
            return Err(OuterRemainderR1csError::Shape);
        }
        for value in tau.iter().chain(point).chain(std::iter::once(uniskip)) {
            value.validate_indices(builder)?;
        }
        let _ = openings.try_map_cells(|_, value| value.validate_indices(builder))?;
        Ok(())
    }

    pub fn constrain(
        &self,
        builder: &mut R1csBuilder<Fr>,
        tau: &[Fp128Var],
        uniskip: &Fp128Var,
        point: &[Fp128Var],
        openings: &OuterRemainderOutputClaims<Fp128Var>,
    ) -> Result<SpartanOuterRemainderVars, OuterRemainderR1csError> {
        self.validate(builder, tau, uniskip, point, openings)?;
        let (tau_high, tau_low) = tau.split_last().ok_or(OuterRemainderR1csError::Shape)?;
        let stream = point.first().ok_or(OuterRemainderR1csError::Shape)?;
        let lu = self.lagrange.evaluate(builder, uniskip)?;
        let lt = self.lagrange.evaluate(builder, tau_high)?;
        let zero = Fp128Var::constant(builder, 0)?;
        let one = Fp128Var::constant(builder, 1)?;
        let mut high_kernel = zero.clone();
        for (x, y) in lu.iter().zip(&lt) {
            let term = x.multiply(builder, y)?;
            high_kernel = high_kernel.add(builder, &term)?;
        }
        let reversed: Vec<_> = point.iter().rev().cloned().collect();
        let eq = eq_fp128_bn254(builder, tau_low, &reversed)?;
        let kernel = high_kernel.multiply(builder, &eq)?;
        let complement = one.subtract(builder, stream)?;
        let mut weights = vec![zero.clone(); self.a.len()];
        for (rows, multiplier) in [
            (&SPARTAN_OUTER_FIRST_GROUP_ROWS[..], &complement),
            (&SPARTAN_OUTER_SECOND_GROUP_ROWS[..], stream),
        ] {
            for (&row, basis) in rows.iter().zip(&lu) {
                let weight = weights.get_mut(row).ok_or(OuterRemainderR1csError::Shape)?;
                *weight = multiplier.multiply(builder, basis)?;
            }
        }
        let mut values = Vec::new();
        let _ = openings.try_map_cells(|_, value| {
            values.push(value.clone());
            Ok::<(), OuterRemainderR1csError>(())
        })?;
        let az = self.linear_form(builder, MatrixSide::A, &weights, &values, &one, &zero)?;
        let bz = self.linear_form(builder, MatrixSide::B, &weights, &values, &one, &zero)?;
        let claim = kernel.multiply(builder, &az)?.multiply(builder, &bz)?;
        Ok(SpartanOuterRemainderVars {
            kernel,
            az,
            bz,
            claim,
        })
    }
    fn linear_form(
        &self,
        builder: &mut R1csBuilder<Fr>,
        side: MatrixSide,
        weights: &[Fp128Var],
        values: &[Fp128Var],
        one: &Fp128Var,
        zero: &Fp128Var,
    ) -> Result<Fp128Var, OuterRemainderR1csError> {
        let rows = match side {
            MatrixSide::A => &self.a,
            MatrixSide::B => &self.b,
        };
        let mut sum = zero.clone();
        for (row, weight) in rows.iter().zip(weights) {
            let mut evaluation = zero.clone();
            for (column, coefficient) in row {
                let value = match column {
                    Column::One => one,
                    Column::Opening(i) => values.get(*i).ok_or(OuterRemainderR1csError::Shape)?,
                };
                let coefficient = Fp128Var::constant(builder, *coefficient)?;
                let term = value.multiply(builder, &coefficient)?;
                evaluation = evaluation.add(builder, &term)?;
            }
            let term = weight.multiply(builder, &evaluation)?;
            sum = sum.add(builder, &term)?;
        }
        Ok(sum)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "native sparse formula oracle and private-cell linkage regression"
)]
mod tests {
    use super::*;
    use crate::protocols::jolt::r1cs::{
        JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges,
    };
    use crate::OutputClaims;
    use jolt_field::Ring;
    #[test]
    fn private_outer_formula_matches_native_and_preserves_claim_order() {
        let mut counter = 0u64;
        let native_openings = OuterRemainderOutputClaims::<()>::default()
            .try_map_cells(|_, ()| {
                counter += 1;
                Ok::<_, ()>(Prime128OffsetA7F7::from_u64(counter))
            })
            .unwrap();
        let native_values = native_openings.opening_values();
        let native_tau = [2, 3, 5, 7].map(Prime128OffsetA7F7::from_u64);
        let native_point = [11, 13, 17].map(Prime128OffsetA7F7::from_u64);
        let u = Prime128OffsetA7F7::from_u64(19);
        let native = JoltSpartanOuterRemainder::new(JoltSpartanOuterRemainderChallenges {
            tau: &native_tau,
            uniskip: u,
            remainder: &native_point,
        })
        .unwrap();
        let mut builder = R1csBuilder::new();
        let tau = native_tau
            .map(|x| Fp128Var::allocate(&mut builder, Some(x.to_canonical_u128())).unwrap());
        let point = native_point
            .map(|x| Fp128Var::allocate(&mut builder, Some(x.to_canonical_u128())).unwrap());
        let uniskip = Fp128Var::allocate(&mut builder, Some(u.to_canonical_u128())).unwrap();
        let mut order = Vec::new();
        let openings = native_openings
            .try_map_cells(|id, value| {
                order.push(*id);
                Fp128Var::allocate(&mut builder, Some(value.to_canonical_u128()))
            })
            .unwrap();
        assert_eq!(order, native_openings.canonical_order());
        let result = SpartanOuterRemainderR1cs::new(2)
            .unwrap()
            .constrain(&mut builder, &tau, &uniskip, &point, &openings)
            .unwrap();
        let mut witness = builder.witness().unwrap();
        assert_eq!(
            witness[result.claim.variable().index()],
            Fr::from_u128(
                native
                    .expected_output_claim(&native_values)
                    .unwrap()
                    .to_canonical_u128()
            )
        );
        let matrices = builder.into_matrices();
        matrices.check_witness(&witness).unwrap();
        witness[openings.product.variable().index()] += Fr::from_u64(1);
        assert!(matrices.check_witness(&witness).is_err());
    }
}
