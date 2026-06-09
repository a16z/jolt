use jolt_field::Field;
use num_traits::{One, Zero};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PolyR1csError {
    #[error("eq polynomial arity mismatch: left has {left} coordinates, right has {right}")]
    EqArityMismatch { left: usize, right: usize },
    #[error(
        "multilinear evaluation table length mismatch: {num_vars} variables require {expected} evaluations, got {got}"
    )]
    EvaluationTableLengthMismatch {
        num_vars: usize,
        expected: usize,
        got: usize,
    },
    #[error("cannot materialize a hypercube of dimension {num_vars}")]
    HypercubeTooLarge { num_vars: usize },
}

pub trait PolynomialScalarGadget: Clone {
    type ConstraintBuilder;
    type Scalar: Field;

    fn constant(scalar: Self::Scalar) -> Self;
    fn add(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self;
    fn sub(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self;
    fn mul(&self, builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self;
}

pub fn eq_eval<S>(
    builder: &mut S::ConstraintBuilder,
    left: &[S],
    right: &[S],
) -> Result<S, PolyR1csError>
where
    S: PolynomialScalarGadget,
{
    if left.len() != right.len() {
        return Err(PolyR1csError::EqArityMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    let mut result = S::constant(S::Scalar::one());
    for (left_coordinate, right_coordinate) in left.iter().zip(right) {
        let both_one = left_coordinate.mul(builder, right_coordinate);
        let one_minus_left = S::constant(S::Scalar::one()).sub(builder, left_coordinate);
        let one_minus_right = S::constant(S::Scalar::one()).sub(builder, right_coordinate);
        let both_zero = one_minus_left.mul(builder, &one_minus_right);
        let coordinate_eq = both_one.add(builder, &both_zero);
        result = result.mul(builder, &coordinate_eq);
    }

    Ok(result)
}

pub fn eq_evals<S>(builder: &mut S::ConstraintBuilder, point: &[S]) -> Vec<S>
where
    S: PolynomialScalarGadget,
{
    scaled_eq_evals(builder, point, &S::constant(S::Scalar::one()))
}

pub fn scaled_eq_evals<S>(
    builder: &mut S::ConstraintBuilder,
    point: &[S],
    scaling_factor: &S,
) -> Vec<S>
where
    S: PolynomialScalarGadget,
{
    let mut table = vec![scaling_factor.clone()];
    for coordinate in point {
        let mut next_table = Vec::with_capacity(table.len() * 2);
        for entry in &table {
            let selected = entry.mul(builder, coordinate);
            next_table.push(entry.sub(builder, &selected));
            next_table.push(selected);
        }
        table = next_table;
    }
    table
}

pub fn inner_product<S>(
    builder: &mut S::ConstraintBuilder,
    left: &[S],
    right: &[S],
) -> Result<S, PolyR1csError>
where
    S: PolynomialScalarGadget,
{
    if left.len() != right.len() {
        return Err(PolyR1csError::EqArityMismatch {
            left: left.len(),
            right: right.len(),
        });
    }

    let mut result = S::constant(S::Scalar::zero());
    for (left_scalar, right_scalar) in left.iter().zip(right) {
        let term = left_scalar.mul(builder, right_scalar);
        result = result.add(builder, &term);
    }
    Ok(result)
}

pub fn multilinear_eval<S>(
    builder: &mut S::ConstraintBuilder,
    evaluations: &[S],
    point: &[S],
) -> Result<S, PolyR1csError>
where
    S: PolynomialScalarGadget,
{
    let expected = hypercube_len(point.len())?;
    if evaluations.len() != expected {
        return Err(PolyR1csError::EvaluationTableLengthMismatch {
            num_vars: point.len(),
            expected,
            got: evaluations.len(),
        });
    }

    let weights = eq_evals(builder, point);
    inner_product(builder, evaluations, &weights)
}

fn hypercube_len(num_vars: usize) -> Result<usize, PolyR1csError> {
    1usize
        .checked_shl(num_vars.try_into().unwrap_or(u32::MAX))
        .ok_or(PolyR1csError::HypercubeTooLarge { num_vars })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::*;
    use crate::{EqPolynomial, Polynomial};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PlainScalar<F: Field>(F);

    impl<F: Field> PolynomialScalarGadget for PlainScalar<F> {
        type ConstraintBuilder = ();
        type Scalar = F;

        fn constant(scalar: Self::Scalar) -> Self {
            Self(scalar)
        }

        fn add(&self, _builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
            Self(self.0 + rhs.0)
        }

        fn sub(&self, _builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
            Self(self.0 - rhs.0)
        }

        fn mul(&self, _builder: &mut Self::ConstraintBuilder, rhs: &Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    #[test]
    fn eq_eval_matches_plain_eq_mle() {
        let mut builder = ();
        let left_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let right_values = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
        let left = left_values
            .iter()
            .copied()
            .map(PlainScalar)
            .collect::<Vec<_>>();
        let right = right_values
            .iter()
            .copied()
            .map(PlainScalar)
            .collect::<Vec<_>>();

        let result = eq_eval(&mut builder, &left, &right).expect("eq eval succeeds");
        let expected = EqPolynomial::new(left_values.to_vec()).evaluate(&right_values);

        assert_eq!(result, PlainScalar(expected));
    }

    #[test]
    fn eq_evals_match_plain_table_order() {
        let mut builder = ();
        let point_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let point = point_values
            .iter()
            .copied()
            .map(PlainScalar)
            .collect::<Vec<_>>();

        let evals = eq_evals(&mut builder, &point)
            .into_iter()
            .map(|scalar| scalar.0)
            .collect::<Vec<_>>();
        let expected = EqPolynomial::new(point_values.to_vec()).evaluations();

        assert_eq!(evals, expected);
    }

    #[test]
    fn multilinear_eval_matches_plain_evaluation() {
        let mut builder = ();
        let evaluation_values = (0..8)
            .map(|index| Fr::from_u64((3 * index + 2) as u64))
            .collect::<Vec<_>>();
        let point_values = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let evaluations = evaluation_values
            .iter()
            .copied()
            .map(PlainScalar)
            .collect::<Vec<_>>();
        let point = point_values
            .iter()
            .copied()
            .map(PlainScalar)
            .collect::<Vec<_>>();

        let result =
            multilinear_eval(&mut builder, &evaluations, &point).expect("evaluation succeeds");
        let expected = Polynomial::new(evaluation_values).evaluate(&point_values);

        assert_eq!(result, PlainScalar(expected));
    }

    #[test]
    fn dimension_errors_are_typed() {
        let mut builder = ();
        let x = PlainScalar(Fr::from_u64(2));
        let y = PlainScalar(Fr::from_u64(3));

        assert_eq!(
            eq_eval(
                &mut builder,
                std::slice::from_ref(&x),
                &[x.clone(), y.clone()]
            ),
            Err(PolyR1csError::EqArityMismatch { left: 1, right: 2 })
        );
        assert_eq!(
            inner_product(&mut builder, std::slice::from_ref(&x), &[x.clone(), y]),
            Err(PolyR1csError::EqArityMismatch { left: 1, right: 2 })
        );
        assert_eq!(
            multilinear_eval(
                &mut builder,
                std::slice::from_ref(&x),
                std::slice::from_ref(&x)
            ),
            Err(PolyR1csError::EvaluationTableLengthMismatch {
                num_vars: 1,
                expected: 2,
                got: 1,
            })
        );
    }
}
