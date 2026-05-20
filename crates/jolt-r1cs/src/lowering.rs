use jolt_claims::{Expr, Source};
use jolt_field::Field;
use thiserror::Error;

use crate::{LinearCombination, R1csBuilder, Variable};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ClaimLoweringError {
    #[error("missing opening source")]
    MissingOpening,
    #[error("missing challenge source")]
    MissingChallenge,
    #[error("missing public source")]
    MissingPublic,
}

pub trait ClaimSources<F> {
    type Opening;
    type Challenge;
    type Public;

    fn opening(&mut self, id: &Self::Opening) -> Result<Variable, ClaimLoweringError>;
    fn challenge(&mut self, id: &Self::Challenge) -> Result<F, ClaimLoweringError>;
    fn public(&mut self, id: &Self::Public) -> Result<F, ClaimLoweringError>;
}

#[derive(Clone, Debug, Default)]
pub struct ClaimSourceTable<F, O, P = (), C = usize> {
    openings: Vec<(O, Variable)>,
    challenges: Vec<(C, F)>,
    publics: Vec<(P, F)>,
}

impl<F, O, P, C> ClaimSourceTable<F, O, P, C> {
    pub fn new() -> Self {
        Self {
            openings: Vec::new(),
            challenges: Vec::new(),
            publics: Vec::new(),
        }
    }

    pub fn insert_opening(&mut self, id: O, variable: Variable) {
        self.openings.push((id, variable));
    }

    pub fn insert_challenge(&mut self, id: C, value: F) {
        self.challenges.push((id, value));
    }

    pub fn insert_public(&mut self, id: P, value: F) {
        self.publics.push((id, value));
    }
}

impl<F: Copy, O: PartialEq, P: PartialEq, C: PartialEq> ClaimSources<F>
    for ClaimSourceTable<F, O, P, C>
{
    type Opening = O;
    type Challenge = C;
    type Public = P;

    fn opening(&mut self, id: &Self::Opening) -> Result<Variable, ClaimLoweringError> {
        self.openings
            .iter()
            .find_map(|(candidate, variable)| (candidate == id).then_some(*variable))
            .ok_or(ClaimLoweringError::MissingOpening)
    }

    fn challenge(&mut self, id: &Self::Challenge) -> Result<F, ClaimLoweringError> {
        self.challenges
            .iter()
            .find_map(|(candidate, value)| (candidate == id).then_some(*value))
            .ok_or(ClaimLoweringError::MissingChallenge)
    }

    fn public(&mut self, id: &Self::Public) -> Result<F, ClaimLoweringError> {
        self.publics
            .iter()
            .find_map(|(candidate, value)| (candidate == id).then_some(*value))
            .ok_or(ClaimLoweringError::MissingPublic)
    }
}

pub fn lower_claim_expr<F, R>(
    builder: &mut R1csBuilder<F>,
    expression: &Expr<F, R::Opening, R::Public, R::Challenge>,
    sources: &mut R,
) -> Result<LinearCombination<F>, ClaimLoweringError>
where
    F: Field,
    R: ClaimSources<F>,
{
    let mut result = LinearCombination::zero();

    for term in &expression.terms {
        let mut coefficient = term.coefficient;
        let mut factors = Vec::new();

        for source in &term.factors {
            match source {
                Source::Opening(id) => factors.push(sources.opening(id)?),
                Source::Challenge(id) => coefficient *= sources.challenge(id)?,
                Source::Public(id) => coefficient *= sources.public(id)?,
            }
        }

        result = result + lower_product(builder, coefficient, &factors);
    }

    Ok(result)
}

pub fn assert_claim_expr_eq<F, R, Expected>(
    builder: &mut R1csBuilder<F>,
    expression: &Expr<F, R::Opening, R::Public, R::Challenge>,
    expected: Expected,
    sources: &mut R,
) -> Result<(), ClaimLoweringError>
where
    F: Field,
    R: ClaimSources<F>,
    Expected: Into<LinearCombination<F>>,
{
    let actual = lower_claim_expr(builder, expression, sources)?;
    builder.assert_equal(actual, expected);
    Ok(())
}

fn lower_product<F: Field>(
    builder: &mut R1csBuilder<F>,
    coefficient: F,
    factors: &[Variable],
) -> LinearCombination<F> {
    if coefficient.is_zero() {
        return LinearCombination::zero();
    }

    let Some((&first, rest)) = factors.split_first() else {
        return LinearCombination::constant(coefficient);
    };

    let mut product = LinearCombination::variable(first);
    for &factor in rest {
        let output = builder.multiply(product, factor);
        product = LinearCombination::variable(output);
    }

    product.scale(coefficient)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_claims::{challenge, constant, opening, public, Expr};
    use jolt_field::{Fr, FromPrimitiveInt};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        A,
        B,
        C,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Public {
        Offset,
    }

    #[test]
    fn lowers_expression_to_satisfied_r1cs() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = builder.alloc(Fr::from_u64(3));
        let b = builder.alloc(Fr::from_u64(5));
        let out = builder.alloc(Fr::from_u64(23));

        let mut sources = ClaimSourceTable::new();
        sources.insert_opening(Opening::A, a);
        sources.insert_opening(Opening::B, b);
        sources.insert_challenge(0, Fr::from_u64(2));
        sources.insert_public(Public::Offset, Fr::from_u64(4));

        let expression: Expr<Fr, Opening, Public> =
            opening(Opening::A) * opening(Opening::B) + challenge(0) * public(Public::Offset);

        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("expression lowers");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn lowered_constraint_rejects_bad_witness() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = builder.alloc(Fr::from_u64(3));
        let b = builder.alloc(Fr::from_u64(5));
        let out = builder.alloc(Fr::from_u64(22));

        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        sources.insert_opening(Opening::A, a);
        sources.insert_opening(Opening::B, b);

        let expression: Expr<Fr, Opening> =
            opening(Opening::A) * opening(Opening::B) + constant(Fr::from_u64(8));

        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("expression lowers");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn multi_factor_product_allocates_chain() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = builder.alloc(Fr::from_u64(2));
        let b = builder.alloc(Fr::from_u64(3));
        let c = builder.alloc(Fr::from_u64(4));
        let out = builder.alloc(Fr::from_u64(24));

        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        sources.insert_opening(Opening::A, a);
        sources.insert_opening(Opening::B, b);
        sources.insert_opening(Opening::C, c);

        let expression: Expr<Fr, Opening> =
            opening(Opening::A) * opening(Opening::B) * opening(Opening::C);

        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("expression lowers");

        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert_eq!(matrices.num_constraints, 3);
        assert!(matrices.check_witness(&witness).is_ok());
    }

    #[test]
    fn missing_challenge_is_typed_error() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        let expression: Expr<Fr, Opening> = challenge(2);

        let error = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect_err("challenge is missing");

        assert_eq!(error, ClaimLoweringError::MissingChallenge);
    }

    #[test]
    fn lowers_typed_challenge_sources() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Challenge {
            Alpha,
        }

        let mut builder = R1csBuilder::<Fr>::new();
        let out = builder.alloc(Fr::from_u64(6));
        let mut sources = ClaimSourceTable::<Fr, Opening, (), Challenge>::new();
        sources.insert_challenge(Challenge::Alpha, Fr::from_u64(6));

        let expression: Expr<Fr, Opening, (), Challenge> = challenge(Challenge::Alpha);
        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("typed challenge lowers");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }
}
