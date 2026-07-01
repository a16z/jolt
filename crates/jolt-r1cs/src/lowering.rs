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

    fn opening(&mut self, id: &Self::Opening) -> Result<SourceValue<F>, ClaimLoweringError>;
    fn challenge(&mut self, id: &Self::Challenge) -> Result<SourceValue<F>, ClaimLoweringError>;
    fn public(&mut self, id: &Self::Public) -> Result<SourceValue<F>, ClaimLoweringError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceValue<F> {
    Constant(F),
    LinearCombination(LinearCombination<F>),
}

impl<F: Field> SourceValue<F> {
    pub fn variable(variable: Variable) -> Self {
        Self::LinearCombination(LinearCombination::variable(variable))
    }

    pub fn linear_combination(linear_combination: LinearCombination<F>) -> Self {
        Self::LinearCombination(linear_combination)
    }

    pub fn into_linear_combination(self) -> LinearCombination<F> {
        match self {
            Self::Constant(value) => LinearCombination::constant(value),
            Self::LinearCombination(linear_combination) => linear_combination,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClaimSourceTable<F, O, P = (), C = usize> {
    openings: Vec<(O, SourceValue<F>)>,
    challenges: Vec<(C, SourceValue<F>)>,
    publics: Vec<(P, SourceValue<F>)>,
}

impl<F, O, P, C> ClaimSourceTable<F, O, P, C> {
    pub fn new() -> Self {
        Self {
            openings: Vec::new(),
            challenges: Vec::new(),
            publics: Vec::new(),
        }
    }

    pub fn insert_opening(&mut self, id: O, variable: Variable)
    where
        F: Field,
        O: PartialEq,
    {
        self.insert_opening_source(id, SourceValue::variable(variable));
    }

    pub fn insert_opening_lc(&mut self, id: O, linear_combination: LinearCombination<F>)
    where
        F: Field,
        O: PartialEq,
    {
        self.insert_opening_source(id, SourceValue::linear_combination(linear_combination));
    }

    pub fn insert_opening_source(&mut self, id: O, source: SourceValue<F>)
    where
        O: PartialEq,
    {
        assert!(
            !self.openings.iter().any(|(candidate, _)| candidate == &id),
            "duplicate opening source"
        );
        self.openings.push((id, source));
    }

    pub fn insert_challenge(&mut self, id: C, value: F)
    where
        C: PartialEq,
    {
        self.insert_challenge_source(id, SourceValue::Constant(value));
    }

    pub fn insert_challenge_lc(&mut self, id: C, linear_combination: LinearCombination<F>)
    where
        F: Field,
        C: PartialEq,
    {
        self.insert_challenge_source(id, SourceValue::linear_combination(linear_combination));
    }

    pub fn insert_challenge_source(&mut self, id: C, source: SourceValue<F>)
    where
        C: PartialEq,
    {
        assert!(
            !self
                .challenges
                .iter()
                .any(|(candidate, _)| candidate == &id),
            "duplicate challenge source"
        );
        self.challenges.push((id, source));
    }

    pub fn insert_public(&mut self, id: P, value: F)
    where
        P: PartialEq,
    {
        self.insert_public_source(id, SourceValue::Constant(value));
    }

    pub fn insert_public_lc(&mut self, id: P, linear_combination: LinearCombination<F>)
    where
        F: Field,
        P: PartialEq,
    {
        self.insert_public_source(id, SourceValue::linear_combination(linear_combination));
    }

    pub fn insert_public_source(&mut self, id: P, source: SourceValue<F>)
    where
        P: PartialEq,
    {
        assert!(
            !self.publics.iter().any(|(candidate, _)| candidate == &id),
            "duplicate public source"
        );
        self.publics.push((id, source));
    }
}

impl<F: Clone, O: PartialEq, P: PartialEq, C: PartialEq> ClaimSources<F>
    for ClaimSourceTable<F, O, P, C>
{
    type Opening = O;
    type Challenge = C;
    type Public = P;

    fn opening(&mut self, id: &Self::Opening) -> Result<SourceValue<F>, ClaimLoweringError> {
        self.openings
            .iter()
            .find_map(|(candidate, source)| (candidate == id).then_some(source.clone()))
            .ok_or(ClaimLoweringError::MissingOpening)
    }

    fn challenge(&mut self, id: &Self::Challenge) -> Result<SourceValue<F>, ClaimLoweringError> {
        self.challenges
            .iter()
            .find_map(|(candidate, source)| (candidate == id).then_some(source.clone()))
            .ok_or(ClaimLoweringError::MissingChallenge)
    }

    fn public(&mut self, id: &Self::Public) -> Result<SourceValue<F>, ClaimLoweringError> {
        self.publics
            .iter()
            .find_map(|(candidate, source)| (candidate == id).then_some(source.clone()))
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
            let source = match source {
                Source::Opening(id) => sources.opening(id)?,
                Source::Challenge(id) => sources.challenge(id)?,
                Source::Derived(id) => sources.public(id)?,
            };
            match source {
                SourceValue::Constant(value) => coefficient *= value,
                SourceValue::LinearCombination(linear_combination) => {
                    factors.push(linear_combination);
                }
            }
        }

        result = result + lower_product(builder, coefficient, factors);
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
    factors: Vec<LinearCombination<F>>,
) -> LinearCombination<F> {
    if coefficient.is_zero() {
        return LinearCombination::zero();
    }

    let mut factors = factors.into_iter();
    let Some(mut product) = factors.next() else {
        return LinearCombination::constant(coefficient);
    };

    for factor in factors {
        product = builder.multiply(product, factor);
    }

    product.scale(coefficient)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_claims::{challenge, constant, derived, opening, Expr};
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Challenge {
        Gamma,
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
            opening(Opening::A) * opening(Opening::B) + challenge(0usize) * derived(Public::Offset);

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
    fn lowers_variable_challenge_and_public_sources() {
        let mut builder = R1csBuilder::<Fr>::new();
        let opening_value = builder.alloc(Fr::from_u64(3));
        let challenge_value = builder.alloc(Fr::from_u64(4));
        let public_value = builder.alloc(Fr::from_u64(7));
        let out = builder.alloc(Fr::from_u64(19));

        let mut sources = ClaimSourceTable::<Fr, Opening, Public, Challenge>::new();
        sources.insert_opening(Opening::A, opening_value);
        sources.insert_challenge_lc(
            Challenge::Gamma,
            LinearCombination::variable(challenge_value),
        );
        sources.insert_public_lc(Public::Offset, LinearCombination::variable(public_value));

        let expression: Expr<Fr, Opening, Public, Challenge> =
            opening(Opening::A) * challenge(Challenge::Gamma) + derived(Public::Offset);

        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("variable sources lower");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn variable_source_products_reject_bad_witness() {
        let mut builder = R1csBuilder::<Fr>::new();
        let opening_value = builder.alloc(Fr::from_u64(3));
        let challenge_value = builder.alloc(Fr::from_u64(4));
        let out = builder.alloc(Fr::from_u64(13));

        let mut sources = ClaimSourceTable::<Fr, Opening, (), Challenge>::new();
        sources.insert_opening(Opening::A, opening_value);
        sources.insert_challenge_lc(
            Challenge::Gamma,
            LinearCombination::variable(challenge_value),
        );

        let expression: Expr<Fr, Opening, (), Challenge> =
            opening(Opening::A) * challenge(Challenge::Gamma);

        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("variable sources lower");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn constant_sources_do_not_allocate_product_constraints() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening, Public, Challenge>::new();
        sources.insert_challenge(Challenge::Gamma, Fr::from_u64(4));
        sources.insert_public(Public::Offset, Fr::from_u64(7));

        let expression: Expr<Fr, Opening, Public, Challenge> =
            challenge(Challenge::Gamma) * derived(Public::Offset);
        let lowered = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect("constant sources lower");

        assert_eq!(lowered, LinearCombination::constant(Fr::from_u64(28)));
        assert_eq!(builder.into_matrices().num_constraints, 0);
    }

    #[test]
    fn single_variable_source_stays_linear() {
        let mut builder = R1csBuilder::<Fr>::new();
        let challenge_value = builder.alloc(Fr::from_u64(4));
        let mut sources = ClaimSourceTable::<Fr, Opening, (), Challenge>::new();
        sources.insert_challenge_lc(
            Challenge::Gamma,
            LinearCombination::variable(challenge_value),
        );

        let expression: Expr<Fr, Opening, (), Challenge> = challenge(Challenge::Gamma);
        let lowered = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect("variable challenge lowers");

        assert_eq!(lowered, LinearCombination::variable(challenge_value));
        assert_eq!(builder.into_matrices().num_constraints, 0);
    }

    #[test]
    fn missing_challenge_is_typed_error() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        let expression: Expr<Fr, Opening> = challenge(2usize);

        let error = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect_err("challenge is missing");

        assert_eq!(error, ClaimLoweringError::MissingChallenge);
    }

    #[test]
    fn missing_opening_is_typed_error() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        let expression: Expr<Fr, Opening> = opening(Opening::A);

        let error = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect_err("opening is missing");

        assert_eq!(error, ClaimLoweringError::MissingOpening);
    }

    #[test]
    fn missing_public_is_typed_error() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();
        let expression: Expr<Fr, Opening, Public> = derived(Public::Offset);

        let error = lower_claim_expr(&mut builder, &expression, &mut sources)
            .expect_err("public is missing");

        assert_eq!(error, ClaimLoweringError::MissingPublic);
    }

    #[test]
    #[should_panic(expected = "duplicate opening source")]
    fn duplicate_opening_source_panics() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = builder.alloc(Fr::from_u64(3));
        let b = builder.alloc(Fr::from_u64(5));
        let mut sources = ClaimSourceTable::<Fr, Opening>::new();
        sources.insert_opening(Opening::A, a);
        sources.insert_opening(Opening::A, b);
    }

    #[test]
    fn lowers_typed_challenge_sources() {
        let mut builder = R1csBuilder::<Fr>::new();
        let out = builder.alloc(Fr::from_u64(6));
        let mut sources = ClaimSourceTable::<Fr, Opening, (), Challenge>::new();
        sources.insert_challenge(Challenge::Gamma, Fr::from_u64(6));

        let expression: Expr<Fr, Opening, (), Challenge> = challenge(Challenge::Gamma);
        assert_claim_expr_eq(&mut builder, &expression, out, &mut sources)
            .expect("typed challenge lowers");

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }
}
