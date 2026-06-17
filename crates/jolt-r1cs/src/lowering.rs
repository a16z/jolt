use jolt_claims::{Expr, Source};
use jolt_field::Field;
use num_traits::Zero;
use thiserror::Error;

use crate::{LinearCombination, R1csBuilder, ScalarGadget, Variable};

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

pub trait ScalarClaimSources<S>
where
    S: ScalarGadget,
{
    type Opening;
    type Challenge;
    type Public;

    fn opening(&mut self, id: &Self::Opening) -> Result<ScalarSourceValue<S>, ClaimLoweringError>;
    fn challenge(
        &mut self,
        id: &Self::Challenge,
    ) -> Result<ScalarSourceValue<S>, ClaimLoweringError>;
    fn public(&mut self, id: &Self::Public) -> Result<ScalarSourceValue<S>, ClaimLoweringError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceValue<F> {
    Constant(F),
    LinearCombination(LinearCombination<F>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarSourceValue<S>
where
    S: ScalarGadget,
{
    Constant(S::Scalar),
    Scalar(S),
}

impl<S> ScalarSourceValue<S>
where
    S: ScalarGadget,
{
    pub fn constant(value: S::Scalar) -> Self {
        Self::Constant(value)
    }

    pub fn scalar(value: S) -> Self {
        Self::Scalar(value)
    }
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

#[derive(Clone, Debug, Default)]
pub struct ScalarClaimSourceTable<S, O, P = (), C = usize>
where
    S: ScalarGadget,
{
    openings: Vec<(O, ScalarSourceValue<S>)>,
    challenges: Vec<(C, ScalarSourceValue<S>)>,
    publics: Vec<(P, ScalarSourceValue<S>)>,
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

impl<S, O, P, C> ScalarClaimSourceTable<S, O, P, C>
where
    S: ScalarGadget,
{
    pub fn new() -> Self {
        Self {
            openings: Vec::new(),
            challenges: Vec::new(),
            publics: Vec::new(),
        }
    }

    pub fn insert_opening_constant(&mut self, id: O, value: S::Scalar)
    where
        O: PartialEq,
    {
        self.insert_opening_source(id, ScalarSourceValue::constant(value));
    }

    pub fn insert_opening_scalar(&mut self, id: O, value: S)
    where
        O: PartialEq,
    {
        self.insert_opening_source(id, ScalarSourceValue::scalar(value));
    }

    pub fn insert_opening_source(&mut self, id: O, source: ScalarSourceValue<S>)
    where
        O: PartialEq,
    {
        assert!(
            !self.openings.iter().any(|(candidate, _)| candidate == &id),
            "duplicate opening source"
        );
        self.openings.push((id, source));
    }

    pub fn insert_challenge_constant(&mut self, id: C, value: S::Scalar)
    where
        C: PartialEq,
    {
        self.insert_challenge_source(id, ScalarSourceValue::constant(value));
    }

    pub fn insert_challenge_scalar(&mut self, id: C, value: S)
    where
        C: PartialEq,
    {
        self.insert_challenge_source(id, ScalarSourceValue::scalar(value));
    }

    pub fn insert_challenge_source(&mut self, id: C, source: ScalarSourceValue<S>)
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

    pub fn insert_public_constant(&mut self, id: P, value: S::Scalar)
    where
        P: PartialEq,
    {
        self.insert_public_source(id, ScalarSourceValue::constant(value));
    }

    pub fn insert_public_scalar(&mut self, id: P, value: S)
    where
        P: PartialEq,
    {
        self.insert_public_source(id, ScalarSourceValue::scalar(value));
    }

    pub fn insert_public_source(&mut self, id: P, source: ScalarSourceValue<S>)
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

impl<S, O, P, C> ScalarClaimSources<S> for ScalarClaimSourceTable<S, O, P, C>
where
    S: ScalarGadget,
    O: PartialEq,
    P: PartialEq,
    C: PartialEq,
{
    type Opening = O;
    type Challenge = C;
    type Public = P;

    fn opening(&mut self, id: &Self::Opening) -> Result<ScalarSourceValue<S>, ClaimLoweringError> {
        self.openings
            .iter()
            .find_map(|(candidate, source)| (candidate == id).then_some(source.clone()))
            .ok_or(ClaimLoweringError::MissingOpening)
    }

    fn challenge(
        &mut self,
        id: &Self::Challenge,
    ) -> Result<ScalarSourceValue<S>, ClaimLoweringError> {
        self.challenges
            .iter()
            .find_map(|(candidate, source)| (candidate == id).then_some(source.clone()))
            .ok_or(ClaimLoweringError::MissingChallenge)
    }

    fn public(&mut self, id: &Self::Public) -> Result<ScalarSourceValue<S>, ClaimLoweringError> {
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
                Source::Public(id) => sources.public(id)?,
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

pub fn lower_claim_expr_gadget<S, R>(
    builder: &mut R1csBuilder<S::BuilderField>,
    expression: &Expr<S::Scalar, R::Opening, R::Public, R::Challenge>,
    sources: &mut R,
) -> Result<S, ClaimLoweringError>
where
    S: ScalarGadget,
    R: ScalarClaimSources<S>,
{
    let mut result = S::constant(S::Scalar::zero());

    for term in &expression.terms {
        let mut coefficient = term.coefficient;
        let mut factors = Vec::new();

        for source in &term.factors {
            let source = match source {
                Source::Opening(id) => sources.opening(id)?,
                Source::Challenge(id) => sources.challenge(id)?,
                Source::Public(id) => sources.public(id)?,
            };
            match source {
                ScalarSourceValue::Constant(value) => coefficient *= value,
                ScalarSourceValue::Scalar(value) => factors.push(value),
            }
        }

        let term = lower_gadget_product(builder, coefficient, factors);
        result = result.add(builder, &term);
    }

    Ok(result)
}

pub fn assert_claim_expr_gadget_eq<S, R>(
    builder: &mut R1csBuilder<S::BuilderField>,
    expression: &Expr<S::Scalar, R::Opening, R::Public, R::Challenge>,
    expected: &S,
    sources: &mut R,
) -> Result<(), ClaimLoweringError>
where
    S: ScalarGadget,
    R: ScalarClaimSources<S>,
{
    let actual = lower_claim_expr_gadget(builder, expression, sources)?;
    actual.assert_equal(builder, expected);
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

fn lower_gadget_product<S>(
    builder: &mut R1csBuilder<S::BuilderField>,
    coefficient: S::Scalar,
    factors: Vec<S>,
) -> S
where
    S: ScalarGadget,
{
    if coefficient.is_zero() {
        return S::constant(S::Scalar::zero());
    }

    let mut factors = factors.into_iter();
    let Some(mut product) = factors.next() else {
        return S::constant(coefficient);
    };

    for factor in factors {
        product = product.mul(builder, &factor);
    }

    product.scale_by_constant(builder, coefficient)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_claims::{challenge, constant, opening, public, Expr};
    use jolt_field::{Fq, Fr, FromPrimitiveInt};

    use crate::{AssignedScalar, FqVar};

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
            opening(Opening::A) * challenge(Challenge::Gamma) + public(Public::Offset);

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
            challenge(Challenge::Gamma) * public(Public::Offset);
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
        let expression: Expr<Fr, Opening> = challenge(2);

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
        let expression: Expr<Fr, Opening, Public> = public(Public::Offset);

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

    #[test]
    fn native_gadget_lowering_accepts_formula() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = AssignedScalar::alloc(&mut builder, Fr::from_u64(3));
        let b = AssignedScalar::alloc(&mut builder, Fr::from_u64(5));
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(2));
        let mut sources =
            ScalarClaimSourceTable::<AssignedScalar<Fr>, Opening, Public, Challenge>::new();
        sources.insert_opening_scalar(Opening::A, a);
        sources.insert_opening_scalar(Opening::B, b);
        sources.insert_challenge_scalar(Challenge::Gamma, gamma);
        sources.insert_public_constant(Public::Offset, Fr::from_u64(4));

        let expected = AssignedScalar::constant(Fr::from_u64(47));
        assert_claim_expr_gadget_eq(&mut builder, &sample_expression(), &expected, &mut sources)
            .expect("native scalar-gadget expression lowers");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn native_gadget_lowering_rejects_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = AssignedScalar::alloc(&mut builder, Fr::from_u64(3));
        let b = AssignedScalar::alloc(&mut builder, Fr::from_u64(5));
        let gamma = AssignedScalar::alloc(&mut builder, Fr::from_u64(2));
        let expected = AssignedScalar::alloc(&mut builder, Fr::from_u64(47));
        let targets = [
            ("opening", variable(&a)),
            ("challenge", variable(&gamma)),
            ("expected", variable(&expected)),
        ];
        let mut sources =
            ScalarClaimSourceTable::<AssignedScalar<Fr>, Opening, Public, Challenge>::new();
        sources.insert_opening_scalar(Opening::A, a);
        sources.insert_opening_scalar(Opening::B, b);
        sources.insert_challenge_scalar(Challenge::Gamma, gamma);
        sources.insert_public_constant(Public::Offset, Fr::from_u64(4));

        assert_claim_expr_gadget_eq(&mut builder, &sample_expression(), &expected, &mut sources)
            .expect("native scalar-gadget expression lowers");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn nonnative_gadget_lowering_accepts_formula() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = FqVar::alloc(&mut builder, Fq::from_u64(3));
        let b = FqVar::alloc(&mut builder, Fq::from_u64(5));
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));
        let mut sources = ScalarClaimSourceTable::<FqVar, Opening, Public, Challenge>::new();
        sources.insert_opening_scalar(Opening::A, a);
        sources.insert_opening_scalar(Opening::B, b);
        sources.insert_challenge_scalar(Challenge::Gamma, gamma);
        sources.insert_public_constant(Public::Offset, Fq::from_u64(4));

        let expected = FqVar::constant(Fq::from_u64(47));
        assert_claim_expr_gadget_eq(&mut builder, &sample_expression(), &expected, &mut sources)
            .expect("non-native scalar-gadget expression lowers");

        assert!(builder_accepts(builder));
    }

    #[test]
    fn nonnative_gadget_lowering_rejects_tampering() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = FqVar::alloc(&mut builder, Fq::from_u64(3));
        let b = FqVar::alloc(&mut builder, Fq::from_u64(5));
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));
        let expected = FqVar::alloc(&mut builder, Fq::from_u64(47));
        let targets = [
            ("opening limb", variable(&a.limbs()[0])),
            ("challenge limb", variable(&gamma.limbs()[0])),
            ("expected limb", variable(&expected.limbs()[0])),
        ];
        let mut sources = ScalarClaimSourceTable::<FqVar, Opening, Public, Challenge>::new();
        sources.insert_opening_scalar(Opening::A, a);
        sources.insert_opening_scalar(Opening::B, b);
        sources.insert_challenge_scalar(Challenge::Gamma, gamma);
        sources.insert_public_constant(Public::Offset, Fq::from_u64(4));

        assert_claim_expr_gadget_eq(&mut builder, &sample_expression(), &expected, &mut sources)
            .expect("non-native scalar-gadget expression lowers");

        assert_tampering_rejected(builder, targets);
    }

    #[test]
    fn nonnative_gadget_lowering_rejects_bad_expected_output() {
        let mut builder = R1csBuilder::<Fr>::new();
        let a = FqVar::alloc(&mut builder, Fq::from_u64(3));
        let b = FqVar::alloc(&mut builder, Fq::from_u64(5));
        let gamma = FqVar::alloc(&mut builder, Fq::from_u64(2));
        let mut sources = ScalarClaimSourceTable::<FqVar, Opening, Public, Challenge>::new();
        sources.insert_opening_scalar(Opening::A, a);
        sources.insert_opening_scalar(Opening::B, b);
        sources.insert_challenge_scalar(Challenge::Gamma, gamma);
        sources.insert_public_constant(Public::Offset, Fq::from_u64(4));

        let expected = FqVar::constant(Fq::from_u64(48));
        assert_claim_expr_gadget_eq(&mut builder, &sample_expression(), &expected, &mut sources)
            .expect("non-native scalar-gadget expression lowers");

        assert!(builder_rejects(builder));
    }

    #[test]
    fn gadget_lowering_missing_sources_are_typed_errors() {
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources =
            ScalarClaimSourceTable::<AssignedScalar<Fr>, Opening, Public, Challenge>::new();

        let opening_expr: Expr<Fr, Opening, Public, Challenge> = opening(Opening::A);
        let challenge_expr: Expr<Fr, Opening, Public, Challenge> = challenge(Challenge::Gamma);
        let public_expr: Expr<Fr, Opening, Public, Challenge> = public(Public::Offset);

        assert_eq!(
            lower_claim_expr_gadget(&mut builder, &opening_expr, &mut sources),
            Err(ClaimLoweringError::MissingOpening)
        );
        assert_eq!(
            lower_claim_expr_gadget(&mut builder, &challenge_expr, &mut sources),
            Err(ClaimLoweringError::MissingChallenge)
        );
        assert_eq!(
            lower_claim_expr_gadget(&mut builder, &public_expr, &mut sources),
            Err(ClaimLoweringError::MissingPublic)
        );
    }

    fn sample_expression<F>() -> Expr<F, Opening, Public, Challenge>
    where
        F: Field,
    {
        constant(F::from_u64(2)) * opening(Opening::A) * opening(Opening::B)
            + challenge(Challenge::Gamma) * public(Public::Offset)
            + constant(F::from_u64(9))
    }

    fn builder_accepts<F>(builder: R1csBuilder<F>) -> bool
    where
        F: Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
    }

    fn builder_rejects<F>(builder: R1csBuilder<F>) -> bool
    where
        F: Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_err()
    }

    fn assert_tampering_rejected<F>(
        builder: R1csBuilder<F>,
        targets: impl IntoIterator<Item = (&'static str, Variable)>,
    ) where
        F: Field,
    {
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        for (label, variable) in targets {
            let mut tampered = witness.clone();
            tampered[variable.index()] += F::one();
            assert!(
                matrices.check_witness(&tampered).is_err(),
                "{label} accepted after tampering variable {}",
                variable.index()
            );
        }
    }

    fn variable<F>(scalar: &AssignedScalar<F>) -> Variable
    where
        F: Field,
    {
        assert_eq!(scalar.lc.terms.len(), 1);
        let (variable, coefficient) = scalar
            .lc
            .terms
            .first()
            .copied()
            .expect("linear combination has one term");
        assert_eq!(coefficient, F::one());
        variable
    }
}
