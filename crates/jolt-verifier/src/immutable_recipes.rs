use jolt_claims::immutable_recipes::{evaluate_canonical, EvaluationError, Phase, RecipeError};
use jolt_claims::{Expr, SymbolicSumcheck};
use jolt_field::Ring;
use std::collections::BTreeSet;

use crate::verifier::CheckedInputs;
use crate::VerifierError;

/// Borrowed from the input validation of this verification, never from a table
/// key. A Committed schedule always contains both program reductions; the
/// canonical PrecommittedSchedule constructor sets neither for Full mode.
pub struct RecipeContext<'a> {
    checked: &'a CheckedInputs,
}

impl<'a> RecipeContext<'a> {
    pub(crate) fn from_checked(checked: &'a CheckedInputs) -> Self {
        Self { checked }
    }

    pub fn permits_full(&self) -> bool {
        cfg!(feature = "akita")
            && cfg!(feature = "field-inline")
            && !self.checked.zk
            && self.checked.precommitted.bytecode.is_none()
            && self.checked.precommitted.program_image.is_none()
    }
}

fn invalid_recipe(error: RecipeError) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: "immutable recipe".to_owned(),
        reason: error.to_string(),
    }
}

pub fn evaluate<S: SymbolicSumcheck, F: Ring>(
    symbolic: &S,
    phase: Phase,
    context: &RecipeContext<'_>,
    canonical: impl FnOnce() -> Expr<F, S::OpeningId, S::DerivedId, S::ChallengeId>,
    opening: impl FnMut(&S::OpeningId) -> Result<F, VerifierError>,
    challenge: impl FnMut(&S::ChallengeId) -> Result<F, VerifierError>,
    derived: impl FnMut(&S::DerivedId) -> Result<F, VerifierError>,
) -> Result<F, VerifierError> {
    if context.permits_full() && F::immutable_recipe_field().is_some() {
        if let Some(recipe) = symbolic
            .immutable_recipe::<F>(phase)
            .map_err(invalid_recipe)?
        {
            return recipe
                .try_evaluate(opening, challenge, derived)
                .map_err(|error| match error {
                    EvaluationError::Recipe(error) => invalid_recipe(error),
                    EvaluationError::Resolver(error) => error,
                });
        }
    }
    let expression = canonical();
    #[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
    if context.permits_full() {
        symbolic.immutable_record_expression(phase, &expression);
    }
    if context.permits_full() {
        if let Some(field) = F::immutable_recipe_field() {
            return evaluate_canonical(&expression, field, opening, challenge, derived).map_err(
                |error| match error {
                    EvaluationError::Recipe(error) => invalid_recipe(error),
                    EvaluationError::Resolver(error) => error,
                },
            );
        }
    }
    expression.try_evaluate(opening, challenge, derived)
}

pub fn support<S: SymbolicSumcheck, F: Ring>(
    symbolic: &S,
    context: &RecipeContext<'_>,
) -> Result<BTreeSet<S::OpeningId>, VerifierError>
where
    S::OpeningId: Ord + Clone,
{
    if context.permits_full() && F::immutable_recipe_field().is_some() {
        if let Some(recipe) = symbolic
            .immutable_recipe::<F>(Phase::Support)
            .map_err(invalid_recipe)?
        {
            return Ok(recipe.openings());
        }
    }
    let openings = symbolic.expected_output_openings::<F>();
    #[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
    if context.permits_full() {
        symbolic.immutable_record_support::<F>(&openings);
    }
    Ok(openings)
}

/// Method syntax supplies the same automatic borrow as canonical expression
/// calls, including a symbolic relation returned by reference.
pub trait RecipeEvaluation: SymbolicSumcheck + Sized {
    fn evaluate_recipe<F: Ring>(
        &self,
        phase: Phase,
        context: &RecipeContext<'_>,
        canonical: impl FnOnce() -> Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>,
        opening: impl FnMut(&Self::OpeningId) -> Result<F, VerifierError>,
        challenge: impl FnMut(&Self::ChallengeId) -> Result<F, VerifierError>,
        derived: impl FnMut(&Self::DerivedId) -> Result<F, VerifierError>,
    ) -> Result<F, VerifierError> {
        evaluate(self, phase, context, canonical, opening, challenge, derived)
    }
    fn recipe_support<F: Ring>(
        &self,
        context: &RecipeContext<'_>,
    ) -> Result<BTreeSet<Self::OpeningId>, VerifierError>
    where
        Self::OpeningId: Ord + Clone,
    {
        support::<Self, F>(self, context)
    }
}
impl<S: SymbolicSumcheck> RecipeEvaluation for S {}
