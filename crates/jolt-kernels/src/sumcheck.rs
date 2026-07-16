use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};

use crate::KernelError;

/// The typed prove-side counterpart of a batch member: pairs the object-safe
/// [`ProveRounds`] round interface (what the engine's round loop consumes)
/// with the member's [`ConcreteSumcheck`] relation, so a stage recipe can
/// extract typed output claims after the loop.
///
/// Split from `ProveRounds` because of the dependency direction:
/// `jolt-verifier` depends on `jolt-sumcheck`, so nothing in `jolt-sumcheck`
/// may name `ConcreteSumcheck` — the typed half lives here, above both.
pub trait ProveSumcheck<F: Field>: ProveRounds<F>
where
    SumcheckInputClaims<F, Self::Relation>: InputClaims<F>,
    SumcheckOutputClaims<F, Self::Relation>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, Self::Relation>: SumcheckChallenges<F, JoltChallengeId>,
{
    type Relation: ConcreteSumcheck<F>;

    /// The relation this member proves — the source of its rounds, degree,
    /// point offsets, and claim algebra.
    fn relation(&self) -> &Self::Relation;

    /// Extract the member's typed produced-opening values from its fully
    /// bound state. Call after the engine's round loop has ingested every
    /// challenge.
    fn output_claims(&mut self) -> Result<SumcheckOutputClaims<F, Self::Relation>, KernelError<F>>;

    /// Cross-check any hand-materialized `Derived` leaf tables against the
    /// verifier's `derive_output_term` at the bound point. Call after the
    /// engine's round loop has ingested every challenge; the stage recipes
    /// run it on every member before the aggregate final-claim check, so a
    /// drifted table is attributed to its id rather than surfacing as a
    /// coarse `FinalClaimMismatch`. Kernels without derived tables keep the
    /// no-op default.
    ///
    /// `relation` must be the STAGE's relation instance — the one whose
    /// `derive_opening_points` already ran (some relations capture their
    /// bound point there) and whose `expected_final_claim` the recipe checks
    /// against — not the member's internal copy.
    fn validate_derived_tables(
        &self,
        _relation: &Self::Relation,
        _input_points: &SumcheckInputPoints<F, Self::Relation>,
        _output_points: &SumcheckOutputPoints<F, Self::Relation>,
        _challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), KernelError<F>> {
        Ok(())
    }
}
