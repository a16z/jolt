use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
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
}
