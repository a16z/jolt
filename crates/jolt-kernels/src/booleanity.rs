//! The booleanity slots: the stage-6a address phase and the stage-6b
//! cycle phase.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::relations::booleanity::BooleanityCyclePhaseChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-6a booleanity address-phase slot. `reference_address` and
/// `reference_cycle` are the little-endian reference points carried in
/// `Stage6aCarriedChallenges`; `gamma` is the booleanity batching challenge.
pub trait BooleanityAddressProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BooleanityDimensions,
        reference_address: &[F],
        reference_cycle: &[F],
        gamma: F,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>;
}

/// The stage-6b booleanity cycle-phase slot: a naive member — the cycle
/// `Expr` references each checked opening squared (`x·x − x`), which the
/// pointwise interpreter handles against one table per opening. Each opening
/// table is the address fold of the committed one-hot grid at the 6a bound
/// address point; the `EqAddressCycle` derived table is the (fixed) address
/// eq factor times the reference-cycle eq table (the carried little-endian
/// reference used verbatim, per the address-phase convention).
pub trait BooleanityCycleProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BooleanityDimensions,
        r_address: &[F],
        reference_address: &[F],
        reference_cycle: &[F],
        challenges: &BooleanityCyclePhaseChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = Booleanity<F>>>, KernelError<F>>;
}
