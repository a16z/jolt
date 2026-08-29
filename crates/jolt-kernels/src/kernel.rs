//! The typed prove-side seam of a batch member: [`SumcheckKernel`] (the
//! execution object the generated stage drivers run), its extraction error
//! vocabulary, and [`ProverInputs`] (the prepare-time protocol bundle).
//! Homed here — with the driver generated into `jolt-prover`, nothing in
//! `jolt-verifier` needs to name them, and the verifier crate stays
//! prover-free.

use jolt_claims::protocols::jolt::{JoltDerivedId, JoltOpeningId};
use jolt_claims::MissingOpeningValue;
use jolt_field::{Field, JoltField};
use jolt_sumcheck::ProveRounds;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::VerifierError;

use crate::{MaybeAllocative, ProofSession};

/// Extraction/self-check failures a [`SumcheckKernel`] can surface: the
/// kernel-side error vocabulary the generated prove drivers name. Deliberately
/// small — compute-level failures (witness access, geometry) belong to
/// [`KernelError`](crate::KernelError), which wraps this one; only the
/// failures the *typed extraction seam* can produce live here.
#[derive(Debug, thiserror::Error)]
pub enum SumcheckKernelError<F: Field> {
    /// Relation-level failures (claim wiring, point derivation): kernels run
    /// the verifier's own relation methods as hard self-checks.
    #[error(transparent)]
    Verifier(#[from] VerifierError),

    #[error(transparent)]
    MissingOpeningValue(#[from] MissingOpeningValue<JoltOpeningId>),

    /// Final values were requested before every round was bound.
    #[error("final table values requested with {remaining} unbound rounds")]
    NotFullyBound { remaining: usize },

    /// A bound derived table's final value disagrees with the verifier's
    /// `derive_output_term` at the bound point — the hand-written table
    /// resolver drifted from the relation's scalar path.
    #[error("derived table {id:?} bound to {got}, but derive_output_term gives {expected}")]
    DerivedTableDrift {
        id: JoltDerivedId,
        expected: F,
        got: F,
    },

    /// A contract the kernel's inputs or internal state must uphold was
    /// violated — a bug, never a capability gap.
    #[error("kernel invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}

/// The typed prove-side counterpart of a batch member: pairs the object-safe
/// [`ProveRounds`] round interface (what the engine's round loop consumes)
/// with the member's [`ConcreteSumcheck`] relation, so the generated stage
/// drivers can extract typed output claims after the loop.
///
/// Kernels need not own a relation instance — the stage's relation is the
/// single source of geometry, threaded back in through
/// [`validate_derived_tables`](Self::validate_derived_tables). A kernel that
/// does keep a copy (the naive tier clones the driver-supplied instance)
/// must treat the threaded-in relation as authoritative.
///
/// Kernels are [`MaybeAllocative`](crate::MaybeAllocative): under the
/// `allocative` feature the generated drivers snapshot every live member
/// into a mid-stage heap flamegraph right after `prepare` — the stage's
/// retained-memory peak. Implement it with size arithmetic
/// (`Vec` capacity × element size; see the reference kernels) so `F` stays
/// unbounded.
// No claim-trait where-clauses: `Relation: ConcreteSumcheck<F>` already
// implies them (the ConcreteSumcheck where-clauses are elaborated at every use
// site), and spelling them with the relation's own id families — required for
// non-jolt protocol families — would name `Self` in a bound's type arguments,
// which breaks dyn compatibility.
pub trait SumcheckKernel<F: JoltField>: ProveRounds<F> + MaybeAllocative {
    type Relation: ConcreteSumcheck<F>;

    /// Extract the member's typed produced-opening values from its fully
    /// bound state. Call after the engine's round loop has ingested every
    /// challenge.
    ///
    /// `inputs` is the member's consumed claims — the same struct `prepare`
    /// received. It resolves the relation's *dual-role* openings (ids shared
    /// by the input and output claim structs, e.g. the RAM value-check advice
    /// cells): those never bind in the round loop, so their wire output value
    /// is the consumed input claim read back, mirroring the verifier's
    /// one-cell-two-readers wiring. Kernels without dual-role openings ignore
    /// it.
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>>;

    /// Cross-check any hand-materialized `Derived` leaf tables against the
    /// verifier's `derive_output_term` at the bound point. Call after the
    /// engine's round loop has ingested every challenge; the generated
    /// drivers run it on every member before the aggregate final-claim check,
    /// so a drifted table is attributed to its id rather than surfacing as a
    /// coarse final-claim mismatch. Kernels without derived tables keep the
    /// no-op default.
    ///
    /// `relation` must be the STAGE's relation instance — the one whose
    /// `derive_opening_points` already ran (some relations capture their
    /// bound point there) and whose `expected_final_claim` the driver checks
    /// against — not a kernel-internal copy.
    fn validate_derived_tables(
        &self,
        _relation: &Self::Relation,
        _input_points: &SumcheckInputPoints<F, Self::Relation>,
        _output_points: &SumcheckOutputPoints<F, Self::Relation>,
        _challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        Ok(())
    }

    /// Move any cross-batch residue this kernel carries into the session. The
    /// generated stage drivers call it uniformly on every member, after typed
    /// extraction and derived-table validation (both borrow the kernel; this
    /// call consumes it, so it is necessarily last). The default parks
    /// nothing; the stage-6b precommitted cycle kernels override it to park
    /// their post-cycle bound state as plain owned data for stage 7's
    /// address-phase `prepare` to reclaim.
    fn park_residue(self: Box<Self>, _session: &mut ProofSession) {}
}

/// One batch member's prepare-time protocol inputs, bundled: the stage's
/// relation instance (the typed request — geometry and points live on it, so
/// kernels read accessors instead of restated constructor arguments) plus the
/// member's consumed claim values, consumed opening points, and drawn
/// challenges. All four are pure functions of the relation and the upstream
/// carriers, which is what lets the generated driver construct the bundle
/// mechanically per member. Backend context (session, witness) is compute
/// plumbing, not protocol input — it stays outside, as positional arguments
/// of [`PrepareKernel::prepare`](crate::PrepareKernel::prepare).
pub struct ProverInputs<'a, F, R>
where
    F: JoltField,
    R: ConcreteSumcheck<F>,
{
    pub relation: &'a R,
    pub claims: &'a SumcheckInputClaims<F, R>,
    pub points: &'a SumcheckInputPoints<F, R>,
    pub challenges: &'a ConcreteSumcheckChallenges<F, R>,
}
