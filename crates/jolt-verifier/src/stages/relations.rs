//! Shared per-relation opening-claim plumbing.
//!
//! The claim data model (the `OutputClaims`/`InputClaims` resolvers) lives in
//! `jolt-claims` and is re-exported here so existing
//! `crate::stages::relations::{..}` paths keep resolving. Those traits are
//! implemented by `#[derive(OutputClaims)]` / `#[derive(InputClaims)]` (crate
//! `jolt-claims-derive`) on each relation's cell-generic claim struct: the value
//! resolver on the `F` cell and the opening-point accessors on the `Vec<F>` cell.
//! This makes the canonical opening **order** and **count** a single-sourced
//! consequence of a struct's field declaration order.
//!
//! Transcript I/O stays here: [`OutputAppend::append_openings`] is a thin
//! verifier-side consumer of [`OutputClaims::opening_values`], so `jolt-claims`
//! stays transcript-free while the Fiat-Shamir order remains single-sourced.

pub use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};

/// `#[derive(SumcheckBatch)]` generates a stage's aggregate claim types from a
/// struct of [`ConcreteSumcheck`] instances; re-exported here alongside the
/// per-relation claim plumbing it composes. See `specs/sumcheck-batch-derive.md`.
pub use jolt_verifier_derive::SumcheckBatch;

use std::collections::BTreeSet;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId};
use jolt_claims::{MissingOpeningValue, SymbolicSumcheck};
use jolt_field::{Field, FieldCore};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_transcript::Transcript;

use crate::VerifierError;

/// Transcript-side companion to [`OutputClaims`]: append a relation's produced
/// openings to the Fiat-Shamir transcript in canonical order.
///
/// This lives in `jolt-verifier` (not `jolt-claims`) because it needs a
/// `Transcript`; `jolt-claims` stays transcript-free. It is a blanket extension
/// over every `OutputClaims` implementor, so the Fiat-Shamir order is
/// single-sourced by [`OutputClaims::opening_values`] and cannot disagree with it.
pub trait OutputAppend<F: Field>: OutputClaims<F> {
    /// Append every produced opening to the transcript in canonical
    /// ([`OutputClaims::opening_values`]) order, each under the `b"opening_claim"`
    /// label. This is the Fiat-Shamir order and MUST match the order in which the
    /// prover commits the openings.
    fn append_openings<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

impl<F: Field, C: OutputClaims<F>> OutputAppend<F> for C {}

/// The drawn Fiat-Shamir challenges of a [`ConcreteSumcheck`] instance: a readable
/// alias for the relation's `Challenges<F>` projection through its symbolic
/// relation. This is the struct [`ConcreteSumcheck::draw_challenges`] returns and
/// that [`input_claim`](ConcreteSumcheck::input_claim) /
/// [`expected_output`](ConcreteSumcheck::expected_output) resolve the challenge leg
/// against.
pub type ConcreteSumcheckChallenges<F, S> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Challenges<F>;

/// A [`ConcreteSumcheck`]'s consumed-claim values (wire form; implements [`InputClaims`]).
pub type SumcheckInputClaims<F, S> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Inputs<F>;
/// A [`ConcreteSumcheck`]'s consumed-claim opening points (carries per-field accessors).
pub type SumcheckInputPoints<F, S> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Inputs<::std::vec::Vec<F>>;
/// A [`ConcreteSumcheck`]'s produced-claim values (wire form; implements [`OutputClaims`]).
pub type SumcheckOutputClaims<F, S> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Outputs<F>;
/// A [`ConcreteSumcheck`]'s produced-claim opening points (carries per-field accessors).
pub type SumcheckOutputPoints<F, S> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Outputs<::std::vec::Vec<F>>;

/// A single sumcheck instance, driven identically by the prover (while producing
/// its proof) and the verifier (after checking it).
///
/// Each relation's consumed/produced claims are split into a *Values* form (the
/// serialized wire form, the cell-generic claim struct at `F` — one value per
/// opening) and a *Points* form (the derived opening points, the same struct at
/// `Vec<F>` — one point per opening). Methods that need only points
/// ([`derive_opening_points`](Self::derive_opening_points),
/// [`derive_output_term`](Self::derive_output_term)) take the Points forms and run
/// in both modes; methods that read values ([`input_claim`](Self::input_claim),
/// [`expected_output`](Self::expected_output)) take the Values forms. This makes
/// "a ZK opening carries no value" a compile-time fact.
pub trait ConcreteSumcheck<F: Field>
where
    SumcheckInputClaims<F, Self>: InputClaims<F>,
    SumcheckOutputClaims<F, Self>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, Self>: SumcheckChallenges<F, JoltChallengeId>,
{
    /// The relation's pure symbolic algebra: id types, sumcheck spec, and the
    /// input/output `Expr`s. The concrete instance holds its `Self::Symbolic` and
    /// sources its claim expressions and spec from it.
    type Symbolic: SymbolicSumcheck<
        RelationId = JoltRelationId,
        OpeningId = JoltOpeningId,
        DerivedId = JoltDerivedId,
        ChallengeId = JoltChallengeId,
    >;

    fn symbolic(&self) -> &Self::Symbolic;

    fn id(&self) -> JoltRelationId {
        Self::Symbolic::id()
    }

    fn rounds(&self) -> usize {
        self.symbolic().rounds()
    }

    fn degree(&self) -> usize {
        self.symbolic().degree()
    }

    /// Draw this instance's own (instance-private) Fiat-Shamir challenges from the
    /// transcript, in the exact order the stage's inline draw uses. Batch-level
    /// coefficients and the shared binding vector are NOT drawn here.
    ///
    /// The default draws one `challenge_scalar` per `Challenges` field, in
    /// declaration order, via [`SumcheckChallenges::from_transcript_values`]. This is
    /// the correct draw for the common case — a relation whose challenges are each a
    /// single `challenge_scalar` (and for [`NoChallenges`](::jolt_claims::NoChallenges),
    /// which has no fields, it draws nothing). A `challenge_scalar_powers(n)` draw
    /// reduces to this case: it performs exactly one squeeze and the relation keeps
    /// the degree-1 power, which equals that squeezed scalar. Only relations whose
    /// draw is genuinely different — an extra transcript append (a domain
    /// separator), a value re-roll, or a powers draw whose kept value is not the
    /// squeezed scalar — override this.
    ///
    /// The bound is `SumcheckChallenges` — which every `Challenges` already
    /// implements — so the default needs no separate `Default` derive. It errors
    /// only if the per-field draw cannot populate the struct, which cannot happen for
    /// the infinite `challenge_scalar` stream the default supplies.
    fn draw_challenges<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
    ) -> Result<ConcreteSumcheckChallenges<F, Self>, VerifierError> {
        SumcheckChallenges::from_transcript_values(::core::iter::repeat_with(|| {
            transcript.challenge_scalar()
        }))
        .map_err(VerifierError::from)
    }

    /// This relation's cross-relation opening aliases, as `(aliased, canonical
    /// source)` id pairs: each aliased opening is produced by this relation's
    /// output `Expr` but is the same polynomial, at the same (structurally
    /// identical) point, as the `source` opening produced by another member of the
    /// same stage batch. Aliased openings appear on the wire claims struct as
    /// plain (present) cells but are absorbed/committed once via their source, so
    /// the generated drivers use this set three ways: the absorb skips the aliased
    /// ids, the shape/count arithmetic subtracts them (see
    /// [`wire_output_openings`](Self::wire_output_openings)), and the generated
    /// `validate_aliases` — run by every `expected_final_claim` — enforces the
    /// wire copies equal their sources. That equality check is load-bearing: the
    /// aliased cells are never Fiat-Shamir-absorbed and the batch fold pins only
    /// their random linear combination, so downstream consumers reading a copy
    /// rely on it. BlindFold's `OpeningAlias` wiring is derived from these same
    /// pairs, which is why this is an associated function (no instance state): the
    /// alias structure is a constant of the relation, consumable without
    /// constructing one.
    ///
    /// Point equality is NOT checked at runtime: opening points are derived (not
    /// wire data), and an alias is only declarable when both relations bind the
    /// same batch-point slice and derive it identically — a structural invariant.
    /// The declaration invariants (each aliased id owned + `Expr`-referenced by
    /// the declaring relation, each source absorbed by another member binding an
    /// identical point slice) are pinned by hand-written tests in each declaring
    /// stage (`alias_declarations_are_valid`).
    fn aliased_output_openings() -> Vec<(JoltOpeningId, JoltOpeningId)>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// The opening ids this instance absorbs into the transcript (and commits in
    /// ZK): the output-`Expr`-referenced set minus the aliased openings (absorbed
    /// once via their canonical source). The generated `output_claim_count` sums
    /// these; the generated `validate_output_claims` compares the wire claims
    /// against them. A relation that absorbs openings its own output `Expr` does
    /// not reference (values whose constraining fold happens downstream, e.g. the
    /// product remainder's stage-6a-consumed flags) overrides this to add them.
    fn wire_output_openings(&self) -> BTreeSet<JoltOpeningId>
    where
        Self: Sized,
    {
        let mut openings = self.symbolic().expected_output_openings::<F>();
        for (aliased, _) in Self::aliased_output_openings() {
            let _ = openings.remove(&aliased);
        }
        openings
    }

    /// The offset of this instance's point within the batch challenge vector: the
    /// instance is bound on `batch_point[offset .. offset + rounds]`. Defaults to
    /// the front-loaded suffix (`batch_num_vars - rounds`); the two-phase address
    /// relations (stages 6/7) override to `0` (the prefix), and the stage-2 RAM
    /// relations to their phase-1 offset. Consumed by the generated
    /// `derive_opening_points` driver when slicing each member's point.
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        batch_num_vars.checked_sub(self.rounds()).ok_or_else(|| {
            VerifierError::StageClaimSumcheckFailed {
                stage: self.id(),
                reason: format!(
                    "batch challenge vector has {batch_num_vars} entries, fewer than the \
                     instance's {} rounds",
                    self.rounds()
                ),
            }
        })
    }

    /// The `batch_point[offset .. offset + rounds]` slice this instance is bound
    /// on, where `offset` is [`instance_point_offset`](Self::instance_point_offset)
    /// (the overridable knob; this method is the derived slice, so the
    /// offset/rounds pairing is single-sourced). Called by the generated
    /// `derive_opening_points` when slicing each member's point.
    fn instance_point<'a>(&self, batch_point: &'a [F]) -> Result<&'a [F], VerifierError> {
        let offset = self.instance_point_offset(batch_point.len())?;
        let rounds = self.rounds();
        offset
            .checked_add(rounds)
            .and_then(|end| batch_point.get(offset..end))
            .ok_or(VerifierError::StageClaimSumcheckFailed {
                stage: self.id(),
                reason: format!(
                    "instance point [{offset}, {offset} + {rounds}) exceeds the batch \
                     challenge vector ({} entries)",
                    batch_point.len(),
                ),
            })
    }

    /// Map this instance's sumcheck point and the upstream input points into the
    /// produced openings' points. Value-independent, so it runs in both the clear
    /// and ZK paths; any cross-input consistency required for a well-defined point
    /// (e.g. address agreement) is checked here.
    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &SumcheckInputPoints<F, Self>,
    ) -> Result<SumcheckOutputPoints<F, Self>, VerifierError>;

    /// Resolve a `Derived` in this relation's **input** expression: from the drawn
    /// challenges. The input claim is the claimed sum *before* binding, so no
    /// produced openings and no bound point are available here. Defaults to "no
    /// input deriveds"; overridden by relations that have them (e.g. `RamValCheck`'s
    /// `InitEval`/`InitSelector`).
    fn derive_input_term(
        &self,
        id: &JoltDerivedId,
        _challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: *id })
    }

    /// Resolve a `Derived` in this relation's **output** expression: from the input
    /// points, the produced openings' points (the bound point, post-binding), and the
    /// drawn challenges. The output claim is checked *after* binding, so the produced
    /// openings' points exist — hence `output_points` is non-optional. Most `eq`/`lt`
    /// deriveds live here (they evaluate at this sumcheck's bound point). Defaults to
    /// "no output deriveds"; overridden by relations that have them.
    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &SumcheckInputPoints<F, Self>,
        _output_points: &SumcheckOutputPoints<F, Self>,
        _challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: *id })
    }

    /// The input claim (claimed sum), evaluated from the input `Expr` against the
    /// wired input opening values and the drawn `challenges`. Shared by prover and
    /// verifier; clear only. The challenge leg resolves through the drawn
    /// [`Challenges`](SumcheckChallenges) struct (not a stored scalar), so the value
    /// the verifier folds is exactly the one [`draw_challenges`](Self::draw_challenges)
    /// produced.
    fn input_claim(
        &self,
        input_values: &SumcheckInputClaims<F, Self>,
        challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        self.symbolic().input_expression::<F>().try_evaluate(
            |id| {
                input_values
                    .resolve_input(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: *id })
            },
            |id| self.derive_input_term(id, challenges),
        )
    }

    /// The expected output claim, evaluated from the produced opening *values*, the
    /// produced opening *points* (for output deriveds), the input points, the drawn
    /// `challenges`, and the relation's derived public values. Shared by prover and
    /// verifier; clear only.
    fn expected_output(
        &self,
        input_points: &SumcheckInputPoints<F, Self>,
        output_values: &SumcheckOutputClaims<F, Self>,
        output_points: &SumcheckOutputPoints<F, Self>,
        challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| {
                output_values
                    .resolve_output(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: *id })
            },
            |id| self.derive_output_term(id, input_points, output_points, challenges),
        )
    }
}

/// Extraction/self-check failures a [`SumcheckKernel`] can surface: the
/// kernel-side error vocabulary the generated prove drivers name. Deliberately
/// small — compute-level failures (witness access, geometry) belong to the
/// kernel crate's own error type, which wraps this one; only the failures the
/// *typed extraction seam* can produce live here.
#[derive(Debug, thiserror::Error)]
pub enum SumcheckKernelError<F: FieldCore> {
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
/// Homed here (not in the kernel crate) so the generated `prove_clear`
/// drivers can name it: `jolt-verifier` cannot depend on the kernel crate,
/// which sits above it. Kernels do NOT own a relation instance — the stage's
/// relation is the single source of geometry, threaded back in through
/// [`validate_derived_tables`](Self::validate_derived_tables) — so batch and
/// kernel geometry cannot diverge.
pub trait SumcheckKernel<F: Field>: ProveRounds<F>
where
    SumcheckInputClaims<F, Self::Relation>: InputClaims<F>,
    SumcheckOutputClaims<F, Self::Relation>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, Self::Relation>: SumcheckChallenges<F, JoltChallengeId>,
{
    type Relation: ConcreteSumcheck<F>;

    /// Extract the member's typed produced-opening values from its fully
    /// bound state. Call after the engine's round loop has ingested every
    /// challenge.
    fn output_claims(
        &mut self,
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
}

/// One batch member's prepare-time protocol inputs, bundled: the stage's
/// relation instance (the typed request — geometry and points live on it, so
/// kernels read accessors instead of restated constructor arguments) plus the
/// member's consumed claim values, consumed opening points, and drawn
/// challenges. All four are pure functions of the relation and the upstream
/// carriers, which is what lets the generated driver construct the bundle
/// mechanically per member. Backend context (session, witness) is compute
/// plumbing, not protocol input — it stays outside, as positional arguments
/// of the preparer.
pub struct ProverInputs<'a, F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    pub relation: &'a R,
    pub claims: &'a SumcheckInputClaims<F, R>,
    pub points: &'a SumcheckInputPoints<F, R>,
    pub challenges: &'a ConcreteSumcheckChallenges<F, R>,
}

/// The error home shared by every [`PrepareSumcheck`] impl on one preparer:
/// a single associated error type carrying the `From` bounds the generated
/// drivers need, so a driver bounded on several member relations has one
/// unambiguous `Self::Error`.
pub trait SumcheckPreparer<F: Field> {
    type Error: From<VerifierError> + From<SumcheckError<F>> + From<SumcheckKernelError<F>>;
}

/// The dependency-inverted preparer bound the generated `prove_clear` drivers
/// name, one implementation per member relation: mint the boxed
/// [`SumcheckKernel`] that proves `R` from the member's [`ProverInputs`].
/// `jolt-prover` implements it for every relation on a small context struct
/// that forwards the bundle to its backend's slots — the only place backend
/// field names are spelled.
pub trait PrepareSumcheck<F, R>: SumcheckPreparer<F>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn prepare(
        &mut self,
        inputs: ProverInputs<'_, F, R>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, Self::Error>;
}

/// One member's absorbed opening scalars: its claims' `canonical_order`-aligned
/// values minus the member's [aliased
/// openings](ConcreteSumcheck::aliased_output_openings) (absorbed once via
/// their canonical source relation). Called by the generated `opening_values`
/// per member, in member declaration order.
pub fn absorbed_opening_values<F, I>(claims: &SumcheckOutputClaims<F, I>) -> Vec<F>
where
    F: Field,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F>,
{
    let skip: BTreeSet<_> = I::aliased_output_openings()
        .into_iter()
        .map(|(aliased, _)| aliased)
        .collect();
    claims
        .canonical_order()
        .into_iter()
        .zip(claims.opening_values())
        .filter(|(id, _)| !skip.contains(id))
        .map(|(_, value)| value)
        .collect()
}

/// Assert an optional member's output-claims presence agrees with the instance:
/// reject a present instance missing its claims cell, and reject claims
/// supplied for an absent instance. Called by the generated
/// `validate_output_claims` for each `Option` member (before its shape check),
/// and directly by a stage that curates its own shape checks (stage 6b).
pub fn validate_member_presence<F, I>(
    member: Option<&I>,
    claims: Option<&SumcheckOutputClaims<F, I>>,
) -> Result<(), VerifierError>
where
    F: Field,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F>,
{
    match (member, claims) {
        (Some(_), Some(_)) | (None, None) => Ok(()),
        (Some(member), None) => Err(VerifierError::StageClaimPublicInputFailed {
            stage: member.id(),
            reason: "present instance is missing its output claims".to_string(),
        }),
        (None, Some(claims)) => Err(match claims.canonical_order().into_iter().next() {
            Some(opening) => VerifierError::UnexpectedOpeningClaim { id: opening },
            None => VerifierError::StageClaimPublicInputFailed {
                stage: <I::Symbolic as SymbolicSumcheck>::id(),
                reason: "output claims supplied for an absent instance".to_string(),
            },
        }),
    }
}

/// Enforce one member's declared cross-relation opening aliases: each aliased
/// wire cell (resolved from the DECLARING member's claims) must equal its
/// canonical source opening, resolved across the batch by `resolve_source`
/// (the generated batch-wide resolver). Load-bearing — aliased cells are never
/// Fiat-Shamir absorbed and the batch fold pins only their random linear
/// combination, so downstream consumers reading a copy rely on this equality.
/// Called by the generated `validate_aliases` per member.
pub fn validate_member_aliases<F, I>(
    member: &I,
    claims: &SumcheckOutputClaims<F, I>,
    resolve_source: impl Fn(&JoltOpeningId) -> Option<F>,
) -> Result<(), VerifierError>
where
    F: Field,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F>,
{
    for (aliased, source) in I::aliased_output_openings() {
        let target = claims
            .resolve_output(&aliased)
            .ok_or(VerifierError::MissingOpeningClaim { id: aliased })?;
        let source_value =
            resolve_source(&source).ok_or(VerifierError::MissingOpeningClaim { id: source })?;
        if target != source_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: member.id(),
                left: aliased,
                right: source,
            });
        }
    }
    Ok(())
}

/// Assert one member's wire claims match its expected output shape: the
/// provided `canonical_order` id-set, minus the member's aliased openings
/// (absorbed via their canonical source; their value equality is enforced
/// separately by `validate_aliases`), must equal the member's
/// [`wire_output_openings`](ConcreteSumcheck::wire_output_openings). Called by
/// the generated `validate_output_claims` per member.
pub fn validate_member_output_shape<F, I>(
    member: &I,
    claims: &SumcheckOutputClaims<F, I>,
) -> Result<(), VerifierError>
where
    F: Field,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F>,
{
    let expected = member.wire_output_openings();
    let aliased: BTreeSet<_> = I::aliased_output_openings()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let provided: BTreeSet<_> = claims
        .canonical_order()
        .into_iter()
        .filter(|id| !aliased.contains(id))
        .collect();
    if provided != expected {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: member.id(),
            reason: format!(
                "output claim shape mismatch: expected {} openings, got {}",
                expected.len(),
                provided.len(),
            ),
        });
    }
    Ok(())
}

/// Test-only transcript double for asserting [`ConcreteSumcheck::draw_challenges`]
/// reproduces a stage's inline Fiat-Shamir draw exactly.
///
/// Unlike the `append_openings` recorder, challenge *squeezes*
/// (`challenge`/`challenge_scalar`/`challenge_scalar_powers`) append no bytes, so a
/// byte-chunk recorder cannot observe them. This double instead records an ordered
/// event log that distinguishes a squeeze from a byte-append (e.g. the
/// `ram_val_check` gamma domain separator), and returns a *distinct sequential*
/// scalar from each squeeze so a relation's stored challenge can be checked against
/// the squeeze that produced it.
#[cfg(test)]
pub(crate) mod draw_recording {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_transcript::Transcript;

    /// One observable transcript operation a `draw_challenges` performs.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum DrawEvent {
        /// A challenge squeeze (`challenge`/`challenge_scalar`/the single squeeze
        /// inside `challenge_scalar_powers`). Carries the 1-based squeeze index so
        /// the value a relation kept can be matched to its squeeze.
        Squeeze(u64),
        /// A raw byte append (a domain separator preceding a squeeze).
        Append(Vec<u8>),
    }

    /// A `Transcript` that logs every squeeze and byte-append in order. Each
    /// squeeze returns `Fr(index)` for the 1-based squeeze counter, so the powers
    /// `challenge_scalar_powers` derives are distinct and a stored `gamma` can be
    /// asserted to equal the squeezed value.
    #[derive(Clone, Default)]
    pub(crate) struct DrawRecordingTranscript {
        pub(crate) events: Vec<DrawEvent>,
        squeezes: u64,
    }

    impl Transcript for DrawRecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.events.push(DrawEvent::Append(bytes.to_vec()));
        }

        fn challenge(&mut self) -> Self::Challenge {
            self.squeezes += 1;
            self.events.push(DrawEvent::Squeeze(self.squeezes));
            Fr::from_u64(self.squeezes)
        }

        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }

    /// Run `draw` against a fresh recorder, returning its ordered event log and the
    /// draw's result. A `draw_challenges` and a hand-written replica of the inline
    /// draw, each passed through this, are directly comparable: equal event logs
    /// prove the same squeeze/append sequence, and the recorder's distinct
    /// sequential squeeze values let the returned challenge be checked against the
    /// replica's captured value.
    pub(crate) fn record<R>(
        draw: impl FnOnce(&mut DrawRecordingTranscript) -> R,
    ) -> (Vec<DrawEvent>, R) {
        let mut transcript = DrawRecordingTranscript::default();
        let result = draw(&mut transcript);
        (transcript.events, result)
    }
}

/// The append-order recorder shared by the stage `append_output_claims` ordering
/// locks: unlike the challenge recorder, it observes only byte appends.
#[cfg(test)]
pub(crate) mod append_recording {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_transcript::Transcript;

    /// A minimal `Transcript` double that records each appended byte chunk, so
    /// that append order can be compared without depending on the digest.
    #[derive(Clone, Default)]
    pub(crate) struct RecordingTranscript {
        pub(crate) chunks: Vec<Vec<u8>>,
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }

        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims_derive::{InputClaims, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_riscv::CircuitFlags;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn virt(polynomial: JoltVirtualPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(polynomial, relation)
    }

    fn committed(polynomial: JoltCommittedPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::committed(polynomial, relation)
    }

    use super::append_recording::RecordingTranscript;

    /// The chunk stream produced by appending `opening_values()` one-by-one is
    /// the reference Fiat-Shamir order; `append_openings` must reproduce it.
    fn assert_append_matches_values<C: OutputClaims<Fr>>(claims: &C) {
        let mut via_append = RecordingTranscript::default();
        claims.append_openings(&mut via_append);

        let mut via_values = RecordingTranscript::default();
        for value in claims.opening_values() {
            via_values.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(via_append.chunks, via_values.chunks);
    }

    #[derive(OutputClaims)]
    #[relation(InstructionReadRaf)]
    struct InstructionLeaf<C> {
        #[opening(LookupTableFlag)]
        lookup_table_flags: Vec<C>,
        #[opening(InstructionRa)]
        instruction_ra: Vec<C>,
        #[opening(InstructionRafFlag)]
        instruction_raf_flag: C,
    }

    #[test]
    fn output_leaf_encoders_follow_declaration_order() {
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(1), fr(2)],
            instruction_ra: vec![fr(3), fr(4), fr(5)],
            instruction_raf_flag: fr(6),
        };

        assert_eq!(claims.opening_values().len(), 6);
        assert_eq!(
            claims.opening_values(),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6)],
        );
        assert_eq!(
            claims.canonical_order().len(),
            claims.opening_values().len()
        );
        assert_append_matches_values(&claims);
    }

    #[test]
    fn output_leaf_resolves_indexed_and_scalar_ids() {
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(10), fr(11)],
            instruction_ra: vec![fr(20), fr(21)],
            instruction_raf_flag: fr(30),
        };
        let relation = JoltRelationId::InstructionReadRaf;

        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::LookupTableFlag(1), relation)),
            Some(fr(11)),
        );
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::InstructionRa(0), relation)),
            Some(fr(20)),
        );
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::InstructionRafFlag, relation)),
            Some(fr(30)),
        );
        // Out-of-range index and wrong relation both miss.
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::LookupTableFlag(2), relation)),
            None,
        );
        assert_eq!(
            claims.resolve_output(&virt(
                JoltVirtualPolynomial::InstructionRafFlag,
                JoltRelationId::RamRaClaimReduction,
            )),
            None,
        );
    }

    #[derive(OutputClaims)]
    #[relation(RamReadWriteChecking)]
    struct CommittedLeaf<C> {
        #[opening(committed = RamInc)]
        ram_inc: C,
        #[opening(committed = BytecodeChunk)]
        bytecode_chunks: Vec<C>,
    }

    #[test]
    fn output_leaf_resolves_committed_ids() {
        let claims = CommittedLeaf {
            ram_inc: fr(7),
            bytecode_chunks: vec![fr(8), fr(9)],
        };
        let relation = JoltRelationId::RamReadWriteChecking;

        assert_eq!(claims.opening_values().len(), 3);
        assert_eq!(claims.opening_values(), vec![fr(7), fr(8), fr(9)]);
        assert_eq!(
            claims.resolve_output(&committed(JoltCommittedPolynomial::RamInc, relation)),
            Some(fr(7)),
        );
        assert_eq!(
            claims.resolve_output(&committed(
                JoltCommittedPolynomial::BytecodeChunk(1),
                relation
            )),
            Some(fr(9)),
        );
        assert_append_matches_values(&claims);
    }

    #[derive(OutputClaims)]
    #[relation(SpartanShift)]
    struct PayloadLeaf<C> {
        #[opening(UnexpandedPC)]
        unexpanded_pc: C,
        #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
        is_virtual: C,
    }

    #[test]
    fn output_leaf_resolves_payload_carrying_variant_ids() {
        let claims = PayloadLeaf {
            unexpanded_pc: fr(1),
            is_virtual: fr(2),
        };
        let relation = JoltRelationId::SpartanShift;

        assert_eq!(claims.opening_values().len(), 2);
        assert_eq!(claims.opening_values(), vec![fr(1), fr(2)]);
        assert_eq!(
            claims.resolve_output(&virt(JoltVirtualPolynomial::UnexpandedPC, relation)),
            Some(fr(1)),
        );
        assert_eq!(
            claims.resolve_output(&virt(
                JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
                relation,
            )),
            Some(fr(2)),
        );
        // A different flag payload is a different opening and misses.
        assert_eq!(
            claims.resolve_output(&virt(
                JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
                relation,
            )),
            None,
        );
        assert_append_matches_values(&claims);
    }

    #[derive(OutputClaims)]
    #[relation(RamValCheck)]
    struct OptionalOutput<C> {
        #[opening(untrusted_advice)]
        untrusted: Option<C>,
        #[opening(committed = RamInc)]
        ram_inc: C,
    }

    #[test]
    fn output_leaf_handles_optional_fields() {
        let relation = JoltRelationId::RamValCheck;
        let present = OptionalOutput {
            untrusted: Some(fr(7)),
            ram_inc: fr(8),
        };
        assert_eq!(present.opening_values().len(), 2);
        assert_eq!(present.opening_values(), vec![fr(7), fr(8)]);
        assert_eq!(
            present.resolve_output(&JoltOpeningId::untrusted_advice(relation)),
            Some(fr(7)),
        );
        assert_eq!(
            present.resolve_output(&committed(JoltCommittedPolynomial::RamInc, relation)),
            Some(fr(8)),
        );
        assert_append_matches_values(&present);

        // An absent optional opening drops out of the count, the value stream,
        // the transcript appends, and id resolution.
        let absent = OptionalOutput {
            untrusted: None,
            ram_inc: fr(8),
        };
        assert_eq!(absent.opening_values().len(), 1);
        assert_eq!(absent.opening_values(), vec![fr(8)]);
        assert_eq!(
            absent.resolve_output(&JoltOpeningId::untrusted_advice(relation)),
            None,
        );
        assert_append_matches_values(&absent);
    }

    #[test]
    fn from_opening_values_reassembles_by_id() {
        // Round-trip: a hand-built instance's (canonical_order, opening_values)
        // pairs feed a map resolver; the assembled struct reproduces both.
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(1), fr(2)],
            instruction_ra: vec![fr(3), fr(4), fr(5)],
            instruction_raf_flag: fr(6),
        };
        let source: std::collections::BTreeMap<_, _> = claims
            .canonical_order()
            .into_iter()
            .zip(claims.opening_values())
            .collect();

        let rebuilt =
            InstructionLeaf::<Fr>::from_opening_values(|id| source.get(id).copied()).unwrap();
        assert_eq!(rebuilt.canonical_order(), claims.canonical_order());
        assert_eq!(rebuilt.opening_values(), claims.opening_values());
    }

    #[test]
    fn from_opening_values_tracks_option_presence_and_errors_on_missing_scalar() {
        let relation = JoltRelationId::RamValCheck;
        let advice_id = JoltOpeningId::untrusted_advice(relation);
        let ram_inc_id = committed(JoltCommittedPolynomial::RamInc, relation);

        // Present `Option`: both ids resolve.
        let present = OptionalOutput::<Fr>::from_opening_values(|id| {
            (*id == advice_id)
                .then(|| fr(7))
                .or_else(|| (*id == ram_inc_id).then(|| fr(8)))
        })
        .unwrap();
        assert_eq!(present.opening_values(), vec![fr(7), fr(8)]);

        // Absent `Option`: only the plain field resolves.
        let absent =
            OptionalOutput::<Fr>::from_opening_values(|id| (*id == ram_inc_id).then(|| fr(8)))
                .unwrap();
        assert_eq!(absent.opening_values(), vec![fr(8)]);
        assert_eq!(absent.resolve_output(&advice_id), None);

        // A plain field that fails to resolve is an error naming its id.
        let missing =
            OptionalOutput::<Fr>::from_opening_values(|id| (*id == advice_id).then(|| fr(7)));
        assert!(
            matches!(missing, Err(jolt_claims::MissingOpeningValue { id }) if id == ram_inc_id)
        );
    }

    #[test]
    fn canonical_order_lists_ids_in_declaration_order() {
        // A struct mixing `Vec` (element-wise) and scalar leaves: the ids appear in
        // field-declaration order, each `Vec` expanded by index, and the list lines
        // up one-for-one with `opening_values()`.
        let relation = JoltRelationId::InstructionReadRaf;
        let claims = InstructionLeaf {
            lookup_table_flags: vec![fr(1), fr(2)],
            instruction_ra: vec![fr(3), fr(4), fr(5)],
            instruction_raf_flag: fr(6),
        };
        assert_eq!(
            claims.canonical_order(),
            vec![
                virt(JoltVirtualPolynomial::LookupTableFlag(0), relation),
                virt(JoltVirtualPolynomial::LookupTableFlag(1), relation),
                virt(JoltVirtualPolynomial::InstructionRa(0), relation),
                virt(JoltVirtualPolynomial::InstructionRa(1), relation),
                virt(JoltVirtualPolynomial::InstructionRa(2), relation),
                virt(JoltVirtualPolynomial::InstructionRafFlag, relation),
            ],
        );
        // The canonical order is the id of each value at the same index.
        assert_eq!(
            claims.canonical_order().len(),
            claims.opening_values().len()
        );
        for id in claims.canonical_order() {
            assert!(claims.resolve_output(&id).is_some());
        }
    }

    #[test]
    fn canonical_order_skips_absent_options() {
        // An `Option` leaf contributes its id only when `Some`, so a present and an
        // absent struct list different ids — the order tracks instance presence.
        let relation = JoltRelationId::RamValCheck;
        let present = OptionalOutput {
            untrusted: Some(fr(7)),
            ram_inc: fr(8),
        };
        assert_eq!(
            present.canonical_order(),
            vec![
                JoltOpeningId::untrusted_advice(relation),
                committed(JoltCommittedPolynomial::RamInc, relation),
            ],
        );

        let absent = OptionalOutput {
            untrusted: None,
            ram_inc: fr(8),
        };
        assert_eq!(
            absent.canonical_order(),
            vec![committed(JoltCommittedPolynomial::RamInc, relation)],
        );
    }

    #[test]
    fn input_canonical_order_lists_ids_in_declaration_order() {
        // The `InputClaims` derive emits `canonical_order` too: same polynomial
        // across three producing relations, listed in field order.
        let inputs = ReductionInputs {
            raf: fr(1),
            read_write: fr(2),
            val_check: fr(3),
        };
        assert_eq!(
            inputs.canonical_order(),
            vec![
                virt(
                    JoltVirtualPolynomial::RamRa,
                    JoltRelationId::RamRafEvaluation
                ),
                virt(
                    JoltVirtualPolynomial::RamRa,
                    JoltRelationId::RamReadWriteChecking,
                ),
                virt(JoltVirtualPolynomial::RamRa, JoltRelationId::RamValCheck),
            ],
        );
    }

    #[test]
    fn output_leaf_point_accessors_follow_fields() {
        // The point cell (`C = Vec<F>`) exposes per-field accessors returning the
        // derived opening points: scalar `&[F]`, `Vec` `&[Vec<F>]`.
        let points = InstructionLeaf::<Vec<Fr>> {
            lookup_table_flags: vec![vec![fr(10)], vec![fr(11)]],
            instruction_ra: vec![vec![fr(12), fr(13)]],
            instruction_raf_flag: vec![fr(14)],
        };
        assert_eq!(
            points.lookup_table_flags(),
            &[vec![fr(10)], vec![fr(11)]] as &[Vec<Fr>]
        );
        assert_eq!(
            points.instruction_ra(),
            &[vec![fr(12), fr(13)]] as &[Vec<Fr>]
        );
        assert_eq!(points.instruction_raf_flag(), &[fr(14)] as &[Fr]);
    }

    #[test]
    fn output_leaf_option_point_accessor() {
        // The `Option` point accessor surfaces the point only when `Some`.
        let present = OptionalOutput::<Vec<Fr>> {
            untrusted: Some(vec![fr(7)]),
            ram_inc: vec![fr(8)],
        };
        assert_eq!(present.untrusted(), Some(&[fr(7)] as &[Fr]));
        assert_eq!(present.ram_inc(), &[fr(8)] as &[Fr]);

        let absent = OptionalOutput::<Vec<Fr>> {
            untrusted: None,
            ram_inc: vec![fr(8)],
        };
        assert_eq!(absent.untrusted(), None);
    }

    #[test]
    fn input_leaf_point_accessors_follow_fields() {
        // The `InputClaims` derive emits point accessors on the `Vec<F>` cell too.
        let points = ReductionInputs::<Vec<Fr>> {
            raf: vec![fr(1)],
            read_write: vec![fr(2)],
            val_check: vec![fr(3)],
        };
        assert_eq!(points.raf(), &[fr(1)] as &[Fr]);
        assert_eq!(points.read_write(), &[fr(2)] as &[Fr]);
        assert_eq!(points.val_check(), &[fr(3)] as &[Fr]);
    }

    #[derive(InputClaims)]
    struct ReductionInputs<C> {
        #[opening(RamRa, from = RamRafEvaluation)]
        raf: C,
        #[opening(RamRa, from = RamReadWriteChecking)]
        read_write: C,
        #[opening(RamRa, from = RamValCheck)]
        val_check: C,
    }

    #[test]
    fn input_leaf_resolves_same_polynomial_across_relations() {
        let inputs = ReductionInputs {
            raf: fr(1),
            read_write: fr(2),
            val_check: fr(3),
        };

        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamRafEvaluation
            )),
            Some(fr(1)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamReadWriteChecking,
            )),
            Some(fr(2)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamValCheck
            )),
            Some(fr(3)),
        );
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::RamRa,
                JoltRelationId::RamRaClaimReduction,
            )),
            None,
        );
    }

    #[derive(InputClaims)]
    struct OptionalInputs<C> {
        #[opening(LookupOutput, from = InstructionClaimReduction)]
        lookup_output: Option<C>,
        #[opening(LeftLookupOperand, from = InstructionClaimReduction)]
        left_lookup_operand: C,
    }

    #[test]
    fn input_leaf_surfaces_option_fields_directly() {
        let relation = JoltRelationId::InstructionClaimReduction;
        let present = OptionalInputs {
            lookup_output: Some(fr(9)),
            left_lookup_operand: fr(8),
        };
        assert_eq!(
            present.resolve_input(&virt(JoltVirtualPolynomial::LookupOutput, relation)),
            Some(fr(9)),
        );
        assert_eq!(
            present.resolve_input(&virt(JoltVirtualPolynomial::LeftLookupOperand, relation)),
            Some(fr(8)),
        );

        let absent = OptionalInputs {
            lookup_output: None,
            left_lookup_operand: fr(8),
        };
        assert_eq!(
            absent.resolve_input(&virt(JoltVirtualPolynomial::LookupOutput, relation)),
            None,
        );
    }
}

#[cfg(test)]
// `Fixture*Sumchecks` exist only to exercise `#[derive(SumcheckBatch)]`.
#[expect(clippy::unwrap_used)]
mod sumcheck_batch_derive_tests {
    use super::SumcheckBatch;
    use crate::stages::stage5::{
        InstructionReadRaf, InstructionReadRafOutputClaims, RegistersValEvaluation,
        RegistersValEvaluationOutputClaims,
    };
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::{Field, Fr, FromPrimitiveInt};

    fn instruction_read_raf() -> InstructionReadRaf<Fr> {
        InstructionReadRaf::new(InstructionReadRafDimensions::try_from((5, 128, 3)).unwrap())
    }

    fn registers_val_evaluation() -> RegistersValEvaluation<Fr> {
        RegistersValEvaluation::new(TraceDimensions::new(4))
    }

    #[derive(SumcheckBatch)]
    // The generated absorb resolves the alias skip-sets statically (no instance
    // state), so this alias-free fixture's members are never read.
    #[expect(dead_code)]
    struct FixtureSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    #[test]
    fn output_aggregate_opening_values_follow_declaration_order() {
        let fr = Fr::from_u64;
        let sumchecks = FixtureSumchecks {
            instruction_read_raf: instruction_read_raf(),
            registers_val_evaluation: registers_val_evaluation(),
        };
        let claims = FixtureOutputClaims::<Fr> {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: vec![fr(1), fr(2)],
                instruction_ra: vec![fr(3)],
                instruction_raf_flag: fr(4),
            },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: fr(5),
                rd_wa: fr(6),
            },
        };

        assert_eq!(
            sumchecks.opening_values(&claims),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6)],
        );
    }

    #[derive(SumcheckBatch)]
    struct FixtureOptionSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: Option<RegistersValEvaluation<F>>,
    }

    #[test]
    fn output_aggregate_chains_present_and_skips_absent_option_members() {
        let fr = Fr::from_u64;
        let instruction = || InstructionReadRafOutputClaims {
            lookup_table_flags: vec![fr(1)],
            instruction_ra: vec![fr(2)],
            instruction_raf_flag: fr(3),
        };

        let with_registers = FixtureOptionSumchecks {
            instruction_read_raf: instruction_read_raf(),
            registers_val_evaluation: Some(registers_val_evaluation()),
        };
        let present = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: Some(RegistersValEvaluationOutputClaims {
                rd_inc: fr(4),
                rd_wa: fr(5),
            }),
        };
        assert_eq!(
            with_registers.opening_values(&present),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5)]
        );

        let without_registers = FixtureOptionSumchecks {
            instruction_read_raf: instruction_read_raf(),
            registers_val_evaluation: None,
        };
        let absent = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: None,
        };
        assert_eq!(
            without_registers.opening_values(&absent),
            vec![fr(1), fr(2), fr(3)]
        );
    }

    /// Wire claims supplied for an `Option` member whose instance did not run are
    /// rejected by the generated `validate_output_claims` (attributed to the first
    /// supplied opening id), and the well-formed absent case still validates.
    #[test]
    fn validate_output_claims_rejects_claims_for_absent_member() {
        use jolt_claims::protocols::jolt::geometry::instruction::read_raf_output_openings;

        let fr = Fr::from_u64;
        let dimensions = InstructionReadRafDimensions::try_from((5, 128, 3)).unwrap();
        let sumchecks = FixtureOptionSumchecks::<Fr> {
            instruction_read_raf: InstructionReadRaf::new(dimensions),
            registers_val_evaluation: None,
        };

        // Shape-correct instruction claims (sized from the geometry), so the absent
        // member's supplied claims are the only defect.
        let openings = read_raf_output_openings(dimensions);
        let instruction = || InstructionReadRafOutputClaims {
            lookup_table_flags: vec![fr(0); openings.lookup_table_flags.len()],
            instruction_ra: vec![fr(0); openings.instruction_ra.len()],
            instruction_raf_flag: fr(0),
        };

        let unexpected = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: Some(RegistersValEvaluationOutputClaims {
                rd_inc: fr(1),
                rd_wa: fr(2),
            }),
        };
        assert!(matches!(
            sumchecks.validate_output_claims(&unexpected),
            Err(crate::VerifierError::UnexpectedOpeningClaim { .. })
        ));

        let well_formed = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: None,
        };
        assert!(sumchecks.validate_output_claims(&well_formed).is_ok());
    }

    // The opt-out fixture: `#[sumcheck_batch(no_opening_values)]` must still
    // generate the five aggregate structs but emit NO `opening_values` /
    // `append_output_claims` on the source struct. The inherent `opening_values`
    // below would collide with a generated one (the compiler rejects two inherent
    // methods of the same name), so this module compiling at all proves the
    // opt-out suppressed it.
    #[derive(SumcheckBatch)]
    #[sumcheck_batch(no_opening_values)]
    // The custom absorb below never reads the members (no aliased sets to consult).
    #[expect(dead_code)]
    struct FixtureCustomSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    impl FixtureCustomSumchecks<Fr> {
        /// A curated order distinct from the generated declaration order, to prove
        /// this is the one in effect (the generated method would chain instruction
        /// then registers; this reverses them).
        #[expect(
            clippy::unused_self,
            reason = "the signature mirrors the generated method it collides with"
        )]
        fn opening_values(&self, claims: &FixtureCustomOutputClaims<Fr>) -> Vec<Fr> {
            use crate::stages::relations::OutputClaims as _;
            claims
                .registers_val_evaluation
                .opening_values()
                .into_iter()
                .chain(claims.instruction_read_raf.opening_values())
                .collect()
        }
    }

    #[test]
    fn no_opening_values_suppresses_generated_impl() {
        let fr = Fr::from_u64;
        let sumchecks = FixtureCustomSumchecks {
            instruction_read_raf: instruction_read_raf(),
            registers_val_evaluation: registers_val_evaluation(),
        };
        let claims = FixtureCustomOutputClaims::<Fr> {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: vec![fr(1)],
                instruction_ra: vec![fr(2)],
                instruction_raf_flag: fr(3),
            },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: fr(4),
                rd_wa: fr(5),
            },
        };

        // The hand-written curated order (registers first), proving no generated
        // `opening_values` (which would be instruction-first) shadows or collides.
        assert_eq!(
            sumchecks.opening_values(&claims),
            vec![fr(4), fr(5), fr(1), fr(2), fr(3)]
        );
    }

    // The draw opt-out fixture: `#[sumcheck_batch(no_draw_challenges)]` must emit
    // NO `draw_challenges` on the source struct (a stage whose member challenges
    // have stage-level provenance hand-assembles its aggregate; the generated draw
    // would squeeze at the wrong transcript position if it existed). The inherent
    // `draw_challenges` below — with a deliberately incompatible signature — would
    // collide with a generated one, so this module compiling at all proves the
    // opt-out suppressed it.
    #[derive(SumcheckBatch)]
    #[sumcheck_batch(no_draw_challenges)]
    #[expect(dead_code)]
    struct FixtureNoDrawSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    impl<F: Field> FixtureNoDrawSumchecks<F> {
        #[expect(dead_code, clippy::unused_self)]
        fn draw_challenges(&self) {}
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod begin_batch_tests {
    use super::draw_recording::{record, DrawEvent};
    use super::ConcreteSumcheck as _;
    use crate::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
    use jolt_field::{Field, Fr, FromPrimitiveInt, MulPow2};
    use jolt_sumcheck::{append_sumcheck_claim, BatchMember, ClearSumcheckRecorder};
    use jolt_transcript::Transcript;

    #[derive(super::SumcheckBatch)]
    struct HeadFixtureSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: Option<RegistersValEvaluation<F>>,
    }

    fn fixture(registers: bool) -> HeadFixtureSumchecks<Fr> {
        HeadFixtureSumchecks {
            instruction_read_raf: InstructionReadRaf::new(
                InstructionReadRafDimensions::try_from((5, 128, 3)).unwrap(),
            ),
            registers_val_evaluation: registers
                .then(|| RegistersValEvaluation::new(TraceDimensions::new(4))),
        }
    }

    fn instruction_inputs() -> InstructionReadRafInputClaims<Fr> {
        let fr = Fr::from_u64;
        InstructionReadRafInputClaims {
            lookup_output: fr(2),
            left_lookup_operand: fr(3),
            right_lookup_operand: fr(5),
        }
    }

    /// `begin_batch` with a clear recorder must reproduce the exact head
    /// Fiat-Shamir sequence `verify_clear` performed before the factoring —
    /// per-member `input_claim` absorbed under `b"sumcheck_claim"` in
    /// declaration order, then one coefficient squeeze per member — and pack
    /// the prelude's engine and named views consistently.
    #[test]
    fn begin_batch_matches_head_replica_and_packs_prelude() {
        let sumchecks = fixture(true);
        let inputs = HeadFixtureInputClaims::<Fr> {
            instruction_read_raf: instruction_inputs(),
            registers_val_evaluation: Some(RegistersValEvaluationInputClaims {
                registers_val: Fr::from_u64(7),
            }),
        };
        let (_, challenges) = record(|t| sumchecks.draw_challenges(t));
        let challenges = challenges.unwrap();

        let (events, head) = record(|t| {
            let mut recorder = ClearSumcheckRecorder::<Fr, Fr>::new();
            sumchecks.begin_batch(&inputs, &challenges, &mut recorder, t)
        });
        let (batch, coefficients) = head.unwrap();

        // The replica head: input_claim is transcript-pure, so only the absorbs
        // and coefficient squeezes are observable events.
        let instruction_sum = sumchecks
            .instruction_read_raf
            .input_claim(
                &inputs.instruction_read_raf,
                &challenges.instruction_read_raf,
            )
            .unwrap();
        let registers_sum = sumchecks
            .registers_val_evaluation
            .as_ref()
            .unwrap()
            .input_claim(
                inputs.registers_val_evaluation.as_ref().unwrap(),
                challenges.registers_val_evaluation.as_ref().unwrap(),
            )
            .unwrap();
        let (replica_events, (instruction_coeff, registers_coeff)) = record(|t| {
            append_sumcheck_claim(t, &instruction_sum);
            append_sumcheck_claim(t, &registers_sum);
            (t.challenge_scalar(), t.challenge_scalar())
        });
        assert_eq!(events, replica_events);

        let instruction_rounds = sumchecks.instruction_read_raf.rounds();
        let registers_rounds = sumchecks
            .registers_val_evaluation
            .as_ref()
            .unwrap()
            .rounds();
        let max_num_vars = instruction_rounds.max(registers_rounds);
        assert_eq!(
            batch.members,
            vec![
                BatchMember {
                    input_claim: instruction_sum,
                    coefficient: instruction_coeff,
                    rounds: instruction_rounds,
                    offset: max_num_vars - instruction_rounds,
                },
                BatchMember {
                    input_claim: registers_sum,
                    coefficient: registers_coeff,
                    rounds: registers_rounds,
                    offset: max_num_vars - registers_rounds,
                },
            ],
        );
        assert_eq!(batch.max_num_vars, max_num_vars);
        assert_eq!(
            batch.claimed_sum,
            instruction_coeff * instruction_sum.mul_pow_2(max_num_vars - instruction_rounds)
                + registers_coeff * registers_sum.mul_pow_2(max_num_vars - registers_rounds),
        );
        assert_eq!(coefficients.instruction_read_raf, instruction_coeff);
        assert_eq!(coefficients.registers_val_evaluation, Some(registers_coeff));
    }

    /// An absent `Option` member contributes no absorb, no coefficient squeeze,
    /// and no batch entry.
    #[test]
    fn begin_batch_skips_absent_option_member() {
        let sumchecks = fixture(false);
        let inputs = HeadFixtureInputClaims::<Fr> {
            instruction_read_raf: instruction_inputs(),
            registers_val_evaluation: None,
        };
        let (_, challenges) = record(|t| sumchecks.draw_challenges(t));
        let challenges = challenges.unwrap();

        let (events, head) = record(|t| {
            let mut recorder = ClearSumcheckRecorder::<Fr, Fr>::new();
            sumchecks.begin_batch(&inputs, &challenges, &mut recorder, t)
        });
        let (batch, coefficients) = head.unwrap();

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, DrawEvent::Squeeze(_)))
                .count(),
            1,
        );
        assert_eq!(batch.members.len(), 1);
        assert_eq!(batch.max_num_vars, sumchecks.instruction_read_raf.rounds());
        assert_eq!(coefficients.registers_val_evaluation, None);
    }
}

/// Twin-transcript engine locks: toy members driven through
/// `jolt_sumcheck::prove_batch` (and the uni-skip provers) against the
/// GENERATED `verify_clear` / `verify_zk` drivers and the shared
/// `uniskip::verify_clear`, asserting byte-identical transcript states. This
/// pins the prove-side engine to the generated verifier before any real stage
/// depends on it.
#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod engine_twin_tests {
    use super::ConcreteSumcheck as _;
    use crate::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
    use crate::stages::uniskip;
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
    use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
    use jolt_field::{Field, Fr, FromPrimitiveInt};
    use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
    use jolt_r1cs::constraints::jolt::{
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    };
    use jolt_sumcheck::{
        prove_batch, prove_uniskip_clear, CenteredIntegerDomain, ClearRound, ClearSumcheckRecorder,
        CommittedSumcheckRecorder, ProveRounds, SumcheckDomain, SumcheckError, SumcheckRecorder,
        OPENING_CLAIM_TRANSCRIPT_LABEL,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    #[derive(super::SumcheckBatch)]
    struct TwinFixtureSumchecks<F: Field> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    /// Small geometry so a dense toy prover is feasible: the instruction
    /// member gets 8 rounds (6 address + 2 cycle), the registers member 3 —
    /// so the generated-driver twins also exercise front-loaded padding.
    fn fixture() -> TwinFixtureSumchecks<Fr> {
        TwinFixtureSumchecks {
            instruction_read_raf: InstructionReadRaf::new(
                InstructionReadRafDimensions::try_from((2, 6, 2)).unwrap(),
            ),
            registers_val_evaluation: RegistersValEvaluation::new(TraceDimensions::new(3)),
        }
    }

    fn inputs() -> TwinFixtureInputClaims<Fr> {
        let fr = Fr::from_u64;
        TwinFixtureInputClaims {
            instruction_read_raf: InstructionReadRafInputClaims {
                lookup_output: fr(2),
                left_lookup_operand: fr(3),
                right_lookup_operand: fr(5),
            },
            registers_val_evaluation: RegistersValEvaluationInputClaims {
                registers_val: fr(7),
            },
        }
    }

    /// A dense multilinear toy batch member with a prescribed total sum
    /// (HighToLow binding) — degree 1, which every relation's degree bound
    /// admits.
    struct DenseMember {
        evals: Vec<Fr>,
        num_rounds: usize,
    }

    impl DenseMember {
        fn with_sum(num_rounds: usize, sum: Fr, seed: u64) -> Self {
            let size = 1u64 << num_rounds;
            let mut evals: Vec<Fr> = (0..size)
                .map(|i| Fr::from_u64(seed + 31 * i + 11))
                .collect();
            let current: Fr = evals.iter().copied().sum();
            evals[0] += sum - current;
            Self { evals, num_rounds }
        }
    }

    impl DenseMember {
        fn bind(&mut self, challenge: Fr) {
            let half = self.evals.len() / 2;
            for i in 0..half {
                self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
            }
            self.evals.truncate(half);
        }
    }

    impl ProveRounds<Fr> for DenseMember {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn prove_round(
            &mut self,
            bind: Option<Fr>,
            _round: usize,
            previous_claim: Fr,
        ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
            if let Some(challenge) = bind {
                self.bind(challenge);
            }
            let half = self.evals.len() / 2;
            let eval_0: Fr = self.evals[..half].iter().copied().sum();
            let eval_1: Fr = self.evals[half..].iter().copied().sum();
            assert_eq!(eval_0 + eval_1, previous_claim);
            Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
        }

        fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
            self.bind(bind);
            Ok(())
        }
    }

    fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
        let generator = Bn254::g1_generator();
        let generators = (2..2 + capacity)
            .map(|k| generator.scalar_mul(&Fr::from_u64(k)))
            .collect();
        PedersenSetup::new(generators, generator.scalar_mul(&Fr::from_u64(99)))
    }

    /// Synthetic stand-ins for a stage's flattened output-claim values.
    fn synthetic_output_values() -> Vec<Fr> {
        vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)]
    }

    #[test]
    fn clear_engine_twin_matches_generated_verify_clear() {
        // Prover: draw → sums → begin_batch(clear) → prove_batch → finish.
        let sumchecks = fixture();
        let inputs = inputs();
        let mut prover_transcript = Blake2bTranscript::new(b"engine-twin");
        let prover_challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();

        let instruction_sum = sumchecks
            .instruction_read_raf
            .input_claim(
                &inputs.instruction_read_raf,
                &prover_challenges.instruction_read_raf,
            )
            .unwrap();
        let registers_sum = sumchecks
            .registers_val_evaluation
            .input_claim(
                &inputs.registers_val_evaluation,
                &prover_challenges.registers_val_evaluation,
            )
            .unwrap();
        let mut instruction_member =
            DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 5);
        let mut registers_member = DenseMember::with_sum(
            sumchecks.registers_val_evaluation.rounds(),
            registers_sum,
            91,
        );

        let mut recorder = ClearSumcheckRecorder::<Fr, Bn254G1>::new();
        let (batch, prover_coefficients) = sumchecks
            .begin_batch(
                &inputs,
                &prover_challenges,
                &mut recorder,
                &mut prover_transcript,
            )
            .unwrap();
        let mut members: Vec<&mut dyn ProveRounds<Fr>> =
            vec![&mut instruction_member, &mut registers_member];
        let proved =
            prove_batch(&batch, &mut members, &mut recorder, &mut prover_transcript).unwrap();
        let output_values = synthetic_output_values();
        let recorded = recorder
            .finish(&output_values, &mut prover_transcript)
            .unwrap();

        // Verifier: draw → generated verify_clear → output-claim absorbs.
        let mut verifier_transcript = Blake2bTranscript::new(b"engine-twin");
        let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let verified = sumchecks
            .verify_clear(
                &inputs,
                &verifier_challenges,
                &recorded.proof,
                &mut verifier_transcript,
            )
            .unwrap();
        for value in &output_values {
            verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, value);
        }

        assert_eq!(verified.reduction.value, proved.final_claim);
        assert_eq!(
            verified.reduction.point.as_slice(),
            proved.challenges.as_slice()
        );
        assert_eq!(verified.coefficients, prover_coefficients);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn committed_engine_twin_matches_generated_verify_zk() {
        type VC = Pedersen<Bn254G1>;
        let setup = pedersen_setup(8);

        // Prover: draw → sums → begin_batch(committed; claim absorbs no-op) →
        // prove_batch → finish (output-claim row commitments absorbed).
        let sumchecks = fixture();
        let inputs = inputs();
        let mut prover_transcript = Blake2bTranscript::new(b"engine-zk-twin");
        let prover_challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();

        let instruction_sum = sumchecks
            .instruction_read_raf
            .input_claim(
                &inputs.instruction_read_raf,
                &prover_challenges.instruction_read_raf,
            )
            .unwrap();
        let registers_sum = sumchecks
            .registers_val_evaluation
            .input_claim(
                &inputs.registers_val_evaluation,
                &prover_challenges.registers_val_evaluation,
            )
            .unwrap();
        let mut instruction_member =
            DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 23);
        let mut registers_member = DenseMember::with_sum(
            sumchecks.registers_val_evaluation.rounds(),
            registers_sum,
            57,
        );

        let mut recorder =
            CommittedSumcheckRecorder::<Fr, VC, _>::new(&setup, rand_core::OsRng).unwrap();
        let (batch, prover_coefficients) = sumchecks
            .begin_batch(
                &inputs,
                &prover_challenges,
                &mut recorder,
                &mut prover_transcript,
            )
            .unwrap();
        let mut members: Vec<&mut dyn ProveRounds<Fr>> =
            vec![&mut instruction_member, &mut registers_member];
        let proved =
            prove_batch(&batch, &mut members, &mut recorder, &mut prover_transcript).unwrap();
        let recorded = recorder
            .finish(&synthetic_output_values(), &mut prover_transcript)
            .unwrap();
        assert!(recorded.committed_witness.is_some());

        // Verifier: draw → generated verify_zk (coefficient draws, committed
        // round consistency, output-claim commitment absorbs).
        let mut verifier_transcript = Blake2bTranscript::new(b"engine-zk-twin");
        let _verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let consistency = sumchecks
            .verify_zk(&recorded.proof, &mut verifier_transcript)
            .unwrap();

        assert_eq!(consistency.challenges(), proved.challenges);
        assert_eq!(
            consistency.batching_coefficients,
            vec![
                prover_coefficients.instruction_read_raf,
                prover_coefficients.registers_val_evaluation,
            ],
        );
        assert_eq!(consistency.max_num_vars, batch.max_num_vars);
        assert_eq!(consistency.max_degree, batch.max_degree);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn uniskip_prover_twin_matches_uniskip_verify_clear() {
        let degree = SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE;
        let domain_size = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
        let poly = UnivariatePoly::new(
            (0..=degree as u64)
                .map(|k| Fr::from_u64(3 * k + 2))
                .collect(),
        );
        let coefficients = CenteredIntegerDomain::new(domain_size)
            .round_sum_coefficients(UnivariatePolynomial::degree(&poly))
            .unwrap();
        let input_claim = <UnivariatePoly<Fr> as ClearRound<Fr>>::coefficient_linear_combination(
            &poly,
            &coefficients,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"uniskip-stage-twin");
        let proved = prove_uniskip_clear::<Fr, Bn254G1, _>(
            poly,
            input_claim,
            degree,
            domain_size,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::new(b"uniskip-stage-twin");
        let challenge = uniskip::verify_clear(
            &proved.proof,
            &uniskip::UniskipParams::spartan_outer(),
            input_claim,
            proved.output_claim,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(challenge, proved.challenge);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }
}

/// Twin locks for the GENERATED `prove_clear` driver against a hand-rolled toy
/// stage: three self-consistent dense relations — a plain member, an `Option`
/// member (exercised absent and present), and a `#[sumcheck(external)]`
/// member — driven end to end (head → prepare → round loop → typed extraction
/// → shape validation → final-claim self-check → finish) and byte-compared
/// against the generated `verify_clear` on a twin transcript.
#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod prove_clear_tests {
    use core::marker::PhantomData;

    use jolt_claims::protocols::jolt::{
        JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims::{opening, NoChallenges, OutputClaims as _, SymbolicSumcheck};
    use jolt_field::{Field, Fr, FromPrimitiveInt, RingCore};
    use jolt_poly::UnivariatePoly;
    use jolt_sumcheck::{
        ClearSumcheckRecorder, CommittedSumcheckRecorder, ProveRounds, SumcheckError,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use super::{
        ConcreteSumcheck, PrepareSumcheck, ProverInputs, SumcheckKernel, SumcheckKernelError,
        SumcheckOutputClaims, SumcheckPreparer,
    };
    use crate::VerifierError;

    /// Declare one toy dense relation: a single produced opening, a single
    /// consumed claim carrying the true table sum, no challenges, degree 1.
    macro_rules! toy_relation {
        (
            $symbolic:ident, $relation:ident, $inputs:ident, $outputs:ident,
            rel = $rel:ident, output = $output:ident, input = $input:ident
        ) => {
            #[derive(Clone, Debug, Default, PartialEq, Eq, jolt_claims::InputClaims)]
            struct $inputs<C> {
                #[opening($input, from = $rel)]
                claimed_sum: C,
            }

            #[derive(
                Clone,
                Debug,
                PartialEq,
                Eq,
                jolt_claims::OutputClaims,
                serde::Serialize,
                serde::Deserialize,
            )]
            #[relation($rel)]
            struct $outputs<C> {
                #[opening($output)]
                value: C,
            }

            struct $symbolic {
                rounds: usize,
            }

            impl SymbolicSumcheck for $symbolic {
                type RelationId = JoltRelationId;
                type OpeningId = JoltOpeningId;
                type DerivedId = jolt_claims::protocols::jolt::JoltDerivedId;
                type ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId;
                type Shape = usize;
                type Challenges<F> = NoChallenges<F>;
                type Inputs<C> = $inputs<C>;
                type Outputs<C> = $outputs<C>;

                fn new(shape: usize) -> Self {
                    Self { rounds: shape }
                }

                fn id() -> JoltRelationId {
                    JoltRelationId::$rel
                }

                fn rounds(&self) -> usize {
                    self.rounds
                }

                fn degree(&self) -> usize {
                    1
                }

                fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$input,
                        JoltRelationId::$rel,
                    ))
                }

                fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
                    opening(JoltOpeningId::virtual_polynomial(
                        JoltVirtualPolynomial::$output,
                        JoltRelationId::$rel,
                    ))
                }
            }

            struct $relation<F: Field> {
                symbolic: $symbolic,
                _field: PhantomData<F>,
            }

            impl<F: Field> $relation<F> {
                fn new(rounds: usize) -> Self {
                    Self {
                        symbolic: $symbolic::new(rounds),
                        _field: PhantomData,
                    }
                }
            }

            impl<F: Field> ConcreteSumcheck<F> for $relation<F> {
                type Symbolic = $symbolic;

                fn symbolic(&self) -> &$symbolic {
                    &self.symbolic
                }

                fn derive_opening_points(
                    &self,
                    sumcheck_point: &[F],
                    _input_points: &$inputs<Vec<F>>,
                ) -> Result<$outputs<Vec<F>>, VerifierError> {
                    Ok($outputs {
                        value: sumcheck_point.to_vec(),
                    })
                }
            }
        };
    }

    toy_relation!(
        AlphaSymbolic,
        ToyAlpha,
        ToyAlphaInputs,
        ToyAlphaOutputs,
        rel = RegistersValEvaluation,
        output = LookupOutput,
        input = UnexpandedPC
    );
    toy_relation!(
        BetaSymbolic,
        ToyBeta,
        ToyBetaInputs,
        ToyBetaOutputs,
        rel = RamValCheck,
        output = LeftLookupOperand,
        input = UnexpandedPC
    );
    toy_relation!(
        GammaSymbolic,
        ToyGamma,
        ToyGammaInputs,
        ToyGammaOutputs,
        rel = SpartanShift,
        output = RightLookupOperand,
        input = UnexpandedPC
    );

    #[derive(super::SumcheckBatch)]
    struct ToyDriverSumchecks<F: Field> {
        alpha: ToyAlpha<F>,
        beta: Option<ToyBeta<F>>,
        gamma: ToyGamma<F>,
    }

    /// A dense multilinear kernel with a prescribed total sum (HighToLow
    /// binding, degree 1): the single produced opening is the fully bound
    /// table value, which is exactly the relation's `expected_output`.
    struct DenseKernel<R> {
        evals: Vec<Fr>,
        num_rounds: usize,
        _relation: PhantomData<fn() -> R>,
    }

    impl<R> DenseKernel<R> {
        fn with_sum(num_rounds: usize, sum: Fr, seed: u64) -> Self {
            let size = 1u64 << num_rounds;
            let mut evals: Vec<Fr> = (0..size)
                .map(|i| Fr::from_u64(seed + 31 * i + 11))
                .collect();
            let current: Fr = evals.iter().copied().sum();
            evals[0] += sum - current;
            Self {
                evals,
                num_rounds,
                _relation: PhantomData,
            }
        }

        fn bind(&mut self, challenge: Fr) {
            let half = self.evals.len() / 2;
            for i in 0..half {
                self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
            }
            self.evals.truncate(half);
        }
    }

    impl<R> ProveRounds<Fr> for DenseKernel<R> {
        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn prove_round(
            &mut self,
            bind: Option<Fr>,
            _round: usize,
            previous_claim: Fr,
        ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
            if let Some(challenge) = bind {
                self.bind(challenge);
            }
            let half = self.evals.len() / 2;
            let eval_0: Fr = self.evals[..half].iter().copied().sum();
            let eval_1: Fr = self.evals[half..].iter().copied().sum();
            assert_eq!(eval_0 + eval_1, previous_claim);
            Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
        }

        fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
            self.bind(bind);
            Ok(())
        }
    }

    impl<R> SumcheckKernel<Fr> for DenseKernel<R>
    where
        R: ConcreteSumcheck<Fr>,
        SumcheckOutputClaims<Fr, R>: super::OutputClaims<Fr>,
        super::SumcheckInputClaims<Fr, R>: super::InputClaims<Fr>,
        super::ConcreteSumcheckChallenges<Fr, R>:
            super::SumcheckChallenges<Fr, jolt_claims::protocols::jolt::JoltChallengeId>,
    {
        type Relation = R;

        fn output_claims(
            &mut self,
        ) -> Result<SumcheckOutputClaims<Fr, R>, SumcheckKernelError<Fr>> {
            assert_eq!(self.evals.len(), 1, "kernel extracted before fully bound");
            let value = self.evals[0];
            SumcheckOutputClaims::<Fr, R>::from_opening_values(|_| Some(value))
                .map_err(SumcheckKernelError::from)
        }
    }

    #[derive(Debug)]
    #[expect(dead_code, reason = "payloads exist for unwrap's Debug output")]
    enum ToyError {
        Verifier(VerifierError),
        Sumcheck(SumcheckError<Fr>),
        Kernel(SumcheckKernelError<Fr>),
    }

    impl From<VerifierError> for ToyError {
        fn from(error: VerifierError) -> Self {
            Self::Verifier(error)
        }
    }

    impl From<SumcheckError<Fr>> for ToyError {
        fn from(error: SumcheckError<Fr>) -> Self {
            Self::Sumcheck(error)
        }
    }

    impl From<SumcheckKernelError<Fr>> for ToyError {
        fn from(error: SumcheckKernelError<Fr>) -> Self {
            Self::Kernel(error)
        }
    }

    /// The toy preparer: mints dense kernels whose tables sum to the member's
    /// consumed claim (read off the `ProverInputs` bundle, like a real backend
    /// slot reads the relation), and records the prepare call order.
    struct ToyPreparer {
        calls: Vec<&'static str>,
    }

    impl SumcheckPreparer<Fr> for ToyPreparer {
        type Error = ToyError;
    }

    impl PrepareSumcheck<Fr, ToyAlpha<Fr>> for ToyPreparer {
        fn prepare(
            &mut self,
            inputs: ProverInputs<'_, Fr, ToyAlpha<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ToyAlpha<Fr>>>, ToyError> {
            self.calls.push("alpha");
            Ok(Box::new(DenseKernel::<ToyAlpha<Fr>>::with_sum(
                inputs.relation.rounds(),
                inputs.claims.claimed_sum,
                5,
            )))
        }
    }

    impl PrepareSumcheck<Fr, ToyBeta<Fr>> for ToyPreparer {
        fn prepare(
            &mut self,
            inputs: ProverInputs<'_, Fr, ToyBeta<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ToyBeta<Fr>>>, ToyError> {
            self.calls.push("beta");
            Ok(Box::new(DenseKernel::<ToyBeta<Fr>>::with_sum(
                inputs.relation.rounds(),
                inputs.claims.claimed_sum,
                91,
            )))
        }
    }

    impl PrepareSumcheck<Fr, ToyGamma<Fr>> for ToyPreparer {
        fn prepare(
            &mut self,
            inputs: ProverInputs<'_, Fr, ToyGamma<Fr>>,
        ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = ToyGamma<Fr>>>, ToyError> {
            self.calls.push("gamma");
            Ok(Box::new(DenseKernel::<ToyGamma<Fr>>::with_sum(
                inputs.relation.rounds(),
                inputs.claims.claimed_sum,
                23,
            )))
        }
    }

    const ALPHA_ROUNDS: usize = 3;
    const BETA_ROUNDS: usize = 2;
    const GAMMA_ROUNDS: usize = 3;

    fn fixture(beta: bool) -> ToyDriverSumchecks<Fr> {
        ToyDriverSumchecks {
            alpha: ToyAlpha::new(ALPHA_ROUNDS),
            beta: beta.then(|| ToyBeta::new(BETA_ROUNDS)),
            gamma: ToyGamma::new(GAMMA_ROUNDS),
        }
    }

    fn inputs(beta: bool) -> ToyDriverInputClaims<Fr> {
        let fr = Fr::from_u64;
        ToyDriverInputClaims {
            alpha: ToyAlphaInputs {
                claimed_sum: fr(1234),
            },
            beta: beta.then(|| ToyBetaInputs {
                claimed_sum: fr(777),
            }),
            gamma: ToyGammaInputs {
                claimed_sum: fr(4242),
            },
        }
    }

    /// Drive the generated `prove_clear` and its `verify_clear` twin, assert
    /// byte-identical transcript states, and return the driver's output.
    fn drive(beta: bool) -> (ProvedToyDriver<Fr, Fr>, Vec<&'static str>) {
        let sumchecks = fixture(beta);
        let inputs = inputs(beta);
        let mut preparer = ToyPreparer { calls: Vec::new() };

        let mut prover_transcript = Blake2bTranscript::new(b"prove-clear-twin");
        let challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();
        let input_points = sumchecks.empty_input_points();
        let proved = sumchecks
            .prove_clear(
                &mut preparer,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<Fr, Fr>::new(),
                &mut prover_transcript,
            )
            .unwrap();

        // Verifier twin: generated draw + verify_clear + output-claim absorbs.
        let mut verifier_transcript = Blake2bTranscript::new(b"prove-clear-twin");
        let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
        let verified = sumchecks
            .verify_clear(
                &inputs,
                &verifier_challenges,
                &proved.recorded.proof,
                &mut verifier_transcript,
            )
            .unwrap();
        sumchecks.append_output_claims(&mut verifier_transcript, &proved.output_claims);

        assert_eq!(verified.reduction.value, proved.final_claim);
        assert_eq!(
            sumchecks
                .expected_final_claim(
                    &verified.coefficients,
                    &input_points,
                    &proved.output_claims,
                    &sumchecks
                        .derive_opening_points(&verified.reduction.point, &input_points)
                        .unwrap(),
                    &verifier_challenges,
                )
                .unwrap(),
            proved.final_claim,
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
        (proved, preparer.calls)
    }

    #[test]
    fn prove_clear_twin_with_present_option_member() {
        let (proved, calls) = drive(true);
        // Prepare ran in declaration order.
        assert_eq!(calls, vec!["alpha", "beta", "gamma"]);
        assert!(proved.output_claims.beta.is_some());
        // Typed extraction filled every slot, external included.
        assert_eq!(proved.output_claims.alpha.opening_values().len(), 1);
        assert_eq!(proved.output_claims.gamma.opening_values().len(), 1);
    }

    #[test]
    fn prove_clear_twin_with_absent_option_member() {
        let (proved, calls) = drive(false);
        assert_eq!(calls, vec!["alpha", "gamma"]);
        assert!(proved.output_claims.beta.is_none());
    }

    /// `prove_clear` is generic over the recorder: this compiles it against
    /// the committed recorder even though nothing wires the ZK path yet.
    #[expect(dead_code, reason = "compile-only recorder-generality witness")]
    fn prove_clear_type_checks_with_committed_recorder(
        sumchecks: &ToyDriverSumchecks<Fr>,
        preparer: &mut ToyPreparer,
        inputs: &ToyDriverInputClaims<Fr>,
        input_points: &ToyDriverInputPoints<Fr>,
        challenges: &ToyDriverChallenges<Fr>,
        recorder: CommittedSumcheckRecorder<
            '_,
            Fr,
            jolt_crypto::Pedersen<jolt_crypto::Bn254G1>,
            rand_core::OsRng,
        >,
        transcript: &mut Blake2bTranscript,
    ) -> Result<ProvedToyDriver<Fr, jolt_crypto::Bn254G1>, ToyError> {
        sumchecks.prove_clear(
            preparer,
            inputs,
            input_points,
            challenges,
            recorder,
            transcript,
        )
    }
}
