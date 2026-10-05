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
//! Transcript I/O stays here: [`receive_member_claims`] reads a relation's
//! produced claims into the shape of its derived output points, in the
//! canonical order the claims struct's declaration defines, on the routes of
//! [`ClaimRoutes`]. `jolt-claims` stays transcript-free while the wire order
//! remains single-sourced in each claims struct.

pub use jolt_claims::{InputClaims, MapCells, OutputClaims, SumcheckChallenges};

/// `#[derive(SumcheckBatch)]` generates a stage's aggregate claim types from a
/// struct of [`ConcreteSumcheck`] instances; re-exported here alongside the
/// per-relation claim plumbing it composes. See `specs/sumcheck-batch-derive.md`.
pub use jolt_verifier_derive::SumcheckBatch;

use core::convert::Infallible;
use core::fmt::Debug;
use std::collections::BTreeMap;

use jolt_blindfold::OpeningAlias;
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::stages::ids::{VerifierChallengeId, VerifierDerivedId};
use crate::VerifierError;

/// Re-exported for the `#[derive(SumcheckBatch)]`-generated batch-wide alias
/// resolver, whose closure is typed at the composite id so members from any
/// protocol family can chain into it.
pub use jolt_claims::protocols::composed::ComposedOpeningId;

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

/// A [`ConcreteSumcheck`]'s symbolic relation.
pub type SymbolicOf<F, S> = <S as ConcreteSumcheck<F>>::Symbolic;
/// A [`ConcreteSumcheck`]'s relation-id type, projected through its symbolic relation.
pub type RelationIdOf<F, S> = <SymbolicOf<F, S> as SymbolicSumcheck>::RelationId;
/// A [`ConcreteSumcheck`]'s opening-id type, projected through its symbolic relation.
pub type OpeningIdOf<F, S> = <SymbolicOf<F, S> as SymbolicSumcheck>::OpeningId;
/// A [`ConcreteSumcheck`]'s derived-id type, projected through its symbolic relation.
pub type DerivedIdOf<F, S> = <SymbolicOf<F, S> as SymbolicSumcheck>::DerivedId;
/// A [`ConcreteSumcheck`]'s challenge-id type, projected through its symbolic relation.
pub type ChallengeIdOf<F, S> = <SymbolicOf<F, S> as SymbolicSumcheck>::ChallengeId;

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
pub trait ConcreteSumcheck<F: JoltField>: Clone + Send + Sync
where
    SumcheckInputClaims<F, Self>: InputClaims<F, OpeningIdOf<F, Self>>,
    SumcheckOutputClaims<F, Self>: OutputClaims<F, OpeningIdOf<F, Self>>,
    ConcreteSumcheckChallenges<F, Self>: SumcheckChallenges<F, ChallengeIdOf<F, Self>>,
    RelationIdOf<F, Self>: Debug + Copy,
    OpeningIdOf<F, Self>: Copy + Ord + Debug + Into<ComposedOpeningId>,
    DerivedIdOf<F, Self>: Copy + Debug + Into<VerifierDerivedId>,
    ChallengeIdOf<F, Self>: Copy + Debug + Into<VerifierChallengeId>,
{
    /// The relation's pure symbolic algebra: id types, sumcheck spec, and the
    /// input/output `Expr`s. The concrete instance holds its `Self::Symbolic` and
    /// sources its claim expressions and spec from it. The id family is the
    /// symbolic relation's own — every id this trait exposes is a projection of
    /// it, so relations from any protocol family (jolt, field-inline) implement
    /// one trait.
    type Symbolic: SymbolicSumcheck;

    fn symbolic(&self) -> &Self::Symbolic;

    fn id(&self) -> RelationIdOf<F, Self> {
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
    /// The default draws one exactly uniform `challenge` per `Challenges` field,
    /// in declaration order, via [`SumcheckChallenges::from_transcript_values`].
    /// This is the correct draw for the common case — a relation whose challenges
    /// are each a single `challenge` (and for
    /// [`NoChallenges`](::jolt_claims::NoChallenges), which has no fields, it draws
    /// nothing). A `challenge_powers(n)` draw reduces to this case: it performs
    /// exactly one draw and the relation keeps the degree-1 power, which equals
    /// that drawn scalar. Only relations whose draw is genuinely different — an
    /// extra transcript absorb (a domain separator), a value re-roll, or a powers
    /// draw whose kept value is not the drawn scalar — override this.
    ///
    /// The bound is `SumcheckChallenges` — which every `Challenges` already
    /// implements — so the default needs no separate `Default` derive. It errors
    /// only if the per-field draw cannot populate the struct, which cannot happen for
    /// the infinite `challenge` stream the default supplies.
    fn draw_challenges<C: Channel>(
        &self,
        transcript: &mut C,
    ) -> Result<ConcreteSumcheckChallenges<F, Self>, VerifierError> {
        SumcheckChallenges::from_transcript_values(::core::iter::repeat_with(|| {
            transcript.challenge()
        }))
        .map_err(VerifierError::from)
    }

    /// This relation's cross-relation opening aliases, as `(aliased, canonical
    /// source)` id pairs: each aliased opening is produced by this relation's
    /// output `Expr` but is the same polynomial, at the same (structurally
    /// identical) point, as the `source` opening produced by another member of the
    /// same stage batch. Aliased openings appear on the wire claims struct as
    /// plain (present) cells but are absorbed/committed once via their source, so
    /// the generated drivers use this set three ways: these cells default to
    /// [`ClaimRoute::Alias`], so they are neither sent nor committed; BlindFold
    /// binds them through the generated layout's `OpeningAlias` rows; and the generated
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
    fn aliased_output_openings() -> Vec<(OpeningIdOf<F, Self>, OpeningIdOf<F, Self>)>
    where
        Self: Sized,
    {
        Vec::new()
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
                stage: format!("{:?}", self.id()),
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
                stage: format!("{:?}", self.id()),
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
        id: &DerivedIdOf<F, Self>,
        _challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
    }

    /// Resolve a `Derived` in this relation's **output** expression: from the input
    /// points, the produced openings' points (the bound point, post-binding), and the
    /// drawn challenges. The output claim is checked *after* binding, so the produced
    /// openings' points exist — hence `output_points` is non-optional. Most `eq`/`lt`
    /// deriveds live here (they evaluate at this sumcheck's bound point). Defaults to
    /// "no output deriveds"; overridden by relations that have them.
    fn derive_output_term(
        &self,
        id: &DerivedIdOf<F, Self>,
        _input_points: &SumcheckInputPoints<F, Self>,
        _output_points: &SumcheckOutputPoints<F, Self>,
        _challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
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
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
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
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| self.derive_output_term(id, input_points, output_points, challenges),
        )
    }
}

/// Resolve one batch member's produced opening by composite id: downcast the
/// composite to the member's own opening-id family, then resolve within the
/// member's claims. A foreign-family or unknown id is a miss (`None`), so the
/// generated batch-wide alias resolver can chain members across protocol
/// families. Called by the generated `validate_aliases` per member.
pub fn resolve_member_opening<F, I>(
    claims: &SumcheckOutputClaims<F, I>,
    id: &ComposedOpeningId,
) -> Option<F>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F, OpeningIdOf<F, I>>,
    OpeningIdOf<F, I>: TryFrom<ComposedOpeningId>,
{
    let native = OpeningIdOf::<F, I>::try_from(*id).ok()?;
    claims.resolve_output(&native)
}

/// How one produced claim cell reaches the verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimRoute {
    /// Sent after the stage's rounds (clear) or committed in the stage's
    /// output-claim rows (ZK).
    Sent,
    /// A copy of an earlier cell of the same batch: never sent or committed,
    /// filled from that source.
    Alias(ComposedOpeningId),
    /// Sent earlier in the stage as part of a typed staged message (clear),
    /// committed in the output-claim rows (ZK).
    Staged,
}

/// One stage batch's claim routes. A cell takes the route set here, else its
/// member's static alias ([`ConcreteSumcheck::aliased_output_openings`]),
/// else [`ClaimRoute::Sent`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaimRoutes {
    routes: BTreeMap<ComposedOpeningId, ClaimRoute>,
}

impl ClaimRoutes {
    /// Routes cell `id` by `route`, overriding its member's static route.
    pub fn set(&mut self, id: impl Into<ComposedOpeningId>, route: ClaimRoute) {
        let _ = self.routes.insert(id.into(), route);
    }

    fn of(
        &self,
        id: &ComposedOpeningId,
        static_aliases: &BTreeMap<ComposedOpeningId, ComposedOpeningId>,
    ) -> ClaimRoute {
        self.routes.get(id).copied().unwrap_or_else(|| {
            static_aliases
                .get(id)
                .map_or(ClaimRoute::Sent, |source| ClaimRoute::Alias(*source))
        })
    }
}

fn static_aliases<F, I>() -> BTreeMap<ComposedOpeningId, ComposedOpeningId>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    OpeningIdOf<F, I>: Into<ComposedOpeningId>,
{
    I::aliased_output_openings()
        .into_iter()
        .map(|(aliased, source)| (aliased.into(), source.into()))
        .collect()
}

/// Receive one batch member's output claims in the shape of its derived
/// output `points`, in the claims struct's canonical order: each `Sent` cell
/// is read from the transcript, each `Alias` cell copies its already received
/// source, and each `Staged` cell takes the value the stage received earlier.
/// `received` holds the batch's values by id (pre-filled with the staged
/// ones) and gains every cell read here. Called by the generated
/// `receive_output_claims` per member, in declaration order.
pub fn receive_member_claims<F, I, H>(
    points: &SumcheckOutputPoints<F, I>,
    routes: &ClaimRoutes,
    received: &mut BTreeMap<ComposedOpeningId, F>,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<SumcheckOutputClaims<F, I>, VerifierError>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    H: Sponge,
    SumcheckOutputPoints<F, I>:
        MapCells<Vec<F>, F, OpeningIdOf<F, I>, Mapped = SumcheckOutputClaims<F, I>>,
    OpeningIdOf<F, I>: Copy + Into<ComposedOpeningId>,
{
    let static_aliases = static_aliases::<F, I>();
    points.try_map_cells(&mut |id, _point| {
        let id: ComposedOpeningId = (*id).into();
        let source = match routes.of(&id, &static_aliases) {
            ClaimRoute::Sent => {
                let value: F = transcript.receive()?;
                let _ = received.insert(id, value);
                return Ok(value);
            }
            ClaimRoute::Alias(source) => source,
            ClaimRoute::Staged => id,
        };
        received
            .get(&source)
            .copied()
            .ok_or(VerifierError::MissingOpeningClaim { id: source })
    })
}

/// One member's claim values on the given routes, in canonical order:
/// `Sent` cells, plus `Staged` cells when `with_staged`. Clear proofs send
/// `with_staged = false` after the rounds; committed proofs commit
/// `with_staged = true`.
pub fn member_claim_values<F, I>(
    claims: &SumcheckOutputClaims<F, I>,
    routes: &ClaimRoutes,
    with_staged: bool,
) -> Vec<F>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F, OpeningIdOf<F, I>>,
    OpeningIdOf<F, I>: Copy + Into<ComposedOpeningId>,
{
    let static_aliases = static_aliases::<F, I>();
    claims
        .canonical_order()
        .into_iter()
        .zip(claims.opening_values())
        .filter(|(id, _)| match routes.of(&(*id).into(), &static_aliases) {
            ClaimRoute::Sent => true,
            ClaimRoute::Staged => with_staged,
            ClaimRoute::Alias(_) => false,
        })
        .map(|(_, value)| value)
        .collect()
}

/// A stage's committed claim layout over its derived output points: the
/// committed (`Sent` and `Staged`) cells' ids in row order, and each `Alias`
/// cell bound to its source row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommittedClaimLayout {
    pub ids: Vec<ComposedOpeningId>,
    pub aliases: Vec<OpeningAlias<ComposedOpeningId>>,
}

/// Appends one member's committed cells to `layout`, in canonical order over
/// the member's derived output `points`.
pub fn extend_committed_layout<F, I>(
    layout: &mut CommittedClaimLayout,
    points: &SumcheckOutputPoints<F, I>,
    routes: &ClaimRoutes,
) where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputPoints<F, I>:
        MapCells<Vec<F>, F, OpeningIdOf<F, I>, Mapped = SumcheckOutputClaims<F, I>>,
    OpeningIdOf<F, I>: Copy + Into<ComposedOpeningId>,
{
    let static_aliases = static_aliases::<F, I>();
    for id in point_cell_ids::<F, I>(points).into_iter().map(Into::into) {
        match routes.of(&id, &static_aliases) {
            ClaimRoute::Alias(source) => layout.aliases.push(OpeningAlias::new(id, source)),
            ClaimRoute::Sent | ClaimRoute::Staged => layout.ids.push(id),
        }
    }
}

/// The ids of a member's output-point cells, in canonical order.
fn point_cell_ids<F, I>(points: &SumcheckOutputPoints<F, I>) -> Vec<OpeningIdOf<F, I>>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputPoints<F, I>:
        MapCells<Vec<F>, F, OpeningIdOf<F, I>, Mapped = SumcheckOutputClaims<F, I>>,
    OpeningIdOf<F, I>: Copy,
{
    let mut ids = Vec::new();
    let Ok(_) = points.try_map_cells(&mut |id, _point| {
        ids.push(*id);
        Ok::<F, Infallible>(F::zero())
    });
    ids
}

/// Assert a member's claim values have the shape of its derived output
/// points: the same canonical ids, so every `Vec` family has its length and
/// every `Option` cell its presence. The prover's self-check before sending.
pub fn validate_member_output_shape<F, I>(
    member: &I,
    claims: &SumcheckOutputClaims<F, I>,
    points: &SumcheckOutputPoints<F, I>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F, OpeningIdOf<F, I>>,
    SumcheckOutputPoints<F, I>:
        MapCells<Vec<F>, F, OpeningIdOf<F, I>, Mapped = SumcheckOutputClaims<F, I>>,
    OpeningIdOf<F, I>: Copy + PartialEq,
    RelationIdOf<F, I>: Debug,
{
    let point_ids = point_cell_ids::<F, I>(points);
    if claims.canonical_order() != point_ids {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", member.id()),
            reason: format!(
                "output claim shape mismatch: the points have {} cells, the claims {}",
                point_ids.len(),
                claims.canonical_order().len(),
            ),
        });
    }
    Ok(())
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
    // The resolver is keyed by the composite id so a mixed-family batch can
    // serve every member through one closure; alias PAIRS stay within the
    // declaring member's own family (`aliased_output_openings` returns its
    // family's ids on both sides), so cross-family aliasing remains
    // unrepresentable at the declaration level.
    resolve_source: impl Fn(&ComposedOpeningId) -> Option<F>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    I: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, I>: OutputClaims<F, OpeningIdOf<F, I>>,
    OpeningIdOf<F, I>: Copy + Into<ComposedOpeningId>,
    RelationIdOf<F, I>: Debug,
{
    for (aliased, source) in I::aliased_output_openings() {
        let target = claims
            .resolve_output(&aliased)
            .ok_or(VerifierError::MissingOpeningClaim { id: aliased.into() })?;
        let source_value = resolve_source(&source.into())
            .ok_or(VerifierError::MissingOpeningClaim { id: source.into() })?;
        if target != source_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: format!("{:?}", member.id()),
                left: aliased.into(),
                right: source.into(),
            });
        }
    }
    Ok(())
}

/// Project a composite-family derived id onto one relation's own public enum —
/// the typed destructure every `derive_output_term` starts with, driven by the
/// id family's generated `TryFrom` inverses of its `From` embeddings. A foreign
/// relation's id is the same `MissingStageClaimDerived` miss the hand-written
/// destructures returned. Family-generic: no protocol ids appear here.
pub fn project_public<D, P>(id: &D) -> Result<P, VerifierError>
where
    D: Copy + Into<VerifierDerivedId>,
    P: TryFrom<D>,
{
    P::try_from(*id).map_err(|_| VerifierError::MissingStageClaimDerived { id: (*id).into() })
}

/// Wrap a point-geometry failure in the uniform stage error, keyed by the
/// relation's Debug-formatted id. Shared by the relations whose
/// `derive_opening_points` / `derive_output_term` residue is point geometry, so
/// none carries its own error-wrapping helper.
pub fn stage_claim_failed(stage: impl Debug, reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{stage:?}"),
        reason: reason.to_string(),
    }
}

/// Test-only transcripts for pinning a production Fiat-Shamir draw against
/// its documented sequence. Every transcript starts from the same protocol and
/// session, so two of them yield equal challenges and equal sponge fingerprints
/// exactly when they performed the same operations.
#[cfg(test)]
pub(crate) mod test_transcript {
    use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript};

    pub(crate) type TestTranscript = ProverTranscript<Blake2b512>;

    pub(crate) fn fresh() -> TestTranscript {
        ProverTranscript::new(
            &ProtocolId::new::<Blake2b512>("jolt-verifier/unit-tests"),
            b"unit-tests",
        )
    }

    /// Runs `production` and `documented` on identical fresh transcripts and
    /// asserts they leave the sponge in the same state, which pins the number,
    /// kind, and order of their operations. Returns both results so the caller
    /// can compare the values `production` kept against the documented draws.
    pub(crate) fn assert_same_draws<A, B>(
        production: impl FnOnce(&mut TestTranscript) -> A,
        documented: impl FnOnce(&mut TestTranscript) -> B,
    ) -> (A, B) {
        let mut left = fresh();
        let produced = production(&mut left);
        let mut right = fresh();
        let expected = documented(&mut right);
        assert_eq!(
            left.preview().squeeze::<32>(),
            right.preview().squeeze::<32>(),
            "production and documented draws diverge"
        );
        (produced, expected)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims_derive::{InputClaims, OutputClaims};
    use jolt_field::{Fr, Ring};
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

        // An absent optional opening drops out of the count, the value stream,
        // and id resolution.
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
    use super::{ClaimRoutes, SumcheckBatch};
    use crate::stages::stage5::{
        InstructionReadRaf, InstructionReadRafOutputClaims, RegistersValEvaluation,
        RegistersValEvaluationOutputClaims,
    };
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::{Fr, JoltField, Ring};

    #[derive(SumcheckBatch)]
    #[sumcheck_batch(crate = "crate")]
    // The generated claim plumbing reads only the claims and routes, so this
    // fixture's members are never read.
    #[expect(dead_code)]
    struct FixtureSumchecks<F: JoltField> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    #[test]
    fn wire_claims_follow_declaration_order() {
        let fr = Fr::from_u64;
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
            FixtureSumchecks::wire_claim_values(&claims, &ClaimRoutes::default()),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6)],
        );
    }

    #[derive(SumcheckBatch)]
    #[sumcheck_batch(crate = "crate")]
    struct FixtureOptionSumchecks<F: JoltField> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: Option<RegistersValEvaluation<F>>,
    }

    #[test]
    fn wire_claims_chain_present_and_skip_absent_option_members() {
        let fr = Fr::from_u64;
        let instruction = || InstructionReadRafOutputClaims {
            lookup_table_flags: vec![fr(1)],
            instruction_ra: vec![fr(2)],
            instruction_raf_flag: fr(3),
        };

        let present = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: Some(RegistersValEvaluationOutputClaims {
                rd_inc: fr(4),
                rd_wa: fr(5),
            }),
        };
        assert_eq!(
            FixtureOptionSumchecks::wire_claim_values(&present, &ClaimRoutes::default()),
            vec![fr(1), fr(2), fr(3), fr(4), fr(5)]
        );

        let absent = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: None,
        };
        assert_eq!(
            FixtureOptionSumchecks::wire_claim_values(&absent, &ClaimRoutes::default()),
            vec![fr(1), fr(2), fr(3)]
        );
    }

    /// Claims supplied for an `Option` member whose instance did not run are
    /// rejected by the generated `validate_output_shape`, and the well-formed
    /// absent case still validates.
    #[test]
    fn validate_output_shape_rejects_claims_for_absent_member() {
        use super::ConcreteSumcheck as _;

        use jolt_claims::protocols::jolt::geometry::instruction::read_raf_output_openings;

        let fr = Fr::from_u64;
        let dimensions = InstructionReadRafDimensions::try_from((2, 6, 2)).unwrap();
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
        let challenges = vec![fr(7); sumchecks.instruction_read_raf.rounds()];
        let points = sumchecks
            .derive_opening_points(&challenges, &sumchecks.empty_input_points())
            .unwrap();
        assert!(matches!(
            sumchecks.validate_output_shape(&unexpected, &points),
            Err(crate::VerifierError::StageClaimSumcheckFailed { .. })
        ));

        let well_formed = FixtureOptionOutputClaims::<Fr> {
            instruction_read_raf: instruction(),
            registers_val_evaluation: None,
        };
        assert!(sumchecks
            .validate_output_shape(&well_formed, &points)
            .is_ok());
    }

    // The draw opt-out fixture: `#[sumcheck_batch(no_draw_challenges)]` must emit
    // NO `draw_challenges` on the source struct (a stage whose member challenges
    // have stage-level provenance hand-assembles its aggregate; the generated draw
    // would squeeze at the wrong transcript position if it existed). The inherent
    // `draw_challenges` below — with a deliberately incompatible signature — would
    // collide with a generated one, so this module compiling at all proves the
    // opt-out suppressed it.
    #[derive(SumcheckBatch)]
    #[sumcheck_batch(no_draw_challenges, crate = "crate")]
    #[expect(dead_code)]
    struct FixtureNoDrawSumchecks<F: JoltField> {
        instruction_read_raf: InstructionReadRaf<F>,
        registers_val_evaluation: RegistersValEvaluation<F>,
    }

    impl<F: JoltField> FixtureNoDrawSumchecks<F> {
        #[expect(dead_code, clippy::unused_self)]
        fn draw_challenges(&self) {}
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod begin_batch_tests {
    use super::test_transcript::{assert_same_draws, fresh};
    use super::ConcreteSumcheck as _;
    use crate::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
    use jolt_field::{Fr, JoltField, Ring};
    use jolt_sumcheck::{BatchMember, ClearSumcheckRecorder};
    use jolt_transcript::Channel;

    #[derive(super::SumcheckBatch)]
    #[sumcheck_batch(crate = "crate")]
    struct HeadFixtureSumchecks<F: JoltField> {
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

    /// `begin_batch` with a clear recorder absorbs every member's
    /// `input_claim` as public values in declaration order, then draws one
    /// uniform batching coefficient per member, and packs the prelude from
    /// exactly those values.
    #[test]
    fn begin_batch_absorbs_input_claims_then_draws_coefficients() {
        let sumchecks = fixture(true);
        let inputs = HeadFixtureInputClaims::<Fr> {
            instruction_read_raf: instruction_inputs(),
            registers_val_evaluation: Some(RegistersValEvaluationInputClaims {
                registers_val: Fr::from_u64(7),
            }),
        };
        let challenges = sumchecks.draw_challenges(&mut fresh()).unwrap();

        // `input_claim` is transcript-pure, so the expected sums come from the
        // members directly.
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
        let (head, (instruction_coeff, registers_coeff)) = assert_same_draws(
            |t| {
                let mut recorder = ClearSumcheckRecorder::<Fr>::new();
                sumchecks.begin_batch(&inputs, &challenges, &mut recorder, t)
            },
            |t| {
                t.public_all(&[instruction_sum, registers_sum]);
                (t.challenge::<Fr>(), t.challenge::<Fr>())
            },
        );
        let (batch, coefficients) = head.unwrap();

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

    /// An absent `Option` member contributes no absorb, no coefficient draw,
    /// and no batch entry.
    #[test]
    fn begin_batch_skips_absent_option_member() {
        let sumchecks = fixture(false);
        let inputs = HeadFixtureInputClaims::<Fr> {
            instruction_read_raf: instruction_inputs(),
            registers_val_evaluation: None,
        };
        let challenges = sumchecks.draw_challenges(&mut fresh()).unwrap();
        let instruction_sum = sumchecks
            .instruction_read_raf
            .input_claim(
                &inputs.instruction_read_raf,
                &challenges.instruction_read_raf,
            )
            .unwrap();

        let (head, instruction_coeff) = assert_same_draws(
            |t| {
                let mut recorder = ClearSumcheckRecorder::<Fr>::new();
                sumchecks.begin_batch(&inputs, &challenges, &mut recorder, t)
            },
            |t| {
                t.public_all(&[instruction_sum]);
                t.challenge::<Fr>()
            },
        );
        let (batch, coefficients) = head.unwrap();

        assert_eq!(batch.members.len(), 1);
        assert_eq!(batch.max_num_vars, sumchecks.instruction_read_raf.rounds());
        assert_eq!(coefficients.instruction_read_raf, instruction_coeff);
        assert_eq!(coefficients.registers_val_evaluation, None);
    }
}
