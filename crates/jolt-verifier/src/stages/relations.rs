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
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
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

    /// This relation's dual-role openings, as `(wire output, consumed input)` id
    /// pairs: each output cell's value equals the member's own consumed `input`
    /// claim — the same evaluation, staged on this relation's wire claims for
    /// downstream consumers — and is NOT a leaf of the output `Expr` (the
    /// relation's fold never reads it, so no kernel has a table for it).
    /// `RamValCheck`'s staged advice / program-image `Val_init` contributions are
    /// the concrete case: consumed by its input `Expr`'s init decomposition and
    /// re-staged as stage-4 wire openings.
    ///
    /// The generic kernel layer resolves the pairs mechanically: the naive
    /// prover snapshots each declared `input` value off the member's consumed
    /// claims at construction and attaches it to the `output` cell during typed
    /// extraction, so no hand-written kernel re-states the relation's wire
    /// structure. An input cell that is absent (an `Option` claim whose
    /// commitment is not present) attaches nothing — presence follows the
    /// consumed claim.
    ///
    /// Like [`aliased_output_openings`](Self::aliased_output_openings), this is
    /// an associated function: the pairing is a constant of the relation,
    /// consumable without constructing one (presence is resolved against the
    /// instance's claims at attach time).
    fn input_carried_outputs() -> Vec<(JoltOpeningId, JoltOpeningId)>
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
