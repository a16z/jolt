//! Shared per-relation opening-claim plumbing.
//!
//! The claim data model (the opening cells, the `OutputClaims`/`InputClaims`
//! resolvers, and the value↔point zip) lives in `jolt-claims` and is re-exported
//! here so existing `crate::stages::relations::{..}` paths keep resolving. Those
//! traits are implemented by `#[derive(OutputClaims)]` / `#[derive(InputClaims)]`
//! (crate `jolt-claims-derive`) on each relation's claim struct, making the
//! canonical opening **order** and **count** a single-sourced consequence of a
//! struct's field declaration order.
//!
//! Transcript I/O stays here: [`OutputAppend::append_openings`] is a thin
//! verifier-side consumer of [`OutputClaims::opening_values`], so `jolt-claims`
//! stays transcript-free while the Fiat-Shamir order remains single-sourced.

pub use jolt_claims::{
    zip_openings, GetPoint, GetValue, InputClaims, OpeningClaim, OutputClaims, SumcheckChallenges,
    ZipOpenings,
};

use jolt_claims::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId, JoltSumcheckDomain,
    JoltSumcheckSpec,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::VerifierError;

/// Reject a relation's sumcheck spec that isn't a positive-degree
/// Boolean-hypercube sumcheck. Every stage that runs a compressed-Boolean batched
/// sumcheck applies this guard to its relation specs before trusting them; sharing
/// it keeps the two error conditions identical across stages.
pub fn check_relation_boolean_hypercube(
    stage: JoltRelationId,
    spec: &JoltSumcheckSpec,
) -> Result<(), VerifierError> {
    if spec.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage,
            degree: spec.degree,
        });
    }
    if !matches!(spec.domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage });
    }
    Ok(())
}

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

/// The consumed-claim struct of a [`ConcreteSumcheck`] instance, projected through
/// its symbolic relation's [`Inputs`](SymbolicSumcheck::Inputs) GAT at cell `C`.
pub type ConcreteSumcheckInputs<F, S, C> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Inputs<C>;
/// The produced-claim struct of a [`ConcreteSumcheck`] instance, projected through
/// its symbolic relation's [`Outputs`](SymbolicSumcheck::Outputs) GAT at cell `C`.
pub type ConcreteSumcheckOutputs<F, S, C> =
    <<S as ConcreteSumcheck<F>>::Symbolic as SymbolicSumcheck>::Outputs<C>;

/// A single sumcheck instance, driven identically by the prover (while producing
/// its proof) and the verifier (after checking it).
///
/// Each relation's consumed/produced claim structs are generic over a *cell*:
/// `OpeningClaim<F>` (point + value) on the clear path, `Vec<F>` (point only) on
/// the ZK path, and `F` (value only) for the serialized wire form. Methods that
/// need only points ([`derive_opening_points`](Self::derive_opening_points)) are
/// generic over any [`GetPoint`] cell and run in both modes; methods that read
/// values pin the `OpeningClaim<F>` cell and run only on the clear path. This
/// makes "a ZK opening carries no value" a compile-time fact.
pub trait ConcreteSumcheck<F: Field>
where
    ConcreteSumcheckInputs<F, Self, OpeningClaim<F>>: InputClaims<F>,
    ConcreteSumcheckOutputs<F, Self, OpeningClaim<F>>: OutputClaims<F>,
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

    /// The symbolic relation backing this instance.
    fn symbolic(&self) -> &Self::Symbolic;

    fn id(&self) -> JoltRelationId {
        Self::Symbolic::id()
    }

    /// The sumcheck spec (rounds, degree, domain), from the symbolic relation.
    fn spec(&self) -> JoltSumcheckSpec {
        self.symbolic().spec()
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

    /// Map this instance's sumcheck point and the upstream input points into the
    /// produced openings' points. Value-independent, so it runs in both the clear
    /// and ZK paths; any cross-input consistency required for a well-defined point
    /// (e.g. address agreement) is checked here.
    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        inputs: &ConcreteSumcheckInputs<F, Self, C>,
    ) -> Result<ConcreteSumcheckOutputs<F, Self, Vec<F>>, VerifierError>;

    /// Resolve a `Derived` in this relation's **input** expression: from the input
    /// points and the drawn challenges. The input claim is the claimed sum *before*
    /// binding, so no produced openings and no bound point are available here.
    /// Defaults to "no input deriveds"; overridden by relations that have them
    /// (e.g. `RamValCheck`'s `InitEval`/`InitSelector`).
    fn derive_input_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &ConcreteSumcheckInputs<F, Self, C>,
        _challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: *id })
    }

    /// Resolve a `Derived` in this relation's **output** expression: from the input
    /// points, the produced openings' points (the bound point, post-binding), and the
    /// drawn challenges. The output claim is checked *after* binding, so the produced
    /// openings exist — hence `outputs` is non-optional. Most `eq`/`lt` deriveds live
    /// here (they evaluate at this sumcheck's bound point). Defaults to "no output
    /// deriveds"; overridden by relations that have them.
    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &ConcreteSumcheckInputs<F, Self, C>,
        _outputs: &ConcreteSumcheckOutputs<F, Self, OpeningClaim<F>>,
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
        inputs: &ConcreteSumcheckInputs<F, Self, OpeningClaim<F>>,
        challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        self.symbolic().input_expression::<F>().try_evaluate(
            |id| {
                inputs
                    .resolve_input(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: *id })
            },
            |id| self.derive_input_term(id, inputs, challenges),
        )
    }

    /// The expected output claim, evaluated from the output `Expr` against the
    /// produced opening values, the drawn `challenges`, and the relation's derived
    /// public values. The input points feed those derivations but the input
    /// *values* are not needed, so the inputs are taken over any [`GetPoint`] cell.
    /// Shared by prover and verifier; clear only.
    fn expected_output<C: GetPoint<F>>(
        &self,
        inputs: &ConcreteSumcheckInputs<F, Self, C>,
        outputs: &ConcreteSumcheckOutputs<F, Self, OpeningClaim<F>>,
        challenges: &ConcreteSumcheckChallenges<F, Self>,
    ) -> Result<F, VerifierError> {
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| {
                outputs
                    .resolve_output(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: *id })
            },
            |id| self.derive_output_term(id, inputs, outputs, challenges),
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    };
    use jolt_claims_derive::{InputClaims, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_riscv::{CircuitFlags, CIRCUIT_FLAGS};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn virt(polynomial: JoltVirtualPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::virtual_polynomial(polynomial, relation)
    }

    fn committed(polynomial: JoltCommittedPolynomial, relation: JoltRelationId) -> JoltOpeningId {
        JoltOpeningId::committed(polynomial, relation)
    }

    /// A minimal `Transcript` double that records each appended byte chunk, so
    /// that append order can be compared without depending on the digest.
    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
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
    fn zip_openings_pairs_values_with_points() {
        // `Vec` families zip element-wise; scalar leaves take their single point.
        let values = InstructionLeaf {
            lookup_table_flags: vec![fr(1), fr(2)],
            instruction_ra: vec![fr(3)],
            instruction_raf_flag: fr(4),
        };
        let points = InstructionLeaf {
            lookup_table_flags: vec![vec![fr(10)], vec![fr(11)]],
            instruction_ra: vec![vec![fr(12), fr(13)]],
            instruction_raf_flag: vec![fr(14)],
        };
        let zipped: InstructionLeaf<OpeningClaim<Fr>> = zip_openings(&values, &points);
        assert_eq!(
            zipped.lookup_table_flags,
            vec![
                OpeningClaim {
                    point: vec![fr(10)],
                    value: fr(1),
                },
                OpeningClaim {
                    point: vec![fr(11)],
                    value: fr(2),
                },
            ],
        );
        assert_eq!(
            zipped.instruction_ra,
            vec![OpeningClaim {
                point: vec![fr(12), fr(13)],
                value: fr(3),
            }],
        );
        assert_eq!(
            zipped.instruction_raf_flag,
            OpeningClaim {
                point: vec![fr(14)],
                value: fr(4),
            },
        );
    }

    #[test]
    fn zip_openings_follows_option_presence() {
        // A `Some` value pairs with its point; a `None` value stays `None`.
        let present: OptionalOutput<OpeningClaim<Fr>> = zip_openings(
            &OptionalOutput {
                untrusted: Some(fr(5)),
                ram_inc: fr(6),
            },
            &OptionalOutput {
                untrusted: Some(vec![fr(7)]),
                ram_inc: vec![fr(8)],
            },
        );
        assert_eq!(
            present.untrusted,
            Some(OpeningClaim {
                point: vec![fr(7)],
                value: fr(5),
            })
        );
        assert_eq!(
            present.ram_inc,
            OpeningClaim {
                point: vec![fr(8)],
                value: fr(6),
            }
        );

        let absent: OptionalOutput<OpeningClaim<Fr>> = zip_openings(
            &OptionalOutput {
                untrusted: None,
                ram_inc: fr(6),
            },
            &OptionalOutput {
                untrusted: None,
                ram_inc: vec![fr(8)],
            },
        );
        assert_eq!(absent.untrusted, None);
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

    // A `Vec` field whose payload annotation is an *array* indexes the family by
    // enum element: `op_flags[i]` is `OpFlags(CIRCUIT_FLAGS[i])`. Exercised in
    // both derives.
    #[derive(InputClaims)]
    struct EnumIndexedInputs<C> {
        #[opening(OpFlags(CIRCUIT_FLAGS), from = SpartanOuter)]
        op_flags: Vec<C>,
    }

    #[derive(OutputClaims)]
    #[relation(SpartanOuter)]
    struct EnumIndexedOutputs<C> {
        #[opening(OpFlags(CIRCUIT_FLAGS))]
        op_flags: Vec<C>,
    }

    #[test]
    fn enum_indexed_vec_resolves_by_flag() {
        let relation = JoltRelationId::SpartanOuter;
        let values: Vec<Fr> = (0..CIRCUIT_FLAGS.len()).map(|i| fr(i as u64)).collect();

        let inputs = EnumIndexedInputs {
            op_flags: values.clone(),
        };
        let outputs = EnumIndexedOutputs {
            op_flags: values.clone(),
        };
        for (i, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
            let id = virt(JoltVirtualPolynomial::OpFlags(flag), relation);
            assert_eq!(inputs.resolve_input(&id), Some(fr(i as u64)));
            assert_eq!(outputs.resolve_output(&id), Some(fr(i as u64)));
        }

        // Same flag, different relation misses; the encoders follow declaration order.
        assert_eq!(
            inputs.resolve_input(&virt(
                JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
                JoltRelationId::SpartanShift,
            )),
            None,
        );
        assert_eq!(outputs.opening_values().len(), CIRCUIT_FLAGS.len());
        assert_eq!(outputs.opening_values(), values);
        assert_append_matches_values(&outputs);
    }
}
