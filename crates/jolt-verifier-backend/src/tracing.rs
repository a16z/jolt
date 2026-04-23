//! AST-recording [`FieldBackend`] + [`CommitmentBackend`] for symbolic
//! execution of the verifier.
//!
//! `Tracing` runs the verifier through the backend traits while doing no
//! field arithmetic and no PCS verification at all. Every wrap, every
//! constant, every arithmetic op, every assertion, every Fiat-Shamir
//! transcript op, and every PCS commitment / opening check becomes a node
//! (or an assertion) in an [`AstGraph`]. The resulting graph is the
//! verifier's *symbolic execution trace*: a side-effect-free, deterministic
//! record of every operation the verifier would have performed on the
//! supplied (still abstract) inputs.
//!
//! ## Downstream consumers
//!
//! An explicit [`AstGraph`] keeps three downstream consumers plug-in
//! compatible:
//!
//! 1. **Recursion / SNARK composition.** Lower the AST to a circuit (R1CS,
//!    AIR, Plonkish, …) inside an outer proof system. The lowering pass owns
//!    the encoding decisions; the verifier source code does not.
//! 2. **Theorem prover export (Lean/Coq).** Walk the AST and emit
//!    proof-friendly definitions; see `jolt-ir`'s Lean emitter for the
//!    pattern.
//! 3. **Inspection / fuzzing / golden tests.** The graph is `Debug + Clone`
//!    so test code can examine it, snapshot it, or replay a smaller instance
//!    through [`Native`](crate::Native) for differential checks.
//!
//! ## Generic over `PCS`
//!
//! [`AstGraph<PCS>`], [`AstOp<PCS>`], and [`Tracing<PCS>`] are parameterised
//! by the [`CommitmentScheme`](jolt_openings::CommitmentScheme). Commitment
//! values ([`PCS::Output`](jolt_openings::CommitmentScheme::Output)) and
//! opening proofs ([`PCS::Proof`](jolt_openings::CommitmentScheme::Proof))
//! are statically typed inhabitants of the AST: a Lean / R1CS lowering of
//! `AstGraph<DoryScheme>` reads off `DoryProof`s directly, with the
//! soundness obligation per [`AstOp::OpeningCheck`] node being
//! "`DoryScheme.verify` accepts these inputs" with no marshalling.
//!
//! Field-only tracing (no commitment ops) still picks a concrete `PCS` so
//! the AST is type-honest about which scheme it would use if a commitment
//! showed up. The standard choice is
//! [`MockCommitmentScheme<F>`](jolt_openings::mock::MockCommitmentScheme):
//! its `Field = F`, so all field-side methods take `F` directly and
//! [`replay`] takes `&()` for the verifier setup.
//!
//! ## Transcript operations
//!
//! The Fiat-Shamir transcript is a stateful sponge: it absorbs bytes (proof
//! data, public inputs, labels, commitment serialisations) and squeezes
//! field-valued challenges. To make the AST a *complete* record of the
//! verifier's observable behaviour, transcript ops are themselves AST nodes:
//!
//! - [`AstOp::TranscriptInit`] starts a new transcript with a domain label.
//! - [`AstOp::TranscriptAbsorbBytes`] threads in a chunk of bytes (the
//!   verifier's labeled-domain encodings, field-element absorbs — anything
//!   the underlying `Blake2bTranscript` would feed to its hash that isn't
//!   covered by a commitment-shaped node).
//! - [`AstOp::TranscriptAbsorbCommitment`] is the structured-absorb variant
//!   for PCS commitments: one node per logical absorb tying back to the
//!   originating [`AstOp::CommitmentWrap`]. The inner `Blake2bTranscript`
//!   is still driven through the standard `LabelWithCount +
//!   AppendToTranscript` byte sequence so squeezed challenges replay
//!   bit-identically against the [`Native`](crate::Native) backend.
//! - [`AstOp::TranscriptChallengeState`] advances the sponge by one squeeze
//!   and represents the post-state.
//! - [`AstOp::TranscriptChallengeValue`] is the squeezed field-element value;
//!   subsequent arithmetic ops reference this node.
//!
//! Splitting "post-state" and "value" into two nodes keeps the AST a pure
//! DAG: state nodes thread through `TranscriptInit` /
//! `TranscriptAbsorbBytes` / `TranscriptAbsorbCommitment` / `OpeningCheck`
//! edges, value nodes thread through arithmetic edges. Downstream consumers
//! can keep them in two separate variable spaces (state hashes vs.
//! field-element witnesses) when lowering to a recursive verifier.
//!
//! ## Provenance
//!
//! Every wrapped scalar carries its [`ScalarOrigin`] and a static label, and
//! every wrapped commitment carries its [`CommitmentOrigin`] and a static
//! label, so downstream consumers can:
//!
//! - mark public-input rows separately from witness rows (R1CS lowering),
//! - render the graph with human-readable variable names (Lean / debug),
//! - audit which proof fields, commitments, and challenges actually flow
//!   into the final assertions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jolt_field::Field;
use jolt_openings::{
    BackendError, CommitmentBackend, CommitmentOrigin, CommitmentScheme, FieldBackend,
    OpeningsError, ScalarOrigin,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use num_traits::Zero;

/// Stable identifier into [`AstGraph::nodes`].
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AstNodeId(pub u32);

/// A single recorded operation.
///
/// Operands are referenced by [`AstNodeId`] so the graph is a DAG even if the
/// verifier reuses a scalar across many ops. Transcript ops (including
/// commitment absorbs and opening checks) thread their `prev_state`
/// through the same id space so a transcript history reads as a chain of
/// state nodes.
///
/// ## `PCS`-generic variants
///
/// Three variants ([`AstOp::CommitmentWrap`],
/// [`AstOp::TranscriptAbsorbCommitment`], [`AstOp::OpeningCheck`]) inline
/// inhabitants of [`PCS::Output`](jolt_openings::CommitmentScheme::Output)
/// and [`PCS::Proof`](jolt_openings::CommitmentScheme::Proof). The values
/// are boxed to keep `size_of::<AstOp<PCS>>()` independent of the
/// underlying PCS proof size.
#[derive(Clone, Debug)]
pub enum AstOp<PCS: CommitmentScheme>
where
    PCS::Output: AppendToTranscript,
{
    /// Wrapped input scalar — provenance plus a static label.
    Wrap {
        /// Where the scalar came from (public, proof, transcript).
        origin: ScalarOrigin,
        /// Caller-supplied label, propagated for inspection.
        label: &'static str,
    },
    /// Integer constant baked into the verifier's code path.
    Constant(i128),
    /// Negation `-a`.
    Neg(AstNodeId),
    /// Addition `a + b`.
    Add(AstNodeId, AstNodeId),
    /// Subtraction `a - b`.
    Sub(AstNodeId, AstNodeId),
    /// Multiplication `a * b`.
    Mul(AstNodeId, AstNodeId),
    /// Squaring `a * a`. Recorded distinctly so backends that emit specialized
    /// `square` can recognize the intent.
    Square(AstNodeId),
    /// Multiplicative inverse `a^{-1}`.
    Inverse {
        /// Operand to invert.
        operand: AstNodeId,
        /// Caller-supplied debug context.
        ctx: &'static str,
    },
    /// Initial transcript state, seeded with a domain-separation label.
    ///
    /// Produces a *state* node id (consumed only by other transcript ops,
    /// never by arithmetic).
    TranscriptInit {
        /// Domain-separation label passed to `Blake2bTranscript::new`.
        label: &'static [u8],
    },
    /// Absorbs a byte string into the transcript.
    ///
    /// `prev_state` is the state node id before the absorb; the new node id
    /// is the post-absorb state. The verifier's higher-level encodings
    /// (`Label`, `LabelWithCount`, field-element `to_bytes`) all decompose
    /// into one or more `TranscriptAbsorbBytes` nodes carrying the exact
    /// byte buffer that `Blake2bTranscript::append_bytes` consumed.
    /// Commitment absorbs use the structured
    /// [`AstOp::TranscriptAbsorbCommitment`] variant instead.
    TranscriptAbsorbBytes {
        /// Previous transcript state node.
        prev_state: AstNodeId,
        /// Bytes fed to `Blake2bTranscript::append_bytes`.
        bytes: Vec<u8>,
    },
    /// Squeeze: post-state side of `Blake2bTranscript::challenge`.
    ///
    /// Produces a *state* node id that subsequent absorbs and squeezes
    /// reference. Paired with a [`AstOp::TranscriptChallengeValue`] that
    /// refers back to this node and represents the field-element output.
    TranscriptChallengeState {
        /// Pre-squeeze transcript state node.
        prev_state: AstNodeId,
    },
    /// Squeeze: field-element value side of `Blake2bTranscript::challenge`.
    ///
    /// `state` points to the [`AstOp::TranscriptChallengeState`] that carries
    /// the corresponding state advance. Arithmetic ops reference *this* node,
    /// transcript-state plumbing references the state node.
    TranscriptChallengeValue {
        /// Companion state node produced in the same squeeze.
        state: AstNodeId,
    },
    /// Wrapped input commitment — inlines a
    /// [`PCS::Output`](jolt_openings::CommitmentScheme::Output) plus
    /// provenance and a static label.
    ///
    /// Boxed so `size_of::<AstOp<PCS>>()` does not balloon for schemes
    /// with large commitments.
    CommitmentWrap {
        /// The commitment value, statically typed by the AST's `PCS`.
        value: Box<PCS::Output>,
        /// Where the commitment came from (vk-pinned vs. proof).
        origin: CommitmentOrigin,
        /// Caller-supplied label, propagated for inspection.
        label: &'static str,
    },
    /// Absorbs a commitment into the transcript via the standard
    /// `LabelWithCount + AppendToTranscript` two-step.
    ///
    /// One structured node per logical absorb so downstream consumers
    /// see "absorb commitment" as a single step. The inner
    /// `Blake2bTranscript` is driven through the matching byte sequence
    /// so squeezed challenges remain bit-identical across backends.
    ///
    /// `prev_state` is the transcript state node before the absorb; this
    /// node id is the post-absorb state. `commitment` references the
    /// originating [`AstOp::CommitmentWrap`] node.
    TranscriptAbsorbCommitment {
        /// Previous transcript state node.
        prev_state: AstNodeId,
        /// Originating [`AstOp::CommitmentWrap`] node id.
        commitment: AstNodeId,
        /// Domain label fed to `LabelWithCount`.
        label: &'static [u8],
    },
    /// Records a PCS opening verification: `commitment` opens to `claim`
    /// at multilinear evaluation point `point`, with `proof` (boxed
    /// inline) supplying the [`PCS::Proof`](jolt_openings::CommitmentScheme::Proof).
    ///
    /// `prev_state` is the transcript state going into the verify call;
    /// this node id is the post-verify transcript state (since
    /// `<PCS as CommitmentScheme>::verify` may absorb / squeeze).
    ///
    /// The corresponding [`AstAssertion::OpeningHolds`] obligation
    /// references this node and fires at replay time, invoking
    /// `<PCS as CommitmentScheme>::verify` directly.
    OpeningCheck {
        /// Pre-verify transcript state node.
        prev_state: AstNodeId,
        /// Originating [`AstOp::CommitmentWrap`] node id.
        commitment: AstNodeId,
        /// Scalar nodes describing the multilinear evaluation point. Each
        /// entry references an arithmetic node (typically a transcript
        /// challenge or a constant).
        point: Vec<AstNodeId>,
        /// Scalar node carrying the claimed evaluation value.
        claim: AstNodeId,
        /// Opening proof, statically typed by the AST's `PCS`.
        proof: Box<PCS::Proof>,
    },
}

/// A recorded verifier obligation.
///
/// Assertions are kept on the side from [`AstOp`] so a graph can carry
/// both the *value DAG* and the *constraint set*. R1CSGen-style consumers
/// lower each variant into the appropriate constraint shape; native
/// replays check each one against concrete witness values.
///
/// Variants:
/// - [`AstAssertion::Equality`] is the bread-and-butter `lhs == rhs`
///   assertion produced by [`FieldBackend::assert_eq`].
/// - [`AstAssertion::OpeningHolds`] discharges a PCS opening obligation
///   recorded as an [`AstOp::OpeningCheck`] node. Recursion lowerings
///   replace it with the in-circuit verifier for the named scheme; native
///   replay invokes the PCS's `verify` directly with the supplied vk.
#[derive(Clone, Debug)]
pub enum AstAssertion {
    /// Equality constraint `lhs == rhs`.
    Equality {
        /// Left-hand side of `lhs == rhs`.
        lhs: AstNodeId,
        /// Right-hand side of `lhs == rhs`.
        rhs: AstNodeId,
        /// Caller-supplied debug context.
        ctx: &'static str,
    },
    /// PCS opening obligation: the named [`AstOp::OpeningCheck`] node
    /// must verify successfully for the AST's static `PCS` type.
    OpeningHolds {
        /// `OpeningCheck` node carrying the commitment / point / claim /
        /// proof bundle.
        check: AstNodeId,
        /// Caller-supplied debug context.
        ctx: &'static str,
    },
}

impl AstAssertion {
    /// Returns the caller-supplied debug context, regardless of variant.
    pub fn ctx(&self) -> &'static str {
        match self {
            AstAssertion::Equality { ctx, .. } | AstAssertion::OpeningHolds { ctx, .. } => ctx,
        }
    }
}

/// Symbolic execution trace produced by [`Tracing`].
///
/// `nodes` is append-only — the index into the vector is the [`AstNodeId`].
/// `assertions` accumulate in the order the verifier issued them.
#[derive(Clone, Debug)]
pub struct AstGraph<PCS: CommitmentScheme>
where
    PCS::Output: AppendToTranscript,
{
    /// All recorded value-producing operations.
    pub nodes: Vec<AstOp<PCS>>,
    /// All `assert_eq` and `OpeningHolds` calls, recorded in issue order.
    pub assertions: Vec<AstAssertion>,
}

impl<PCS: CommitmentScheme> Default for AstGraph<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<PCS: CommitmentScheme> AstGraph<PCS>
where
    PCS::Output: AppendToTranscript,
{
    /// Returns an empty graph (no nodes, no assertions).
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            assertions: Vec::new(),
        }
    }

    /// Returns `nodes.len()` — handy for snapshot tests.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `assertions.len()`.
    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }

    /// Pushes a node and returns its assigned id.
    fn push(&mut self, op: AstOp<PCS>) -> AstNodeId {
        let id = AstNodeId(self.nodes.len() as u32);
        self.nodes.push(op);
        id
    }
}

/// Per-scalar handle returned by [`Tracing`] methods.
///
/// `Copy` because it is just a node id plus a phantom marker; the actual
/// graph state lives on the [`Tracing`] backend.
#[derive(Copy, Clone, Debug)]
pub struct AstScalar<F: jolt_field::Field> {
    /// Position in the owning graph.
    pub id: AstNodeId,
    _marker: std::marker::PhantomData<F>,
}

impl<F: jolt_field::Field> AstScalar<F> {
    pub(crate) fn new(id: AstNodeId) -> Self {
        Self {
            id,
            _marker: std::marker::PhantomData,
        }
    }
}

/// AST-recording backend, generic over a [`CommitmentScheme`].
///
/// `Tracing` itself is just a handle to the shared graph — clone it freely
/// and every clone records into the same DAG. Transcripts produced by
/// [`Tracing::new_transcript`] also record into this same graph, so
/// arithmetic, transcript history, and commitment checks all sit in the
/// same node-id space.
///
/// Alongside the [`AstGraph`], `Tracing` keeps the concrete wrap *scalar*
/// values it received (proof / public field elements that flow into the
/// arithmetic). Commitment values and opening proofs live inline on
/// [`AstOp::CommitmentWrap::value`] and [`AstOp::OpeningCheck::proof`];
/// they are not duplicated here. Graph consumers (Lean export, R1CS lower)
/// ignore the scalar value list; differential testing against
/// [`Native`](crate::Native) uses it via [`replay`] without the caller
/// having to reconstruct the wrap sequence.
///
/// Internally the graph lives behind `Arc<Mutex<…>>` (rather than
/// `Rc<RefCell<…>>`) because [`Transcript`] requires `Sync + Send + 'static`
/// and the [`TracingTranscript`] shares the same handle.
#[derive(Clone, Debug)]
pub struct Tracing<PCS: CommitmentScheme>
where
    PCS::Output: AppendToTranscript,
{
    graph: Arc<Mutex<AstGraph<PCS>>>,
    wrap_values: Arc<Mutex<Vec<PCS::Field>>>,
}

impl<PCS: CommitmentScheme> Default for Tracing<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<PCS: CommitmentScheme> Tracing<PCS>
where
    PCS::Output: AppendToTranscript,
{
    /// Constructs a fresh `Tracing` backend with an empty graph.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Mutex::new(AstGraph::new())),
            wrap_values: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a deep clone of the recorded graph (immutable snapshot).
    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    pub fn snapshot(&self) -> AstGraph<PCS> {
        self.graph.lock().expect("AstGraph mutex poisoned").clone()
    }

    /// Returns a clone of the wrap *scalar* values in the order they were
    /// recorded. Commitment values and opening proofs are *not* part of
    /// this list — they live inline in the AST nodes themselves.
    ///
    /// Pair with [`replay`] to re-execute the trace against [`Native`] for
    /// differential testing. Production graph consumers (Lean, R1CS) should
    /// ignore this and walk [`AstGraph`] directly.
    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    pub fn wrap_values(&self) -> Vec<PCS::Field> {
        self.wrap_values
            .lock()
            .expect("wrap_values mutex poisoned")
            .clone()
    }

    /// Borrows the underlying graph for read-only inspection.
    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    pub fn with_graph<R>(&self, f: impl FnOnce(&AstGraph<PCS>) -> R) -> R {
        f(&self.graph.lock().expect("AstGraph mutex poisoned"))
    }

    /// Returns a fresh transcript that records into this graph.
    ///
    /// Provided as an inherent method so callers that already hold a
    /// [`Tracing`] handle can construct extra transcripts without going
    /// through [`FieldBackend::new_transcript`].
    pub fn new_transcript(&self, label: &'static [u8]) -> TracingTranscript<PCS> {
        TracingTranscript::with_graph(label, self.graph.clone())
    }
}

impl<PCS: CommitmentScheme> FieldBackend for Tracing<PCS>
where
    PCS::Output: AppendToTranscript,
{
    type F = PCS::Field;
    type Scalar = AstScalar<PCS::Field>;
    type Transcript = TracingTranscript<PCS>;

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn wrap(
        &mut self,
        value: PCS::Field,
        origin: ScalarOrigin,
        label: &'static str,
    ) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Wrap { origin, label });
        self.wrap_values
            .lock()
            .expect("wrap_values mutex poisoned")
            .push(value);
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn const_i128(&mut self, v: i128) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Constant(v));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn add(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Add(a.id, b.id));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn sub(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Sub(a.id, b.id));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn mul(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Mul(a.id, b.id));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn neg(&mut self, a: &Self::Scalar) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Neg(a.id));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn square(&mut self, a: &Self::Scalar) -> Self::Scalar {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Square(a.id));
        AstScalar::new(id)
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn inverse(
        &mut self,
        a: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<Self::Scalar, BackendError> {
        let id = self
            .graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .push(AstOp::Inverse { operand: a.id, ctx });
        Ok(AstScalar::new(id))
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn assert_eq(
        &mut self,
        a: &Self::Scalar,
        b: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<(), BackendError> {
        self.graph
            .lock()
            .expect("AstGraph mutex poisoned")
            .assertions
            .push(AstAssertion::Equality {
                lhs: a.id,
                rhs: b.id,
                ctx,
            });
        Ok(())
    }

    fn unwrap(&self, _scalar: &Self::Scalar) -> Option<PCS::Field> {
        // Tracing intentionally does not expose concrete values through the
        // backend interface — that would let downstream code branch on
        // witness data and leak symbolic faithfulness. Replay reads the
        // wrap-value list explicitly via [`Tracing::wrap_values`].
        None
    }

    fn new_transcript(&mut self, label: &'static [u8]) -> Self::Transcript {
        TracingTranscript::with_graph(label, self.graph.clone())
    }

    fn squeeze(
        &mut self,
        transcript: &mut Self::Transcript,
        _label: &'static str,
    ) -> (Self::F, Self::Scalar) {
        let value = transcript.challenge();
        let value_node = transcript
            .take_last_squeeze_value()
            .expect("TracingTranscript::challenge always pushes a value node");
        (value, AstScalar::new(value_node))
    }
}

/// AST-recording [`CommitmentBackend`] for [`Tracing`].
///
/// Every commitment operation pushes a structured node into the shared
/// [`AstGraph<PCS>`] *and* drives the inner `Blake2bTranscript` of the
/// supplied [`TracingTranscript`] for byte-level Fiat-Shamir parity with
/// [`Native`](crate::Native). `verify_opening` is intentionally *deferred*:
/// it records an [`AstOp::OpeningCheck`] node plus a corresponding
/// [`AstAssertion::OpeningHolds`] obligation, and always returns `Ok(())`.
/// The actual `<PCS as CommitmentScheme>::verify` invocation happens in
/// [`replay`] (or in a downstream lowering pass).
///
/// This separation is the load-bearing invariant of the trait: tracing must
/// not short-circuit on a bad witness, because the *graph itself* is the
/// artifact downstream consumers (Lean export, R1CS lowering) depend on
/// being well-formed. Replay then closes the loop for differential
/// testing against [`Native`].
impl<PCS: CommitmentScheme> CommitmentBackend<PCS> for Tracing<PCS>
where
    PCS::Output: AppendToTranscript,
{
    type Commitment = AstNodeId;

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        origin: CommitmentOrigin,
        label: &'static str,
    ) -> Self::Commitment {
        let mut g = self.graph.lock().expect("AstGraph mutex poisoned");
        g.push(AstOp::CommitmentWrap {
            value: Box::new(value),
            origin,
            label,
        })
    }

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn unwrap_commitment(&self, commitment: &Self::Commitment) -> PCS::Output {
        // Looks up the inlined PCS::Output on the originating CommitmentWrap
        // node. Used by per-PCS verify_batch_with_backend impls to feed
        // commitments into PCS::combine and rewrap the result. Panics if the
        // referenced node is not a CommitmentWrap (graph corruption).
        let g = self.graph.lock().expect("AstGraph mutex poisoned");
        match &g.nodes[commitment.0 as usize] {
            AstOp::CommitmentWrap { value, .. } => (**value).clone(),
            _ => panic!(
                "Tracing::unwrap_commitment: node #{} is not CommitmentWrap",
                commitment.0,
            ),
        }
    }

    fn absorb_commitment(
        &mut self,
        transcript: &mut Self::Transcript,
        commitment: &Self::Commitment,
        label: &'static [u8],
    ) {
        // Drive the inner Blake2bTranscript through the same byte sequence
        // Native produces (LabelWithCount + AppendToTranscript) so squeezed
        // challenges replay bit-identically. The per-byte AST nodes are
        // suppressed; the graph carries one structured
        // TranscriptAbsorbCommitment node tying the new state back to the
        // originating CommitmentWrap.
        let value = transcript.with_commitment_value(commitment, |value: &PCS::Output| {
            (value.serialized_len(), value.clone())
        });
        let (serialized_len, owned_value) = value;
        transcript.silent_append_bytes_for_label(label, serialized_len);
        transcript.silent_append_to_transcript(&owned_value);
        let prev_state = transcript.current_state_node();
        let new_state = transcript.push_node(AstOp::TranscriptAbsorbCommitment {
            prev_state,
            commitment: *commitment,
            label,
        });
        transcript.set_state_node(new_state);
    }

    fn verify_opening(
        &mut self,
        _vk: &PCS::VerifierSetup,
        commitment: &Self::Commitment,
        point: &[Self::Scalar],
        claim: &Self::Scalar,
        proof: &PCS::Proof,
        transcript: &mut Self::Transcript,
    ) -> Result<(), OpeningsError> {
        let prev_state = transcript.current_state_node();
        let point_ids: Vec<AstNodeId> = point.iter().map(|s| s.id).collect();
        let check_node = transcript.push_node(AstOp::OpeningCheck {
            prev_state,
            commitment: *commitment,
            point: point_ids,
            claim: claim.id,
            proof: Box::new(proof.clone()),
        });
        // The OpeningCheck node *is* the post-verify transcript state: the
        // replay path will run <PCS as CommitmentScheme>::verify against
        // the live transcript and that call may mutate the sponge.
        transcript.set_state_node(check_node);
        transcript.push_assertion(AstAssertion::OpeningHolds {
            check: check_node,
            ctx: "verify_opening",
        });
        Ok(())
    }
}

/// Transcript wrapper that records every absorb/squeeze into a shared
/// [`AstGraph`].
///
/// Internally holds a real [`Blake2bTranscript`] so all challenges are
/// produced by the actual Fiat-Shamir hash function. This keeps higher-level
/// PCS / sumcheck verification working transparently against this
/// transcript: every `transcript.append_bytes` and `transcript.challenge`
/// they perform also lands in the AST.
///
/// The shared graph handle is [`Arc<Mutex<AstGraph<PCS>>>`] so the type
/// satisfies [`Sync + Send + 'static`] (required by the [`Transcript`] trait).
pub struct TracingTranscript<PCS: CommitmentScheme>
where
    PCS::Output: AppendToTranscript,
{
    inner: Blake2bTranscript<PCS::Field>,
    graph: Arc<Mutex<AstGraph<PCS>>>,
    state_node: AstNodeId,
    last_squeeze_value: Option<AstNodeId>,
}

impl<PCS: CommitmentScheme> TracingTranscript<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn with_graph(label: &'static [u8], graph: Arc<Mutex<AstGraph<PCS>>>) -> Self {
        let state_node = {
            #[expect(
                clippy::expect_used,
                reason = "AstGraph mutex is internal; poisoning would itself be a bug"
            )]
            let mut g = graph.lock().expect("AstGraph mutex poisoned");
            g.push(AstOp::TranscriptInit { label })
        };
        Self {
            inner: Blake2bTranscript::<PCS::Field>::new(label),
            graph,
            state_node,
            last_squeeze_value: None,
        }
    }

    /// Returns the AST node id of the most recent squeezed challenge value
    /// and clears it. Used by [`FieldBackend::squeeze`] to bridge a transcript
    /// challenge into a backend [`AstScalar`].
    pub fn take_last_squeeze_value(&mut self) -> Option<AstNodeId> {
        self.last_squeeze_value.take()
    }

    /// Returns the current state node id.
    pub fn current_state_node(&self) -> AstNodeId {
        self.state_node
    }

    /// Drives the inner [`Blake2bTranscript`] through a [`LabelWithCount`]
    /// header *without* recording a [`AstOp::TranscriptAbsorbBytes`] node
    /// in the graph. Used by [`AstOp::TranscriptAbsorbCommitment`] paths
    /// that record a single structured node.
    pub(crate) fn silent_append_bytes_for_label(&mut self, label: &'static [u8], count: u64) {
        self.inner.append(&LabelWithCount(label, count));
    }

    /// Mirror of [`silent_append_bytes_for_label`] for [`AppendToTranscript`]
    /// payloads that carry their own framing (e.g. a commitment's
    /// canonical serialisation). Forwards to the inner Blake2b without
    /// recording any byte-level AST node.
    pub(crate) fn silent_append_to_transcript<A: AppendToTranscript>(&mut self, value: &A) {
        value.append_to_transcript(&mut self.inner);
    }

    /// Replaces the recorded state-node id without touching the inner
    /// Blake2b. Used by structured-absorb paths after they push their
    /// own state-producing node and need subsequent ops to thread from
    /// the new id.
    pub(crate) fn set_state_node(&mut self, node: AstNodeId) {
        self.state_node = node;
    }

    /// Pushes a node into the shared graph. Convenience for the
    /// `CommitmentBackend<PCS>` impl on [`Tracing`] so it does not need
    /// to expose the mutex through the public surface.
    pub(crate) fn push_node(&mut self, op: AstOp<PCS>) -> AstNodeId {
        #[expect(
            clippy::expect_used,
            reason = "AstGraph mutex is internal; poisoning would itself be a bug"
        )]
        let mut g = self.graph.lock().expect("AstGraph mutex poisoned");
        g.push(op)
    }

    /// Pushes an assertion into the shared graph. Companion to
    /// [`Self::push_node`].
    pub(crate) fn push_assertion(&mut self, assertion: AstAssertion) {
        #[expect(
            clippy::expect_used,
            reason = "AstGraph mutex is internal; poisoning would itself be a bug"
        )]
        let mut g = self.graph.lock().expect("AstGraph mutex poisoned");
        g.assertions.push(assertion);
    }

    /// Borrows the inlined `PCS::Output` value on a `CommitmentWrap` node
    /// referenced by the supplied [`AstNodeId`] and runs `f` against it
    /// while the graph mutex is held.
    ///
    /// # Panics
    ///
    /// Panics if `commitment` does not refer to an
    /// [`AstOp::CommitmentWrap`] node.
    pub(crate) fn with_commitment_value<R>(
        &self,
        commitment: &AstNodeId,
        f: impl FnOnce(&PCS::Output) -> R,
    ) -> R {
        #[expect(
            clippy::expect_used,
            reason = "AstGraph mutex is internal; poisoning would itself be a bug"
        )]
        let g = self.graph.lock().expect("AstGraph mutex poisoned");
        match &g.nodes[commitment.0 as usize] {
            AstOp::CommitmentWrap { value, .. } => f(value),
            // SAFETY (panic-message): we don't `Debug`-format the variant
            // because `AstOp<PCS>: Debug` would force a `PCS: Debug` bound
            // (the auto-derived `Debug` puts the bound on the type
            // parameter, not on the actually-used associated types).
            _ => panic!(
                "with_commitment_value: node #{} is not CommitmentWrap",
                commitment.0,
            ),
        }
    }
}

impl<PCS: CommitmentScheme> Clone for TracingTranscript<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            graph: self.graph.clone(),
            state_node: self.state_node,
            last_squeeze_value: self.last_squeeze_value,
        }
    }
}

impl<PCS: CommitmentScheme> Default for TracingTranscript<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn default() -> Self {
        // Default constructs an *isolated* graph (rather than panicking)
        // only to satisfy the [`Transcript`] trait bound. The primary
        // construction path is [`Tracing::new_transcript`], which shares
        // the backend's graph; downstream code must use that for graph
        // aggregation.
        Self::with_graph(
            b"jolt_tracing_default",
            Arc::new(Mutex::new(AstGraph::new())),
        )
    }
}

impl<PCS: CommitmentScheme> std::fmt::Debug for TracingTranscript<PCS>
where
    PCS::Output: AppendToTranscript,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracingTranscript")
            .field("state_node", &self.state_node)
            .field("last_squeeze_value", &self.last_squeeze_value)
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<PCS: CommitmentScheme> Transcript for TracingTranscript<PCS>
where
    PCS::Output: AppendToTranscript,
{
    type Challenge = PCS::Field;

    fn new(label: &'static [u8]) -> Self {
        Self::with_graph(label, Arc::new(Mutex::new(AstGraph::new())))
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
        let new_state = {
            #[expect(
                clippy::expect_used,
                reason = "AstGraph mutex is internal; poisoning would itself be a bug"
            )]
            let mut g = self.graph.lock().expect("AstGraph mutex poisoned");
            g.push(AstOp::TranscriptAbsorbBytes {
                prev_state: self.state_node,
                bytes: bytes.to_vec(),
            })
        };
        self.state_node = new_state;
    }

    fn challenge(&mut self) -> PCS::Field {
        let value = self.inner.challenge();
        let (state_id, value_id) = {
            #[expect(
                clippy::expect_used,
                reason = "AstGraph mutex is internal; poisoning would itself be a bug"
            )]
            let mut g = self.graph.lock().expect("AstGraph mutex poisoned");
            let state_id = g.push(AstOp::TranscriptChallengeState {
                prev_state: self.state_node,
            });
            let value_id = g.push(AstOp::TranscriptChallengeValue { state: state_id });
            (state_id, value_id)
        };
        self.state_node = state_id;
        self.last_squeeze_value = Some(value_id);
        value
    }

    fn state(&self) -> &[u8; 32] {
        self.inner.state()
    }
}

/// Replays a tracing graph against a concrete witness assignment.
///
/// `wrap_values` supplies a concrete `PCS::Field` for each [`AstOp::Wrap`]
/// node, in the order the wraps appeared in `graph.nodes`. Transcript ops
/// re-run a real [`Blake2bTranscript`] against the recorded byte stream
/// (including commitment absorbs), so squeezed challenges are derived
/// deterministically by Fiat-Shamir replay rather than supplied externally.
/// Opening checks invoke `<PCS as CommitmentScheme>::verify` against the
/// supplied verifier setup `vk`; failures surface as
/// [`BackendError::OpeningCheckFailed`].
///
/// The function returns the field values associated with every node (state
/// nodes are recorded as zero — they have no meaningful field semantics)
/// and confirms that every recorded assertion holds.
///
/// This is the bridge that lets a snapshot taken with [`Tracing`] be
/// validated against a regular [`Native`](crate::Native) execution: if the
/// graph is faithful, then for any consistent witness it must reproduce the
/// same value at every node and pass every assertion.
///
/// # Errors
///
/// - [`BackendError::AssertionFailed`] for a violated equality assertion.
/// - [`BackendError::InverseOfZero`] for an [`AstOp::Inverse`] of zero.
/// - [`BackendError::OpeningCheckFailed`] when `<PCS as
///   CommitmentScheme>::verify` rejects an [`AstOp::OpeningCheck`] that
///   carries an [`AstAssertion::OpeningHolds`] obligation.
///
/// # Panics
///
/// Panics if `wrap_values` does not contain exactly one entry per
/// [`AstOp::Wrap`] node, if the graph references a transcript-state
/// operand that has no live transcript (graph corruption), or if a
/// transcript challenge node references a non-state operand.
pub fn replay<PCS: CommitmentScheme>(
    graph: &AstGraph<PCS>,
    wrap_values: &[PCS::Field],
    vk: &PCS::VerifierSetup,
) -> Result<Vec<PCS::Field>, BackendError>
where
    PCS::Output: AppendToTranscript,
{
    let mut values: Vec<PCS::Field> = vec![PCS::Field::zero(); graph.nodes.len()];
    let mut transcripts: Vec<Option<Blake2bTranscript<PCS::Field>>> = vec![None; graph.nodes.len()];
    let mut opening_results: HashMap<AstNodeId, Result<(), OpeningsError>> = HashMap::new();
    let mut wrap_cursor = 0usize;

    for (idx, op) in graph.nodes.iter().enumerate() {
        match op {
            AstOp::Wrap { .. } => {
                assert!(
                    wrap_cursor < wrap_values.len(),
                    "replay: not enough wrap_values (need at least {})",
                    wrap_cursor + 1
                );
                values[idx] = wrap_values[wrap_cursor];
                wrap_cursor += 1;
            }
            AstOp::Constant(c) => values[idx] = PCS::Field::from_i128(*c),
            AstOp::Neg(a) => values[idx] = -values[a.0 as usize],
            AstOp::Add(a, b) => values[idx] = values[a.0 as usize] + values[b.0 as usize],
            AstOp::Sub(a, b) => values[idx] = values[a.0 as usize] - values[b.0 as usize],
            AstOp::Mul(a, b) => values[idx] = values[a.0 as usize] * values[b.0 as usize],
            AstOp::Square(a) => values[idx] = values[a.0 as usize].square(),
            AstOp::Inverse { operand, ctx } => {
                values[idx] = values[operand.0 as usize]
                    .inverse()
                    .ok_or(BackendError::InverseOfZero(ctx))?;
            }
            AstOp::TranscriptInit { label } => {
                transcripts[idx] = Some(Blake2bTranscript::<PCS::Field>::new(label));
            }
            AstOp::TranscriptAbsorbBytes { prev_state, bytes } => {
                let mut t = transcripts[prev_state.0 as usize]
                    .take()
                    .expect("replay: transcript state already consumed");
                t.append_bytes(bytes);
                transcripts[idx] = Some(t);
            }
            AstOp::TranscriptChallengeState { prev_state } => {
                let mut t = transcripts[prev_state.0 as usize]
                    .take()
                    .expect("replay: transcript state already consumed");
                let value = t.challenge();
                values[idx] = value;
                transcripts[idx] = Some(t);
            }
            AstOp::TranscriptChallengeValue { state } => {
                values[idx] = values[state.0 as usize];
            }
            AstOp::CommitmentWrap { .. } => {
                // No field-side semantics; commitment value is held inline.
                // Opening checks resolve `commitment` -> this node directly.
            }
            AstOp::TranscriptAbsorbCommitment {
                prev_state,
                commitment,
                label,
            } => {
                // Resolve the commitment value inlined on the originating
                // CommitmentWrap node.
                let value = match &graph.nodes[commitment.0 as usize] {
                    AstOp::CommitmentWrap { value, .. } => value,
                    _ => panic!(
                        "replay: TranscriptAbsorbCommitment refers to non-CommitmentWrap node #{}",
                        commitment.0
                    ),
                };
                let mut t = transcripts[prev_state.0 as usize]
                    .take()
                    .expect("replay: transcript state already consumed");
                t.append(&LabelWithCount(label, value.serialized_len()));
                value.append_to_transcript(&mut t);
                transcripts[idx] = Some(t);
            }
            AstOp::OpeningCheck {
                prev_state,
                commitment,
                point,
                claim,
                proof,
            } => {
                let commitment_value = match &graph.nodes[commitment.0 as usize] {
                    AstOp::CommitmentWrap { value, .. } => value,
                    _ => panic!(
                        "replay: OpeningCheck refers to non-CommitmentWrap node #{}",
                        commitment.0
                    ),
                };
                let point_values: Vec<PCS::Field> =
                    point.iter().map(|p| values[p.0 as usize]).collect();
                let claim_value = values[claim.0 as usize];
                let mut t = transcripts[prev_state.0 as usize]
                    .take()
                    .expect("replay: transcript state already consumed");
                let result = PCS::verify(
                    commitment_value,
                    &point_values,
                    claim_value,
                    proof,
                    vk,
                    &mut t,
                );
                let _ = opening_results.insert(AstNodeId(idx as u32), result);
                transcripts[idx] = Some(t);
            }
        }
    }
    assert_eq!(
        wrap_cursor,
        wrap_values.len(),
        "replay: too many wrap_values (consumed {wrap_cursor}, given {})",
        wrap_values.len()
    );

    for assertion in &graph.assertions {
        match assertion {
            AstAssertion::Equality { lhs, rhs, ctx } => {
                if values[lhs.0 as usize] != values[rhs.0 as usize] {
                    return Err(BackendError::AssertionFailed(ctx));
                }
            }
            AstAssertion::OpeningHolds { check, ctx } => match opening_results.get(check) {
                Some(Ok(())) => {}
                Some(Err(source)) => {
                    return Err(BackendError::OpeningCheckFailed {
                        ctx,
                        source: source.clone(),
                    });
                }
                None => panic!(
                    "replay: OpeningHolds references node #{} which is not an OpeningCheck",
                    check.0
                ),
            },
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use crate::helpers::eq_eval;
    use crate::native::Native;
    use jolt_openings::CommitmentBackend;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    /// Field-only tracing tests pick a concrete (trivial) PCS so the
    /// `AstGraph<PCS>` shape is type-honest. `MockCommitmentScheme<F>`
    /// has `Field = F` and `VerifierSetup = ()`, so field-side methods
    /// take `F` directly and `replay` takes `&()`.
    type Mock = MockCommitmentScheme<Fr>;

    #[test]
    fn records_basic_dag_shape() {
        let mut t = Tracing::<Mock>::new();
        let a = t.wrap_proof(Fr::from_u64(2), "a");
        let b = t.wrap_proof(Fr::from_u64(3), "b");
        let sum = t.add(&a, &b);
        let prod = t.mul(&a, &b);
        let _ = t.sub(&sum, &prod);

        let g = t.snapshot();
        assert_eq!(g.node_count(), 5);
        assert!(matches!(
            g.nodes[0],
            AstOp::Wrap {
                origin: ScalarOrigin::Proof,
                ..
            }
        ));
        assert!(matches!(g.nodes[2], AstOp::Add(..)));
        assert!(matches!(g.nodes[3], AstOp::Mul(..)));
        assert!(matches!(g.nodes[4], AstOp::Sub(..)));
        assert_eq!(g.assertion_count(), 0);
    }

    #[test]
    fn assertions_are_recorded_not_evaluated() {
        let mut t = Tracing::<Mock>::new();
        let a = t.wrap_proof(Fr::from_u64(2), "a");
        let three = t.const_i128(3);
        // Tracing must not panic on a "wrong" assert; it just records.
        t.assert_eq(&a, &three, "demo").unwrap();
        let g = t.snapshot();
        assert_eq!(g.assertion_count(), 1);
        assert_eq!(g.assertions[0].ctx(), "demo");
        assert!(matches!(g.assertions[0], AstAssertion::Equality { .. }));
    }

    /// `eq_eval` is a non-trivial helper that mixes wraps, constants, subs
    /// and muls. Tracing it and replaying should give the same value as the
    /// direct `eq` formula.
    #[test]
    fn tracing_eq_eval_then_replay_matches_native() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xeed);
        for n in 0..5 {
            let a_vals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let b_vals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

            let mut native = Native::<Fr>::new();
            let na: Vec<Fr> = a_vals.iter().map(|v| native.wrap_proof(*v, "a")).collect();
            let nb: Vec<Fr> = b_vals.iter().map(|v| native.wrap_proof(*v, "b")).collect();
            let native_value = eq_eval(&mut native, &na, &nb);

            let mut tracer = Tracing::<Mock>::new();
            let ta: Vec<_> = a_vals.iter().map(|v| tracer.wrap_proof(*v, "a")).collect();
            let tb: Vec<_> = b_vals.iter().map(|v| tracer.wrap_proof(*v, "b")).collect();
            let traced_handle = eq_eval(&mut tracer, &ta, &tb);

            let graph = tracer.snapshot();
            let wraps = tracer.wrap_values();
            let values = replay::<Mock>(&graph, &wraps, &()).unwrap();
            assert_eq!(values[traced_handle.id.0 as usize], native_value, "n = {n}");
        }
    }

    /// `assert_eq` records an entry; replay catches a witness that violates
    /// it.
    #[test]
    fn replay_reports_assertion_failure() {
        let mut tracer = Tracing::<Mock>::new();
        let a = tracer.wrap_proof(Fr::from_u64(0), "a");
        let b = tracer.wrap_proof(Fr::from_u64(0), "b");
        tracer.assert_eq(&a, &b, "ab").unwrap();
        let graph = tracer.snapshot();

        // Witness consistent with the trace: replay succeeds.
        let _ = replay::<Mock>(&graph, &[Fr::from_u64(7), Fr::from_u64(7)], &()).unwrap();

        // Inconsistent witness: replay fails.
        let err = replay::<Mock>(&graph, &[Fr::from_u64(7), Fr::from_u64(8)], &()).unwrap_err();
        assert!(matches!(err, BackendError::AssertionFailed("ab")));
    }

    /// A short transcript script (init → absorb → squeeze → absorb → squeeze)
    /// is captured as discrete AST nodes, and replay reproduces the same
    /// challenge values as a freshly-driven [`Blake2bTranscript`].
    #[test]
    fn transcript_ops_round_trip_through_replay() {
        let tracer = Tracing::<Mock>::new();
        let mut transcript = tracer.new_transcript(b"jolt_test");
        transcript.append_bytes(b"hello");
        let c1 = transcript.challenge();
        transcript.append_bytes(b"world");
        let c2 = transcript.challenge();

        // Capture node ids of the squeezed values so we can index after replay.
        // Node layout (in order): TranscriptInit, AbsorbBytes("hello"),
        // ChallengeState, ChallengeValue(c1), AbsorbBytes("world"),
        // ChallengeState, ChallengeValue(c2).
        let g = tracer.snapshot();
        assert_eq!(g.node_count(), 7);
        assert!(matches!(g.nodes[0], AstOp::TranscriptInit { .. }));
        assert!(matches!(g.nodes[1], AstOp::TranscriptAbsorbBytes { .. }));
        assert!(matches!(g.nodes[2], AstOp::TranscriptChallengeState { .. }));
        assert!(matches!(g.nodes[3], AstOp::TranscriptChallengeValue { .. }));
        assert!(matches!(g.nodes[4], AstOp::TranscriptAbsorbBytes { .. }));
        assert!(matches!(g.nodes[5], AstOp::TranscriptChallengeState { .. }));
        assert!(matches!(g.nodes[6], AstOp::TranscriptChallengeValue { .. }));

        // Replay re-derives the challenges by simulating the transcript.
        let values = replay::<Mock>(&g, &tracer.wrap_values(), &()).unwrap();
        assert_eq!(values[3], c1, "first challenge");
        assert_eq!(values[6], c2, "second challenge");

        // Ground truth: a clean Blake2bTranscript produces identical challenges.
        let mut reference = Blake2bTranscript::<Fr>::new(b"jolt_test");
        reference.append_bytes(b"hello");
        let ref1: Fr = reference.challenge();
        reference.append_bytes(b"world");
        let ref2: Fr = reference.challenge();
        assert_eq!(c1, ref1);
        assert_eq!(c2, ref2);
    }

    /// Squeezes routed through `FieldBackend::squeeze` produce the same
    /// challenge value as the underlying transcript, *and* the returned
    /// `AstScalar` resolves to that same value on replay.
    #[test]
    fn backend_squeeze_links_transcript_value_into_arithmetic() {
        let mut tracer = Tracing::<Mock>::new();
        let mut transcript = tracer.new_transcript(b"squeeze_test");
        let (challenge_f, challenge_w) = tracer.squeeze(&mut transcript, "alpha");

        // Use the squeezed challenge in a tiny arithmetic op to confirm the
        // backend links the value node into the field DAG.
        let two = tracer.const_i128(2);
        let doubled = tracer.mul(&challenge_w, &two);

        let g = tracer.snapshot();
        let values = replay::<Mock>(&g, &tracer.wrap_values(), &()).unwrap();
        assert_eq!(values[challenge_w.id.0 as usize], challenge_f);
        assert_eq!(values[doubled.id.0 as usize], challenge_f + challenge_f);
    }

    /// End-to-end commitment-shaped tracing: wrap a commitment, absorb it,
    /// run an opening check via the [`CommitmentBackend`] surface, and
    /// confirm that replay invokes `<MockCommitmentScheme as
    /// CommitmentScheme>::verify` correctly (both happy path and tampered
    /// claim paths).
    #[test]
    fn tracing_commitment_round_trip_through_replay() {
        let mut tracer = Tracing::<Mock>::new();
        let mut transcript = tracer.new_transcript(b"commit_round_trip");

        let evaluations: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let poly = Polynomial::<Fr>::new(evaluations);
        let (commitment_value, _hint) =
            <Mock as jolt_openings::CommitmentScheme>::commit(&poly, &());

        // Build the proof against the *prover-side* transcript so its
        // Fiat-Shamir state mirrors what `Native` would do; then we feed
        // the *same* domain label to the tracing transcript and replay
        // the absorb+verify pair.
        let point = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let eval = poly.evaluate(&point);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"commit_round_trip");
        // Mirror the verifier-side absorb on the prover side so the
        // transcripts stay in sync.
        prover_transcript.append(&LabelWithCount(b"C", commitment_value.serialized_len()));
        commitment_value.append_to_transcript(&mut prover_transcript);
        let proof = <Mock as jolt_openings::CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &(),
            None,
            &mut prover_transcript,
        );

        // Wrap commitment, absorb, then the opening check on the tracer.
        let commitment_node = <Tracing<Mock> as CommitmentBackend<Mock>>::wrap_commitment(
            &mut tracer,
            commitment_value,
            CommitmentOrigin::Proof,
            "C",
        );
        <Tracing<Mock> as CommitmentBackend<Mock>>::absorb_commitment(
            &mut tracer,
            &mut transcript,
            &commitment_node,
            b"C",
        );

        let point_w: Vec<_> = point.iter().map(|p| tracer.wrap_proof(*p, "z")).collect();
        let claim_w = tracer.wrap_proof(eval, "claim");

        <Tracing<Mock> as CommitmentBackend<Mock>>::verify_opening(
            &mut tracer,
            &(),
            &commitment_node,
            &point_w,
            &claim_w,
            &proof,
            &mut transcript,
        )
        .expect("Tracing::verify_opening defers checking; should always return Ok");

        // Drop the transcript to release the shared graph borrow.
        drop(transcript);

        // The tracing graph should now contain at least one of each
        // commitment-shaped variant plus the OpeningHolds assertion.
        let graph = tracer.snapshot();
        let mut has_wrap = false;
        let mut has_absorb = false;
        let mut has_check = false;
        for n in &graph.nodes {
            match n {
                AstOp::CommitmentWrap { .. } => has_wrap = true,
                AstOp::TranscriptAbsorbCommitment { .. } => has_absorb = true,
                AstOp::OpeningCheck { .. } => has_check = true,
                _ => {}
            }
        }
        assert!(has_wrap, "graph must record CommitmentWrap");
        assert!(has_absorb, "graph must record TranscriptAbsorbCommitment");
        assert!(has_check, "graph must record OpeningCheck");
        assert!(
            graph
                .assertions
                .iter()
                .any(|a| matches!(a, AstAssertion::OpeningHolds { .. })),
            "graph must record OpeningHolds assertion"
        );

        // Happy-path replay: PCS::verify accepts.
        let _ = replay::<Mock>(&graph, &tracer.wrap_values(), &())
            .expect("replay must accept honest commitment + proof");
    }

    /// Replay surfaces a structured `OpeningCheckFailed` error when the
    /// underlying `<PCS as CommitmentScheme>::verify` rejects the proof.
    #[test]
    fn replay_reports_opening_check_failure() {
        let mut tracer = Tracing::<Mock>::new();
        let mut transcript = tracer.new_transcript(b"commit_failure");

        let evaluations: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let poly = Polynomial::<Fr>::new(evaluations);
        let (commitment_value, _hint) =
            <Mock as jolt_openings::CommitmentScheme>::commit(&poly, &());

        let point = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let eval = poly.evaluate(&point);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"commit_failure");
        prover_transcript.append(&LabelWithCount(b"C", commitment_value.serialized_len()));
        commitment_value.append_to_transcript(&mut prover_transcript);
        let proof = <Mock as jolt_openings::CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &(),
            None,
            &mut prover_transcript,
        );

        let commitment_node = <Tracing<Mock> as CommitmentBackend<Mock>>::wrap_commitment(
            &mut tracer,
            commitment_value,
            CommitmentOrigin::Proof,
            "C",
        );
        <Tracing<Mock> as CommitmentBackend<Mock>>::absorb_commitment(
            &mut tracer,
            &mut transcript,
            &commitment_node,
            b"C",
        );

        let point_w: Vec<_> = point.iter().map(|p| tracer.wrap_proof(*p, "z")).collect();
        // Tamper the claim: claim a wrong evaluation. The tracer happily
        // records this; replay should reject it via `MockPCS::verify`.
        let tampered_claim = eval + Fr::from_u64(1);
        let claim_w = tracer.wrap_proof(tampered_claim, "claim");

        <Tracing<Mock> as CommitmentBackend<Mock>>::verify_opening(
            &mut tracer,
            &(),
            &commitment_node,
            &point_w,
            &claim_w,
            &proof,
            &mut transcript,
        )
        .expect("Tracing::verify_opening defers checking; should always return Ok");

        drop(transcript);

        let graph = tracer.snapshot();
        let err = replay::<Mock>(&graph, &tracer.wrap_values(), &())
            .expect_err("tampered claim must be rejected at replay");
        assert!(
            matches!(err, BackendError::OpeningCheckFailed { .. }),
            "expected OpeningCheckFailed, got {err:?}",
        );
    }
}
