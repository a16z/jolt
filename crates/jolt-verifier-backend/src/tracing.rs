//! AST-recording [`FieldBackend`] for symbolic execution of the verifier.
//!
//! `Tracing` runs the verifier code through the [`FieldBackend`] interface
//! while doing no field arithmetic at all. Every wrap, every constant, every
//! arithmetic op, every assertion, **and every Fiat-Shamir transcript op**
//! becomes a node (or a constraint) in an [`AstGraph`]. The resulting graph
//! is the verifier's *symbolic execution trace*: a side-effect-free,
//! deterministic record of which operations the verifier would have
//! performed on the supplied (still abstract) inputs.
//!
//! ## Why a separate AST instead of trait-style "do work in the ZK target"
//!
//! By emitting an explicit [`AstGraph`] we keep three downstream consumers
//! plug-in compatible:
//!
//! 1. **Recursion / SNARK composition.** Lower the AST to a circuit (R1CS,
//!    AIR, Plonkish, …) inside an outer proof system. The lowering pass owns
//!    the encoding decisions; the verifier source code does not.
//! 2. **Theorem prover export (Lean/Coq).** Walk the AST and emit
//!    proof-friendly definitions, see `jolt-ir`'s Lean emitter for the
//!    pattern.
//! 3. **Inspection / fuzzing / golden tests.** The graph is `Debug + Clone`
//!    so test code can examine it, snapshot it, or replay a smaller instance
//!    through [`Native`](crate::Native) for differential checks.
//!
//! ## Transcript operations
//!
//! The Fiat-Shamir transcript is a stateful sponge: it absorbs bytes (proof
//! data, public inputs, labels) and squeezes field-valued challenges. To
//! make the AST a *complete* record of the verifier's observable behaviour,
//! transcript ops are themselves AST nodes:
//!
//! - [`AstOp::TranscriptInit`] starts a new transcript with a domain label.
//! - [`AstOp::TranscriptAbsorbBytes`] threads in a chunk of bytes (the
//!   verifier's labeled-domain encodings, PCS commitment serialisations,
//!   field-element absorbs — anything that the underlying `Blake2bTranscript`
//!   would feed to its hash).
//! - [`AstOp::TranscriptChallengeState`] advances the sponge by one squeeze
//!   and represents the post-state.
//! - [`AstOp::TranscriptChallengeValue`] is the squeezed field-element value;
//!   it is the node that subsequent arithmetic ops reference.
//!
//! Splitting "post-state" and "value" into two nodes keeps the AST a pure
//! DAG: state nodes thread through `TranscriptAbsorbBytes`/`TranscriptInit`
//! edges, value nodes thread through arithmetic edges. Downstream consumers
//! can keep them in two separate variable spaces (state hashes vs.
//! field-element witnesses) when lowering to a recursive verifier.
//!
//! ## Provenance
//!
//! Every wrapped scalar carries its [`ScalarOrigin`] and a static label, so
//! downstream consumers can:
//!
//! - mark public-input rows separately from witness rows (R1CS lowering),
//! - render the graph with human-readable variable names (Lean / debug),
//! - audit which proof fields and challenges actually flow into the final
//!   assertions.

use std::sync::{Arc, Mutex};

use jolt_field::Field;
use jolt_transcript::{Blake2bTranscript, Transcript};

use crate::backend::{FieldBackend, ScalarOrigin};
use crate::error::BackendError;

/// Stable identifier into [`AstGraph::nodes`].
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AstNodeId(pub u32);

/// A single recorded operation.
///
/// Operands are referenced by [`AstNodeId`] so the graph is a DAG even if the
/// verifier reuses a scalar across many ops. Transcript ops thread their
/// `prev_state` through the same id space so a transcript history reads as
/// a chain of state nodes.
#[derive(Clone, Debug)]
pub enum AstOp {
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
    /// (`Label`, `LabelWithCount`, field-element `to_bytes`, PCS commitment
    /// serialisations) all decompose into one or more `TranscriptAbsorbBytes`
    /// nodes carrying the exact byte buffer that `Blake2bTranscript::append_bytes`
    /// consumed.
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
}

/// A recorded equality assertion `lhs == rhs`.
///
/// Assertions are kept on the side from [`AstOp`] so a graph can carry both
/// the *value DAG* and the *constraint set*. R1CSGen-style consumers turn
/// each into a `lhs - rhs == 0` constraint; native replays check them by
/// equality.
#[derive(Clone, Debug)]
pub struct AstAssertion {
    /// Left-hand side of `lhs == rhs`.
    pub lhs: AstNodeId,
    /// Right-hand side of `lhs == rhs`.
    pub rhs: AstNodeId,
    /// Caller-supplied debug context.
    pub ctx: &'static str,
}

/// Symbolic execution trace produced by [`Tracing`].
///
/// `nodes` is append-only — the index into the vector is the [`AstNodeId`].
/// `assertions` accumulate in the order the verifier issued them.
#[derive(Clone, Debug, Default)]
pub struct AstGraph {
    /// All recorded value-producing operations.
    pub nodes: Vec<AstOp>,
    /// All `assert_eq` calls, recorded in issue order.
    pub assertions: Vec<AstAssertion>,
}

impl AstGraph {
    /// Returns an empty graph (no nodes, no assertions).
    pub fn new() -> Self {
        Self::default()
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
    fn push(&mut self, op: AstOp) -> AstNodeId {
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
pub struct AstScalar<F: Field> {
    /// Position in the owning graph.
    pub id: AstNodeId,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> AstScalar<F> {
    pub(crate) fn new(id: AstNodeId) -> Self {
        Self {
            id,
            _marker: std::marker::PhantomData,
        }
    }
}

/// AST-recording backend.
///
/// `Tracing` itself is just a handle to the shared graph — clone it freely
/// and every clone records into the same DAG. Transcripts produced by
/// [`Tracing::new_transcript`] also record into this same graph, so
/// arithmetic and transcript history sit in the same node-id space.
///
/// In addition to the side-effect-free [`AstGraph`], `Tracing` keeps an
/// internal sidecar of the concrete wrap values it received. The graph and
/// the value list are independent: graph consumers (Lean export, R1CS lower)
/// ignore the values, while differential testing against [`Native`](crate::Native)
/// uses them via [`replay`] without the caller having to reconstruct the
/// wrap sequence.
///
/// Internally the graph lives behind `Arc<Mutex<…>>` (rather than
/// `Rc<RefCell<…>>`) because [`Transcript`] requires `Sync + Send + 'static`
/// and the [`TracingTranscript`] shares the same handle.
#[derive(Clone, Debug)]
pub struct Tracing<F: Field> {
    graph: Arc<Mutex<AstGraph>>,
    wrap_values: Arc<Mutex<Vec<F>>>,
}

impl<F: Field> Default for Tracing<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> Tracing<F> {
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
    pub fn snapshot(&self) -> AstGraph {
        self.graph.lock().expect("AstGraph mutex poisoned").clone()
    }

    /// Returns a clone of the wrap values in the order they were recorded.
    ///
    /// Pair with [`replay`] to re-execute the trace against [`Native`] for
    /// differential testing. Production graph consumers (Lean, R1CS) should
    /// ignore this and walk [`AstGraph`] directly.
    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    pub fn wrap_values(&self) -> Vec<F> {
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
    pub fn with_graph<R>(&self, f: impl FnOnce(&AstGraph) -> R) -> R {
        f(&self.graph.lock().expect("AstGraph mutex poisoned"))
    }

    /// Returns a fresh transcript that records into this graph.
    ///
    /// Provided as an inherent method so callers that already hold a
    /// [`Tracing`] handle can construct extra transcripts without going
    /// through [`FieldBackend::new_transcript`].
    pub fn new_transcript(&self, label: &'static [u8]) -> TracingTranscript<F> {
        TracingTranscript::with_graph(label, self.graph.clone())
    }
}

impl<F: Field> FieldBackend for Tracing<F> {
    type F = F;
    type Scalar = AstScalar<F>;
    type Transcript = TracingTranscript<F>;

    #[expect(
        clippy::expect_used,
        reason = "AstGraph mutex is internal; poisoning would itself be a bug"
    )]
    fn wrap(&mut self, value: F, origin: ScalarOrigin, label: &'static str) -> Self::Scalar {
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
            .push(AstAssertion {
                lhs: a.id,
                rhs: b.id,
                ctx,
            });
        Ok(())
    }

    fn unwrap(&self, _scalar: &Self::Scalar) -> Option<F> {
        // Tracing intentionally does not expose concrete values through the
        // backend interface — that would let downstream code branch on
        // witness data and leak symbolic faithfulness. Replay uses the
        // sidecar wrap-value list explicitly via [`Tracing::wrap_values`].
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

/// Transcript wrapper that records every absorb/squeeze into a shared
/// [`AstGraph`].
///
/// Internally holds a real [`Blake2bTranscript`] so all challenges are
/// produced by the actual Fiat-Shamir hash function. This keeps higher-level
/// PCS / sumcheck verification working transparently against this
/// transcript: every `transcript.append_bytes` and `transcript.challenge`
/// they perform also lands in the AST.
///
/// The shared graph handle is [`Arc<Mutex<AstGraph>>`] so the type satisfies
/// [`Sync + Send + 'static`] (required by the [`Transcript`] trait).
pub struct TracingTranscript<F: Field> {
    inner: Blake2bTranscript<F>,
    graph: Arc<Mutex<AstGraph>>,
    state_node: AstNodeId,
    last_squeeze_value: Option<AstNodeId>,
}

impl<F: Field> TracingTranscript<F> {
    fn with_graph(label: &'static [u8], graph: Arc<Mutex<AstGraph>>) -> Self {
        let state_node = {
            #[expect(
                clippy::expect_used,
                reason = "AstGraph mutex is internal; poisoning would itself be a bug"
            )]
            let mut g = graph.lock().expect("AstGraph mutex poisoned");
            g.push(AstOp::TranscriptInit { label })
        };
        Self {
            inner: Blake2bTranscript::<F>::new(label),
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
}

impl<F: Field> Clone for TracingTranscript<F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            graph: self.graph.clone(),
            state_node: self.state_node,
            last_squeeze_value: self.last_squeeze_value,
        }
    }
}

impl<F: Field> Default for TracingTranscript<F> {
    fn default() -> Self {
        // Default constructs an *isolated* graph rather than panicking; the
        // primary construction path is `Tracing::new_transcript`, which
        // shares the backend's graph. Default exists only to satisfy the
        // [`Transcript`] trait bound; downstream code must avoid relying on
        // it for graph aggregation.
        Self::with_graph(
            b"jolt_tracing_default",
            Arc::new(Mutex::new(AstGraph::new())),
        )
    }
}

impl<F: Field> std::fmt::Debug for TracingTranscript<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracingTranscript")
            .field("state_node", &self.state_node)
            .field("last_squeeze_value", &self.last_squeeze_value)
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<F: Field> Transcript for TracingTranscript<F> {
    type Challenge = F;

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

    fn challenge(&mut self) -> F {
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
/// `wrap_values` supplies a concrete `F` for each [`AstOp::Wrap`] node, in
/// the order the wraps appeared in `graph.nodes`. Transcript ops re-run a
/// real [`Blake2bTranscript`] against the recorded byte stream, so squeezed
/// challenges are derived deterministically by Fiat-Shamir replay rather
/// than supplied externally. The function returns the field values
/// associated with every node (state nodes are recorded as zero — they
/// have no meaningful field semantics) and confirms that every recorded
/// assertion holds.
///
/// This is the bridge that lets a snapshot taken with [`Tracing`] be
/// validated against a regular [`Native`](crate::Native) execution: if the
/// graph is faithful, then for any consistent witness it must reproduce the
/// same value at every node and pass every assertion.
///
/// # Errors
///
/// Returns [`BackendError::AssertionFailed`] if a recorded assertion does
/// not hold. Returns [`BackendError::InverseOfZero`] if an
/// [`AstOp::Inverse`] node references a zero operand.
///
/// # Panics
///
/// Panics if `wrap_values` does not contain exactly one entry per
/// [`AstOp::Wrap`] node, if the graph references a transcript-state operand
/// that has no live transcript (graph corruption), or if a transcript
/// challenge node references a non-state operand.
pub fn replay<F: Field>(graph: &AstGraph, wrap_values: &[F]) -> Result<Vec<F>, BackendError> {
    let mut values: Vec<F> = vec![F::zero(); graph.nodes.len()];
    let mut transcripts: Vec<Option<Blake2bTranscript<F>>> = vec![None; graph.nodes.len()];
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
            AstOp::Constant(c) => values[idx] = F::from_i128(*c),
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
                transcripts[idx] = Some(Blake2bTranscript::<F>::new(label));
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
        }
    }
    assert_eq!(
        wrap_cursor,
        wrap_values.len(),
        "replay: too many wrap_values (consumed {wrap_cursor}, given {})",
        wrap_values.len()
    );

    for assertion in &graph.assertions {
        if values[assertion.lhs.0 as usize] != values[assertion.rhs.0 as usize] {
            return Err(BackendError::AssertionFailed(assertion.ctx));
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
    use jolt_field::{Field, Fr};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn records_basic_dag_shape() {
        let mut t = Tracing::<Fr>::new();
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
        let mut t = Tracing::<Fr>::new();
        let a = t.wrap_proof(Fr::from_u64(2), "a");
        let three = t.const_i128(3);
        // Tracing must not panic on a "wrong" assert; it just records.
        t.assert_eq(&a, &three, "demo").unwrap();
        let g = t.snapshot();
        assert_eq!(g.assertion_count(), 1);
        assert_eq!(g.assertions[0].ctx, "demo");
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

            let mut tracer = Tracing::<Fr>::new();
            let ta: Vec<_> = a_vals.iter().map(|v| tracer.wrap_proof(*v, "a")).collect();
            let tb: Vec<_> = b_vals.iter().map(|v| tracer.wrap_proof(*v, "b")).collect();
            let traced_handle = eq_eval(&mut tracer, &ta, &tb);

            let graph = tracer.snapshot();
            let wraps = tracer.wrap_values();
            let values = replay(&graph, &wraps).unwrap();
            assert_eq!(values[traced_handle.id.0 as usize], native_value, "n = {n}");
        }
    }

    /// `assert_eq` records an entry; replay catches a witness that violates
    /// it.
    #[test]
    fn replay_reports_assertion_failure() {
        let mut tracer = Tracing::<Fr>::new();
        let a = tracer.wrap_proof(Fr::from_u64(0), "a");
        let b = tracer.wrap_proof(Fr::from_u64(0), "b");
        tracer.assert_eq(&a, &b, "ab").unwrap();
        let graph = tracer.snapshot();

        // Witness consistent with the trace: replay succeeds.
        let _ = replay(&graph, &[Fr::from_u64(7), Fr::from_u64(7)]).unwrap();

        // Inconsistent witness: replay fails.
        let err = replay(&graph, &[Fr::from_u64(7), Fr::from_u64(8)]).unwrap_err();
        assert!(matches!(err, BackendError::AssertionFailed("ab")));
    }

    /// A short transcript script (init → absorb → squeeze → absorb → squeeze)
    /// is captured as discrete AST nodes, and replay reproduces the same
    /// challenge values as a freshly-driven [`Blake2bTranscript`].
    #[test]
    fn transcript_ops_round_trip_through_replay() {
        let tracer = Tracing::<Fr>::new();
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
        let values = replay(&g, &tracer.wrap_values()).unwrap();
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
        let mut tracer = Tracing::<Fr>::new();
        let mut transcript = tracer.new_transcript(b"squeeze_test");
        let (challenge_f, challenge_w) = tracer.squeeze(&mut transcript, "alpha");

        // Use the squeezed challenge in a tiny arithmetic op to confirm the
        // backend links the value node into the field DAG.
        let two = tracer.const_i128(2);
        let doubled = tracer.mul(&challenge_w, &two);

        let g = tracer.snapshot();
        let values = replay(&g, &tracer.wrap_values()).unwrap();
        assert_eq!(values[challenge_w.id.0 as usize], challenge_f);
        assert_eq!(values[doubled.id.0 as usize], challenge_f + challenge_f);
    }
}
