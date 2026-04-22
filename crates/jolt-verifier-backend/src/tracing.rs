//! AST-recording [`FieldBackend`] for symbolic execution of the verifier.
//!
//! `Tracing` runs the verifier code through the [`FieldBackend`] interface
//! while doing no field arithmetic at all. Every wrap, every constant, every
//! arithmetic op, and every assertion becomes a node (or a constraint) in an
//! [`AstGraph`]. The resulting graph is the verifier's *symbolic execution
//! trace*: a side-effect-free, deterministic record of which operations the
//! verifier would have performed on the supplied (still abstract) inputs.
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
//! ## Provenance
//!
//! Every wrapped scalar carries its [`ScalarOrigin`] and a static label, so
//! downstream consumers can:
//!
//! - mark public-input rows separately from witness rows (R1CS lowering),
//! - render the graph with human-readable variable names (Lean / debug),
//! - audit which proof fields and challenges actually flow into the final
//!   assertions.

use std::cell::RefCell;
use std::rc::Rc;

use jolt_field::Field;

use crate::backend::{FieldBackend, ScalarOrigin};
use crate::error::BackendError;

/// Stable identifier into [`AstGraph::nodes`].
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AstNodeId(pub u32);

/// A single recorded operation.
///
/// Operands are referenced by [`AstNodeId`] so the graph is a DAG even if the
/// verifier reuses a scalar across many ops.
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
    fn new(id: AstNodeId) -> Self {
        Self {
            id,
            _marker: std::marker::PhantomData,
        }
    }
}

/// AST-recording backend.
///
/// `Tracing` itself is just a handle to the shared graph — clone it freely
/// and every clone records into the same DAG. This makes it easy to thread
/// through code that wants to take ownership of a backend per stage.
///
/// In addition to the side-effect-free [`AstGraph`], `Tracing` keeps an
/// internal sidecar of the concrete wrap values it received. The graph and
/// the value list are independent: graph consumers (Lean export, R1CS lower)
/// ignore the values, while differential testing against [`Native`] uses
/// them via [`replay`] without the caller having to reconstruct the wrap
/// sequence.
#[derive(Clone, Debug)]
pub struct Tracing<F: Field> {
    graph: Rc<RefCell<AstGraph>>,
    wrap_values: Rc<RefCell<Vec<F>>>,
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
            graph: Rc::new(RefCell::new(AstGraph::new())),
            wrap_values: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Returns a deep clone of the recorded graph (immutable snapshot).
    pub fn snapshot(&self) -> AstGraph {
        self.graph.borrow().clone()
    }

    /// Returns a clone of the wrap values in the order they were recorded.
    ///
    /// Pair with [`replay`] to re-execute the trace against [`Native`] for
    /// differential testing. Production graph consumers (Lean, R1CS) should
    /// ignore this and walk [`AstGraph`] directly.
    pub fn wrap_values(&self) -> Vec<F> {
        self.wrap_values.borrow().clone()
    }

    /// Borrows the underlying graph for read-only inspection.
    pub fn with_graph<R>(&self, f: impl FnOnce(&AstGraph) -> R) -> R {
        f(&self.graph.borrow())
    }
}

impl<F: Field> FieldBackend for Tracing<F> {
    type F = F;
    type Scalar = AstScalar<F>;

    fn wrap(&mut self, value: F, origin: ScalarOrigin, label: &'static str) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Wrap { origin, label });
        self.wrap_values.borrow_mut().push(value);
        AstScalar::new(id)
    }

    fn const_i128(&mut self, v: i128) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Constant(v));
        AstScalar::new(id)
    }

    fn add(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Add(a.id, b.id));
        AstScalar::new(id)
    }

    fn sub(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Sub(a.id, b.id));
        AstScalar::new(id)
    }

    fn mul(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Mul(a.id, b.id));
        AstScalar::new(id)
    }

    fn neg(&mut self, a: &Self::Scalar) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Neg(a.id));
        AstScalar::new(id)
    }

    fn square(&mut self, a: &Self::Scalar) -> Self::Scalar {
        let id = self.graph.borrow_mut().push(AstOp::Square(a.id));
        AstScalar::new(id)
    }

    fn inverse(
        &mut self,
        a: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<Self::Scalar, BackendError> {
        let id = self
            .graph
            .borrow_mut()
            .push(AstOp::Inverse { operand: a.id, ctx });
        Ok(AstScalar::new(id))
    }

    fn assert_eq(
        &mut self,
        a: &Self::Scalar,
        b: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<(), BackendError> {
        self.graph.borrow_mut().assertions.push(AstAssertion {
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
}

/// Replays a tracing graph against a concrete witness assignment.
///
/// `wrap_values` supplies a concrete `F` for each wrap node, in the order
/// the wraps appeared in `graph.nodes`. The function returns the field
/// values associated with every node and confirms that every recorded
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
/// not hold. Panics if the graph references operands out of order (which
/// the [`Tracing`] backend never produces by construction).
///
/// # Panics
///
/// Panics if `wrap_values` does not contain exactly one entry per
/// [`AstOp::Wrap`] node, or if an [`AstOp::Inverse`] node references a zero
/// operand.
pub fn replay<F: Field>(graph: &AstGraph, wrap_values: &[F]) -> Result<Vec<F>, BackendError> {
    let mut values: Vec<F> = Vec::with_capacity(graph.nodes.len());
    let mut wrap_cursor = 0usize;
    for (idx, op) in graph.nodes.iter().enumerate() {
        let v = match op {
            AstOp::Wrap { .. } => {
                assert!(
                    wrap_cursor < wrap_values.len(),
                    "replay: not enough wrap_values (need at least {})",
                    wrap_cursor + 1
                );
                let v = wrap_values[wrap_cursor];
                wrap_cursor += 1;
                v
            }
            AstOp::Constant(c) => F::from_i128(*c),
            AstOp::Neg(a) => -values[a.0 as usize],
            AstOp::Add(a, b) => values[a.0 as usize] + values[b.0 as usize],
            AstOp::Sub(a, b) => values[a.0 as usize] - values[b.0 as usize],
            AstOp::Mul(a, b) => values[a.0 as usize] * values[b.0 as usize],
            AstOp::Square(a) => values[a.0 as usize].square(),
            AstOp::Inverse { operand, ctx } => values[operand.0 as usize]
                .inverse()
                .ok_or(BackendError::InverseOfZero(ctx))?,
        };
        debug_assert_eq!(values.len(), idx);
        values.push(v);
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
}
