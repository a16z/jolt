//! Circuit transpilation backend.
//!
//! Provides [`CircuitEmitter`], a trait for emitting circuit constraints from an
//! expression tree. Each target framework (gnark, bellman, plonky2, etc.)
//! implements this trait to produce its native wire/constraint representation.
//!
//! `CircuitEmitter` is structurally identical to [`ExprVisitor`](crate::ExprVisitor)
//! but uses circuit-domain naming (`wire`, `constant`, `variable`) instead of
//! visitor-pattern naming (`visit_constant`, `visit_var`). The `Expr::to_circuit`
//! method dispatches through the visitor infrastructure with caching, so each
//! subexpression is emitted exactly once (critical for DAGs after CSE).
//!
//! # Implementing a target
//!
//! ```ignore
//! struct GnarkEmitter { /* codegen state */ }
//!
//! impl CircuitEmitter for GnarkEmitter {
//!     type Wire = String; // Go expression text
//!
//!     fn constant(&mut self, val: i128) -> String {
//!         format!("big.NewInt({})", val)
//!     }
//!     fn variable(&mut self, var: Var) -> String {
//!         match var {
//!             Var::Opening(i) => format!("circuit.Opening_{i}"),
//!             Var::Challenge(i) => format!("circuit.Challenge_{i}"),
//!         }
//!     }
//!     fn neg(&mut self, inner: String) -> String { format!("api.Neg({inner})") }
//!     fn add(&mut self, l: String, r: String) -> String { format!("api.Add({l}, {r})") }
//!     fn sub(&mut self, l: String, r: String) -> String { format!("api.Sub({l}, {r})") }
//!     fn mul(&mut self, l: String, r: String) -> String { format!("api.Mul({l}, {r})") }
//! }
//! ```

use crate::expr::{Expr, Var};
use crate::visitor::ExprVisitor;

/// Trait for emitting circuit constraints from an expression tree.
///
/// Each method corresponds to a single arithmetic operation in the expression.
/// The `Wire` associated type represents a handle to a value in the target
/// circuit — a Go expression string for gnark, an `AllocatedNum` for bellman,
/// a wire index for plonky2, etc.
///
/// Implementations may accumulate side effects (constraint emission, code
/// generation) in `&mut self`.
pub trait CircuitEmitter {
    /// A handle to a value in the target circuit.
    type Wire;

    /// Emit a constant value.
    fn constant(&mut self, val: i128) -> Self::Wire;

    /// Emit a variable reference (opening or challenge).
    fn variable(&mut self, var: Var) -> Self::Wire;

    /// Emit a negation: `-inner`.
    fn neg(&mut self, inner: Self::Wire) -> Self::Wire;

    /// Emit an addition: `lhs + rhs`.
    fn add(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit a subtraction: `lhs - rhs`.
    fn sub(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;

    /// Emit a multiplication: `lhs * rhs`.
    fn mul(&mut self, lhs: Self::Wire, rhs: Self::Wire) -> Self::Wire;
}

/// Adapter that bridges `CircuitEmitter` to `ExprVisitor`.
struct EmitterAdapter<'a, E: CircuitEmitter> {
    emitter: &'a mut E,
}

impl<E: CircuitEmitter> ExprVisitor for EmitterAdapter<'_, E> {
    type Output = E::Wire;

    #[inline]
    fn visit_constant(&mut self, val: i128) -> E::Wire {
        self.emitter.constant(val)
    }
    #[inline]
    fn visit_var(&mut self, var: Var) -> E::Wire {
        self.emitter.variable(var)
    }
    #[inline]
    fn visit_neg(&mut self, inner: E::Wire) -> E::Wire {
        self.emitter.neg(inner)
    }
    #[inline]
    fn visit_add(&mut self, lhs: E::Wire, rhs: E::Wire) -> E::Wire {
        self.emitter.add(lhs, rhs)
    }
    #[inline]
    fn visit_sub(&mut self, lhs: E::Wire, rhs: E::Wire) -> E::Wire {
        self.emitter.sub(lhs, rhs)
    }
    #[inline]
    fn visit_mul(&mut self, lhs: E::Wire, rhs: E::Wire) -> E::Wire {
        self.emitter.mul(lhs, rhs)
    }
}

impl Expr {
    /// Emit circuit constraints by traversing the expression with a
    /// [`CircuitEmitter`].
    ///
    /// Uses cached traversal so each subexpression is emitted exactly once —
    /// critical for DAGs produced by CSE.
    pub fn to_circuit<E: CircuitEmitter>(&self, emitter: &mut E) -> E::Wire
    where
        E::Wire: Clone,
    {
        let mut adapter = EmitterAdapter { emitter };
        self.visit_cached(&mut adapter)
    }

    /// Emit circuit constraints using tree traversal (no caching).
    ///
    /// Shared subexpressions will be emitted multiple times. Use `to_circuit`
    /// for DAGs after CSE.
    pub fn to_circuit_tree<E: CircuitEmitter>(&self, emitter: &mut E) -> E::Wire
    where
        E::Wire: Clone,
    {
        let mut adapter = EmitterAdapter { emitter };
        self.visit(&mut adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

    /// Mock emitter that records operations as strings (simulates gnark-style codegen).
    struct RecordingEmitter {
        ops: Vec<String>,
        next_wire: usize,
    }

    impl RecordingEmitter {
        fn new() -> Self {
            Self {
                ops: Vec::new(),
                next_wire: 0,
            }
        }

        fn wire(&mut self, label: &str) -> String {
            let w = format!("w{}_{}", self.next_wire, label);
            self.next_wire += 1;
            w
        }
    }

    impl CircuitEmitter for RecordingEmitter {
        type Wire = String;

        fn constant(&mut self, val: i128) -> String {
            let w = self.wire("const");
            self.ops.push(format!("{w} = const({val})"));
            w
        }
        fn variable(&mut self, var: Var) -> String {
            match var {
                Var::Opening(i) => {
                    let w = self.wire("open");
                    self.ops.push(format!("{w} = opening[{i}]"));
                    w
                }
                Var::Challenge(i) => {
                    let w = self.wire("chal");
                    self.ops.push(format!("{w} = challenge[{i}]"));
                    w
                }
            }
        }
        fn neg(&mut self, inner: String) -> String {
            let w = self.wire("neg");
            self.ops.push(format!("{w} = neg({inner})"));
            w
        }
        fn add(&mut self, lhs: String, rhs: String) -> String {
            let w = self.wire("add");
            self.ops.push(format!("{w} = add({lhs}, {rhs})"));
            w
        }
        fn sub(&mut self, lhs: String, rhs: String) -> String {
            let w = self.wire("sub");
            self.ops.push(format!("{w} = sub({lhs}, {rhs})"));
            w
        }
        fn mul(&mut self, lhs: String, rhs: String) -> String {
            let w = self.wire("mul");
            self.ops.push(format!("{w} = mul({lhs}, {rhs})"));
            w
        }
    }

    #[test]
    fn emit_booleanity() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let mut emitter = RecordingEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        // Bottom-up: h, gamma, h*h, h*h-h, gamma*(h*h-h)
        assert_eq!(emitter.ops.len(), 5);
        assert!(root.contains("mul"));
    }

    #[test]
    fn emit_constant_only() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));

        let mut emitter = RecordingEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        assert_eq!(emitter.ops.len(), 1);
        assert_eq!(emitter.ops[0], format!("{root} = const(42)"));
    }

    #[test]
    fn emit_cse_shares_subexpressions() {
        // (a+b) * (a+b) — after CSE, (a+b) should be emitted once
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let expr = b.build((a1 + b1) * (a2 + b2));
        let optimized = expr.eliminate_common_subexpressions();

        let mut emitter = RecordingEmitter::new();
        let _ = optimized.to_circuit(&mut emitter);

        // With CSE: a, b, a+b (once), mul — 4 ops instead of 5
        let add_count = emitter.ops.iter().filter(|op| op.contains("add(")).count();
        assert_eq!(add_count, 1, "CSE should emit a+b only once");
    }

    #[test]
    fn emit_negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        let mut emitter = RecordingEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        assert_eq!(emitter.ops.len(), 2);
        assert!(root.contains("neg"));
    }

    #[test]
    fn emit_traversal_order() {
        // a + b*c — should emit a, b, c, b*c, then add
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build(a + bv * c);

        let mut emitter = RecordingEmitter::new();
        let _ = expr.to_circuit(&mut emitter);

        // Verify bottom-up: the mul must come before the add
        let mul_pos = emitter
            .ops
            .iter()
            .position(|op| op.contains("mul("))
            .unwrap();
        let add_pos = emitter
            .ops
            .iter()
            .position(|op| op.contains("add("))
            .unwrap();
        assert!(mul_pos < add_pos);
    }

    #[test]
    fn tree_vs_cached_traversal() {
        // Without CSE: to_circuit and to_circuit_tree should produce the same
        // number of operations for a tree expression (no sharing)
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a * bv + a);

        let mut emitter1 = RecordingEmitter::new();
        let _ = expr.to_circuit_tree(&mut emitter1);

        let mut emitter2 = RecordingEmitter::new();
        let _ = expr.to_circuit(&mut emitter2);

        // Tree has 4 arena nodes but `a` at index 0 is used by both Mul and Add.
        // Tree traversal visits it twice (4 ops); cached visits it once (3 ops).
        assert!(emitter1.ops.len() >= emitter2.ops.len());
    }
}
