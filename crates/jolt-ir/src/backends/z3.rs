//! Z3 SMT solver backend.
//!
//! Translates [`Expr`] trees into Z3 integer arithmetic expressions via the
//! [`CircuitEmitter`] trait. This enables formal verification of claim
//! definitions and constraint systems.
//!
//! Feature-gated behind `z3`. Enable with:
//!
//! ```toml
//! jolt-ir = { path = "...", features = ["z3"] }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use jolt_ir::ExprBuilder;
//! use jolt_ir::backends::z3::Z3Emitter;
//!
//! let mut emitter = Z3Emitter::new();
//! let b = ExprBuilder::new();
//! let h = b.opening(0);
//! let gamma = b.challenge(0);
//! let expr = b.build(gamma * (h * h - h));
//!
//! let z3_int = expr.to_circuit(&mut emitter);
//! // z3_int is a z3::ast::Int that can be used in Z3 solver assertions
//! ```

use std::collections::BTreeMap;

use z3::ast::Int;

use crate::backends::circuit::CircuitEmitter;
use crate::expr::Var;

/// Z3 SMT emitter for [`Expr`] trees.
///
/// Implements [`CircuitEmitter`] with `Wire = z3::ast::Int`, producing Z3
/// integer expressions. Opening and challenge variables are created lazily
/// as fresh Z3 symbolic integers on first access.
///
/// Pre-bind variables to concrete values with [`bind_opening`] /
/// [`bind_challenge`] before emission for partial evaluation.
///
/// [`bind_opening`]: Z3Emitter::bind_opening
/// [`bind_challenge`]: Z3Emitter::bind_challenge
pub struct Z3Emitter {
    openings: BTreeMap<u32, Int>,
    challenges: BTreeMap<u32, Int>,
    opening_prefix: String,
    challenge_prefix: String,
}

impl Z3Emitter {
    pub fn new() -> Self {
        Self {
            openings: BTreeMap::new(),
            challenges: BTreeMap::new(),
            opening_prefix: "opening".into(),
            challenge_prefix: "challenge".into(),
        }
    }

    /// Set a custom prefix for opening variable names.
    ///
    /// Useful when multiple emitters coexist in the same Z3 context to avoid
    /// name collisions (e.g., `"cpu1_opening"` vs `"cpu2_opening"`).
    pub fn with_opening_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.opening_prefix = prefix.into();
        self
    }

    /// Set a custom prefix for challenge variable names.
    pub fn with_challenge_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.challenge_prefix = prefix.into();
        self
    }

    /// Bind an opening variable to a concrete Z3 integer.
    ///
    /// When the expression references `Opening(id)`, this value is returned
    /// instead of a fresh symbolic variable.
    pub fn bind_opening(&mut self, id: u32, val: Int) {
        let _ = self.openings.insert(id, val);
    }

    /// Bind a challenge variable to a concrete Z3 integer.
    pub fn bind_challenge(&mut self, id: u32, val: Int) {
        let _ = self.challenges.insert(id, val);
    }

    /// Returns all opening variables (symbolic or bound) accessed during
    /// emission.
    pub fn openings(&self) -> &BTreeMap<u32, Int> {
        &self.openings
    }

    /// Returns all challenge variables (symbolic or bound) accessed during
    /// emission.
    pub fn challenges(&self) -> &BTreeMap<u32, Int> {
        &self.challenges
    }

    fn get_opening(&mut self, id: u32) -> Int {
        self.openings
            .entry(id)
            .or_insert_with(|| Int::new_const(format!("{}_{}", self.opening_prefix, id)))
            .clone()
    }

    fn get_challenge(&mut self, id: u32) -> Int {
        self.challenges
            .entry(id)
            .or_insert_with(|| Int::new_const(format!("{}_{}", self.challenge_prefix, id)))
            .clone()
    }
}

impl Default for Z3Emitter {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitEmitter for Z3Emitter {
    type Wire = Int;

    fn constant(&mut self, val: i128) -> Int {
        // IR constants are structural (0, 1, -1, chunk sizes, register counts)
        // and always fit in i64.
        if let Ok(v) = i64::try_from(val) {
            Int::from_i64(v)
        } else {
            // Large constants: parse from decimal string.
            val.to_string()
                .parse::<Int>()
                .expect("z3 failed to parse constant")
        }
    }

    fn variable(&mut self, var: Var) -> Int {
        match var {
            Var::Opening(id) => self.get_opening(id),
            Var::Challenge(id) => self.get_challenge(id),
        }
    }

    fn neg(&mut self, inner: Int) -> Int {
        -inner
    }

    fn add(&mut self, lhs: Int, rhs: Int) -> Int {
        lhs + rhs
    }

    fn sub(&mut self, lhs: Int, rhs: Int) -> Int {
        lhs - rhs
    }

    fn mul(&mut self, lhs: Int, rhs: Int) -> Int {
        lhs * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use z3::{SatResult, Solver};

    #[test]
    fn booleanity_symbolic() {
        // γ · (H² − H) = 0 should be SAT when H ∈ {0, 1}
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let mut emitter = Z3Emitter::new();
        let z3_expr = expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        let h_var = emitter.openings().get(&0).unwrap();
        solver.assert(h_var.eq(Int::from(0)) | h_var.eq(Int::from(1)));
        let gamma_var = emitter.challenges().get(&0).unwrap();
        solver.assert(gamma_var.ne(Int::from(0)));
        solver.assert(z3_expr.eq(Int::from(0)));

        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn booleanity_violated() {
        // γ · (H² − H) = 0 with H = 2, γ = 1 → 1·(4-2) = 2 ≠ 0
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(0, Int::from(2));
        emitter.bind_challenge(0, Int::from(1));
        let z3_expr = expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        solver.assert(z3_expr.eq(Int::from(0)));

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn weighted_sum_consistency() {
        // c0·o0 + c1·o1 + c2·o2 — two evaluations with same inputs must match
        let build = || {
            let b = ExprBuilder::new();
            let o0 = b.opening(0);
            let o1 = b.opening(1);
            let o2 = b.opening(2);
            let c0 = b.challenge(0);
            let c1 = b.challenge(1);
            let c2 = b.challenge(2);
            b.build(c0 * o0 + c1 * o1 + c2 * o2)
        };

        let expr = build();
        let mut e1 = Z3Emitter::new().with_opening_prefix("a");
        let mut e2 = Z3Emitter::new().with_opening_prefix("b");

        let z1 = expr.to_circuit(&mut e1);
        let z2 = expr.to_circuit(&mut e2);

        let solver = Solver::new();
        for i in 0..3 {
            let c1 = e1.challenges().get(&i).unwrap();
            let c2 = e2.challenges().get(&i).unwrap();
            solver.assert(c1.eq(c2));
        }
        for i in 0..3 {
            let o1 = e1.openings().get(&i).unwrap();
            let o2 = e2.openings().get(&i).unwrap();
            solver.assert(o1.eq(o2));
        }
        // Outputs differ → should be UNSAT
        solver.assert(z1.ne(z2));

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn concrete_evaluation() {
        // 3·x + 7 with x=5 → 22
        let b = ExprBuilder::new();
        let x = b.opening(0);
        let expr = b.build(b.constant(3) * x + b.constant(7));

        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(0, Int::from(5));
        let z3_expr = expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        solver.assert(z3_expr.eq(Int::from(22)));
        assert_eq!(solver.check(), SatResult::Sat);

        let solver2 = Solver::new();
        solver2.assert(z3_expr.eq(Int::from(23)));
        assert_eq!(solver2.check(), SatResult::Unsat);
    }

    #[test]
    fn negation() {
        // -x = 0 → x = 0
        let b = ExprBuilder::new();
        let x = b.opening(0);
        let expr = b.build(-x);

        let mut emitter = Z3Emitter::new();
        let z3_expr = expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        solver.assert(z3_expr.eq(Int::from(0)));
        assert_eq!(solver.check(), SatResult::Sat);

        let model = solver.get_model().unwrap();
        let x_var = emitter.openings().get(&0).unwrap();
        let x_val = model.eval(x_var, true).unwrap().as_i64().unwrap();
        assert_eq!(x_val, 0);
    }

    #[test]
    fn product_constraint() {
        // left · right = output with 6·7 = 42
        let b = ExprBuilder::new();
        let left = b.opening(0);
        let right = b.opening(1);
        let output = b.opening(2);
        let expr = b.build(left * right - output);

        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(0, Int::from(6));
        emitter.bind_opening(1, Int::from(7));
        emitter.bind_opening(2, Int::from(42));
        let z3_expr = expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        solver.assert(z3_expr.eq(Int::from(0)));
        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn claim_definition_booleanity() {
        use crate::zkvm::claims::ram;

        // hamming_booleanity: eq·H² + neg_eq·H where neg_eq = -eq
        // This equals eq·(H² − H) when the constraint neg_eq = -eq holds.
        let claim = ram::hamming_booleanity();

        // H = 1 (boolean), neg_eq = -eq → eq·1 + (-eq)·1 = 0 ✓
        let mut emitter = Z3Emitter::new();
        emitter.bind_opening(0, Int::from(1));
        let z3_expr = claim.expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        // Enforce neg_eq = -eq (challenge(1) = -challenge(0))
        let eq_var = emitter.challenges().get(&0).unwrap();
        let neg_eq_var = emitter.challenges().get(&1).unwrap();
        solver.assert(neg_eq_var.eq(-eq_var));
        solver.assert(z3_expr.eq(Int::from(0)));
        assert_eq!(solver.check(), SatResult::Sat);

        // H = 3 (non-boolean), neg_eq = -eq, eq ≠ 0 → eq·9 + (-eq)·3 = 6·eq ≠ 0
        let mut emitter2 = Z3Emitter::new();
        emitter2.bind_opening(0, Int::from(3));
        let z3_expr2 = claim.expr.to_circuit(&mut emitter2);

        let solver2 = Solver::new();
        let eq2 = emitter2.challenges().get(&0).unwrap();
        let neg_eq2 = emitter2.challenges().get(&1).unwrap();
        solver2.assert(neg_eq2.eq(-eq2));
        solver2.assert(eq2.ne(Int::from(0)));
        solver2.assert(z3_expr2.eq(Int::from(0)));
        assert_eq!(solver2.check(), SatResult::Unsat);
    }

    #[test]
    fn claim_definition_shift() {
        use crate::zkvm::claims::spartan;

        // shift: c0·unexpanded_pc + c1·pc + c2·is_virtual + c3·is_first + c4·noop + c5
        // Consistency: same inputs → same output
        let claim = spartan::shift();

        let mut e1 = Z3Emitter::new()
            .with_opening_prefix("s1")
            .with_challenge_prefix("sc1");
        let mut e2 = Z3Emitter::new()
            .with_opening_prefix("s2")
            .with_challenge_prefix("sc2");
        let z1 = claim.expr.to_circuit(&mut e1);
        let z2 = claim.expr.to_circuit(&mut e2);

        let solver = Solver::new();
        // Bind same challenges
        for i in 0..=5 {
            let c1 = e1.challenges().get(&i).unwrap();
            let c2 = e2.challenges().get(&i).unwrap();
            solver.assert(c1.eq(c2));
        }
        // Bind same openings
        for i in 0..=4 {
            let o1 = e1.openings().get(&i).unwrap();
            let o2 = e2.openings().get(&i).unwrap();
            solver.assert(o1.eq(o2));
        }
        // Outputs must match
        solver.assert(z1.ne(z2));
        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn claim_definition_product_virtual() {
        use crate::zkvm::claims::spartan;

        // product_virtual_remainder: multi-term degree-2 expression
        let claim = spartan::product_virtual_remainder();

        // Concrete check: all openings = 0, all challenges = 1 → result = 0
        let mut emitter = Z3Emitter::new();
        for i in 0..8 {
            emitter.bind_opening(i, Int::from(0));
        }
        for i in 0..=5 {
            emitter.bind_challenge(i, Int::from(1));
        }
        let z3_expr = claim.expr.to_circuit(&mut emitter);

        let solver = Solver::new();
        solver.assert(z3_expr.eq(Int::from(0)));
        assert_eq!(solver.check(), SatResult::Sat);
    }
}
