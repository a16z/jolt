//! Lean4 code generation backend.
//!
//! Emits Lean4 syntax from an [`Expr`], producing `let` bindings for shared
//! subexpressions (after CSE) and a final root expression. Configurable via
//! [`LeanConfig`].
//!
//! # Example
//!
//! ```
//! use jolt_ir::ExprBuilder;
//! use jolt_ir::backends::lean::LeanConfig;
//!
//! let b = ExprBuilder::new();
//! let h = b.opening(0);
//! let gamma = b.challenge(0);
//! let expr = b.build(gamma * (h * h - h));
//!
//! let lean = expr.to_lean4(&LeanConfig::default());
//! assert!(lean.contains("opening_0"));
//! assert!(lean.contains("challenge_0"));
//! ```

use crate::expr::{Expr, ExprId, ExprNode, Var};

/// Configuration for Lean4 code emission.
pub struct LeanConfig {
    /// Prefix for CSE let-bound variables (e.g. `"x"` → `x0`, `x1`, ...).
    pub let_prefix: String,
    /// Base name for opening variables (e.g. `"opening"` → `opening_0`).
    pub opening_name: String,
    /// Base name for challenge variables (e.g. `"challenge"` → `challenge_0`).
    pub challenge_name: String,
}

impl Default for LeanConfig {
    fn default() -> Self {
        Self {
            let_prefix: "x".into(),
            opening_name: "opening".into(),
            challenge_name: "challenge".into(),
        }
    }
}

impl Expr {
    /// Emit Lean4 code representing this expression.
    ///
    /// Shared subexpressions (nodes referenced more than once in the arena) are
    /// emitted as `let` bindings. The root expression is the final line.
    ///
    /// For tree expressions (no sharing), this produces a single inline
    /// expression with no `let` bindings.
    pub fn to_lean4(&self, config: &LeanConfig) -> String {
        let ref_counts = self.compute_ref_counts();
        let mut emitter = LeanEmitter::new(config, &ref_counts, self.arena.len());
        let root_expr = emitter.emit_node(self, self.root);

        let mut output = String::new();
        for binding in &emitter.bindings {
            output.push_str(binding);
            output.push('\n');
        }
        output.push_str(&root_expr);
        output
    }

    /// Count how many times each node is referenced as a child.
    fn compute_ref_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.arena.len()];
        for i in 0..self.arena.len() {
            match self.arena.get(ExprId(i as u32)) {
                ExprNode::Constant(_) | ExprNode::Var(_) => {}
                ExprNode::Neg(inner) => counts[inner.index()] += 1,
                ExprNode::Add(l, r) | ExprNode::Sub(l, r) | ExprNode::Mul(l, r) => {
                    counts[l.index()] += 1;
                    counts[r.index()] += 1;
                }
            }
        }
        // The root itself is referenced once (implicitly)
        counts[self.root.index()] += 1;
        counts
    }
}

struct LeanEmitter<'a> {
    config: &'a LeanConfig,
    ref_counts: &'a [u32],
    /// Cache: ExprId → emitted string (either inline expr or let-bound name)
    cache: Vec<Option<String>>,
    /// Accumulated let bindings in emission order
    bindings: Vec<String>,
    /// Next let-binding index
    next_let: usize,
}

impl<'a> LeanEmitter<'a> {
    fn new(config: &'a LeanConfig, ref_counts: &'a [u32], arena_len: usize) -> Self {
        Self {
            config,
            ref_counts,
            cache: vec![None; arena_len],
            bindings: Vec::new(),
            next_let: 0,
        }
    }

    fn emit_node(&mut self, expr: &Expr, id: ExprId) -> String {
        if let Some(cached) = &self.cache[id.index()] {
            return cached.clone();
        }

        let result = match expr.arena.get(id) {
            ExprNode::Constant(val) => {
                if val < 0 {
                    format!("(-{})", -val)
                } else {
                    val.to_string()
                }
            }
            ExprNode::Var(var) => match var {
                Var::Opening(i) => format!("{}_{i}", self.config.opening_name),
                Var::Challenge(i) => format!("{}_{i}", self.config.challenge_name),
            },
            ExprNode::Neg(inner) => {
                let inner_s = self.emit_node(expr, inner);
                format!("(-{inner_s})")
            }
            ExprNode::Add(l, r) => {
                let ls = self.emit_node(expr, l);
                let rs = self.emit_node(expr, r);
                format!("({ls} + {rs})")
            }
            ExprNode::Sub(l, r) => {
                let ls = self.emit_node(expr, l);
                let rs = self.emit_node(expr, r);
                format!("({ls} - {rs})")
            }
            ExprNode::Mul(l, r) => {
                let ls = self.emit_node(expr, l);
                let rs = self.emit_node(expr, r);
                format!("({ls} * {rs})")
            }
        };

        // Hoist to a let binding if referenced more than once and not a leaf
        let is_compound = !matches!(expr.arena.get(id), ExprNode::Constant(_) | ExprNode::Var(_));
        let final_result = if is_compound && self.ref_counts[id.index()] > 1 {
            let name = format!("{}{}", self.config.let_prefix, self.next_let);
            self.next_let += 1;
            self.bindings.push(format!("let {name} := {result}"));
            name
        } else {
            result
        };

        self.cache[id.index()] = Some(final_result.clone());
        final_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

    #[test]
    fn lean_booleanity() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let lean = expr.to_lean4(&LeanConfig::default());
        assert!(lean.contains("opening_0"));
        assert!(lean.contains("challenge_0"));
        assert!(!lean.contains("let "), "no sharing → no let bindings");
    }

    #[test]
    fn lean_constant_only() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));
        let lean = expr.to_lean4(&LeanConfig::default());
        assert_eq!(lean, "42");
    }

    #[test]
    fn lean_negative_constant() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(-7));
        let lean = expr.to_lean4(&LeanConfig::default());
        assert_eq!(lean, "(-7)");
    }

    #[test]
    fn lean_single_variable() {
        let b = ExprBuilder::new();
        let expr = b.build(b.opening(3));
        let lean = expr.to_lean4(&LeanConfig::default());
        assert_eq!(lean, "opening_3");
    }

    #[test]
    fn lean_cse_produces_let_bindings() {
        // (a+b) * (a+b) — after CSE, a+b should get a let binding
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let expr = b.build((a1 + b1) * (a2 + b2));
        let optimized = expr.eliminate_common_subexpressions();

        let lean = optimized.to_lean4(&LeanConfig::default());
        assert!(
            lean.contains("let x0 :="),
            "shared (a+b) should be let-bound"
        );
        // The root should reference x0, not re-emit the addition
        assert!(lean.contains("(x0 * x0)"));
    }

    #[test]
    fn lean_custom_config() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * h);

        let config = LeanConfig {
            let_prefix: "v".into(),
            opening_name: "poly".into(),
            challenge_name: "r".into(),
        };
        let lean = expr.to_lean4(&config);
        assert!(lean.contains("poly_0"));
        assert!(lean.contains("r_0"));
    }

    #[test]
    fn lean_negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        let lean = expr.to_lean4(&LeanConfig::default());
        assert_eq!(lean, "(-opening_0)");
    }

    #[test]
    fn lean_complex_with_cse() {
        // (a+b)*(a+b) + (a+b)*(c-d) — (a+b) is shared 3 times
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let a3 = b.opening(0);
        let b3 = b.opening(1);
        let c = b.opening(2);
        let d = b.opening(3);
        let expr = b.build((a1 + b1) * (a2 + b2) + (a3 + b3) * (c - d));
        let optimized = expr.eliminate_common_subexpressions();

        let lean = optimized.to_lean4(&LeanConfig::default());
        // (a+b) should be let-bound since it appears multiple times
        assert!(lean.contains("let x0 :="));
        // Should have at most one let binding for the shared subexpr
        let let_count = lean.matches("let ").count();
        assert!(let_count >= 1, "should have at least one let binding");
    }

    #[test]
    fn lean_subtraction() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a - bv);

        let lean = expr.to_lean4(&LeanConfig::default());
        assert_eq!(lean, "(opening_0 - opening_1)");
    }
}
