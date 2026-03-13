use std::collections::BTreeMap;

use jolt_ir::{CircuitEmitter, Var};

/// Gnark Go code emitter for `jolt-ir` expressions.
///
/// Implements [`CircuitEmitter`] to produce gnark `frontend.API` calls.
/// Each arithmetic operation becomes a Go assignment with a CSE variable name,
/// and the final result is the root wire.
///
/// The emitter accumulates Go assignment lines in order. Call [`finish`] to
/// collect them into a complete Go code block.
///
/// # Constant handling
///
/// Constants in the range `i64::MIN..=i64::MAX` are emitted as Go integer
/// literals. Values outside that range use `bigInt("...")` — the caller must
/// ensure the `bigInt` helper exists in the generated Go file:
///
/// ```go
/// func bigInt(s string) *big.Int {
///     n, _ := new(big.Int).SetString(s, 10)
///     return n
/// }
/// ```
///
/// [`finish`]: GnarkEmitter::finish
pub struct GnarkEmitter {
    opening_names: BTreeMap<u32, String>,
    challenge_names: BTreeMap<u32, String>,
    cse_prefix: String,
    lines: Vec<String>,
    next_var: usize,
}

impl GnarkEmitter {
    pub fn new() -> Self {
        Self {
            opening_names: BTreeMap::new(),
            challenge_names: BTreeMap::new(),
            cse_prefix: "cse".into(),
            lines: Vec::new(),
            next_var: 0,
        }
    }

    /// Set a custom CSE prefix (e.g., `"cse_3"` for constraint 3).
    ///
    /// Per-constraint prefixing prevents variable name collisions when
    /// multiple expressions are emitted into the same Go function.
    pub fn with_cse_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.cse_prefix = prefix.into();
        self
    }

    /// Map an opening index to a custom Go field name.
    ///
    /// Without this, opening `i` becomes `circuit.Opening_i`.
    pub fn with_opening_name(mut self, index: u32, name: impl Into<String>) -> Self {
        let _ = self.opening_names.insert(index, name.into());
        self
    }

    /// Map a challenge index to a custom Go field name.
    ///
    /// Without this, challenge `i` becomes `circuit.Challenge_i`.
    pub fn with_challenge_name(mut self, index: u32, name: impl Into<String>) -> Self {
        let _ = self.challenge_names.insert(index, name.into());
        self
    }

    /// Collect all emitted Go assignments into a single code block.
    ///
    /// The `root` wire (returned by `to_circuit`) is not re-assigned here —
    /// it's already the last CSE variable. The caller can use it in an
    /// assertion like `api.AssertIsEqual(root, 0)`.
    pub fn finish(&self, _root: &str) -> String {
        self.lines.join("\n")
    }

    /// Collect all emitted lines plus an `api.AssertIsEqual(root, 0)` assertion.
    pub fn finish_with_assert_zero(&self, root: &str) -> String {
        let mut out = self.lines.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        use std::fmt::Write;
        let _ = write!(out, "api.AssertIsEqual({root}, 0)");
        out
    }

    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    fn alloc_var(&mut self) -> String {
        let name = format!("{}_{}", self.cse_prefix, self.next_var);
        self.next_var += 1;
        name
    }

    fn format_constant(val: i128) -> String {
        // Small constants: Go integer literal
        if val >= i64::MIN as i128 && val <= i64::MAX as i128 {
            return val.to_string();
        }
        // Large constants: bigInt("...") helper
        format!("bigInt(\"{val}\")")
    }

    fn opening_field_name(&self, index: u32) -> String {
        if let Some(name) = self.opening_names.get(&index) {
            format!("circuit.{name}")
        } else {
            format!("circuit.Opening_{index}")
        }
    }

    fn challenge_field_name(&self, index: u32) -> String {
        if let Some(name) = self.challenge_names.get(&index) {
            format!("circuit.{name}")
        } else {
            format!("circuit.Challenge_{index}")
        }
    }
}

impl Default for GnarkEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitEmitter for GnarkEmitter {
    type Wire = String;

    fn constant(&mut self, val: i128) -> String {
        // Constants are inlined — no CSE variable needed
        Self::format_constant(val)
    }

    fn variable(&mut self, var: Var) -> String {
        // Variables are circuit field references — no CSE variable needed
        match var {
            Var::Opening(i) => self.opening_field_name(i),
            Var::Challenge(i) => self.challenge_field_name(i),
        }
    }

    fn neg(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Neg({inner})"));
        v
    }

    fn add(&mut self, lhs: String, rhs: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Add({lhs}, {rhs})"));
        v
    }

    fn sub(&mut self, lhs: String, rhs: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Sub({lhs}, {rhs})"));
        v
    }

    fn mul(&mut self, lhs: String, rhs: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Mul({lhs}, {rhs})"));
        v
    }
}

/// Convert a Rust identifier to a valid Go exported name.
///
/// Non-alphanumeric characters become `_`, and each underscore-delimited
/// segment is PascalCased.
///
/// # Examples
///
/// ```
/// use jolt_wrapper::gnark::sanitize_go_name;
///
/// assert_eq!(sanitize_go_name("stage_1_hash"), "Stage_1_Hash");
/// assert_eq!(sanitize_go_name("commitment[0]"), "Commitment_0");
/// assert_eq!(sanitize_go_name("x-y"), "X_Y");
/// assert_eq!(sanitize_go_name("simple"), "Simple");
/// ```
pub fn sanitize_go_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();

    cleaned
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let mut result = first.to_uppercase().to_string();
                    result.extend(chars);
                    result
                }
            }
        })
        .collect::<Vec<_>>()
        .join("_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_ir::ExprBuilder;

    #[test]
    fn booleanity_codegen() {
        // γ · (H² − H)
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("api.Mul(circuit.Opening_0, circuit.Opening_0)"));
        assert!(code.contains("api.Sub("));
        assert!(code.contains("api.Mul(circuit.Challenge_0,"));
    }

    #[test]
    fn constant_only() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        assert_eq!(root, "42");
        assert!(emitter.lines().is_empty());
    }

    #[test]
    fn large_constant() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(i128::MAX));

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        assert!(root.starts_with("bigInt(\""));
        assert!(root.contains(&i128::MAX.to_string()));
    }

    #[test]
    fn negative_constant() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a + b.constant(-3));

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("api.Add(circuit.Opening_0, -3)"));
    }

    #[test]
    fn custom_opening_names() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let v = b.opening(1);
        let expr = b.build(h + v);

        let mut emitter = GnarkEmitter::new()
            .with_opening_name(0, "Stage1_H")
            .with_opening_name(1, "Stage1_V");
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("circuit.Stage1_H"));
        assert!(code.contains("circuit.Stage1_V"));
        assert!(!code.contains("Opening_"), "should use custom names");
        assert!(root.starts_with("cse_"));
    }

    #[test]
    fn custom_challenge_names() {
        let b = ExprBuilder::new();
        let gamma = b.challenge(0);
        let tau = b.challenge(1);
        let expr = b.build(gamma * tau);

        let mut emitter = GnarkEmitter::new()
            .with_challenge_name(0, "Gamma")
            .with_challenge_name(1, "Tau");
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("api.Mul(circuit.Gamma, circuit.Tau)"));
        assert!(root.starts_with("cse_"));
    }

    #[test]
    fn cse_prefix_scoping() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let c = b.opening(1);
        let expr = b.build(a + c);

        let mut emitter = GnarkEmitter::new().with_cse_prefix("cse_3");
        let root = expr.to_circuit(&mut emitter);

        assert!(root.starts_with("cse_3_"), "prefix should scope the vars");
    }

    #[test]
    fn cse_shares_subexpressions() {
        // (a+b) * (a+b) after CSE — (a+b) emitted once
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let expr = b.build((a1 + b1) * (a2 + b2));
        let optimized = expr.eliminate_common_subexpressions();

        let mut emitter = GnarkEmitter::new();
        let _root = optimized.to_circuit(&mut emitter);

        let add_count = emitter
            .lines()
            .iter()
            .filter(|l| l.contains("api.Add("))
            .count();
        assert_eq!(add_count, 1, "CSE should emit a+b only once");
    }

    #[test]
    fn assert_zero_output() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let c = b.opening(1);
        let expr = b.build(a - c);

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);
        let code = emitter.finish_with_assert_zero(&root);

        assert!(code.contains("api.Sub("));
        assert!(code.ends_with(&format!("api.AssertIsEqual({root}, 0)")));
    }

    #[test]
    fn negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("api.Neg(circuit.Opening_0)"));
    }

    #[test]
    fn complex_ram_style() {
        // init + gamma * (rs - ws) — typical RAM checking formula
        let b = ExprBuilder::new();
        let init = b.opening(0);
        let rs = b.opening(1);
        let ws = b.opening(2);
        let gamma = b.challenge(0);
        let expr = b.build(init + gamma * (rs - ws));

        let mut emitter = GnarkEmitter::new()
            .with_opening_name(0, "Init")
            .with_opening_name(1, "ReadSum")
            .with_opening_name(2, "WriteSum")
            .with_challenge_name(0, "Gamma");
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        assert!(code.contains("api.Sub(circuit.ReadSum, circuit.WriteSum)"));
        assert!(code.contains("api.Mul(circuit.Gamma,"));
        assert!(code.contains("api.Add(circuit.Init,"));
        assert!(root.starts_with("cse_"));
    }

    #[test]
    fn sanitize_go_name_cases() {
        assert_eq!(sanitize_go_name("stage_1_hash"), "Stage_1_Hash");
        assert_eq!(sanitize_go_name("x-y"), "X_Y");
        assert_eq!(sanitize_go_name("simple"), "Simple");
        assert_eq!(sanitize_go_name("ABC"), "ABC");
        assert_eq!(sanitize_go_name("a__b"), "A_B");
        assert_eq!(sanitize_go_name("123"), "123");
    }

    #[test]
    fn default_trait() {
        let emitter = GnarkEmitter::default();
        assert!(emitter.lines().is_empty());
    }

    #[test]
    fn weighted_sum_codegen() {
        // w0*o0 + w1*o1 + w2*o2 — weighted sum pattern
        let b = ExprBuilder::new();
        let o0 = b.opening(0);
        let o1 = b.opening(1);
        let o2 = b.opening(2);
        let w0 = b.challenge(0);
        let w1 = b.challenge(1);
        let w2 = b.challenge(2);
        let expr = b.build(w0 * o0 + w1 * o1 + w2 * o2);

        let mut emitter = GnarkEmitter::new();
        let root = expr.to_circuit(&mut emitter);

        let code = emitter.finish(&root);
        let mul_count = emitter
            .lines()
            .iter()
            .filter(|l| l.contains("api.Mul("))
            .count();
        let add_count = emitter
            .lines()
            .iter()
            .filter(|l| l.contains("api.Add("))
            .count();
        assert_eq!(mul_count, 3);
        assert_eq!(add_count, 2);
        assert!(root.starts_with("cse_"));
        // Verify the code is well-formed: every line has := assignment
        for line in emitter.lines() {
            assert!(line.contains(":="), "every line should be an assignment");
        }
        drop(code);
    }
}
