//! Gnark implementation of [`AstEmitter`].
//!
//! Emits gnark `frontend.API` calls from an [`AstBundle`](crate::bundle::AstBundle).
//! Each operation becomes a Go assignment with a CSE variable name.

use std::collections::HashMap;

use crate::ast_emitter::AstEmitter;
use crate::scalar_ops;

/// Gnark `AstEmitter` that produces Go `frontend.API` calls.
///
/// Emits one Go assignment per arena node. Constants are inlined as Go integer
/// literals or `bigInt("...")` for large values. Variables reference circuit
/// struct fields.
pub struct GnarkAstEmitter {
    /// Custom names for variables by index.
    var_names: HashMap<u32, String>,
    /// CSE variable prefix.
    cse_prefix: String,
    /// Accumulated Go assignment lines.
    lines: Vec<String>,
    /// Assertion lines.
    assertions: Vec<String>,
    /// Next CSE variable counter.
    next_var: usize,
}

impl GnarkAstEmitter {
    pub fn new() -> Self {
        Self {
            var_names: HashMap::new(),
            cse_prefix: "v".into(),
            lines: Vec::new(),
            assertions: Vec::new(),
            next_var: 0,
        }
    }

    /// Set a custom CSE prefix.
    pub fn with_cse_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.cse_prefix = prefix.into();
        self
    }

    /// Map a variable index to a custom Go field name.
    pub fn with_var_name(mut self, index: u32, name: impl Into<String>) -> Self {
        let _ = self.var_names.insert(index, name.into());
        self
    }

    /// Return the accumulated Go assignment lines.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Return the accumulated assertion lines.
    pub fn assertion_lines(&self) -> &[String] {
        &self.assertions
    }

    /// Collect all lines (assignments + assertions) into a Go code block.
    pub fn finish(&self) -> String {
        let mut out = self.lines.join("\n");
        if !self.assertions.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&self.assertions.join("\n"));
        }
        out
    }

    fn alloc_var(&mut self) -> String {
        let name = format!("{}_{}", self.cse_prefix, self.next_var);
        self.next_var += 1;
        name
    }

    fn format_constant(val: [u64; 4]) -> String {
        // Try to fit in i64 first
        if val[1] == 0 && val[2] == 0 && val[3] == 0 && val[0] <= i64::MAX as u64 {
            return val[0].to_string();
        }
        // Large constant: use bigInt("decimal_string")
        let decimal = scalar_ops::to_decimal_string(val);
        format!("bigInt(\"{decimal}\")")
    }

    fn var_field_name(&self, index: u32, name: &str) -> String {
        if let Some(custom) = self.var_names.get(&index) {
            format!("circuit.{custom}")
        } else {
            let go_name = super::emitter::sanitize_go_name(name);
            format!("circuit.{go_name}")
        }
    }
}

impl Default for GnarkAstEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl AstEmitter for GnarkAstEmitter {
    type Wire = String;

    fn constant(&mut self, val: [u64; 4]) -> String {
        Self::format_constant(val)
    }

    fn variable(&mut self, index: u32, name: &str) -> String {
        self.var_field_name(index, name)
    }

    fn neg(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Neg({inner})"));
        v
    }

    fn inv(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Inverse({inner})"));
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

    fn div(&mut self, lhs: String, rhs: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := api.Div({lhs}, {rhs})"));
        v
    }

    fn poseidon(&mut self, state: String, n_rounds: String, data: String) -> String {
        let v = self.alloc_var();
        self.lines
            .push(format!("{v} := poseidonHash({state}, {n_rounds}, {data})"));
        v
    }

    fn byte_reverse(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := byteReverse({inner})"));
        v
    }

    fn truncate_128(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := truncate128({inner})"));
        v
    }

    fn mul_two_pow_192(&mut self, inner: String) -> String {
        let v = self.alloc_var();
        self.lines.push(format!("{v} := mulTwoPow192({inner})"));
        v
    }

    fn assert_zero(&mut self, expr: String) {
        self.assertions
            .push(format!("api.AssertIsEqual({expr}, 0)"));
    }

    fn assert_equal(&mut self, lhs: String, rhs: String) {
        self.assertions
            .push(format!("api.AssertIsEqual({lhs}, {rhs})"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;
    use crate::ast_emitter::AstEmitter;
    use crate::bundle::VarAllocator;
    use crate::symbolic::SymbolicField;

    #[test]
    fn small_constant() {
        let mut emitter = GnarkAstEmitter::new();
        let wire = emitter.constant(scalar_ops::from_u64(42));
        assert_eq!(wire, "42");
    }

    #[test]
    fn large_constant() {
        let mut emitter = GnarkAstEmitter::new();
        let wire = emitter.constant(scalar_ops::MODULUS);
        assert!(wire.starts_with("bigInt(\""));
    }

    #[test]
    fn variable_default_name() {
        let mut emitter = GnarkAstEmitter::new();
        let wire = emitter.variable(0, "my_var");
        assert_eq!(wire, "circuit.My_Var");
    }

    #[test]
    fn variable_custom_name() {
        let mut emitter = GnarkAstEmitter::new().with_var_name(0, "CustomField");
        let wire = emitter.variable(0, "ignored");
        assert_eq!(wire, "circuit.CustomField");
    }

    #[test]
    fn arithmetic_lines() {
        let mut emitter = GnarkAstEmitter::new();
        let a = emitter.variable(0, "a");
        let b = emitter.variable(1, "b");
        let sum = emitter.add(a, b);
        assert!(emitter.lines().last().unwrap().contains("api.Add("));
        assert!(sum.starts_with("v_"));
    }

    #[test]
    fn assertions() {
        let mut emitter = GnarkAstEmitter::new();
        let a = emitter.variable(0, "a");
        let b = emitter.variable(1, "b");
        emitter.assert_equal(a.clone(), b.clone());
        emitter.assert_zero(a);

        assert_eq!(emitter.assertion_lines().len(), 2);
        assert!(emitter.assertion_lines()[0].contains("AssertIsEqual"));
        assert!(emitter.assertion_lines()[1].contains("AssertIsEqual"));
    }

    #[test]
    fn full_pipeline() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let result = x * y;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(result.into_edge());

        let bundle = allocator.finish();

        let mut emitter = GnarkAstEmitter::new();
        bundle.emit(&mut emitter);

        let code = emitter.finish();
        assert!(code.contains("api.Mul("));
        assert!(code.contains("AssertIsEqual("));
    }

    #[test]
    fn byte_reverse_codegen() {
        let _session = ArenaSession::new();
        use crate::arena::{self, Atom, Node};

        let x = SymbolicField::variable(0, "x");
        let reversed_id = arena::alloc(Node::ByteReverse(x.into_edge()));
        let reversed_edge = Atom::Node(reversed_id);

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        allocator.assert_zero(reversed_edge);

        let bundle = allocator.finish();
        let mut emitter = GnarkAstEmitter::new();
        bundle.emit(&mut emitter);

        let code = emitter.finish();
        assert!(
            code.contains("byteReverse("),
            "expected byteReverse call: {code}"
        );
        assert!(
            code.contains("circuit.X"),
            "expected circuit.X reference: {code}"
        );
        assert!(code.contains("AssertIsEqual("));
    }

    #[test]
    fn truncate_128_codegen() {
        let _session = ArenaSession::new();
        use crate::arena::{self, Atom, Node};

        let x = SymbolicField::variable(0, "x");
        let truncated_id = arena::alloc(Node::Truncate128(x.into_edge()));
        let truncated_edge = Atom::Node(truncated_id);

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        allocator.assert_zero(truncated_edge);

        let bundle = allocator.finish();
        let mut emitter = GnarkAstEmitter::new();
        bundle.emit(&mut emitter);

        let code = emitter.finish();
        assert!(
            code.contains("truncate128("),
            "expected truncate128 call: {code}"
        );
        assert!(
            code.contains("circuit.X"),
            "expected circuit.X reference: {code}"
        );
        assert!(code.contains("AssertIsEqual("));
    }

    #[test]
    fn mul_two_pow_192_codegen() {
        let _session = ArenaSession::new();
        use crate::arena::{self, Atom, Node};

        let x = SymbolicField::variable(0, "x");
        let scaled_id = arena::alloc(Node::MulTwoPow192(x.into_edge()));
        let scaled_edge = Atom::Node(scaled_id);

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        allocator.assert_zero(scaled_edge);

        let bundle = allocator.finish();
        let mut emitter = GnarkAstEmitter::new();
        bundle.emit(&mut emitter);

        let code = emitter.finish();
        assert!(
            code.contains("mulTwoPow192("),
            "expected mulTwoPow192 call: {code}"
        );
        assert!(
            code.contains("circuit.X"),
            "expected circuit.X reference: {code}"
        );
        assert!(code.contains("AssertIsEqual("));
    }

    #[test]
    fn special_ops_compose_with_arithmetic() {
        let _session = ArenaSession::new();
        use crate::arena::{self, Atom, Node};

        // byte_reverse(x) * truncate128(y) + mul_two_pow_192(x)
        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");

        let rev_id = arena::alloc(Node::ByteReverse(x.into_edge()));
        let trunc_id = arena::alloc(Node::Truncate128(y.into_edge()));
        let scaled_id = arena::alloc(Node::MulTwoPow192(x.into_edge()));

        let rev = SymbolicField::from_edge(Atom::Node(rev_id));
        let trunc = SymbolicField::from_edge(Atom::Node(trunc_id));
        let scaled = SymbolicField::from_edge(Atom::Node(scaled_id));

        let result = rev * trunc + scaled;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(result.into_edge());

        let bundle = allocator.finish();
        let mut emitter = GnarkAstEmitter::new();
        bundle.emit(&mut emitter);

        let code = emitter.finish();
        assert!(code.contains("byteReverse("));
        assert!(code.contains("truncate128("));
        assert!(code.contains("mulTwoPow192("));
        assert!(code.contains("api.Mul("));
        assert!(code.contains("api.Add("));
    }
}
