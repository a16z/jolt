//! Two-phase CSE and Go expression codegen.
//!
//! [`MemoizedCodeGen`] performs common subexpression elimination across
//! multiple constraints, then emits Go code with per-constraint CSE variable
//! namespacing.

use std::collections::BTreeMap;

use crate::bundle::{AstBundle, SerializedEdge, SerializedNode};

use super::ast_emitter::GnarkAstEmitter;

/// Two-phase codegen: first pass counts node uses for CSE, second pass emits.
pub struct MemoizedCodeGen {
    /// How many times each node is referenced (for CSE decisions).
    ref_counts: BTreeMap<u32, usize>,
    /// CSE variable prefix.
    cse_prefix: String,
}

impl MemoizedCodeGen {
    pub fn new() -> Self {
        Self {
            ref_counts: BTreeMap::new(),
            cse_prefix: "v".into(),
        }
    }

    pub fn with_cse_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.cse_prefix = prefix.into();
        self
    }

    pub fn ref_counts(&self) -> &BTreeMap<u32, usize> {
        &self.ref_counts
    }

    /// Phase 1: count references to identify shared subexpressions.
    pub fn count_refs(&mut self, bundle: &AstBundle) {
        for node in &bundle.nodes {
            self.count_edge_refs(node);
        }
        for assertion in &bundle.assertions {
            match assertion {
                crate::bundle::Assertion::Zero(e) => self.inc_ref(e),
                crate::bundle::Assertion::Equal(l, r) => {
                    self.inc_ref(l);
                    self.inc_ref(r);
                }
            }
        }
    }

    fn count_edge_refs(&mut self, node: &SerializedNode) {
        match node {
            SerializedNode::Var { .. } | SerializedNode::Challenge { .. } => {}
            SerializedNode::Neg(e)
            | SerializedNode::Inv(e)
            | SerializedNode::ByteReverse(e)
            | SerializedNode::Truncate128(e)
            | SerializedNode::MulTwoPow192(e) => self.inc_ref(e),
            SerializedNode::Add(l, r)
            | SerializedNode::Sub(l, r)
            | SerializedNode::Mul(l, r)
            | SerializedNode::Div(l, r) => {
                self.inc_ref(l);
                self.inc_ref(r);
            }
            SerializedNode::Poseidon {
                state,
                n_rounds,
                data,
            } => {
                self.inc_ref(state);
                self.inc_ref(n_rounds);
                self.inc_ref(data);
            }
        }
    }

    fn inc_ref(&mut self, edge: &SerializedEdge) {
        if let SerializedEdge::Node(id) = edge {
            *self.ref_counts.entry(*id).or_insert(0) += 1;
        }
    }

    /// Phase 2: emit Go code using `GnarkAstEmitter`.
    ///
    /// Returns the complete Go code block (assignments + assertions).
    pub fn emit(self, bundle: &AstBundle) -> String {
        let mut emitter = GnarkAstEmitter::new().with_cse_prefix(&self.cse_prefix);
        bundle.emit(&mut emitter);
        emitter.finish()
    }

    /// Emit to a provided `GnarkAstEmitter`, allowing external control.
    pub fn emit_to(self, bundle: &AstBundle, emitter: &mut GnarkAstEmitter) {
        bundle.emit(emitter);
    }
}

impl Default for MemoizedCodeGen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;
    use crate::bundle::VarAllocator;
    use crate::symbolic::SymbolicField;

    #[test]
    fn ref_counting() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        // x*y used twice (in two assertions)
        let product = x * y;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(product.into_edge());
        allocator.assert_zero(product.into_edge());

        let bundle = allocator.finish();

        let mut codegen = MemoizedCodeGen::new();
        codegen.count_refs(&bundle);

        assert!(!codegen.ref_counts().is_empty());
    }

    #[test]
    fn emit_produces_go_code() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let result = x + y;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(result.into_edge());

        let bundle = allocator.finish();

        let codegen = MemoizedCodeGen::new().with_cse_prefix("c0");
        let code = codegen.emit(&bundle);

        assert!(code.contains("api.Add("));
        assert!(code.contains("AssertIsEqual("));
    }
}
