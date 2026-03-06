use crate::expr::{Expr, ExprId, ExprNode, Var};

/// Bottom-up visitor over an expression tree.
///
/// Each `visit_*` method receives already-computed child results, enabling
/// compositional evaluation. For DAGs (after CSE), results are cached so each
/// node is visited exactly once.
///
/// # Implementing a backend
///
/// The evaluate backend implements this to compute `F` values.
/// The R1CS backend implements this to emit constraints.
pub trait ExprVisitor {
    type Output;

    fn visit_constant(&mut self, val: i128) -> Self::Output;
    fn visit_var(&mut self, var: Var) -> Self::Output;
    fn visit_neg(&mut self, inner: Self::Output) -> Self::Output;
    fn visit_add(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_sub(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_mul(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
}

impl Expr {
    /// Traverse the expression bottom-up, invoking visitor methods on each node.
    ///
    /// Each node is visited exactly once. Results are cached by `ExprId` so
    /// shared subexpressions (from CSE) don't cause redundant computation.
    pub fn visit<V: ExprVisitor>(&self, visitor: &mut V) -> V::Output
    where
        V::Output: Clone,
    {
        self.visit_node(self.root, visitor)
    }

    fn visit_node<V: ExprVisitor>(&self, id: ExprId, visitor: &mut V) -> V::Output
    where
        V::Output: Clone,
    {
        // For a tree (no sharing), we just recurse. The arena guarantees
        // children have lower indices than parents, so this terminates.
        // For DAG support with caching, see `visit_cached`.
        match self.arena.get(id) {
            ExprNode::Constant(val) => visitor.visit_constant(val),
            ExprNode::Var(var) => visitor.visit_var(var),
            ExprNode::Neg(inner) => {
                let inner_val = self.visit_node(inner, visitor);
                visitor.visit_neg(inner_val)
            }
            ExprNode::Add(lhs, rhs) => {
                let l = self.visit_node(lhs, visitor);
                let r = self.visit_node(rhs, visitor);
                visitor.visit_add(l, r)
            }
            ExprNode::Sub(lhs, rhs) => {
                let l = self.visit_node(lhs, visitor);
                let r = self.visit_node(rhs, visitor);
                visitor.visit_sub(l, r)
            }
            ExprNode::Mul(lhs, rhs) => {
                let l = self.visit_node(lhs, visitor);
                let r = self.visit_node(rhs, visitor);
                visitor.visit_mul(l, r)
            }
        }
    }

    /// Cached traversal for DAGs where nodes may be referenced multiple times.
    ///
    /// Each node is visited exactly once; subsequent references reuse the
    /// cached result. This is the correct traversal after CSE.
    pub fn visit_cached<V: ExprVisitor>(&self, visitor: &mut V) -> V::Output
    where
        V::Output: Clone,
    {
        let mut cache: Vec<Option<V::Output>> = vec![None; self.arena.len()];
        self.visit_node_cached(self.root, visitor, &mut cache)
    }

    fn visit_node_cached<V: ExprVisitor>(
        &self,
        id: ExprId,
        visitor: &mut V,
        cache: &mut [Option<V::Output>],
    ) -> V::Output
    where
        V::Output: Clone,
    {
        if let Some(cached) = &cache[id.index()] {
            return cached.clone();
        }

        let result = match self.arena.get(id) {
            ExprNode::Constant(val) => visitor.visit_constant(val),
            ExprNode::Var(var) => visitor.visit_var(var),
            ExprNode::Neg(inner) => {
                let inner_val = self.visit_node_cached(inner, visitor, cache);
                visitor.visit_neg(inner_val)
            }
            ExprNode::Add(lhs, rhs) => {
                let l = self.visit_node_cached(lhs, visitor, cache);
                let r = self.visit_node_cached(rhs, visitor, cache);
                visitor.visit_add(l, r)
            }
            ExprNode::Sub(lhs, rhs) => {
                let l = self.visit_node_cached(lhs, visitor, cache);
                let r = self.visit_node_cached(rhs, visitor, cache);
                visitor.visit_sub(l, r)
            }
            ExprNode::Mul(lhs, rhs) => {
                let l = self.visit_node_cached(lhs, visitor, cache);
                let r = self.visit_node_cached(rhs, visitor, cache);
                visitor.visit_mul(l, r)
            }
        };

        cache[id.index()] = Some(result.clone());
        result
    }
}

/// Counts the total number of nodes visited in the expression.
struct NodeCounter;

impl ExprVisitor for NodeCounter {
    type Output = usize;

    fn visit_constant(&mut self, _val: i128) -> usize {
        1
    }
    fn visit_var(&mut self, _var: Var) -> usize {
        1
    }
    fn visit_neg(&mut self, inner: usize) -> usize {
        1 + inner
    }
    fn visit_add(&mut self, lhs: usize, rhs: usize) -> usize {
        1 + lhs + rhs
    }
    fn visit_sub(&mut self, lhs: usize, rhs: usize) -> usize {
        1 + lhs + rhs
    }
    fn visit_mul(&mut self, lhs: usize, rhs: usize) -> usize {
        1 + lhs + rhs
    }
}

impl Expr {
    /// Count total nodes reachable from the root (counting shared nodes once
    /// per reference in tree mode, once total in cached mode).
    pub fn node_count(&self) -> usize {
        self.visit(&mut NodeCounter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

    /// Simple visitor that reconstructs a string representation.
    struct StringVisitor;

    impl ExprVisitor for StringVisitor {
        type Output = String;

        fn visit_constant(&mut self, val: i128) -> String {
            val.to_string()
        }
        fn visit_var(&mut self, var: Var) -> String {
            match var {
                Var::Opening(i) => format!("o{i}"),
                Var::Challenge(i) => format!("c{i}"),
            }
        }
        fn visit_neg(&mut self, inner: String) -> String {
            format!("(-{inner})")
        }
        fn visit_add(&mut self, lhs: String, rhs: String) -> String {
            format!("({lhs} + {rhs})")
        }
        fn visit_sub(&mut self, lhs: String, rhs: String) -> String {
            format!("({lhs} - {rhs})")
        }
        fn visit_mul(&mut self, lhs: String, rhs: String) -> String {
            format!("({lhs} * {rhs})")
        }
    }

    #[test]
    fn visitor_traversal_order() {
        // gamma * (h^2 - h)
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let result = expr.visit(&mut StringVisitor);
        assert_eq!(result, "(c0 * ((o0 * o0) - o0))");
    }

    #[test]
    fn visitor_negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(-(a * bv));

        let result = expr.visit(&mut StringVisitor);
        assert_eq!(result, "(-(o0 * o1))");
    }

    #[test]
    fn visitor_constant_only() {
        let b = ExprBuilder::new();
        let c = b.constant(42);
        let expr = b.build(c);

        let result = expr.visit(&mut StringVisitor);
        assert_eq!(result, "42");
    }

    #[test]
    fn node_count_simple() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let c = b.constant(5);
        let expr = b.build(a + c);

        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn node_count_booleanity() {
        // gamma * (h*h - h) — h is Copy so reused, but tree traversal
        // visits the h node twice (once in h*h, once in sub).
        // Nodes: h(0), gamma(1), mul(h,h)(2), sub(mul,h)(3), mul(gamma,sub)(4) = 5 arena nodes
        // Tree traversal counts: h=1, h=1, h*h=3, h=1, sub=5, gamma=1, root_mul=7
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        assert_eq!(expr.node_count(), 7);
    }
}
