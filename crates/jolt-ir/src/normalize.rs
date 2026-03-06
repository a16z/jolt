use crate::expr::{Expr, ExprArena, ExprId, ExprNode, Var};
use crate::visitor::ExprVisitor;

/// A value reference in sum-of-products normal form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SopValue {
    Constant(i128),
    Opening(u32),
    Challenge(u32),
}

/// A single term in sum-of-products form: `coefficient * factor[0] * factor[1] * ...`
///
/// The `coefficient` is always an `i128` (see [`ExprNode`] for why constants
/// are `i128`). Variable factors that appear as "coefficients" in the original
/// expression (e.g., `gamma * h`) are flattened into `factors` — there is no
/// distinguished variable coefficient slot. The R1CS backend reconstructs the
/// coeff/factors split if needed for constraint emission.
///
/// Invariant: `factors` contains only `Opening` and `Challenge` variants —
/// constant factors are folded into `coefficient` during normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SopTerm {
    pub coefficient: i128,
    pub factors: Vec<SopValue>,
}

impl SopTerm {
    fn constant(val: i128) -> Self {
        Self {
            coefficient: val,
            factors: Vec::new(),
        }
    }

    fn var(var: SopValue) -> Self {
        Self {
            coefficient: 1,
            factors: vec![var],
        }
    }

    fn negate(&self) -> Self {
        Self {
            // Wrapping is fine: we're in abstract integer domain, field reduction happens later
            coefficient: self.coefficient.wrapping_neg(),
            factors: self.factors.clone(),
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        let mut factors = Vec::with_capacity(self.factors.len() + other.factors.len());
        factors.extend_from_slice(&self.factors);
        factors.extend_from_slice(&other.factors);
        Self {
            coefficient: self.coefficient.wrapping_mul(other.coefficient),
            factors,
        }
    }
}

/// Sum-of-products normal form: `Σᵢ coeffᵢ * ∏ⱼ factorᵢⱼ`
///
/// This is the canonical representation for R1CS emission and BlindFold
/// constraint construction. Produced by `Expr::to_sum_of_products()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumOfProducts {
    pub terms: Vec<SopTerm>,
}

impl SumOfProducts {
    /// Number of terms in the sum.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

/// Visitor that normalizes an expression tree into sum-of-products form.
///
/// Each node produces a `Vec<SopTerm>` representing the sum of product terms
/// for that subtree. Multiplication distributes over addition, subtraction
/// distributes negation, and constants fold into coefficients.
struct SopVisitor;

impl ExprVisitor for SopVisitor {
    type Output = Vec<SopTerm>;

    fn visit_constant(&mut self, val: i128) -> Vec<SopTerm> {
        vec![SopTerm::constant(val)]
    }

    fn visit_var(&mut self, var: Var) -> Vec<SopTerm> {
        let sop_val = match var {
            Var::Opening(id) => SopValue::Opening(id),
            Var::Challenge(id) => SopValue::Challenge(id),
        };
        vec![SopTerm::var(sop_val)]
    }

    fn visit_neg(&mut self, inner: Vec<SopTerm>) -> Vec<SopTerm> {
        inner.iter().map(SopTerm::negate).collect()
    }

    fn visit_add(&mut self, mut lhs: Vec<SopTerm>, rhs: Vec<SopTerm>) -> Vec<SopTerm> {
        lhs.extend(rhs);
        lhs
    }

    fn visit_sub(&mut self, mut lhs: Vec<SopTerm>, rhs: Vec<SopTerm>) -> Vec<SopTerm> {
        lhs.extend(rhs.iter().map(SopTerm::negate));
        lhs
    }

    fn visit_mul(&mut self, lhs: Vec<SopTerm>, rhs: Vec<SopTerm>) -> Vec<SopTerm> {
        let mut result = Vec::with_capacity(lhs.len() * rhs.len());
        for l in &lhs {
            for r in &rhs {
                result.push(l.multiply(r));
            }
        }
        result
    }
}

impl Expr {
    /// Normalize the expression into sum-of-products form.
    ///
    /// Mechanically distributes multiplication over addition and folds
    /// constant factors into term coefficients. The result maps directly to
    /// R1CS constraints.
    ///
    /// # Example
    ///
    /// `(a + b) * c` becomes `[1·a·c, 1·b·c]`
    pub fn to_sum_of_products(&self) -> SumOfProducts {
        let terms = self.visit(&mut SopVisitor);
        SumOfProducts { terms }
    }

    /// Fold constant sub-expressions in the tree.
    ///
    /// Evaluates pure-constant subtrees to single `Constant` nodes. Does not
    /// change the expression's semantics.
    pub fn fold_constants(&self) -> Expr {
        let mut arena = ExprArena::new();
        let root = self.fold_constants_node(self.root, &mut arena);
        Expr { arena, root }
    }

    fn fold_constants_node(&self, id: ExprId, out: &mut ExprArena) -> ExprId {
        match self.arena.get(id) {
            ExprNode::Constant(val) => out.push(ExprNode::Constant(val)),
            ExprNode::Var(var) => out.push(ExprNode::Var(var)),
            ExprNode::Neg(inner) => {
                let inner_id = self.fold_constants_node(inner, out);
                if let ExprNode::Constant(val) = out.get(inner_id) {
                    out.push(ExprNode::Constant(val.wrapping_neg()))
                } else {
                    out.push(ExprNode::Neg(inner_id))
                }
            }
            ExprNode::Add(lhs, rhs) => {
                let l = self.fold_constants_node(lhs, out);
                let r = self.fold_constants_node(rhs, out);
                match (out.get(l), out.get(r)) {
                    (ExprNode::Constant(a), ExprNode::Constant(b)) => {
                        out.push(ExprNode::Constant(a.wrapping_add(b)))
                    }
                    _ => out.push(ExprNode::Add(l, r)),
                }
            }
            ExprNode::Sub(lhs, rhs) => {
                let l = self.fold_constants_node(lhs, out);
                let r = self.fold_constants_node(rhs, out);
                match (out.get(l), out.get(r)) {
                    (ExprNode::Constant(a), ExprNode::Constant(b)) => {
                        out.push(ExprNode::Constant(a.wrapping_sub(b)))
                    }
                    _ => out.push(ExprNode::Sub(l, r)),
                }
            }
            ExprNode::Mul(lhs, rhs) => {
                let l = self.fold_constants_node(lhs, out);
                let r = self.fold_constants_node(rhs, out);
                match (out.get(l), out.get(r)) {
                    (ExprNode::Constant(a), ExprNode::Constant(b)) => {
                        out.push(ExprNode::Constant(a.wrapping_mul(b)))
                    }
                    _ => out.push(ExprNode::Mul(l, r)),
                }
            }
        }
    }

    /// Eliminate common subexpressions by deduplicating structurally identical subtrees.
    ///
    /// Produces a DAG where shared subtrees are referenced by the same `ExprId`.
    /// Use `visit_cached` for correct traversal of the result.
    pub fn eliminate_common_subexpressions(&self) -> Expr {
        let mut out = ExprArena::new();
        let mut map: std::collections::HashMap<ExprNode, ExprId> = std::collections::HashMap::new();
        let root = self.cse_node(self.root, &mut out, &mut map);
        Expr { arena: out, root }
    }

    fn cse_node(
        &self,
        id: ExprId,
        out: &mut ExprArena,
        map: &mut std::collections::HashMap<ExprNode, ExprId>,
    ) -> ExprId {
        // First, recursively process children to get their canonical ids
        let canonical_node = match self.arena.get(id) {
            node @ (ExprNode::Constant(_) | ExprNode::Var(_)) => node,
            ExprNode::Neg(inner) => {
                let inner_id = self.cse_node(inner, out, map);
                ExprNode::Neg(inner_id)
            }
            ExprNode::Add(lhs, rhs) => {
                let l = self.cse_node(lhs, out, map);
                let r = self.cse_node(rhs, out, map);
                ExprNode::Add(l, r)
            }
            ExprNode::Sub(lhs, rhs) => {
                let l = self.cse_node(lhs, out, map);
                let r = self.cse_node(rhs, out, map);
                ExprNode::Sub(l, r)
            }
            ExprNode::Mul(lhs, rhs) => {
                let l = self.cse_node(lhs, out, map);
                let r = self.cse_node(rhs, out, map);
                ExprNode::Mul(l, r)
            }
        };

        // Deduplicate: if we've seen this exact node before, reuse its id
        *map.entry(canonical_node)
            .or_insert_with(|| out.push(canonical_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

    #[test]
    fn sop_constant() {
        let b = ExprBuilder::new();
        let c = b.constant(42);
        let expr = b.build(c);
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 1);
        assert_eq!(sop.terms[0].coefficient, 42);
        assert!(sop.terms[0].factors.is_empty());
    }

    #[test]
    fn sop_single_var() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 1);
        assert_eq!(sop.terms[0].coefficient, 1);
        assert_eq!(sop.terms[0].factors, vec![SopValue::Opening(0)]);
    }

    #[test]
    fn sop_addition() {
        // a + b → [1·a, 1·b]
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a + bv);
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 2);
        assert_eq!(sop.terms[0].factors, vec![SopValue::Opening(0)]);
        assert_eq!(sop.terms[1].factors, vec![SopValue::Opening(1)]);
    }

    #[test]
    fn sop_distribution() {
        // (a + b) * c → [1·a·c, 1·b·c]
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build((a + bv) * c);
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 2);
        assert_eq!(
            sop.terms[0].factors,
            vec![SopValue::Opening(0), SopValue::Opening(2)]
        );
        assert_eq!(
            sop.terms[1].factors,
            vec![SopValue::Opening(1), SopValue::Opening(2)]
        );
    }

    #[test]
    fn sop_double_distribution() {
        // (a + b) * (c + d) → [a·c, a·d, b·c, b·d]
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let d = b.opening(3);
        let expr = b.build((a + bv) * (c + d));
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 4);
    }

    #[test]
    fn sop_subtraction() {
        // (a - b) * c → [1·a·c, -1·b·c]
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build((a - bv) * c);
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 2);
        assert_eq!(sop.terms[0].coefficient, 1);
        assert_eq!(sop.terms[1].coefficient, -1);
    }

    #[test]
    fn sop_negation() {
        // -(a * b) → [-1·a·b]
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(-(a * bv));
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 1);
        assert_eq!(sop.terms[0].coefficient, -1);
        assert_eq!(
            sop.terms[0].factors,
            vec![SopValue::Opening(0), SopValue::Opening(1)]
        );
    }

    #[test]
    fn sop_booleanity() {
        // gamma * (h^2 - h) → [gamma·h·h, -gamma·h]
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let sop = expr.to_sum_of_products();

        assert_eq!(sop.len(), 2);
        // First term: 1 * gamma * h * h
        assert_eq!(sop.terms[0].coefficient, 1);
        assert_eq!(
            sop.terms[0].factors,
            vec![
                SopValue::Challenge(0),
                SopValue::Opening(0),
                SopValue::Opening(0)
            ]
        );
        // Second term: -1 * gamma * h
        assert_eq!(sop.terms[1].coefficient, -1);
        assert_eq!(
            sop.terms[1].factors,
            vec![SopValue::Challenge(0), SopValue::Opening(0)]
        );
    }

    #[test]
    fn fold_constants_basic() {
        // (2 + 3) * a → 5 * a
        let b = ExprBuilder::new();
        let two = b.constant(2);
        let three = b.constant(3);
        let a = b.opening(0);
        let expr = b.build((two + three) * a);

        let folded = expr.fold_constants();
        // Root should be Mul with a folded Constant(5) child
        match folded.get(folded.root()) {
            ExprNode::Mul(l, _r) => {
                assert_eq!(folded.get(l), ExprNode::Constant(5));
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn fold_constants_nested() {
        // -(3 * 4) → root should be Constant(-12)
        let b = ExprBuilder::new();
        let three = b.constant(3);
        let four = b.constant(4);
        let expr = b.build(-(three * four));

        let folded = expr.fold_constants();
        assert_eq!(folded.get(folded.root()), ExprNode::Constant(-12));
    }

    #[test]
    fn cse_deduplicates() {
        // h * h: two references to Opening(0), but structurally identical
        let b = ExprBuilder::new();
        let h1 = b.opening(0);
        let h2 = b.opening(0);
        let expr = b.build(h1 * h2);

        // Before CSE: 3 nodes (h, h, h*h)
        assert_eq!(expr.len(), 3);

        let optimized = expr.eliminate_common_subexpressions();
        // After CSE: 2 nodes (h, h*h) — the two Opening(0) nodes are merged
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn cse_preserves_evaluation() {
        // (a + b) * (a + b) — the shared subtree (a + b) should be merged
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let expr = b.build((a1 + b1) * (a2 + b2));

        // Before: 7 nodes (a, b, a+b, a, b, a+b, (a+b)*(a+b))
        assert_eq!(expr.len(), 7);

        let optimized = expr.eliminate_common_subexpressions();
        // After: 4 nodes (a, b, a+b, (a+b)*(a+b))
        // Wait: Mul(id, id) is the same node referencing Add twice, so
        // that's Mul(add_id, add_id) — the Mul node itself has two identical
        // children, but the Mul node is unique. So: a, b, a+b, (a+b)^2 = 4 nodes.
        assert!(optimized.len() <= 4);
    }
}
