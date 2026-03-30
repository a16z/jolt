use jolt_field::Field;

use crate::expr::{Expr, ExprArena, ExprId, ExprNode, Var};
use crate::visitor::ExprVisitor;

/// A variable reference in a composition formula term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Factor {
    /// Reference to an input polynomial buffer by index.
    Input(u32),
    /// Reference to a symbolic challenge slot, resolved at runtime.
    Challenge(u32),
}

/// A single term: `coefficient × factor[0] × factor[1] × …`
///
/// Constants from the original expression are folded into `coefficient`
/// during normalization — `factors` contains only [`Input`](Factor::Input)
/// and [`Challenge`](Factor::Challenge) references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductTerm {
    pub coefficient: i128,
    pub factors: Vec<Factor>,
}

impl ProductTerm {
    fn constant(val: i128) -> Self {
        Self {
            coefficient: val,
            factors: Vec::new(),
        }
    }

    fn var(var: Factor) -> Self {
        Self {
            coefficient: 1,
            factors: vec![var],
        }
    }

    fn negate(&self) -> Self {
        Self {
            // Wrapping is fine: abstract integer domain, field reduction happens later
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

    /// Evaluate this term with concrete input and challenge values.
    pub fn evaluate<F: Field>(&self, inputs: &[F], challenges: &[F]) -> F {
        let mut val = F::from_i128(self.coefficient);
        for factor in &self.factors {
            val *= match factor {
                Factor::Input(i) => inputs[*i as usize],
                Factor::Challenge(i) => challenges[*i as usize],
            };
        }
        val
    }

    /// Number of [`Factor::Input`] references in this term.
    #[inline]
    pub fn input_degree(&self) -> usize {
        self.factors
            .iter()
            .filter(|f| matches!(f, Factor::Input(_)))
            .count()
    }
}

/// Normalized sum-of-products representation of a composition polynomial.
///
/// `Σᵢ coeffᵢ × ∏ⱼ factorᵢⱼ`
///
/// This is the canonical form consumed by compute backends for kernel
/// compilation, R1CS emission, and BlindFold constraint construction.
/// Produced from an [`Expr`] via [`Expr::to_composition_formula()`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionFormula {
    pub terms: Vec<ProductTerm>,
    /// Number of distinct input polynomial slots.
    pub num_inputs: usize,
    /// Number of distinct challenge slots.
    pub num_challenges: usize,
}

impl CompositionFormula {
    /// Build from terms, computing `num_inputs` and `num_challenges` from the terms.
    pub fn from_terms(terms: Vec<ProductTerm>) -> Self {
        let mut max_input: Option<u32> = None;
        let mut max_challenge: Option<u32> = None;
        for term in &terms {
            for factor in &term.factors {
                match factor {
                    Factor::Input(id) => {
                        max_input = Some(max_input.map_or(*id, |m: u32| m.max(*id)));
                    }
                    Factor::Challenge(id) => {
                        max_challenge = Some(max_challenge.map_or(*id, |m: u32| m.max(*id)));
                    }
                }
            }
        }
        Self {
            terms,
            num_inputs: max_input.map_or(0, |m| m as usize + 1),
            num_challenges: max_challenge.map_or(0, |m| m as usize + 1),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Maximum input-variable degree across all terms.
    ///
    /// This is the degree of the composition as a polynomial in the input
    /// variables (challenge factors are runtime constants, not variables).
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.input_degree())
            .max()
            .unwrap_or(0)
    }

    /// Evaluate with concrete input and challenge values.
    pub fn evaluate<F: Field>(&self, inputs: &[F], challenges: &[F]) -> F {
        self.terms
            .iter()
            .map(|t| t.evaluate(inputs, challenges))
            .sum()
    }

    /// Detect a pure product-sum structure.
    ///
    /// Returns `Some((d, p))` if every term is a product of exactly `d` input
    /// factors with coefficient 1 and no challenge factors, and there are `p`
    /// such terms. This pattern maps to Toom-Cook evaluation.
    pub fn as_product_sum(&self) -> Option<(usize, usize)> {
        if self.terms.is_empty() {
            return None;
        }
        let d = self.terms[0].factors.len();
        if d == 0 {
            return None;
        }
        for term in &self.terms {
            if term.coefficient != 1 {
                return None;
            }
            if term.factors.len() != d {
                return None;
            }
            if term
                .factors
                .iter()
                .any(|f| matches!(f, Factor::Challenge(_)))
            {
                return None;
            }
        }
        Some((d, self.terms.len()))
    }

    /// Detect a linear-combination pattern: every non-zero term has exactly
    /// one input factor and at least one challenge factor.
    ///
    /// This pattern enables pre-combination: `g[i] = Σ_j weight_j * poly_j[i]`
    /// so the kernel only computes `eq × g`.
    pub fn is_linear_combination(&self) -> bool {
        let mut has_nonzero = false;
        for term in &self.terms {
            if term.coefficient == 0 {
                continue;
            }
            has_nonzero = true;
            let n_inputs = term
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Input(_)))
                .count();
            let n_challenges = term
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Challenge(_)))
                .count();
            if n_inputs != 1 || n_challenges < 1 {
                return false;
            }
        }
        has_nonzero
    }

    /// Detect the `eq(x) · g(x)` pattern: a single degree-2 term
    /// `Input(0) × Input(1)` with unit coefficient and no challenge factors.
    ///
    /// This formula matches `as_product_sum()` (d=2, p=1) but uses a
    /// standard-grid kernel, not Toom-Cook.
    pub fn is_eq_product(&self) -> bool {
        self.terms.len() == 1 && {
            let t = &self.terms[0];
            t.coefficient == 1
                && t.factors.len() == 2
                && t.factors.contains(&Factor::Input(0))
                && t.factors.contains(&Factor::Input(1))
        }
    }

    /// Detect a Hamming booleanity pattern: `challenge × input × (input − 1)`.
    ///
    /// In normalized form this is exactly 2 terms over 1 distinct input:
    /// one with `input²` and one with `input`, each scaled by a challenge.
    pub fn is_hamming_booleanity(&self) -> bool {
        if self.terms.len() != 2 {
            return false;
        }

        let input_factors = |idx: usize| -> Vec<u32> {
            self.terms[idx]
                .factors
                .iter()
                .filter_map(|f| match f {
                    Factor::Input(id) => Some(*id),
                    Factor::Challenge(_) => None,
                })
                .collect()
        };
        let challenge_count = |idx: usize| -> usize {
            self.terms[idx]
                .factors
                .iter()
                .filter(|f| matches!(f, Factor::Challenge(_)))
                .count()
        };

        let i0 = input_factors(0);
        let i1 = input_factors(1);

        let (sq_idx, lin_idx) = if i0.len() == 2 && i1.len() == 1 {
            (0, 1)
        } else if i0.len() == 1 && i1.len() == 2 {
            (1, 0)
        } else {
            return false;
        };

        let sq = input_factors(sq_idx);
        let lin = input_factors(lin_idx);

        // Same input variable in both terms
        sq[0] == sq[1] && sq[0] == lin[0]
            // Both terms have exactly 1 challenge factor
            && challenge_count(sq_idx) == 1
            && challenge_count(lin_idx) == 1
    }

    /// Per-input pre-combination weights for linear-combination formulas.
    ///
    /// For each term `coeff × challenge_product × input(k)`, computes
    /// `coeff × Π challenges[c_id]` and accumulates by input index.
    /// Panics if `!self.is_linear_combination()`.
    pub fn linear_combination_weights<F: Field>(&self, challenges: &[F]) -> Vec<F> {
        debug_assert!(self.is_linear_combination());
        let mut weights = vec![F::zero(); self.num_inputs];
        for term in &self.terms {
            if term.coefficient == 0 {
                continue;
            }
            let input_id = term
                .factors
                .iter()
                .find_map(|f| match f {
                    Factor::Input(id) => Some(*id as usize),
                    Factor::Challenge(_) => None,
                })
                .expect("linear combination term must have an input factor");

            let mut w = F::from_i128(term.coefficient);
            for factor in &term.factors {
                if let Factor::Challenge(id) = factor {
                    w *= challenges[*id as usize];
                }
            }
            weights[input_id] += w;
        }
        weights
    }

    /// Extract the eq-scale factor from a Hamming booleanity formula.
    ///
    /// Returns `coefficient × Π challenges[c_id]` from the squared term.
    /// Panics if `!self.is_hamming_booleanity()`.
    pub fn hamming_eq_scale<F: Field>(&self, challenges: &[F]) -> F {
        debug_assert!(self.is_hamming_booleanity());
        let sq_term = self
            .terms
            .iter()
            .find(|t| {
                t.factors
                    .iter()
                    .filter(|f| matches!(f, Factor::Input(_)))
                    .count()
                    == 2
            })
            .expect("hamming booleanity must have a squared term");

        let mut scale = F::from_i128(sq_term.coefficient);
        for factor in &sq_term.factors {
            if let Factor::Challenge(id) = factor {
                scale *= challenges[*id as usize];
            }
        }
        scale
    }

    /// Challenge values by slot index, for kernels that bake challenges at
    /// compile time.
    pub fn challenge_values<F: Field>(&self, challenges: &[F]) -> Vec<F> {
        (0..self.num_challenges)
            .map(|i| challenges.get(i).copied().unwrap_or_else(F::zero))
            .collect()
    }
}

/// Visitor that normalizes an expression tree into composition formula form.
struct CompositionVisitor;

impl ExprVisitor for CompositionVisitor {
    type Output = Vec<ProductTerm>;

    fn visit_constant(&mut self, val: i128) -> Vec<ProductTerm> {
        vec![ProductTerm::constant(val)]
    }

    fn visit_var(&mut self, var: Var) -> Vec<ProductTerm> {
        let factor = match var {
            Var::Opening(id) => Factor::Input(id),
            Var::Challenge(id) => Factor::Challenge(id),
        };
        vec![ProductTerm::var(factor)]
    }

    fn visit_neg(&mut self, inner: Vec<ProductTerm>) -> Vec<ProductTerm> {
        inner.iter().map(ProductTerm::negate).collect()
    }

    fn visit_add(&mut self, mut lhs: Vec<ProductTerm>, rhs: Vec<ProductTerm>) -> Vec<ProductTerm> {
        lhs.extend(rhs);
        lhs
    }

    fn visit_sub(&mut self, mut lhs: Vec<ProductTerm>, rhs: Vec<ProductTerm>) -> Vec<ProductTerm> {
        lhs.extend(rhs.iter().map(ProductTerm::negate));
        lhs
    }

    fn visit_mul(&mut self, lhs: Vec<ProductTerm>, rhs: Vec<ProductTerm>) -> Vec<ProductTerm> {
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
    /// Normalize the expression into a [`CompositionFormula`].
    ///
    /// Distributes multiplication over addition and folds constant factors
    /// into term coefficients.
    ///
    /// # Example
    ///
    /// `(a + b) * c` becomes `[1·a·c, 1·b·c]`
    pub fn to_composition_formula(&self) -> CompositionFormula {
        let terms = self.visit(&mut CompositionVisitor);
        CompositionFormula::from_terms(terms)
    }

    /// Fold constant sub-expressions in the tree.
    ///
    /// Evaluates pure-constant subtrees to single `Constant` nodes.
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
        let mut map: std::collections::BTreeMap<ExprNode, ExprId> =
            std::collections::BTreeMap::new();
        let root = self.cse_node(self.root, &mut out, &mut map);
        Expr { arena: out, root }
    }

    fn cse_node(
        &self,
        id: ExprId,
        out: &mut ExprArena,
        map: &mut std::collections::BTreeMap<ExprNode, ExprId>,
    ) -> ExprId {
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

        *map.entry(canonical_node)
            .or_insert_with(|| out.push(canonical_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;

    #[test]
    fn constant() {
        let b = ExprBuilder::new();
        let c = b.constant(42);
        let expr = b.build(c);
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 1);
        assert_eq!(f.terms[0].coefficient, 42);
        assert!(f.terms[0].factors.is_empty());
        assert_eq!(f.num_inputs, 0);
        assert_eq!(f.num_challenges, 0);
    }

    #[test]
    fn single_var() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 1);
        assert_eq!(f.terms[0].coefficient, 1);
        assert_eq!(f.terms[0].factors, vec![Factor::Input(0)]);
        assert_eq!(f.num_inputs, 1);
    }

    #[test]
    fn addition() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(a + bv);
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 2);
        assert_eq!(f.terms[0].factors, vec![Factor::Input(0)]);
        assert_eq!(f.terms[1].factors, vec![Factor::Input(1)]);
        assert_eq!(f.num_inputs, 2);
    }

    #[test]
    fn distribution() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build((a + bv) * c);
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 2);
        assert_eq!(f.terms[0].factors, vec![Factor::Input(0), Factor::Input(2)]);
        assert_eq!(f.terms[1].factors, vec![Factor::Input(1), Factor::Input(2)]);
    }

    #[test]
    fn subtraction() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let expr = b.build((a - bv) * c);
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 2);
        assert_eq!(f.terms[0].coefficient, 1);
        assert_eq!(f.terms[1].coefficient, -1);
    }

    #[test]
    fn negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let expr = b.build(-(a * bv));
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 1);
        assert_eq!(f.terms[0].coefficient, -1);
        assert_eq!(f.terms[0].factors, vec![Factor::Input(0), Factor::Input(1)]);
    }

    #[test]
    fn booleanity() {
        // gamma * (h^2 - h) → [gamma·h·h, -gamma·h]
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let f = expr.to_composition_formula();

        assert_eq!(f.len(), 2);
        assert_eq!(f.terms[0].coefficient, 1);
        assert_eq!(
            f.terms[0].factors,
            vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(0)]
        );
        assert_eq!(f.terms[1].coefficient, -1);
        assert_eq!(
            f.terms[1].factors,
            vec![Factor::Challenge(0), Factor::Input(0)]
        );
        assert_eq!(f.num_inputs, 1);
        assert_eq!(f.num_challenges, 1);
        assert_eq!(f.degree(), 2);
    }

    #[test]
    fn evaluate_formula() {
        use jolt_field::Fr;
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * a + bv);
        let f = expr.to_composition_formula();

        let inputs = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let challenges = vec![Fr::from_u64(7)];
        // gamma*a + b = 7*3 + 5 = 26
        assert_eq!(f.evaluate(&inputs, &challenges), Fr::from_u64(26));
        // Must match Expr::evaluate
        assert_eq!(
            f.evaluate(&inputs, &challenges),
            expr.evaluate(&inputs, &challenges)
        );
    }

    #[test]
    fn as_product_sum_detected() {
        // 2 groups of 3 consecutive inputs: pure product-sum
        let terms = vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
            },
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(3), Factor::Input(4), Factor::Input(5)],
            },
        ];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.as_product_sum(), Some((3, 2)));
    }

    #[test]
    fn as_product_sum_rejected_with_challenges() {
        let terms = vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Challenge(0)],
        }];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.as_product_sum(), None);
    }

    #[test]
    fn as_product_sum_rejected_with_nonunit_coeff() {
        let terms = vec![ProductTerm {
            coefficient: 2,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }];
        let f = CompositionFormula::from_terms(terms);
        assert_eq!(f.as_product_sum(), None);
    }

    #[test]
    fn is_linear_combination_detected() {
        // c0 * o0 + c1 * o1
        let b = ExprBuilder::new();
        let o0 = b.opening(0);
        let o1 = b.opening(1);
        let c0 = b.challenge(0);
        let c1 = b.challenge(1);
        let expr = b.build(c0 * o0 + c1 * o1);
        let f = expr.to_composition_formula();
        assert!(f.is_linear_combination());
    }

    #[test]
    fn is_linear_combination_rejected_quadratic() {
        // o0 * o1 — no challenges, 2 inputs per term
        let b = ExprBuilder::new();
        let o0 = b.opening(0);
        let o1 = b.opening(1);
        let expr = b.build(o0 * o1);
        let f = expr.to_composition_formula();
        assert!(!f.is_linear_combination());
    }

    #[test]
    fn is_hamming_booleanity_detected() {
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let f = expr.to_composition_formula();
        assert!(f.is_hamming_booleanity());
    }

    #[test]
    fn linear_combination_weights() {
        use jolt_field::Fr;
        let b = ExprBuilder::new();
        let o0 = b.opening(0);
        let o1 = b.opening(1);
        let o2 = b.opening(2);
        let c0 = b.challenge(0);
        let c1 = b.challenge(1);
        let c2 = b.challenge(2);
        let expr = b.build(c0 * o0 + c1 * o1 + c2 * o2);
        let f = expr.to_composition_formula();

        let challenges = vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
        let weights = f.linear_combination_weights(&challenges);
        assert_eq!(
            weights,
            vec![Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)]
        );
    }

    #[test]
    fn hamming_eq_scale_extraction() {
        use jolt_field::Fr;
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let f = expr.to_composition_formula();

        let challenges = vec![Fr::from_u64(7)];
        let scale = f.hamming_eq_scale(&challenges);
        assert_eq!(scale, Fr::from_u64(7));
    }

    #[test]
    fn degree_computation() {
        // gamma * h * h - gamma * h: degree 2 (max input factors in a term)
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));
        let f = expr.to_composition_formula();
        assert_eq!(f.degree(), 2);
    }

    #[test]
    fn fold_constants_basic() {
        let b = ExprBuilder::new();
        let two = b.constant(2);
        let three = b.constant(3);
        let a = b.opening(0);
        let expr = b.build((two + three) * a);

        let folded = expr.fold_constants();
        match folded.get(folded.root()) {
            ExprNode::Mul(l, _r) => {
                assert_eq!(folded.get(l), ExprNode::Constant(5));
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn fold_constants_nested() {
        let b = ExprBuilder::new();
        let three = b.constant(3);
        let four = b.constant(4);
        let expr = b.build(-(three * four));

        let folded = expr.fold_constants();
        assert_eq!(folded.get(folded.root()), ExprNode::Constant(-12));
    }

    #[test]
    fn cse_deduplicates() {
        let b = ExprBuilder::new();
        let h1 = b.opening(0);
        let h2 = b.opening(0);
        let expr = b.build(h1 * h2);
        assert_eq!(expr.len(), 3);

        let optimized = expr.eliminate_common_subexpressions();
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn cse_preserves_evaluation() {
        let b = ExprBuilder::new();
        let a1 = b.opening(0);
        let b1 = b.opening(1);
        let a2 = b.opening(0);
        let b2 = b.opening(1);
        let expr = b.build((a1 + b1) * (a2 + b2));
        assert_eq!(expr.len(), 7);

        let optimized = expr.eliminate_common_subexpressions();
        assert!(optimized.len() <= 4);
    }
}
