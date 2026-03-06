use jolt_field::Field;

use crate::expr::{Expr, Var};
use crate::normalize::{SopValue, SumOfProducts};
use crate::visitor::ExprVisitor;

/// Zero-allocation visitor that evaluates an expression tree over a field `F`.
struct EvaluateVisitor<'a, F> {
    openings: &'a [F],
    challenges: &'a [F],
}

impl<F: Field> ExprVisitor for EvaluateVisitor<'_, F> {
    type Output = F;

    #[inline]
    fn visit_constant(&mut self, val: i128) -> F {
        F::from_i128(val)
    }

    #[inline]
    fn visit_var(&mut self, var: Var) -> F {
        match var {
            Var::Opening(id) => self.openings[id as usize],
            Var::Challenge(id) => self.challenges[id as usize],
        }
    }

    #[inline]
    fn visit_neg(&mut self, inner: F) -> F {
        -inner
    }

    #[inline]
    fn visit_add(&mut self, lhs: F, rhs: F) -> F {
        lhs + rhs
    }

    #[inline]
    fn visit_sub(&mut self, lhs: F, rhs: F) -> F {
        lhs - rhs
    }

    #[inline]
    fn visit_mul(&mut self, lhs: F, rhs: F) -> F {
        lhs * rhs
    }
}

impl Expr {
    /// Evaluate the expression with concrete field values for openings and challenges.
    ///
    /// Replaces the hand-written `SumcheckInstanceParams::input_claim()` methods.
    /// Panics if an `Opening(id)` or `Challenge(id)` index is out of bounds.
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        let mut visitor = EvaluateVisitor {
            openings,
            challenges,
        };
        self.visit(&mut visitor)
    }
}

impl SumOfProducts {
    /// Evaluate the sum-of-products form with concrete field values.
    ///
    /// Each term contributes `coefficient * factor[0] * factor[1] * ...` to
    /// the sum. This must produce the same result as `Expr::evaluate` for the
    /// same expression — that invariant is the critical correctness property.
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F {
        let resolve = |val: &SopValue| -> F {
            match val {
                SopValue::Constant(c) => F::from_i128(*c),
                SopValue::Opening(id) => openings[*id as usize],
                SopValue::Challenge(id) => challenges[*id as usize],
            }
        };

        self.terms
            .iter()
            .map(|term| {
                let coeff = F::from_i128(term.coefficient);
                let product: F = term.factors.iter().map(&resolve).product();
                coeff * product
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ExprBuilder;
    use jolt_field::Fr;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn evaluate_constant() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));
        let result: Fr = expr.evaluate(&[], &[]);
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn evaluate_zero_and_one() {
        let b = ExprBuilder::new();
        let expr = b.build(b.zero() + b.one());
        let result: Fr = expr.evaluate(&[], &[]);
        assert_eq!(result, Fr::from_u64(1));
    }

    #[test]
    fn evaluate_single_opening() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);
        let val = Fr::from_u64(7);
        let result: Fr = expr.evaluate(&[val], &[]);
        assert_eq!(result, val);
    }

    #[test]
    fn evaluate_single_challenge() {
        let b = ExprBuilder::new();
        let c = b.challenge(0);
        let expr = b.build(c);
        let val = Fr::from_u64(13);
        let result: Fr = expr.evaluate(&[], &[val]);
        assert_eq!(result, val);
    }

    #[test]
    fn evaluate_booleanity() {
        // gamma * (h^2 - h) with h=3, gamma=5
        // = 5 * (9 - 3) = 5 * 6 = 30
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let h_val = Fr::from_u64(3);
        let gamma_val = Fr::from_u64(5);
        let result: Fr = expr.evaluate(&[h_val], &[gamma_val]);
        assert_eq!(result, Fr::from_u64(30));
    }

    #[test]
    fn evaluate_negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);
        let val = Fr::from_u64(7);
        let result: Fr = expr.evaluate(&[val], &[]);
        assert_eq!(result, -val);
    }

    #[test]
    fn evaluate_integer_literal_ops() {
        // 2 * h + 1 with h=10 → 21
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let expr = b.build(2i128 * h + 1);

        let result: Fr = expr.evaluate(&[Fr::from_u64(10)], &[]);
        assert_eq!(result, Fr::from_u64(21));
    }

    #[test]
    fn sop_evaluate_constant() {
        let b = ExprBuilder::new();
        let expr = b.build(b.constant(42));
        let sop = expr.to_sum_of_products();
        let result: Fr = sop.evaluate(&[], &[]);
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn sop_evaluate_booleanity() {
        // gamma * (h^2 - h) with h=3, gamma=5 → 30
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        let h_val = Fr::from_u64(3);
        let gamma_val = Fr::from_u64(5);
        let sop = expr.to_sum_of_products();
        let result: Fr = sop.evaluate(&[h_val], &[gamma_val]);
        assert_eq!(result, Fr::from_u64(30));
    }

    /// Critical invariant: `expr.evaluate()` must equal
    /// `expr.to_sum_of_products().evaluate()` for all inputs.
    #[test]
    fn property_expr_equals_sop() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xdead);

        // Test several expression shapes
        let expressions = build_test_expressions();

        for (name, expr, n_openings, n_challenges) in &expressions {
            let openings: Vec<Fr> = (0..*n_openings).map(|_| Fr::random(&mut rng)).collect();
            let challenges: Vec<Fr> = (0..*n_challenges).map(|_| Fr::random(&mut rng)).collect();

            let direct: Fr = expr.evaluate(&openings, &challenges);
            let sop = expr.to_sum_of_products();
            let via_sop: Fr = sop.evaluate(&openings, &challenges);

            assert_eq!(direct, via_sop, "mismatch for expression: {name}");
        }
    }

    /// Same property with many random evaluation points per expression.
    #[test]
    fn property_expr_equals_sop_many_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xcafe);
        let expressions = build_test_expressions();

        for (name, expr, n_openings, n_challenges) in &expressions {
            let sop = expr.to_sum_of_products();
            for _ in 0..50 {
                let openings: Vec<Fr> = (0..*n_openings).map(|_| Fr::random(&mut rng)).collect();
                let challenges: Vec<Fr> =
                    (0..*n_challenges).map(|_| Fr::random(&mut rng)).collect();

                let direct: Fr = expr.evaluate(&openings, &challenges);
                let via_sop: Fr = sop.evaluate(&openings, &challenges);
                assert_eq!(direct, via_sop, "mismatch for expression: {name}");
            }
        }
    }

    use crate::expr::Expr;

    fn build_test_expressions() -> Vec<(&'static str, Expr, usize, usize)> {
        vec![
            {
                let b = ExprBuilder::new();
                let c = b.constant(42);
                ("constant", b.build(c), 0, 0)
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                ("single_var", b.build(a), 1, 0)
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                ("negation", b.build(-a), 1, 0)
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                ("add", b.build(a + bv), 2, 0)
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                ("sub", b.build(a - bv), 2, 0)
            },
            {
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                ("mul", b.build(a * bv), 2, 0)
            },
            {
                // gamma * (h^2 - h)
                let b = ExprBuilder::new();
                let h = b.opening(0);
                let gamma = b.challenge(0);
                ("booleanity", b.build(gamma * (h * h - h)), 1, 1)
            },
            {
                // (a + b) * (c - d)
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let c = b.opening(2);
                let d = b.opening(3);
                ("cross_distribute", b.build((a + bv) * (c - d)), 4, 0)
            },
            {
                // (a + b) * (c + d)
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let c = b.opening(2);
                let d = b.opening(3);
                ("foil", b.build((a + bv) * (c + d)), 4, 0)
            },
            {
                // -(a * b) + c
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let c = b.opening(2);
                ("neg_product_plus", b.build(-(a * bv) + c), 3, 0)
            },
            {
                // alpha * a + beta * b (weighted sum)
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let alpha = b.challenge(0);
                let beta = b.challenge(1);
                ("weighted_sum", b.build(alpha * a + beta * bv), 2, 2)
            },
            {
                // 2 * h + 1 (integer literal ops)
                let b = ExprBuilder::new();
                let h = b.opening(0);
                ("integer_literals", b.build(2i128 * h + 1), 1, 0)
            },
            {
                // h * h (same var squared)
                let b = ExprBuilder::new();
                let h = b.opening(0);
                ("squared", b.build(h * h), 1, 0)
            },
            {
                // challenge-only expression: alpha * beta + gamma
                let b = ExprBuilder::new();
                let alpha = b.challenge(0);
                let beta = b.challenge(1);
                let gamma = b.challenge(2);
                ("challenges_only", b.build(alpha * beta + gamma), 0, 3)
            },
            {
                // deeply nested: ((a + b) * c - d) * (e + f)
                let b = ExprBuilder::new();
                let a = b.opening(0);
                let bv = b.opening(1);
                let c = b.opening(2);
                let d = b.opening(3);
                let e = b.opening(4);
                let f = b.opening(5);
                ("deep_nested", b.build(((a + bv) * c - d) * (e + f)), 6, 0)
            },
        ]
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn evaluate_panics_on_oob_opening() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(a);
        let _: Fr = expr.evaluate(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn evaluate_panics_on_oob_challenge() {
        let b = ExprBuilder::new();
        let c = b.challenge(0);
        let expr = b.build(c);
        let _: Fr = expr.evaluate(&[], &[]);
    }
}
