//! GT multiplication witness generation for Dory recursion

use ark_bn254::{Fq, Fq12};
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

/// GT multiplication witness generation
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiplicationSteps {
    pub lhs: Fq12,             // Left operand
    pub rhs: Fq12,             // Right operand
    pub result: Fq12,          // Product lhs * rhs
    pub quotient_mle: Vec<Fq>, // MLE of quotient polynomial Q(x)
}

impl MultiplicationSteps {
    /// Generate witness for lhs * rhs
    pub fn new(lhs: Fq12, rhs: Fq12) -> Self {
        let result = lhs * rhs;
        let quotient_mle = compute_mul_quotient(lhs, rhs, result);

        Self {
            lhs,
            rhs,
            result,
            quotient_mle,
        }
    }

    /// Verify that the result matches lhs * rhs
    pub fn verify_result(&self) -> bool {
        self.result == self.lhs * self.rhs
    }

    /// Verify constraint at a Boolean cube point
    /// Checks that lhs(x) * rhs(x) - result(x) - Q(x) * g(x) = 0
    pub fn verify_constraint_at_cube_point(&self, cube_index: usize) -> bool {
        if cube_index >= 16 {
            return false;
        }
        let point = index_to_boolean_point(cube_index);

        // Evaluate MLEs
        let lhs_mle = fq12_to_multilinear_evals(&self.lhs);
        let rhs_mle = fq12_to_multilinear_evals(&self.rhs);
        let result_mle = fq12_to_multilinear_evals(&self.result);

        let lhs_eval = eval_mle_at_boolean_point(&lhs_mle, &point);
        let rhs_eval = eval_mle_at_boolean_point(&rhs_mle, &point);
        let result_eval = eval_mle_at_boolean_point(&result_mle, &point);
        let quotient_eval = eval_mle_at_boolean_point(&self.quotient_mle, &point);

        let g_mle = get_g_mle();
        let g_eval = eval_mle_at_boolean_point(&g_mle, &point);

        // Compute constraint: lhs * rhs - result - Q * g
        let constraint = lhs_eval * rhs_eval - result_eval - quotient_eval * g_eval;
        constraint.is_zero()
    }
}

/// Compute quotient MLE for multiplication constraint
/// Q(x) = (lhs(x) * rhs(x) - result(x)) / g(x)
fn compute_mul_quotient(lhs: Fq12, rhs: Fq12, result: Fq12) -> Vec<Fq> {
    let lhs_mle = fq12_to_multilinear_evals(&lhs);
    let rhs_mle = fq12_to_multilinear_evals(&rhs);
    let result_mle = fq12_to_multilinear_evals(&result);
    let g_mle = get_g_mle();

    // Compute the quotient MLE pointwise
    let mut quotient_mle = vec![Fq::zero(); 16];
    for j in 0..16 {
        let product = lhs_mle[j] * rhs_mle[j];
        let numerator = product - result_mle[j];

        // Q(x) = (lhs(x) * rhs(x) - result(x)) / g(x)
        if !g_mle[j].is_zero() {
            quotient_mle[j] = numerator / g_mle[j];
        }
    }

    quotient_mle
}

/// Convert a cube index (0..15) to a Boolean point in {0,1}^4
fn index_to_boolean_point(index: usize) -> Vec<Fq> {
    vec![
        Fq::from((index & 1) as u64),        // bit 0
        Fq::from(((index >> 1) & 1) as u64), // bit 1
        Fq::from(((index >> 2) & 1) as u64), // bit 2
        Fq::from(((index >> 3) & 1) as u64), // bit 3
    ]
}

/// Evaluate an MLE at a Boolean cube point
/// For Boolean points, this is equivalent to indexing but makes the evaluation explicit
fn eval_mle_at_boolean_point(mle: &[Fq], point: &[Fq]) -> Fq {
    // For Boolean points, we could just index, but using eval_multilinear
    // makes it clear we're doing MLE evaluation
    use jolt_optimizations::eval_multilinear;
    eval_multilinear(mle, point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::One;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_multiplication_steps_basic() {
        let mut rng = thread_rng();
        let lhs = Fq12::rand(&mut rng);
        let rhs = Fq12::rand(&mut rng);

        let mul_steps = MultiplicationSteps::new(lhs, rhs);

        assert!(mul_steps.verify_result());
        assert_eq!(mul_steps.result, lhs * rhs);

        // Verify constraint at all Boolean cube points
        for i in 0..16 {
            assert!(
                mul_steps.verify_constraint_at_cube_point(i),
                "Constraint failed at cube point {i}"
            );
        }
    }

    #[test]
    fn test_multiplication_identity() {
        let mut rng = thread_rng();
        let a = Fq12::rand(&mut rng);

        let mul_steps = MultiplicationSteps::new(a, Fq12::one());
        assert_eq!(mul_steps.result, a);
        assert!(mul_steps.verify_result());

        // Verify constraint at all Boolean cube points
        for i in 0..16 {
            assert!(mul_steps.verify_constraint_at_cube_point(i));
        }
    }

    #[test]
    fn test_multiplication_zero() {
        let mut rng = thread_rng();
        let a = Fq12::rand(&mut rng);

        let mul_steps = MultiplicationSteps::new(a, Fq12::zero());
        assert_eq!(mul_steps.result, Fq12::zero());
        assert!(mul_steps.verify_result());

        // Verify constraint at all Boolean cube points
        for i in 0..16 {
            assert!(mul_steps.verify_constraint_at_cube_point(i));
        }
    }
}
