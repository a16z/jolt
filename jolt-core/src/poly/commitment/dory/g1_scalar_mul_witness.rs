//! G1 scalar multiplication witness generation for Dory recursion
//! Implements the double-and-add algorithm for elliptic curve scalar multiplication

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::ops::Add;

/// Build MLE from a vector of field elements (one per step)
/// Pads with zeros to reach the full size of the Boolean hypercube
fn build_mle_from_steps(step_values: &[Fq], num_vars: usize) -> Vec<Fq> {
    let size = 1 << num_vars;
    let mut mle = vec![Fq::zero(); size];

    // Copy the step values into the MLE
    for (i, &value) in step_values.iter().enumerate() {
        if i < size {
            mle[i] = value;
        }
    }

    mle
}

/// G1 scalar multiplication witness generation
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ScalarMultiplicationSteps {
    pub point_base: G1Affine, // Base point P
    pub scalar: Fr,           // Scalar k
    pub result: G1Affine,     // Result Q = [k]P

    // Witness MLEs: x_a[i] and y_a[i] give coords of A_i
    // For constraint i: A_i is current, T_i = [2]A_i is doubled, A_{i+1} is next
    pub x_a_mles: Vec<Vec<Fq>>, // x-coords of accumulator A_0, A_1, ..., A_n
    pub y_a_mles: Vec<Vec<Fq>>, // y-coords of accumulator A_0, A_1, ..., A_n
    pub x_t_mles: Vec<Vec<Fq>>, // x-coords of doubled point T_i = [2]A_i
    pub y_t_mles: Vec<Vec<Fq>>, // y-coords of doubled point T_i
    pub x_a_next_mles: Vec<Vec<Fq>>, // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next_mles: Vec<Vec<Fq>>, // y-coords of A_{i+1} (shifted by 1)

    /// Indicator: 1 if T_i = O (point at infinity), 0 otherwise
    pub t_is_infinity_mles: Vec<Vec<Fq>>,

    pub bits: Vec<bool>, // Scalar bits (MSB first, always 256 bits)
}

impl ScalarMultiplicationSteps {
    /// Generate witness for [scalar]point using double-and-add algorithm
    pub fn new(point: G1Affine, scalar: Fr) -> Self {
        // Get binary representation of scalar (little-endian)
        let scalar_bits = scalar.into_bigint().to_bits_le();

        // IMPORTANT: We must process exactly 256 bits for the sumcheck to work
        // The MLEs are 8-variate (2^8 = 256), so we need all 256 steps
        let bits_msb: Vec<bool> = (0..256).rev().map(|i| scalar_bits[i]).collect();

        // Determine number of variables needed for MLEs
        // For 256-bit scalars, we need 8 variables (2^8 = 256)
        let num_vars = 8;

        // Initialize arrays to collect values for each coordinate across all steps
        let n = bits_msb.len();
        let mut x_a_values = Vec::with_capacity(n + 1); // A_0, A_1, ..., A_n
        let mut y_a_values = Vec::with_capacity(n + 1);
        let mut x_t_values = Vec::with_capacity(n); // T_0, T_1, ..., T_{n-1}
        let mut y_t_values = Vec::with_capacity(n);
        let mut t_is_infinity_values = Vec::with_capacity(n); // Indicator for T_i = O

        // Initialize accumulator with point at infinity (identity)
        let mut accumulator = G1Projective::zero();

        // Store A_0 = O (point at infinity)
        let a_0: G1Affine = accumulator.into();
        let (x_a_0, y_a_0) = if a_0.is_zero() {
            (Fq::zero(), Fq::zero())
        } else {
            (a_0.x, a_0.y)
        };
        x_a_values.push(x_a_0);
        y_a_values.push(y_a_0);

        // Perform double-and-add algorithm and collect values
        // For each bit b_i (i = 1 to 256), compute:
        // T_i = [2]A_{i-1} and A_i = T_i + b_i * P
        for (_i, &bit) in bits_msb.iter().enumerate() {
            // Double: T_{i+1} = [2]A_i
            let doubled = accumulator + accumulator;
            let t_affine: G1Affine = doubled.into();
            let (x_t, y_t) = if t_affine.is_zero() {
                (Fq::zero(), Fq::zero())
            } else {
                (t_affine.x, t_affine.y)
            };
            x_t_values.push(x_t);
            y_t_values.push(y_t);

            // Compute infinity indicator for T_i
            let is_inf = if x_t.is_zero() && y_t.is_zero() {
                Fq::one()
            } else {
                Fq::zero()
            };
            t_is_infinity_values.push(is_inf);

            // Conditional add: A_{i+1} = T_i + b_i * P
            // where b_i is bits_msb[i] (0-indexed array)
            accumulator = if bit {
                doubled.add(&point.into_group())
            } else {
                doubled
            };

            // Store A_{i+1} coordinates
            let a_next: G1Affine = accumulator.into();
            let (x_a_next, y_a_next) = if a_next.is_zero() {
                (Fq::zero(), Fq::zero())
            } else {
                (a_next.x, a_next.y)
            };
            x_a_values.push(x_a_next);
            y_a_values.push(y_a_next);
        }

        // Build MLEs from collected values
        // x_a_values has 257 elements [A_0, ..., A_256], but MLE only needs first 256
        let x_a_mle = build_mle_from_steps(&x_a_values[..256], num_vars);
        let y_a_mle = build_mle_from_steps(&y_a_values[..256], num_vars);
        let x_t_mle = build_mle_from_steps(&x_t_values, num_vars);
        let y_t_mle = build_mle_from_steps(&y_t_values, num_vars);
        let t_is_infinity_mle = build_mle_from_steps(&t_is_infinity_values, num_vars);

        // Build shifted MLEs for A_{i+1} values
        // x_a_values contains [A_0, A_1, ..., A_256] (257 elements)
        // We want x_a_next to contain [A_1, A_2, ..., A_256] (256 elements)
        // This is exactly x_a_values[1..257]
        let x_a_next_values = x_a_values[1..257].to_vec();
        let y_a_next_values = y_a_values[1..257].to_vec();

        let x_a_next_mle = build_mle_from_steps(&x_a_next_values, num_vars);
        let y_a_next_mle = build_mle_from_steps(&y_a_next_values, num_vars);

        let result: G1Affine = accumulator.into();

        Self {
            point_base: point,
            scalar,
            result,
            x_a_mles: vec![x_a_mle],
            y_a_mles: vec![y_a_mle],
            x_t_mles: vec![x_t_mle],
            y_t_mles: vec![y_t_mle],
            x_a_next_mles: vec![x_a_next_mle],
            y_a_next_mles: vec![y_a_next_mle],
            t_is_infinity_mles: vec![t_is_infinity_mle],
            bits: bits_msb,
        }
    }

    /// Verify that the result matches [scalar]point
    pub fn verify_result(&self) -> bool {
        let expected = self.point_base.mul_bigint(self.scalar.into_bigint());
        let expected_affine: G1Affine = expected.into();
        self.result == expected_affine
    }

    /// Verify constraints at a specific step
    /// This checks the four constraints C1-C4 from the spec
    /// Note: The constraints check the unconditional addition A_{i+1}' = T_i + P
    /// The bit multiplication happens in the sumcheck, not here
    pub fn verify_constraint_at_step(&self, step: usize) -> bool {
        if step >= self.bits.len() {
            return false;
        }

        // Get coordinates from MLEs
        // For step i: A_i is current, T_i is doubled, A_{i+1}' is next (unconditional addition)
        let x_a = self.x_a_mles[0][step]; // x-coord of A_i
        let y_a = self.y_a_mles[0][step]; // y-coord of A_i
        let x_t = self.x_t_mles[0][step]; // x-coord of T_i
        let y_t = self.y_t_mles[0][step]; // y-coord of T_i
        let x_a_next = self.x_a_next_mles[0][step]; // x-coord of A_{i+1}' = T_i + P
        let y_a_next = self.y_a_next_mles[0][step]; // y-coord of A_{i+1}' = T_i + P

        // Base point coordinates
        let x_p = self.point_base.x;
        let y_p = self.point_base.y;

        // C1: Doubling x-coordinate constraint
        // 4y_A^2(x_T + 2x_A) - 9x_A^4 = 0
        let c1 = {
            let four = Fq::from(4u64);
            let two = Fq::from(2u64);
            let nine = Fq::from(9u64);

            let y_a_sq = y_a * y_a;
            let x_a_sq = x_a * x_a;
            let x_a_fourth = x_a_sq * x_a_sq;

            four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
        };

        // C2: Doubling y-coordinate constraint
        // 3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0
        let c2 = {
            let three = Fq::from(3u64);
            let two = Fq::from(2u64);

            let x_a_sq = x_a * x_a;
            three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
        };

        // C3: Conditional addition x-coordinate constraint (reformulated without explicit bit)
        // C3 = (x_A' - x_T) * [(x_A' + x_T + x_P)(x_P - x_T)^2 - (y_P - y_T)^2]
        let c3 = {
            let x_diff = x_p - x_t;
            let y_diff = y_p - y_t;
            let x_a_diff = x_a_next - x_t;

            x_a_diff * ((x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff)
        };

        // C4: Conditional addition y-coordinate constraint (reformulated without explicit bit)
        // C4 = (y_A' - y_T) * [x_T(y_P + y_A') - x_P(y_T + y_A') + x_A'(y_T - y_P)]
        let c4 = {
            let y_a_diff = y_a_next - y_t;

            y_a_diff * (x_t * (y_p + y_a_next) - x_p * (y_t + y_a_next) + x_a_next * (y_t - y_p))
        };

        // All constraints should equal zero
        c1.is_zero() && c2.is_zero() && c3.is_zero() && c4.is_zero()
    }

    /// Get the number of steps (bits) in the scalar multiplication
    pub fn num_steps(&self) -> usize {
        self.bits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;

    #[test]
    fn test_scalar_multiplication_witness() {
        let mut rng = ark_std::test_rng();

        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::from(55743u64);

        // Generate witness
        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Verify result
        assert!(witness.verify_result(), "Result verification failed");

        // Verify constraints at each step
        for i in 0..witness.num_steps() {
            let result = witness.verify_constraint_at_step(i);
            assert!(result, "Constraint verification failed at step {}", i);
        }
    }

    #[test]
    fn test_identity_scalar() {
        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::zero();

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Result should be point at infinity
        assert!(witness.result.is_zero());
        assert!(witness.verify_result());
    }

    #[test]
    fn test_scalar_one() {
        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::one();

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Result should be the point itself
        assert_eq!(witness.result, point);
        assert!(witness.verify_result());
    }

    #[test]
    fn test_witness_structure() {
        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::rand(&mut rng);

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Check basic structure
        // We always use 256 bits
        assert_eq!(witness.bits.len(), 256);
        assert_eq!(witness.x_a_mles.len(), 1); // Should have one MLE
        assert_eq!(witness.y_a_mles.len(), 1);
        assert_eq!(witness.x_t_mles.len(), 1);
        assert_eq!(witness.y_t_mles.len(), 1);

        // Check MLE sizes
        assert_eq!(witness.x_a_mles[0].len(), 256); // 2^8 for 8 variables
        assert_eq!(witness.y_a_mles[0].len(), 256);
        assert_eq!(witness.x_t_mles[0].len(), 256);
        assert_eq!(witness.y_t_mles[0].len(), 256);

        // Verify the result
        assert!(witness.verify_result());
    }

    #[test]
    fn test_sumcheck_constraint_evaluation() {
        // This test emulates what the sumcheck prover does:
        // evaluates the constraints over the Boolean hypercube
        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::rand(&mut rng);

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Define the constraint functions (same as in g1_scalar_mul.rs)
        fn compute_c1(x_a: Fq, y_a: Fq, x_t: Fq) -> Fq {
            let four = Fq::from(4u64);
            let two = Fq::from(2u64);
            let nine = Fq::from(9u64);

            let y_a_sq = y_a * y_a;
            let x_a_sq = x_a * x_a;
            let x_a_fourth = x_a_sq * x_a_sq;

            four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
        }

        fn compute_c2(x_a: Fq, y_a: Fq, x_t: Fq, y_t: Fq) -> Fq {
            let three = Fq::from(3u64);
            let two = Fq::from(2u64);

            let x_a_sq = x_a * x_a;
            three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
        }

        fn compute_c3(x_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
            let x_diff = x_p - x_t;
            let y_diff = y_p - y_t;
            let x_a_diff = x_a_next - x_t;

            x_a_diff * ((x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff)
        }

        fn compute_c4(x_a_next: Fq, y_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
            let y_a_diff = y_a_next - y_t;

            y_a_diff * (x_t * (y_p + y_a_next) - x_p * (y_t + y_a_next) + x_a_next * (y_t - y_p))
        }

        // Test constraint evaluation at every point in the Boolean hypercube
        for cube_index in 0..256 {
            // The index directly corresponds to the step number
            let step = cube_index;

            // Get values from MLEs at this step
            let x_a = witness.x_a_mles[0][step];
            let y_a = witness.y_a_mles[0][step];
            let x_t = witness.x_t_mles[0][step];
            let y_t = witness.y_t_mles[0][step];
            let x_a_next = witness.x_a_next_mles[0][step];
            let y_a_next = witness.y_a_next_mles[0][step];
            let _t_is_infinity = witness.t_is_infinity_mles[0][step];
            let x_p = witness.point_base.x;
            let y_p = witness.point_base.y;

            // Compute the 4 constraints
            let c1 = compute_c1(x_a, y_a, x_t);
            let c2 = compute_c2(x_a, y_a, x_t, y_t);
            let c3 = compute_c3(x_a_next, x_t, y_t, x_p, y_p);
            let c4 = compute_c4(x_a_next, y_a_next, x_t, y_t, x_p, y_p);

            // In the sumcheck, these would be batched with delta
            // For this test, just verify each constraint is zero (or satisfies special conditions)

            // Special case: if A_i is the point at infinity
            if x_a.is_zero() && y_a.is_zero() {
                // T_i should also be point at infinity
                assert!(
                    x_t.is_zero() && y_t.is_zero(),
                    "T_{} should be infinity when A_{} is",
                    step,
                    step
                );
                // Constraints don't apply for point at infinity, skip
                continue;
            }

            // Normal case: all constraints should be zero
            assert!(c1.is_zero(), "C1 failed at step {}: {:?}", step, c1);
            assert!(c2.is_zero(), "C2 failed at step {}: {:?}", step, c2);

            // C3 and C4 are satisfied either when:
            // 1. x_a_next = x_t (no addition case), OR
            // 2. The addition formula holds
            let no_addition = x_a_next == x_t && y_a_next == y_t;
            if !no_addition {
                // If we're adding the base point, constraints should still be zero
                assert!(c3.is_zero(), "C3 failed at step {}: {:?}", step, c3);
                assert!(c4.is_zero(), "C4 failed at step {}: {:?}", step, c4);
            }
        }
    }

    #[test]
    fn test_mle_evaluation_sumcheck_style() {
        // This test actually uses MultilinearPolynomial evaluation like the sumcheck does
        use crate::poly::dense_mlpoly::DensePolynomial;
        use crate::poly::multilinear_polynomial::MultilinearPolynomial;

        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::rand(&mut rng);

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Create MultilinearPolynomial objects from the witness MLEs
        let x_a_mle =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(witness.x_a_mles[0].clone()));
        let y_a_mle =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(witness.y_a_mles[0].clone()));
        let x_t_mle =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(witness.x_t_mles[0].clone()));
        let y_t_mle =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(witness.y_t_mles[0].clone()));
        let x_a_next_mle = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            witness.x_a_next_mles[0].clone(),
        ));
        let y_a_next_mle = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            witness.y_a_next_mles[0].clone(),
        ));

        // Helper function to evaluate MLE at a point
        fn eval_multilinear(coeffs: &[Fq], point: &[Fq]) -> Fq {
            // For Boolean points, this is just indexing
            let mut index = 0;
            for (i, &bit) in point.iter().enumerate() {
                if bit == Fq::from(1u64) {
                    index |= 1 << i;
                }
            }
            coeffs[index]
        }

        // Define the constraint functions (same as above)
        fn compute_c1(x_a: Fq, y_a: Fq, x_t: Fq) -> Fq {
            let four = Fq::from(4u64);
            let two = Fq::from(2u64);
            let nine = Fq::from(9u64);

            let y_a_sq = y_a * y_a;
            let x_a_sq = x_a * x_a;
            let x_a_fourth = x_a_sq * x_a_sq;

            four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
        }

        fn compute_c2(x_a: Fq, y_a: Fq, x_t: Fq, y_t: Fq) -> Fq {
            let three = Fq::from(3u64);
            let two = Fq::from(2u64);

            let x_a_sq = x_a * x_a;
            three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
        }

        fn compute_c3(x_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
            let x_diff = x_p - x_t;
            let y_diff = y_p - y_t;
            let x_a_diff = x_a_next - x_t;

            x_a_diff * ((x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff)
        }

        fn compute_c4(x_a_next: Fq, y_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
            let y_a_diff = y_a_next - y_t;

            y_a_diff * (x_t * (y_p + y_a_next) - x_p * (y_t + y_a_next) + x_a_next * (y_t - y_p))
        }

        // Test MLE evaluation at Boolean points
        for cube_index in 0..256 {
            // Convert index to Boolean evaluation point
            let bool_point: Vec<Fq> = (0..8)
                .map(|i| {
                    if (cube_index >> i) & 1 == 1 {
                        Fq::from(1u64)
                    } else {
                        Fq::from(0u64)
                    }
                })
                .collect();

            // Evaluate MLEs at this point using eval_multilinear
            let x_a = match &x_a_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };
            let y_a = match &y_a_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };
            let x_t = match &x_t_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };
            let y_t = match &y_t_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };
            let x_a_next = match &x_a_next_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };
            let y_a_next = match &y_a_next_mle {
                MultilinearPolynomial::LargeScalars(dense) => {
                    eval_multilinear(&dense.Z, &bool_point)
                }
                _ => panic!("Wrong MLE type"),
            };

            let x_p = witness.point_base.x;
            let y_p = witness.point_base.y;

            // Verify the evaluations match direct indexing
            assert_eq!(
                x_a, witness.x_a_mles[0][cube_index],
                "x_a MLE eval mismatch at {}",
                cube_index
            );
            assert_eq!(
                y_a, witness.y_a_mles[0][cube_index],
                "y_a MLE eval mismatch at {}",
                cube_index
            );
            assert_eq!(
                x_t, witness.x_t_mles[0][cube_index],
                "x_t MLE eval mismatch at {}",
                cube_index
            );
            assert_eq!(
                y_t, witness.y_t_mles[0][cube_index],
                "y_t MLE eval mismatch at {}",
                cube_index
            );
            assert_eq!(
                x_a_next, witness.x_a_next_mles[0][cube_index],
                "x_a_next MLE eval mismatch at {}",
                cube_index
            );
            assert_eq!(
                y_a_next, witness.y_a_next_mles[0][cube_index],
                "y_a_next MLE eval mismatch at {}",
                cube_index
            );

            // Get infinity indicator
            let _t_is_infinity = witness.t_is_infinity_mles[0][cube_index];

            // Compute and verify constraints
            let c1 = compute_c1(x_a, y_a, x_t);
            let c2 = compute_c2(x_a, y_a, x_t, y_t);
            let c3 = compute_c3(x_a_next, x_t, y_t, x_p, y_p);
            let c4 = compute_c4(x_a_next, y_a_next, x_t, y_t, x_p, y_p);

            // Special case: if A_i is the point at infinity
            if x_a.is_zero() && y_a.is_zero() {
                assert!(
                    x_t.is_zero() && y_t.is_zero(),
                    "T should be infinity when A is"
                );
                continue;
            }

            // Verify constraints
            assert!(
                c1.is_zero(),
                "C1 failed at cube_index {}: {:?}",
                cube_index,
                c1
            );
            assert!(
                c2.is_zero(),
                "C2 failed at cube_index {}: {:?}",
                cube_index,
                c2
            );

            let no_addition = x_a_next == x_t && y_a_next == y_t;
            if !no_addition {
                assert!(
                    c3.is_zero(),
                    "C3 failed at cube_index {}: {:?}",
                    cube_index,
                    c3
                );
                assert!(
                    c4.is_zero(),
                    "C4 failed at cube_index {}: {:?}",
                    cube_index,
                    c4
                );
            }
        }
    }
}
