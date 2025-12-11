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
    pub point_base: G1Affine,      // Base point P
    pub scalar: Fr,                 // Scalar k
    pub result: G1Affine,          // Result Q = [k]P

    // Witness MLEs: x_a[i] and y_a[i] give coords of A_i
    // For constraint i: A_i is current, T_i = [2]A_i is doubled, A_{i+1} is next
    pub x_a_mles: Vec<Vec<Fq>>,    // x-coords of accumulator A_0, A_1, ..., A_n
    pub y_a_mles: Vec<Vec<Fq>>,    // y-coords of accumulator A_0, A_1, ..., A_n
    pub x_t_mles: Vec<Vec<Fq>>,    // x-coords of doubled point T_i = [2]A_i
    pub y_t_mles: Vec<Vec<Fq>>,    // y-coords of doubled point T_i

    pub bits: Vec<bool>,            // Scalar bits (MSB first after trimming leading zeros)
}

impl ScalarMultiplicationSteps {
    /// Generate witness for [scalar]point using double-and-add algorithm
    pub fn new(point: G1Affine, scalar: Fr) -> Self {
        // Get binary representation of scalar (little-endian)
        let scalar_bits = scalar.into_bigint().to_bits_le();

        // Find the highest set bit to trim leading zeros
        let mut highest_bit = 0;
        for (i, &bit) in scalar_bits.iter().enumerate().rev() {
            if bit {
                highest_bit = i;
                break;
            }
        }

        // Convert to MSB-first and trim leading zeros
        let bits_msb: Vec<bool> = scalar_bits[..=highest_bit]
            .iter()
            .rev()
            .copied()
            .collect();

        // Determine number of variables needed for MLEs
        // For 256-bit scalars, we need 8 variables (2^8 = 256)
        let num_vars = 8;

        // Initialize arrays to collect values for each coordinate across all steps
        let n = bits_msb.len();
        let mut x_a_values = Vec::with_capacity(n + 1);  // A_0, A_1, ..., A_n
        let mut y_a_values = Vec::with_capacity(n + 1);
        let mut x_t_values = Vec::with_capacity(n);      // T_0, T_1, ..., T_{n-1}
        let mut y_t_values = Vec::with_capacity(n);

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
        for &bit in bits_msb.iter() {
            // Double: T_i = [2]A_i
            let doubled = accumulator + accumulator;
            let t_affine: G1Affine = doubled.into();
            let (x_t, y_t) = if t_affine.is_zero() {
                (Fq::zero(), Fq::zero())
            } else {
                (t_affine.x, t_affine.y)
            };
            x_t_values.push(x_t);
            y_t_values.push(y_t);

            // Conditional add: A_{i+1} = T_i + b_i * P
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
        let x_a_mle = build_mle_from_steps(&x_a_values, num_vars);
        let y_a_mle = build_mle_from_steps(&y_a_values, num_vars);
        let x_t_mle = build_mle_from_steps(&x_t_values, num_vars);
        let y_t_mle = build_mle_from_steps(&y_t_values, num_vars);

        let result: G1Affine = accumulator.into();

        Self {
            point_base: point,
            scalar,
            result,
            x_a_mles: vec![x_a_mle],
            y_a_mles: vec![y_a_mle],
            x_t_mles: vec![x_t_mle],
            y_t_mles: vec![y_t_mle],
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
    pub fn verify_constraint_at_step(&self, step: usize) -> bool {
        if step >= self.bits.len() {
            return false;
        }

        // Get the bit for this step
        let bit = self.bits[step];

        // Get coordinates from MLEs
        // For step i: A_i is current, T_i is doubled, A_{i+1} is next
        let x_a = self.x_a_mles[0][step];        // x-coord of A_i
        let y_a = self.y_a_mles[0][step];        // y-coord of A_i
        let x_t = self.x_t_mles[0][step];        // x-coord of T_i
        let y_t = self.y_t_mles[0][step];        // y-coord of T_i
        let x_a_next = self.x_a_mles[0][step + 1];  // x-coord of A_{i+1}
        let y_a_next = self.y_a_mles[0][step + 1];  // y-coord of A_{i+1}

        // Base point coordinates
        let x_p = self.point_base.x;
        let y_p = self.point_base.y;

        // Special case: if A_i is the point at infinity
        if x_a.is_zero() && y_a.is_zero() {
            // When doubling point at infinity, T_i is also point at infinity
            if !x_t.is_zero() || !y_t.is_zero() {
                return false;
            }

            // For conditional add: A_{i+1} = T_i + b_i * P
            // Since T_i = O, we have A_{i+1} = O + b_i * P = b_i * P
            if bit {
                // A_{i+1} should be P
                return x_a_next == x_p && y_a_next == y_p;
            } else {
                // A_{i+1} should be O (point at infinity)
                return x_a_next.is_zero() && y_a_next.is_zero();
            }
        }

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

        // First test with a simple scalar to debug
        let point = G1Affine::rand(&mut rng);
        let scalar = Fr::from(5u64); // Binary: 101 (MSB first)

        // Generate witness
        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Debug output
        println!("Testing scalar = 5, bits = {:?}", witness.bits);
        println!("Base point: ({:?}, {:?})", point.x, point.y);

        // Verify result
        assert!(witness.verify_result(), "Result verification failed");

        // Verify constraints at each step with detailed logging
        for i in 0..witness.num_steps() {
            println!("\nStep {} (bit = {}):", i, witness.bits[i]);
            println!("  A_{}: ({:?}, {:?})", i, witness.x_a_mles[0][i], witness.y_a_mles[0][i]);
            println!("  T_{}: ({:?}, {:?})", i, witness.x_t_mles[0][i], witness.y_t_mles[0][i]);
            println!("  A_{}: ({:?}, {:?})", i+1, witness.x_a_mles[0][i+1], witness.y_a_mles[0][i+1]);

            let result = witness.verify_constraint_at_step(i);
            if !result {
                println!("  CONSTRAINT VERIFICATION FAILED!");
                // Let's check each constraint individually
                let bit = witness.bits[i];
                let x_a = witness.x_a_mles[0][i];
                let y_a = witness.y_a_mles[0][i];
                let x_t = witness.x_t_mles[0][i];
                let y_t = witness.y_t_mles[0][i];
                let x_a_next = witness.x_a_mles[0][i + 1];
                let y_a_next = witness.y_a_mles[0][i + 1];
                let x_p = witness.point_base.x;
                let y_p = witness.point_base.y;

                // Check if A_i is point at infinity
                if x_a.is_zero() && y_a.is_zero() {
                    println!("  A_{} is point at infinity", i);
                } else {
                    // C1: Doubling x-coordinate constraint
                    let four = Fq::from(4u64);
                    let two = Fq::from(2u64);
                    let nine = Fq::from(9u64);
                    let y_a_sq = y_a * y_a;
                    let x_a_sq = x_a * x_a;
                    let x_a_fourth = x_a_sq * x_a_sq;
                    let c1 = four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth;
                    println!("  C1 (doubling x): {:?} (should be 0)", c1);

                    // C2: Doubling y-coordinate constraint
                    let three = Fq::from(3u64);
                    let c2 = three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a);
                    println!("  C2 (doubling y): {:?} (should be 0)", c2);
                }

                // C3 and C4: Conditional addition constraints
                if bit {
                    println!("  Adding base point");
                } else {
                    println!("  Not adding base point");
                }
            }
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
        let scalar = Fr::from(5u64); // Binary: 101

        let witness = ScalarMultiplicationSteps::new(point, scalar);

        // Check basic structure
        assert_eq!(witness.bits, vec![true, false, true]); // MSB first: 101
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

        // Debug: print first few values
        println!("First A_i values:");
        for i in 0..4 {
            println!("  A_{}: ({:?}, {:?})", i, witness.x_a_mles[0][i], witness.y_a_mles[0][i]);
        }
    }
}