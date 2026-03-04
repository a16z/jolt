//! Random linear combination (RLC) batch reduction utilities.
//!
//! RLC is the standard technique for batching multiple polynomial opening
//! claims at the same evaluation point into a single claim. Given polynomials
//! $p_1, \ldots, p_k$ and a random challenge $\rho$, the combined polynomial is:
//!
//! $$p_{\text{combined}} = \sum_{i=0}^{k-1} \rho^i \cdot p_i$$

use jolt_field::Field;

/// Computes the RLC of polynomial evaluation tables.
///
/// Given evaluation tables $p_1, \ldots, p_k$ (each of length $2^n$) and a
/// Fiat-Shamir challenge $\rho$, returns the evaluation table of:
///
/// $$p_{\text{combined}}(x) = p_1(x) + \rho \cdot p_2(x) + \rho^2 \cdot p_3(x) + \cdots + \rho^{k-1} \cdot p_k(x)$$
///
/// # Panics
///
/// Panics if `polynomials` is empty or if the evaluation tables have different lengths.
pub fn rlc_combine<F: Field>(polynomials: &[&[F]], rho: F) -> Vec<F> {
    assert!(!polynomials.is_empty(), "need at least one polynomial");
    let len = polynomials[0].len();
    for (i, p) in polynomials.iter().enumerate().skip(1) {
        assert_eq!(
            p.len(),
            len,
            "polynomial {i} has length {} but expected {len}",
            p.len()
        );
    }

    // Horner's method: iterate from the last polynomial backwards.
    // result = p_k, then result = result * rho + p_{k-1}, ...
    let mut result = polynomials.last().unwrap().to_vec();

    for p in polynomials.iter().rev().skip(1) {
        for (r, &val) in result.iter_mut().zip(p.iter()) {
            *r = *r * rho + val;
        }
    }

    result
}

/// Computes the RLC of scalar evaluations.
///
/// Given claimed evaluations $v_1, \ldots, v_k$ and challenge $\rho$, returns:
///
/// $$v_{\text{combined}} = \sum_{i=0}^{k-1} \rho^i \cdot v_i$$
///
/// Uses Horner's method for $O(k)$ multiplications.
///
/// # Panics
///
/// Panics if `evals` is empty.
pub fn rlc_combine_scalars<F: Field>(evals: &[F], rho: F) -> F {
    assert!(!evals.is_empty(), "need at least one evaluation");

    // Horner: start from the last, accumulate backwards
    let mut result = F::zero();
    for &v in evals.iter().rev() {
        result = result * rho + v;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;

    #[test]
    fn rlc_combine_single_polynomial_is_identity() {
        let evals: Vec<Fr> = (0..4).map(|i| Fr::from_u64(i + 1)).collect();
        let rho = Fr::from_u64(7);
        let result = rlc_combine(&[&evals], rho);
        assert_eq!(result, evals);
    }

    #[test]
    fn rlc_combine_two_polynomials() {
        // p1 = [1, 2, 3, 4], p2 = [5, 6, 7, 8], rho = 3
        // combined[i] = p1[i] + rho * p2[i]
        let p1: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let p2: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let rho = Fr::from_u64(3);

        let result = rlc_combine(&[&p1, &p2], rho);

        for i in 0..4 {
            let expected = p1[i] + rho * p2[i];
            assert_eq!(result[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn rlc_combine_three_polynomials_horner() {
        // p1 = [1], p2 = [2], p3 = [3], rho = 5
        // combined = 1 + 5*2 + 25*3 = 1 + 10 + 75 = 86
        let p1 = [Fr::from_u64(1)];
        let p2 = [Fr::from_u64(2)];
        let p3 = [Fr::from_u64(3)];
        let rho = Fr::from_u64(5);

        let result = rlc_combine(&[&p1[..], &p2[..], &p3[..]], rho);
        assert_eq!(result[0], Fr::from_u64(86));
    }

    #[test]
    fn rlc_combine_scalars_matches_manual() {
        let evals: Vec<Fr> = vec![Fr::from_u64(10), Fr::from_u64(20), Fr::from_u64(30)];
        let rho = Fr::from_u64(2);
        // 10 + 2*20 + 4*30 = 10 + 40 + 120 = 170
        let result = rlc_combine_scalars(&evals, rho);
        assert_eq!(result, Fr::from_u64(170));
    }

    #[test]
    fn rlc_combine_scalars_single() {
        let result = rlc_combine_scalars(&[Fr::from_u64(42)], Fr::from_u64(999));
        assert_eq!(result, Fr::from_u64(42));
    }

    #[test]
    fn rlc_combine_with_zero_rho() {
        let p1: Vec<Fr> = vec![Fr::from_u64(5), Fr::from_u64(10)];
        let p2: Vec<Fr> = vec![Fr::from_u64(99), Fr::from_u64(99)];
        let rho = Fr::from_u64(0);

        let result = rlc_combine(&[&p1, &p2], rho);
        // rho=0 means only p1 contributes
        assert_eq!(result, p1);
    }

    #[test]
    fn rlc_combine_rho_one_equal_weight() {
        // rho = 1: combined[i] = p1[i] + p2[i] + p3[i]
        let p1: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let p2: Vec<Fr> = vec![Fr::from_u64(3), Fr::from_u64(4)];
        let p3: Vec<Fr> = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let rho = Fr::from_u64(1);

        let result = rlc_combine(&[&p1, &p2, &p3], rho);
        assert_eq!(result[0], Fr::from_u64(9)); // 1 + 3 + 5
        assert_eq!(result[1], Fr::from_u64(12)); // 2 + 4 + 6
    }

    #[test]
    fn rlc_combine_scalars_consistent_with_rlc_combine() {
        // Verify that rlc_combine_scalars gives the same result as evaluating
        // the rlc_combine result at a specific point.
        use jolt_poly::{DensePolynomial, MultilinearPolynomial};
        use rand_chacha::rand_core::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(555);
        let num_vars = 3;
        let rho = Fr::from_u64(7);

        let p1 = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let p3 = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let eval1 = p1.evaluate(&point);
        let eval2 = p2.evaluate(&point);
        let eval3 = p3.evaluate(&point);

        // Method 1: combine polynomials first, then evaluate
        let evals1 = p1.evaluations();
        let evals2 = p2.evaluations();
        let evals3 = p3.evaluations();
        let combined = rlc_combine(&[evals1, evals2, evals3], rho);
        let combined_poly = DensePolynomial::new(combined);
        let result_via_poly = combined_poly.evaluate(&point);

        // Method 2: evaluate first, then combine scalars
        let result_via_scalars = rlc_combine_scalars(&[eval1, eval2, eval3], rho);

        assert_eq!(
            result_via_poly, result_via_scalars,
            "rlc_combine then evaluate must equal evaluate then rlc_combine_scalars"
        );
    }
}
