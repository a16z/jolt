//! Generic eq · g product [`SumcheckCompute`] (degree 2).
//!
//! Proves $\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot g(x) = v$ where
//! $g$ is a pre-computed multilinear polynomial and $v$ is the claimed sum.
//!
//! Used by claim reduction stages where $g$ is a random linear combination
//! of the input polynomials.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

/// Sumcheck witness for the product $\widetilde{eq}(r, x) \cdot g(x)$.
///
/// Degree 2 per variable (1 from eq + 1 from g). Each round evaluates
/// at $t = 0, 1, 2$ and interpolates the quadratic round polynomial.
pub struct EqProductCompute<F: Field> {
    g_table: Vec<F>,
    eq_table: Vec<F>,
    num_vars: usize,
    round: usize,
}

impl<F: Field> EqProductCompute<F> {
    /// Creates a witness from the pre-computed $g$ table and eq table.
    ///
    /// Both must have length $2^n$.
    pub fn new(g_table: Vec<F>, eq_table: Vec<F>, num_vars: usize) -> Self {
        let expected = 1usize << num_vars;
        assert_eq!(g_table.len(), expected);
        assert_eq!(eq_table.len(), expected);
        Self {
            g_table,
            eq_table,
            num_vars,
            round: 0,
        }
    }

    #[inline]
    fn current_size(&self) -> usize {
        1usize << (self.num_vars - self.round)
    }
}

impl<F: Field> SumcheckCompute<F> for EqProductCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.current_size() / 2;
        let two = F::from_u64(2);
        let mut evals = [F::zero(); 3];

        for j in 0..half {
            let g0 = self.g_table[2 * j];
            let g1 = self.g_table[2 * j + 1];
            let e0 = self.eq_table[2 * j];
            let e1 = self.eq_table[2 * j + 1];

            let dg = g1 - g0;
            let de = e1 - e0;

            // t=0
            evals[0] += e0 * g0;
            // t=1
            evals[1] += e1 * g1;
            // t=2
            evals[2] += (e0 + two * de) * (g0 + two * dg);
        }

        UnivariatePoly::interpolate_over_integers(&evals)
    }

    fn bind(&mut self, challenge: F) {
        let half = self.current_size() / 2;
        for j in 0..half {
            self.g_table[j] =
                self.g_table[2 * j] + challenge * (self.g_table[2 * j + 1] - self.g_table[2 * j]);
            self.eq_table[j] = self.eq_table[2 * j]
                + challenge * (self.eq_table[2 * j + 1] - self.eq_table[2 * j]);
        }
        self.g_table.truncate(half);
        self.eq_table.truncate(half);
        self.round += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn round_polynomial_consistency_degree_2() {
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let witness = EqProductCompute::new(g_table.clone(), eq_table.clone(), num_vars);
        let poly = witness.round_polynomial();

        // Verify at t=0 and t=1
        let s0: Fr = (0..n / 2).map(|j| eq_table[2 * j] * g_table[2 * j]).sum();
        let s1: Fr = (0..n / 2)
            .map(|j| eq_table[2 * j + 1] * g_table[2 * j + 1])
            .sum();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);
    }

    #[test]
    fn full_prove_verify() {
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum: Fr = eq_table.iter().zip(g_table.iter()).map(|(&e, &g)| e * g).sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let mut witness = EqProductCompute::new(g_table, eq_table, num_vars);

        let mut pt = Blake2bTranscript::new(b"eq_product_test");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"eq_product_test");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }
}
