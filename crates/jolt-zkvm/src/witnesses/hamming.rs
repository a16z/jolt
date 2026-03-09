//! Hamming booleanity [`SumcheckCompute`] witness.
//!
//! Proves that a polynomial $h$ is Boolean-valued on the hypercube:
//!
//! $$\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot h(x) \cdot (h(x) - 1) = 0$$
//!
//! where $r$ is a random evaluation point.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

/// Sumcheck witness for the Hamming booleanity check.
///
/// Maintains two evaluation tables: `h` (the polynomial being checked)
/// and `eq` (the equality polynomial at the random point `r`). Both are
/// iteratively halved via variable binding.
///
/// The per-variable polynomial has degree 3:
/// - degree 1 from $\widetilde{eq}$
/// - degree 2 from $h \cdot (h - 1)$
pub struct HammingBooleanityCompute<F: Field> {
    h_table: Vec<F>,
    eq_table: Vec<F>,
    num_vars: usize,
    round: usize,
}

impl<F: Field> HammingBooleanityCompute<F> {
    /// Creates a new witness from the hamming weight evaluations and
    /// pre-materialized eq polynomial table.
    ///
    /// Both tables must have length $2^{\text{num\_vars}}$.
    ///
    /// # Panics
    ///
    /// Panics if the table lengths don't match $2^{\text{num\_vars}}$.
    pub fn new(h_table: Vec<F>, eq_table: Vec<F>, num_vars: usize) -> Self {
        let expected_len = 1usize << num_vars;
        assert_eq!(
            h_table.len(),
            expected_len,
            "h_table length {} != 2^{num_vars} = {expected_len}",
            h_table.len()
        );
        assert_eq!(
            eq_table.len(),
            expected_len,
            "eq_table length {} != 2^{num_vars} = {expected_len}",
            eq_table.len()
        );
        Self {
            h_table,
            eq_table,
            num_vars,
            round: 0,
        }
    }

    /// Current table size: $2^{n - \text{round}}$.
    #[inline]
    fn current_size(&self) -> usize {
        1usize << (self.num_vars - self.round)
    }
}

impl<F: Field> SumcheckCompute<F> for HammingBooleanityCompute<F> {
    /// Computes the degree-3 round polynomial by evaluating
    /// $\widetilde{eq}(t) \cdot h(t) \cdot (h(t) - 1)$ at $t = 0, 1, 2, 3$
    /// over all remaining hypercube points.
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.current_size() / 2;
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);

        let mut evals = [F::zero(); 4];

        for j in 0..half {
            let h0 = self.h_table[2 * j];
            let h1 = self.h_table[2 * j + 1];
            let e0 = self.eq_table[2 * j];
            let e1 = self.eq_table[2 * j + 1];

            let dh = h1 - h0;
            let de = e1 - e0;

            // t = 0: e0 · h0 · (h0 - 1)
            evals[0] += e0 * h0 * (h0 - one);

            // t = 1: e1 · h1 · (h1 - 1)
            evals[1] += e1 * h1 * (h1 - one);

            // t = 2
            let h2 = h0 + two * dh;
            let e2 = e0 + two * de;
            evals[2] += e2 * h2 * (h2 - one);

            // t = 3
            let h3 = h0 + three * dh;
            let e3 = e0 + three * de;
            evals[3] += e3 * h3 * (h3 - one);
        }

        UnivariatePoly::interpolate_over_integers(&evals)
    }

    /// Fixes the current leading variable to `challenge`, halving the tables.
    fn bind(&mut self, challenge: F) {
        let half = self.current_size() / 2;

        for j in 0..half {
            self.h_table[j] =
                self.h_table[2 * j] + challenge * (self.h_table[2 * j + 1] - self.h_table[2 * j]);
            self.eq_table[j] = self.eq_table[2 * j]
                + challenge * (self.eq_table[2 * j + 1] - self.eq_table[2 * j]);
        }

        self.h_table.truncate(half);
        self.eq_table.truncate(half);
        self.round += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Brute-force computes ∑_x eq(r, x) · h(x) · (h(x) - 1).
    fn brute_force_booleanity_sum(h: &[Fr], eq: &[Fr]) -> Fr {
        h.iter()
            .zip(eq.iter())
            .map(|(&h_val, &eq_val)| eq_val * h_val * (h_val - Fr::one()))
            .sum()
    }

    #[test]
    fn boolean_polynomial_has_zero_sum() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        // h = [0, 1, 1, 0, 1, 0, 0, 1] — all boolean
        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let sum = brute_force_booleanity_sum(&h_table, &eq_table);
        assert!(sum.is_zero(), "boolean polynomial should have sum 0");
    }

    #[test]
    fn non_boolean_polynomial_has_nonzero_sum() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        // h = [0, 2, 1, 0] — h(01) = 2, not boolean
        let h_table: Vec<Fr> = vec![0, 2, 1, 0].into_iter().map(Fr::from_u64).collect();

        let sum = brute_force_booleanity_sum(&h_table, &eq_table);
        assert!(
            !sum.is_zero(),
            "non-boolean polynomial should have nonzero sum"
        );
    }

    #[test]
    fn round_polynomial_consistency() {
        // Verify that the round polynomial evaluates to the correct partial
        // sums at t=0 and t=1.
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let witness = HammingBooleanityCompute::new(h_table.clone(), eq_table.clone(), num_vars);
        let poly = witness.round_polynomial();

        let one = Fr::one();
        let zero = Fr::zero();

        // s(0) = sum over second half=0 of eq * h * (h-1) at x_0 = 0
        let s0: Fr = (0..4)
            .map(|j| eq_table[2 * j] * h_table[2 * j] * (h_table[2 * j] - one))
            .sum();

        // s(1) = sum over second half=1 of eq * h * (h-1) at x_0 = 1
        let s1: Fr = (0..4)
            .map(|j| eq_table[2 * j + 1] * h_table[2 * j + 1] * (h_table[2 * j + 1] - one))
            .sum();

        assert_eq!(poly.evaluate(zero), s0);
        assert_eq!(poly.evaluate(one), s1);

        // s(0) + s(1) should equal the claimed sum (0 for boolean h)
        let total = s0 + s1;
        assert!(total.is_zero());
    }

    #[test]
    fn full_sumcheck_proof_boolean_polynomial() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        // All-boolean h
        let h_table: Vec<Fr> = (0..(1 << num_vars)).map(|i| Fr::from_u64(i % 2)).collect();

        let claimed_sum = brute_force_booleanity_sum(&h_table, &eq_table);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut witness = HammingBooleanityCompute::new(h_table, eq_table, num_vars);

        let mut prover_transcript = Blake2bTranscript::new(b"test_hamming_booleanity");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert_eq!(proof.round_polynomials.len(), num_vars);

        // Verify the proof
        let mut verifier_transcript = Blake2bTranscript::new(b"test_hamming_booleanity");
        let result = jolt_sumcheck::SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    #[test]
    fn full_sumcheck_proof_random_polynomial() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(456);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        // Random h — not necessarily boolean, so claimed_sum != 0
        let h_table: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum = brute_force_booleanity_sum(&h_table, &eq_table);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut witness = HammingBooleanityCompute::new(h_table, eq_table, num_vars);

        let mut prover_transcript = Blake2bTranscript::new(b"test_random");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_random");
        let result = jolt_sumcheck::SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    #[test]
    fn bind_halves_table_size() {
        let num_vars = 3;
        let h_table = vec![Fr::from_u64(1); 8];
        let eq_table = vec![Fr::from_u64(1); 8];
        let mut witness = HammingBooleanityCompute::new(h_table, eq_table, num_vars);

        assert_eq!(witness.current_size(), 8);
        witness.bind(Fr::from_u64(5));
        assert_eq!(witness.current_size(), 4);
        witness.bind(Fr::from_u64(3));
        assert_eq!(witness.current_size(), 2);
        witness.bind(Fr::from_u64(7));
        assert_eq!(witness.current_size(), 1);
    }
}
