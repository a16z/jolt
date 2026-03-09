//! Generic sum-of-products [`SumcheckCompute`] with eq polynomial.
//!
//! Proves:
//! $$\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot
//!   \sum_{k} c_k \prod_{i \in \text{factors}_k} p_i(x) = v$$
//!
//! where $c_k$ are scalar coefficients, $p_i$ are multilinear polynomials,
//! and $\widetilde{eq}$ is the standard equality polynomial.
//!
//! This is the universal [`SumcheckCompute`] that can evaluate any
//! [`ClaimDefinition`](jolt_ir::ClaimDefinition) formula.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

/// A term in a sum-of-products formula.
///
/// Represents $c \cdot \prod_{i \in \text{factors}} p_i(x)$ where
/// `factors` indexes into the polynomial tables array. Duplicate
/// indices are allowed (e.g., `[0, 0]` for $p_0^2$).
#[derive(Clone, Debug)]
pub struct Term<F: Field> {
    /// Scalar coefficient.
    pub coeff: F,
    /// Indices into the polynomial tables. May contain duplicates.
    pub factors: Vec<usize>,
}

/// Generic sum-of-products sumcheck witness.
///
/// Evaluates the formula:
/// $$f(x) = \widetilde{eq}(r, x) \cdot \sum_{k} c_k \prod_{i \in \text{factors}_k} p_i(x)$$
///
/// Per-variable degree = $1 + \max_k |\text{factors}_k|$.
pub struct FormulaCompute<F: Field> {
    poly_tables: Vec<Vec<F>>,
    eq_table: Vec<F>,
    terms: Vec<Term<F>>,
    degree: usize,
    num_vars: usize,
    round: usize,
}

impl<F: Field> FormulaCompute<F> {
    /// Creates a new witness.
    ///
    /// `degree` must equal $1 + \max_k |\text{factors}_k|$ (the per-variable
    /// degree of the formula including the eq polynomial).
    ///
    /// # Panics
    ///
    /// Panics if any table has wrong length or if factor indices are out of bounds.
    pub fn new(
        poly_tables: Vec<Vec<F>>,
        eq_table: Vec<F>,
        terms: Vec<Term<F>>,
        degree: usize,
        num_vars: usize,
    ) -> Self {
        let expected = 1usize << num_vars;
        assert_eq!(eq_table.len(), expected, "eq_table length mismatch");
        for (i, table) in poly_tables.iter().enumerate() {
            assert_eq!(
                table.len(),
                expected,
                "poly_table[{i}] length {} != {expected}",
                table.len()
            );
        }
        for term in &terms {
            for &idx in &term.factors {
                assert!(
                    idx < poly_tables.len(),
                    "factor index {idx} out of bounds (have {} tables)",
                    poly_tables.len()
                );
            }
        }
        Self {
            poly_tables,
            eq_table,
            terms,
            degree,
            num_vars,
            round: 0,
        }
    }

    #[inline]
    fn current_size(&self) -> usize {
        1usize << (self.num_vars - self.round)
    }

    /// Number of polynomial tables.
    pub fn num_polys(&self) -> usize {
        self.poly_tables.len()
    }
}

impl<F: Field> SumcheckCompute<F> for FormulaCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.current_size() / 2;
        let num_points = self.degree + 1;
        let num_polys = self.poly_tables.len();

        let mut evals = vec![F::zero(); num_points];

        let mut poly_at_t = vec![F::zero(); num_polys];

        for j in 0..half {
            // Pre-compute deltas for each polynomial
            let eq0 = self.eq_table[2 * j];
            let eq1 = self.eq_table[2 * j + 1];
            let deq = eq1 - eq0;

            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = F::from_u64(t as u64);

                let eq_t = eq0 + t_f * deq;

                for (i, table) in self.poly_tables.iter().enumerate() {
                    let p0 = table[2 * j];
                    poly_at_t[i] = p0 + t_f * (table[2 * j + 1] - p0);
                }

                let mut formula_val = F::zero();
                for term in &self.terms {
                    let mut product = term.coeff;
                    for &idx in &term.factors {
                        product *= poly_at_t[idx];
                    }
                    formula_val += product;
                }

                *eval += eq_t * formula_val;
            }
        }

        UnivariatePoly::interpolate_over_integers(&evals)
    }

    fn bind(&mut self, challenge: F) {
        let half = self.current_size() / 2;

        for j in 0..half {
            self.eq_table[j] = self.eq_table[2 * j]
                + challenge * (self.eq_table[2 * j + 1] - self.eq_table[2 * j]);
        }
        self.eq_table.truncate(half);

        for table in &mut self.poly_tables {
            for j in 0..half {
                table[j] = table[2 * j] + challenge * (table[2 * j + 1] - table[2 * j]);
            }
            table.truncate(half);
        }

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
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Brute-force evaluation of eq · Σ_term coeff · Π factors.
    fn brute_force<F: Field>(eq: &[F], polys: &[Vec<F>], terms: &[Term<F>]) -> F {
        let n = eq.len();
        (0..n)
            .map(|x| {
                let mut formula_val = F::zero();
                for term in terms {
                    let mut product = term.coeff;
                    for &idx in &term.factors {
                        product *= polys[idx][x];
                    }
                    formula_val += product;
                }
                eq[x] * formula_val
            })
            .sum()
    }

    #[test]
    fn single_linear_term() {
        // eq · g → degree 2
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let terms = vec![Term {
            coeff: Fr::one(),
            factors: vec![0],
        }];

        let claimed_sum = brute_force(&eq_table, &[g.clone()], &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let mut witness = FormulaCompute::new(vec![g], eq_table, terms, 2, num_vars);

        let mut pt = Blake2bTranscript::new(b"formula_linear");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"formula_linear");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn quadratic_term_h_squared() {
        // eq · h · (h - 1) = eq · h^2 - eq · h → degree 3
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let h: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 % 2)).collect();

        let terms = vec![
            Term {
                coeff: Fr::one(),
                factors: vec![0, 0], // h^2
            },
            Term {
                coeff: -Fr::one(),
                factors: vec![0], // -h
            },
        ];

        let claimed_sum = brute_force(&eq_table, &[h.clone()], &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut witness = FormulaCompute::new(vec![h], eq_table, terms, 3, num_vars);

        let mut pt = Blake2bTranscript::new(b"formula_quad");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"formula_quad");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn multi_poly_product() {
        // eq · (c0·a·b + c1·a·c) → degree 3
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let c: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::from_u64(3);
        let c1 = Fr::from_u64(7);

        let terms = vec![
            Term {
                coeff: c0,
                factors: vec![0, 1], // a · b
            },
            Term {
                coeff: c1,
                factors: vec![0, 2], // a · c
            },
        ];

        let polys = vec![a, b, c];
        let claimed_sum = brute_force(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut witness = FormulaCompute::new(polys, eq_table, terms, 3, num_vars);

        let mut pt = Blake2bTranscript::new(b"formula_multi");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"formula_multi");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn gamma_weighted_linear_combination() {
        // eq · (p0 + γ·p1 + γ²·p2) → degree 2
        // This is the standard claim reduction formula
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(456);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let p0: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let p1: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let p2: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let gamma = Fr::from_u64(13);
        let gamma_sq = gamma * gamma;

        let terms = vec![
            Term {
                coeff: Fr::one(),
                factors: vec![0],
            },
            Term {
                coeff: gamma,
                factors: vec![1],
            },
            Term {
                coeff: gamma_sq,
                factors: vec![2],
            },
        ];

        let polys = vec![p0, p1, p2];
        let claimed_sum = brute_force(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let mut witness = FormulaCompute::new(polys, eq_table, terms, 2, num_vars);

        let mut pt = Blake2bTranscript::new(b"claim_reduction");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"claim_reduction");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok());
    }
}
