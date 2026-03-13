//! Kernel descriptor catalog for sumcheck compositions.
//!
//! Each function returns a [`KernelDescriptor`] for a specific sumcheck
//! formula. Stages compile these descriptors into backend-specific kernels
//! and pass them to [`KernelEvaluator`](super::kernel::KernelEvaluator).
//!
//! For arbitrary sum-of-products formulas, use [`Term`] + [`formula_descriptor`]
//! to build descriptors dynamically.

use jolt_field::Field;
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};

/// Descriptor for a sum-of-products composition on the Toom-Cook grid.
///
/// Produces D evaluations at `{1, ..., D-1, ∞}` for each pair position,
/// summing across `num_products` product groups. Used with
/// [`KernelEvaluator::with_toom_cook_eq`](super::kernel::KernelEvaluator::with_toom_cook_eq)
/// for RA virtual sumchecks.
///
/// Input layout: `opening(g*D + k)` for group `g`, factor `k`.
pub fn product_sum(d: usize, num_products: usize) -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products,
        },
        degree: d,
        tensor_split: None,
    }
}

/// Descriptor for `eq(x) · g(x)` — degree 2, 2 inputs.
///
/// Uses a hand-coded kernel that eliminates stack-VM dispatch overhead.
///
/// Input layout: `opening(0) = eq`, `opening(1) = g`.
pub fn eq_product() -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::EqProduct,
        degree: 2,
        tensor_split: None,
    }
}

/// Descriptor for `eq(x) · h(x) · (h(x) − 1)` — degree 3, 2 inputs.
///
/// Uses a hand-coded kernel that eliminates stack-VM dispatch overhead.
///
/// Input layout: `opening(0) = eq`, `opening(1) = h`.
pub fn hamming_booleanity() -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::HammingBooleanity,
        degree: 3,
        tensor_split: None,
    }
}

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

/// Builds a [`KernelDescriptor`] and challenge values from a sum-of-products
/// formula.
///
/// The kernel encodes `eq(x) · Σ_k c_k · Π_{i ∈ factors_k} p_i(x)` where
/// `eq` is `opening(0)` and each `p_i` is `opening(i+1)`. Term coefficients
/// become `challenge(k)`.
///
/// Returns `(descriptor, challenge_values)` where `challenge_values[k] = terms[k].coeff`.
/// Pass these to [`jolt_cpu::compile_with_challenges`] to compile.
///
/// # Panics
///
/// Panics if `terms` is empty.
pub fn formula_descriptor<F: Field>(
    terms: &[Term<F>],
    num_polys: usize,
    degree: usize,
) -> (KernelDescriptor, Vec<F>) {
    assert!(!terms.is_empty(), "formula must have at least one term");
    let b = ExprBuilder::new();
    let eq = b.opening(0);

    let challenges: Vec<F> = terms.iter().map(|t| t.coeff).collect();

    let mut sum = {
        let term = &terms[0];
        let mut product = b.challenge(0);
        for &idx in &term.factors {
            product = product * b.opening(idx as u32 + 1);
        }
        product
    };
    for (k, term) in terms.iter().enumerate().skip(1) {
        let mut product = b.challenge(k as u32);
        for &idx in &term.factors {
            product = product * b.opening(idx as u32 + 1);
        }
        sum = sum + product;
    }

    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(eq * sum),
            num_inputs: num_polys + 1, // +1 for eq
        },
        degree,
        tensor_split: None,
    };
    (desc, challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use jolt_compute::ComputeBackend;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::prover::SumcheckCompute;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::evaluators::kernel::KernelEvaluator;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    #[test]
    fn eq_product_round_polynomial_consistency() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let s0: Fr = (0..n / 2).map(|j| eq_table[2 * j] * g_table[2 * j]).sum();
        let s1: Fr = (0..n / 2)
            .map(|j| eq_table[2 * j + 1] * g_table[2 * j + 1])
            .sum();

        let desc = eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        // P(1) is derived from the claim, so set it before round_polynomial()
        witness.set_claim(s0 + s1);
        let poly = witness.round_polynomial();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);
    }

    #[test]
    fn eq_product_full_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum: Fr = eq_table
            .iter()
            .zip(g_table.iter())
            .map(|(&e, &g)| e * g)
            .sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let desc = eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"eq_product_test");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"eq_product_test");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

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

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let sum = brute_force_booleanity_sum(&h_table, &eq_table);
        assert!(sum.is_zero(), "boolean polynomial should have sum 0");
    }

    #[test]
    fn hamming_round_polynomial_consistency() {
        let backend = cpu();
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let one = Fr::one();
        let s0: Fr = (0..4)
            .map(|j| eq_table[2 * j] * h_table[2 * j] * (h_table[2 * j] - one))
            .sum();
        let s1: Fr = (0..4)
            .map(|j| eq_table[2 * j + 1] * h_table[2 * j + 1] * (h_table[2 * j + 1] - one))
            .sum();

        let desc = hamming_booleanity();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        witness.set_claim(s0 + s1);
        let poly = witness.round_polynomial();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);

        let total = s0 + s1;
        assert!(total.is_zero());
    }

    #[test]
    fn hamming_full_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let h_table: Vec<Fr> = (0..(1 << num_vars)).map(|i| Fr::from_u64(i % 2)).collect();

        let claimed_sum = brute_force_booleanity_sum(&h_table, &eq_table);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let desc = hamming_booleanity();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"test_hamming_booleanity");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"test_hamming_booleanity");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    fn brute_force_formula(eq: &[Fr], polys: &[Vec<Fr>], terms: &[Term<Fr>]) -> Fr {
        let n = eq.len();
        (0..n)
            .map(|x| {
                let mut formula_val = Fr::zero();
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
    fn formula_single_linear_term() {
        let backend = cpu();
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

        let claimed_sum = brute_force_formula(&eq_table, &[g.clone()], &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 1, 2);
        let kernel = jolt_cpu::compile_with_challenges::<Fr>(&desc, &challenges);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"formula_linear");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"formula_linear");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn formula_quadratic_term_h_squared() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let h: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 % 2)).collect();

        let terms = vec![
            Term {
                coeff: Fr::one(),
                factors: vec![0, 0],
            },
            Term {
                coeff: -Fr::one(),
                factors: vec![0],
            },
        ];

        let claimed_sum = brute_force_formula(&eq_table, &[h.clone()], &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 1, 3);
        let kernel = jolt_cpu::compile_with_challenges::<Fr>(&desc, &challenges);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"formula_quad");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"formula_quad");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn formula_multi_poly_product() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let b_vec: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let c: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::from_u64(3);
        let c1 = Fr::from_u64(7);

        let terms = vec![
            Term {
                coeff: c0,
                factors: vec![0, 1],
            },
            Term {
                coeff: c1,
                factors: vec![0, 2],
            },
        ];

        let polys = vec![a.clone(), b_vec.clone(), c.clone()];
        let claimed_sum = brute_force_formula(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 3, 3);
        let kernel = jolt_cpu::compile_with_challenges::<Fr>(&desc, &challenges);
        let inputs = vec![
            backend.upload(&eq_table),
            backend.upload(&a),
            backend.upload(&b_vec),
            backend.upload(&c),
        ];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"formula_multi");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"formula_multi");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn formula_gamma_weighted_linear_combination() {
        let backend = cpu();
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

        let polys = vec![p0.clone(), p1.clone(), p2.clone()];
        let claimed_sum = brute_force_formula(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 3, 2);
        let kernel = jolt_cpu::compile_with_challenges::<Fr>(&desc, &challenges);
        let inputs = vec![
            backend.upload(&eq_table),
            backend.upload(&p0),
            backend.upload(&p1),
            backend.upload(&p2),
        ];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"claim_reduction");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"claim_reduction");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok());
    }
}
