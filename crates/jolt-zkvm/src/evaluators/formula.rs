//! Generic sum-of-products [`SumcheckCompute`] evaluator with eq polynomial.
//!
//! Proves:
//! $$\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot
//!   \sum_{k} c_k \prod_{i \in \text{factors}_k} p_i(x) = v$$
//!
//! where $c_k$ are scalar coefficients, $p_i$ are multilinear polynomials,
//! and $\widetilde{eq}$ is the standard equality polynomial.
//!
//! This is the universal [`SumcheckCompute`] evaluator that can handle any
//! [`ClaimDefinition`](jolt_ir::ClaimDefinition) formula.
//!
//! Thin wrapper around [`KernelEvaluator`](super::kernel::KernelEvaluator)
//! using a `Custom` kernel built from the term coefficients.

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

use super::kernel::KernelEvaluator;

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
/// Pass these to [`jolt_cpu_kernels::compile_with_challenges`] to compile.
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

    // Build: Σ_k challenge(k) · Π_{j} opening(factors[j] + 1)
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

/// Generic sum-of-products sumcheck evaluator.
///
/// Evaluates the formula:
/// $$f(x) = \widetilde{eq}(r, x) \cdot \sum_{k} c_k \prod_{i \in \text{factors}_k} p_i(x)$$
///
/// Per-variable degree = $1 + \max_k |\text{factors}_k|$.
///
/// Delegates all computation to [`KernelEvaluator`] via a compiled kernel
/// where term coefficients are baked as challenges.
pub struct FormulaEvaluator<F: Field, B: ComputeBackend> {
    inner: KernelEvaluator<F, B>,
}

impl<F: Field, B: ComputeBackend> FormulaEvaluator<F, B> {
    /// Creates a new evaluator from pre-uploaded buffers.
    ///
    /// `kernel` must be compiled from the descriptor returned by
    /// [`formula_descriptor`] with the corresponding challenge values.
    pub fn new(
        eq_table: B::Buffer<F>,
        poly_tables: Vec<B::Buffer<F>>,
        kernel: B::CompiledKernel<F>,
        degree: usize,
        backend: Arc<B>,
    ) -> Self {
        let mut inputs = Vec::with_capacity(1 + poly_tables.len());
        inputs.push(eq_table);
        inputs.extend(poly_tables);
        Self {
            inner: KernelEvaluator::with_unit_weights(inputs, kernel, degree + 1, backend),
        }
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for FormulaEvaluator<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        self.inner.round_polynomial()
    }

    fn bind(&mut self, challenge: F) {
        self.inner.bind(challenge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compute::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    fn brute_force(eq: &[Fr], polys: &[Vec<Fr>], terms: &[Term<Fr>]) -> Fr {
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
    fn single_linear_term() {
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

        let claimed_sum = brute_force(&eq_table, &[g.clone()], &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 1, 2);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<Fr>(&desc, &challenges);

        let mut witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            vec![backend.upload(&g)],
            kernel,
            2,
            backend,
        );

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

        let (desc, challenges) = formula_descriptor(&terms, 1, 3);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<Fr>(&desc, &challenges);

        let mut witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            vec![backend.upload(&h)],
            kernel,
            3,
            backend,
        );

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
                factors: vec![0, 1], // a · b
            },
            Term {
                coeff: c1,
                factors: vec![0, 2], // a · c
            },
        ];

        let polys = vec![a.clone(), b_vec.clone(), c.clone()];
        let claimed_sum = brute_force(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 3, 3);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<Fr>(&desc, &challenges);

        let mut witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            vec![
                backend.upload(&a),
                backend.upload(&b_vec),
                backend.upload(&c),
            ],
            kernel,
            3,
            backend,
        );

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
        let claimed_sum = brute_force(&eq_table, &polys, &terms);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let (desc, challenges) = formula_descriptor(&terms, 3, 2);
        let kernel = jolt_cpu_kernels::compile_with_challenges::<Fr>(&desc, &challenges);

        let mut witness = FormulaEvaluator::new(
            backend.upload(&eq_table),
            vec![
                backend.upload(&p0),
                backend.upload(&p1),
                backend.upload(&p2),
            ],
            kernel,
            2,
            backend,
        );

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
