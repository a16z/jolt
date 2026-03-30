//! Cross-integration tests for jolt-cpu with jolt-compute and jolt-ir.
//!
//! Compiles composition formulas from IR, then executes them through the
//! CpuBackend pairwise_reduce pipeline and verifies correctness.

use jolt_compute::{BindingOrder, ComputeBackend, EqInput};
use jolt_cpu::{compile, compile_with_challenges, from_ir_formula, CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use jolt_compiler::{CompositionFormula, Factor, ProductTerm};
use jolt_ir::ExprBuilder;
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn backend() -> CpuBackend {
    CpuBackend
}

fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
    let mut out = vec![Fr::zero(); n];
    kernel.evaluate(lo, hi, &mut out);
    out
}

/// Helper: build a pure product-sum formula with `p` groups of `d` consecutive inputs.
fn product_sum_formula(d: usize, p: usize) -> CompositionFormula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    CompositionFormula::from_terms(terms)
}

/// Reference: compute Toom-Cook evaluations for a sum-of-products composition
/// by evaluating directly at each grid point {1, 2, ..., D-1, ∞}.
fn reference_toom_cook_reduce(
    inputs: &[&Vec<Fr>],
    weights: &[Fr],
    d: usize,
    num_products: usize,
) -> Vec<Fr> {
    let n = inputs[0].len();
    let half = n / 2;
    let total_inputs = d * num_products;
    debug_assert_eq!(inputs.len(), total_inputs);

    let mut result = vec![Fr::from_u64(0); d];

    for (pair_idx, &w) in weights.iter().enumerate().take(half) {
        // Finite grid points 1, 2, ..., D-1
        for (t_idx, res) in result[..(d - 1)].iter_mut().enumerate() {
            let t_f = Fr::from_u64((t_idx + 1) as u64);

            let mut sum = Fr::from_u64(0);
            for g in 0..num_products {
                let mut prod = Fr::one();
                for j in 0..d {
                    let k = g * d + j;
                    let lo = inputs[k][2 * pair_idx];
                    let hi = inputs[k][2 * pair_idx + 1];
                    let delta = hi - lo;
                    prod *= lo + t_f * delta;
                }
                sum += prod;
            }
            *res += w * sum;
        }

        // t = ∞: product of slopes
        let mut sum_inf = Fr::from_u64(0);
        for g in 0..num_products {
            let mut prod = Fr::one();
            for j in 0..d {
                let k = g * d + j;
                let lo = inputs[k][2 * pair_idx];
                let hi = inputs[k][2 * pair_idx + 1];
                prod *= hi - lo;
            }
            sum_inf += prod;
        }
        result[d - 1] += w * sum_inf;
    }

    result
}

// ProductSum through pairwise_reduce

/// Compile a D=4 ProductSum kernel, run through pairwise_reduce, verify result.
#[test]
fn product_sum_d4_via_pairwise_reduce() {
    let b = backend();
    let formula = product_sum_formula(4, 1);
    let kernel = compile::<Fr>(&formula);

    let num_pairs = 4;
    let bufs: Vec<Vec<Fr>> = (0..4)
        .map(|k| {
            (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 100 + i + 1) as u64))
                .collect::<Vec<_>>()
        })
        .map(|data| b.upload(&data))
        .collect();

    let weights = b.upload(&vec![Fr::from_u64(1); num_pairs]);
    let buf_refs: Vec<&Vec<Fr>> = bufs.iter().collect();
    let result = b.pairwise_reduce(
        &buf_refs,
        EqInput::Weighted(&weights),
        &kernel,
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(result.len(), 4); // D outputs (Toom-Cook)

    let expected = reference_toom_cook_reduce(&buf_refs, &vec![Fr::from_u64(1); num_pairs], 4, 1);
    assert_eq!(result, expected);
}

/// D=8 ProductSum with multiple product groups.
#[test]
fn product_sum_d8_multiple_groups() {
    let b = backend();
    let num_products = 2;
    let d = 8;
    let formula = product_sum_formula(d, num_products);
    let kernel = compile::<Fr>(&formula);

    let num_pairs = 2;
    let total_inputs = d * num_products;
    let bufs: Vec<Vec<Fr>> = (0..total_inputs)
        .map(|k| {
            (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 10 + i + 1) as u64))
                .collect::<Vec<_>>()
        })
        .map(|data| b.upload(&data))
        .collect();

    let weights_data = vec![Fr::from_u64(1); num_pairs];
    let weights = b.upload(&weights_data);
    let buf_refs: Vec<&Vec<Fr>> = bufs.iter().collect();
    let result = b.pairwise_reduce(
        &buf_refs,
        EqInput::Weighted(&weights),
        &kernel,
        formula.degree(),
        BindingOrder::LowToHigh,
    );
    assert_eq!(result.len(), d);

    let expected = reference_toom_cook_reduce(&buf_refs, &weights_data, d, num_products);
    assert_eq!(result, expected, "D=8 P=2 Toom-Cook mismatch");
}

// Custom kernel through pairwise_reduce

/// Compile a D=3 P=1 product-sum formula via ExprBuilder and verify Toom-Cook
/// evaluations through pairwise_reduce.
#[test]
fn custom_product_via_pairwise_reduce() {
    let b = backend();
    let formula = product_sum_formula(3, 1);
    let kernel = compile::<Fr>(&formula);

    let num_pairs = 3;
    let bufs: Vec<Vec<Fr>> = (0..3)
        .map(|k| {
            (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 10 + i + 1) as u64))
                .collect::<Vec<_>>()
        })
        .map(|data| b.upload(&data))
        .collect();

    let weights = b.upload(&vec![Fr::from_u64(1); num_pairs]);
    let buf_refs: Vec<&Vec<Fr>> = bufs.iter().collect();

    let result = b.pairwise_reduce(
        &buf_refs,
        EqInput::Weighted(&weights),
        &kernel,
        formula.degree(),
        BindingOrder::LowToHigh,
    );
    assert_eq!(result.len(), 3);

    let expected = reference_toom_cook_reduce(&buf_refs, &vec![Fr::from_u64(1); num_pairs], 3, 1);
    assert_eq!(result, expected, "D=3 P=1 Toom-Cook mismatch");
}

/// Compile eq_product (`o0 * o1`) — dispatches to the standard-grid
/// specialized kernel (not Toom-Cook).
#[test]
fn eq_product_via_pairwise_reduce() {
    let _b = backend();
    let eb = ExprBuilder::new();
    let o0 = eb.opening(0);
    let o1 = eb.opening(1);
    let expr = eb.build(o0 * o1);

    let formula = from_ir_formula(&expr.to_composition_formula());
    let kernel = compile::<Fr>(&formula);

    // Standard grid {0, 2}: result[0] = lo[0]*lo[1], result[1] = (lo+2δ)·(lo+2δ)
    let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
    let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
    let result = eval_kernel(&kernel, &lo, &hi, formula.degree());
    assert_eq!(result[0], Fr::from_u64(15)); // 3*5
    assert_eq!(result[1], Fr::from_u64(187)); // 11*17
}

// Custom kernel with challenges

/// Custom kernel: c0 * o0 * o1 — challenge-weighted product of two openings.
///
/// This formula contains a challenge factor, so `as_product_sum()` returns
/// `None` and the formula goes through the generic formula compiler with the
/// challenge baked in.
#[test]
fn custom_with_challenge_via_pairwise_reduce() {
    let b = backend();
    let eb = ExprBuilder::new();
    let o0 = eb.opening(0);
    let o1 = eb.opening(1);
    let c0 = eb.challenge(0);
    let expr = eb.build(c0 * o0 * o1);

    let formula = from_ir_formula(&expr.to_composition_formula());

    let gamma = Fr::from_u64(42);
    let kernel = compile_with_challenges::<Fr>(&formula, &[gamma]);

    let num_pairs = 3;
    let buf_a = b.upload(
        &(0..num_pairs * 2)
            .map(|i| Fr::from_u64(i as u64 + 1))
            .collect::<Vec<_>>(),
    );
    let buf_b = b.upload(
        &(0..num_pairs * 2)
            .map(|i| Fr::from_u64(i as u64 + 10))
            .collect::<Vec<_>>(),
    );
    let weights = b.upload(&vec![Fr::from_u64(1); num_pairs]);

    // Degree 2 (challenge is a constant): standard grid {0, 2}, 2 evaluations
    let result = b.pairwise_reduce(
        &[&buf_a, &buf_b],
        EqInput::Weighted(&weights),
        &kernel,
        formula.degree(),
        BindingOrder::LowToHigh,
    );
    assert_eq!(result.len(), 2);

    // Verify t=0 (first eval): gamma * Σ lo_a[i] * lo_b[i]
    let mut t0 = Fr::from_u64(0);
    for i in 0..num_pairs {
        let lo_a = Fr::from_u64((i * 2 + 1) as u64);
        let lo_b = Fr::from_u64((i * 2 + 10) as u64);
        t0 += lo_a * lo_b;
    }
    assert_eq!(result[0], gamma * t0, "t=0 mismatch");
}

// Kernel evaluation matches manual polynomial evaluation

/// For a ProductSum kernel, verify the Toom-Cook evaluations match the
/// reference computation at each grid point.
#[test]
fn kernel_toom_cook_consistency() {
    let mut rng = ChaCha20Rng::seed_from_u64(9000);
    let d = 4;
    let num_products = 2;

    let formula = product_sum_formula(d, num_products);
    let kernel = compile::<Fr>(&formula);

    let total_inputs = d * num_products;
    let lo: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();
    let hi: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();

    let evals = eval_kernel(&kernel, &lo, &hi, formula.degree());
    assert_eq!(evals.len(), d);

    // Verify against Toom-Cook grid: {1, 2, ..., D-1, ∞}
    for (t_idx, eval) in evals[..(d - 1)].iter().enumerate() {
        let t = t_idx + 1;
        let t_f = Fr::from_u64(t as u64);
        let mut expected = Fr::from_u64(0);
        for g in 0..num_products {
            let mut prod = Fr::one();
            for j in 0..d {
                let k = g * d + j;
                let val = lo[k] + t_f * (hi[k] - lo[k]);
                prod *= val;
            }
            expected += prod;
        }
        assert_eq!(*eval, expected, "grid point t={t} mismatch");
    }

    // Verify P(∞): product of slopes
    let mut expected_inf = Fr::from_u64(0);
    for g in 0..num_products {
        let mut prod = Fr::one();
        for j in 0..d {
            let k = g * d + j;
            prod *= hi[k] - lo[k];
        }
        expected_inf += prod;
    }
    assert_eq!(evals[d - 1], expected_inf, "P(∞) mismatch");
}
