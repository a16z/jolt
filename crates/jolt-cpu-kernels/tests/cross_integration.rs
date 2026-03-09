//! Cross-integration tests for jolt-cpu-kernels with jolt-compute and jolt-ir.
//!
//! Compiles kernel descriptors from IR, then executes them through the
//! CpuBackend pairwise_reduce pipeline and verifies correctness.

use jolt_compute::{ComputeBackend, CpuBackend, CpuKernel};
use jolt_cpu_kernels::{compile, compile_with_challenges};
use jolt_field::{Field, Fr};
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
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
    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 1,
        },
        degree: 4,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);

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
    let result = b.pairwise_reduce(&buf_refs, &weights, &kernel, desc.num_evals());

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
    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products,
        },
        degree: d,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);

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
    let result = b.pairwise_reduce(&buf_refs, &weights, &kernel, desc.num_evals());
    assert_eq!(result.len(), d);

    let expected = reference_toom_cook_reduce(&buf_refs, &weights_data, d, num_products);
    assert_eq!(result, expected, "D=8 P=2 Toom-Cook mismatch");
}

// Custom kernel through pairwise_reduce

/// Custom kernel: o0 * o1 (simple product of two openings).
#[test]
fn custom_product_via_pairwise_reduce() {
    let b = backend();
    let eb = ExprBuilder::new();
    let o0 = eb.opening(0);
    let o1 = eb.opening(1);
    let expr = eb.build(o0 * o1);

    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            num_inputs: 2,
            expr,
        },
        degree: 2,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);

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

    // Custom degree-2: num_evals = degree + 1 = 3
    let result = b.pairwise_reduce(&[&buf_a, &buf_b], &weights, &kernel, desc.num_evals());
    assert_eq!(result.len(), 3);

    // t=0: sum over pairs of (lo_a * lo_b)
    let mut t0 = Fr::from_u64(0);
    for i in 0..num_pairs {
        let lo_a = Fr::from_u64((i * 2 + 1) as u64);
        let lo_b = Fr::from_u64((i * 2 + 10) as u64);
        t0 += lo_a * lo_b;
    }
    assert_eq!(result[0], t0, "t=0 mismatch");
}

// Custom kernel with challenges

/// Custom kernel: c0 * (o0^2 - o0) — booleanity check weighted by challenge.
#[test]
fn custom_with_challenge_via_pairwise_reduce() {
    let b = backend();
    let eb = ExprBuilder::new();
    let o0 = eb.opening(0);
    let c0 = eb.challenge(0);
    let expr = eb.build(c0 * (o0 * o0 - o0));

    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            num_inputs: 1,
            expr,
        },
        degree: 2,
        tensor_split: None,
    };

    let gamma = Fr::from_u64(42);
    let kernel = compile_with_challenges::<Fr>(&desc, &[gamma]);

    // Boolean values: kernel evaluates to 0 at t=0 and t=1
    let data: Vec<Fr> = vec![
        Fr::from_u64(0),
        Fr::from_u64(1), // pair 0
        Fr::from_u64(1),
        Fr::from_u64(0), // pair 1
        Fr::from_u64(0),
        Fr::from_u64(0), // pair 2
    ];
    let buf = b.upload(&data);
    let weights = b.upload(&[Fr::from_u64(1); 3]);

    // Custom degree-2: num_evals = degree + 1 = 3
    let result = b.pairwise_reduce(&[&buf], &weights, &kernel, desc.num_evals());
    assert_eq!(result.len(), 3);

    // t=0: gamma * (lo^2 - lo) for each pair, with lo in {0, 1, 0}
    assert_eq!(
        result[0],
        Fr::from_u64(0),
        "boolean inputs should give 0 at t=0"
    );
    // t=1: gamma * (hi^2 - hi) for each pair, with hi in {1, 0, 0}
    assert_eq!(
        result[1],
        Fr::from_u64(0),
        "boolean inputs should give 0 at t=1"
    );
}

// Kernel evaluation matches manual polynomial evaluation

/// For a ProductSum kernel, verify the Toom-Cook evaluations match the
/// reference computation at each grid point.
#[test]
fn kernel_toom_cook_consistency() {
    let mut rng = ChaCha20Rng::seed_from_u64(9000);
    let d = 4;
    let num_products = 2;

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products,
        },
        degree: d,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);

    let total_inputs = d * num_products;
    let lo: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();
    let hi: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();

    let evals = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
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
