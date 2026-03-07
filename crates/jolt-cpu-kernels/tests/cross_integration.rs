//! Cross-integration tests for jolt-cpu-kernels with jolt-compute and jolt-ir.
//!
//! Compiles kernel descriptors from IR, then executes them through the
//! CpuBackend pairwise_reduce pipeline and verifies correctness.

use jolt_compute::{ComputeBackend, CpuBackend};
use jolt_cpu_kernels::{compile, compile_with_challenges};
use jolt_field::{Field, Fr};
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn backend() -> CpuBackend {
    CpuBackend
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
    let result = b.pairwise_reduce(&buf_refs, &weights, &kernel, 4);

    assert_eq!(result.len(), 5); // degree + 1

    // Verify t=0: sum over pairs of product of lo values
    let mut t0 = Fr::from_u64(0);
    for pair_idx in 0..num_pairs {
        let mut prod = Fr::from_u64(1);
        for k in 0..4 {
            let lo = Fr::from_u64((k * 100 + pair_idx * 2 + 1) as u64);
            prod *= lo;
        }
        t0 += prod;
    }
    assert_eq!(result[0], t0, "t=0 mismatch");

    // Verify t=1: sum over pairs of product of hi values
    let mut t1 = Fr::from_u64(0);
    for pair_idx in 0..num_pairs {
        let mut prod = Fr::from_u64(1);
        for k in 0..4 {
            let hi = Fr::from_u64((k * 100 + pair_idx * 2 + 2) as u64);
            prod *= hi;
        }
        t1 += prod;
    }
    assert_eq!(result[1], t1, "t=1 mismatch");
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

    let weights = b.upload(&vec![Fr::from_u64(1); num_pairs]);
    let buf_refs: Vec<&Vec<Fr>> = bufs.iter().collect();
    let result = b.pairwise_reduce(&buf_refs, &weights, &kernel, d);
    assert_eq!(result.len(), d + 1);

    let mut t0 = Fr::from_u64(0);
    for pair_idx in 0..num_pairs {
        for group in 0..num_products {
            let mut prod = Fr::from_u64(1);
            for j in 0..d {
                let k = group * d + j;
                let lo = Fr::from_u64((k * 10 + pair_idx * 2 + 1) as u64);
                prod *= lo;
            }
            t0 += prod;
        }
    }
    assert_eq!(result[0], t0, "t=0 mismatch for D=8 P=2");
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

    let result = b.pairwise_reduce(&[&buf_a, &buf_b], &weights, &kernel, 2);
    assert_eq!(result.len(), 3);

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

    let result = b.pairwise_reduce(&[&buf], &weights, &kernel, 2);
    assert_eq!(result.len(), 3);

    // t=0: gamma * (lo^2 - lo) for each pair, with lo in {0, 1, 0}
    assert_eq!(result[0], Fr::from_u64(0), "boolean inputs should give 0 at t=0");
    // t=1: gamma * (hi^2 - hi) for each pair, with hi in {1, 0, 0}
    assert_eq!(result[1], Fr::from_u64(0), "boolean inputs should give 0 at t=1");
}

// Kernel evaluation matches manual polynomial evaluation

/// For a degree-D ProductSum kernel, evaluating the round polynomial at a
/// random point should match summing the product over interpolated values.
#[test]
fn kernel_interpolation_consistency() {
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

    let evals = kernel.evaluate(&lo, &hi, d);
    assert_eq!(evals.len(), d + 1);

    for (t, eval) in evals.iter().enumerate() {
        let t_f = Fr::from_u64(t as u64);
        let mut expected = Fr::from_u64(0);
        for g in 0..num_products {
            let mut prod = Fr::from_u64(1);
            for j in 0..d {
                let k = g * d + j;
                let val = lo[k] + t_f * (hi[k] - lo[k]);
                prod *= val;
            }
            expected += prod;
        }
        assert_eq!(*eval, expected, "grid point t={t} mismatch");
    }
}
