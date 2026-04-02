//! Cross-integration tests for jolt-cpu with jolt-compute.
//!
//! Compiles composition formulas via KernelSpec, then executes them through the
//! CpuBackend reduce pipeline and verifies correctness.

use jolt_compiler::{BindingOrder, Factor, Formula, Iteration, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::{compile, CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn backend() -> CpuBackend {
    CpuBackend
}

fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
    let mut out = vec![Fr::zero(); n];
    kernel.evaluate(lo, hi, &[], &mut out);
    out
}

fn make_spec(formula: &Formula) -> KernelSpec {
    KernelSpec {
        num_evals: formula.degree(),
        formula: formula.clone(),
        iteration: Iteration::Dense,
        binding_order: BindingOrder::LowToHigh,
    }
}

/// Build a pure product-sum formula with `p` groups of `d` consecutive inputs.
fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

/// Compute Toom-Cook evaluations for a dense sum-of-products composition
/// by evaluating directly at each grid point {1, 2, ..., D-1, ∞}.
fn reference_toom_cook_reduce(inputs: &[&Vec<Fr>], d: usize, num_products: usize) -> Vec<Fr> {
    let n = inputs[0].len();
    let half = n / 2;
    let total_inputs = d * num_products;
    debug_assert_eq!(inputs.len(), total_inputs);

    let mut result = vec![Fr::from_u64(0); d];

    for pair_idx in 0..half {
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
            *res += sum;
        }

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
        result[d - 1] += sum_inf;
    }

    result
}

#[test]
fn product_sum_d4_via_reduce() {
    let b = backend();
    let formula = product_sum_formula(4, 1);
    let kernel = compile::<Fr>(&make_spec(&formula));

    let num_pairs = 4;
    let bufs: Vec<Buf<CpuBackend, Fr>> = (0..4)
        .map(|k| {
            let data: Vec<Fr> = (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 100 + i + 1) as u64))
                .collect();
            DeviceBuffer::Field(b.upload(&data))
        })
        .collect();

    let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
    let result = b.reduce(&kernel, &buf_refs, &[]);

    assert_eq!(result.len(), 4);

    let raw_refs: Vec<&Vec<Fr>> = bufs.iter().map(|db| db.as_field()).collect();
    let expected = reference_toom_cook_reduce(&raw_refs, 4, 1);
    assert_eq!(result, expected);
}

/// D=8 ProductSum with multiple product groups.
#[test]
fn product_sum_d8_multiple_groups() {
    let b = backend();
    let num_products = 2;
    let d = 8;
    let formula = product_sum_formula(d, num_products);
    let kernel = compile::<Fr>(&make_spec(&formula));

    let num_pairs = 2;
    let total_inputs = d * num_products;
    let bufs: Vec<Buf<CpuBackend, Fr>> = (0..total_inputs)
        .map(|k| {
            let data: Vec<Fr> = (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 10 + i + 1) as u64))
                .collect();
            DeviceBuffer::Field(b.upload(&data))
        })
        .collect();

    let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
    let result = b.reduce(&kernel, &buf_refs, &[]);
    assert_eq!(result.len(), d);

    let raw_refs: Vec<&Vec<Fr>> = bufs.iter().map(|db| db.as_field()).collect();
    let expected = reference_toom_cook_reduce(&raw_refs, d, num_products);
    assert_eq!(result, expected, "D=8 P=2 Toom-Cook mismatch");
}

#[test]
fn custom_product_via_reduce() {
    let b = backend();
    let formula = product_sum_formula(3, 1);
    let kernel = compile::<Fr>(&make_spec(&formula));

    let num_pairs = 3;
    let bufs: Vec<Buf<CpuBackend, Fr>> = (0..3)
        .map(|k| {
            let data: Vec<Fr> = (0..num_pairs * 2)
                .map(|i| Fr::from_u64((k * 10 + i + 1) as u64))
                .collect();
            DeviceBuffer::Field(b.upload(&data))
        })
        .collect();

    let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
    let result = b.reduce(&kernel, &buf_refs, &[]);
    assert_eq!(result.len(), 3);

    let raw_refs: Vec<&Vec<Fr>> = bufs.iter().map(|db| db.as_field()).collect();
    let expected = reference_toom_cook_reduce(&raw_refs, 3, 1);
    assert_eq!(result, expected, "D=3 P=1 Toom-Cook mismatch");
}

/// Compile eq_product (`o0 * o1`) -- dispatches to Toom-Cook (d=2, p=1).
#[test]
fn eq_product_via_reduce() {
    let formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Input(0), Factor::Input(1)],
    }]);
    let kernel = compile::<Fr>(&make_spec(&formula));

    // Toom-Cook grid {1, ∞}: P(1) = hi[0]*hi[1], P(∞) = delta product
    let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
    let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
    let result = eval_kernel(&kernel, &lo, &hi, formula.degree());
    assert_eq!(result[0], Fr::from_u64(77)); // 7*11
    assert_eq!(result[1], Fr::from_u64(24)); // 4*6
}

/// Custom kernel: c0 * o0 * o1 -- challenge-weighted product of two openings.
///
/// This formula contains a challenge factor, so `as_product_sum()` returns
/// `None` and the formula goes through the generic formula compiler with the
/// challenge baked in.
#[test]
fn custom_with_challenge_via_reduce() {
    let b = backend();
    let formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(1)],
    }]);

    let gamma = Fr::from_u64(42);
    let kernel = compile::<Fr>(&make_spec(&formula));

    let num_pairs = 3;
    let buf_a: Buf<CpuBackend, Fr> = DeviceBuffer::Field(
        b.upload(
            &(0..num_pairs * 2)
                .map(|i| Fr::from_u64(i as u64 + 1))
                .collect::<Vec<_>>(),
        ),
    );
    let buf_b: Buf<CpuBackend, Fr> = DeviceBuffer::Field(
        b.upload(
            &(0..num_pairs * 2)
                .map(|i| Fr::from_u64(i as u64 + 10))
                .collect::<Vec<_>>(),
        ),
    );

    let result = b.reduce(&kernel, &[&buf_a, &buf_b], &[gamma]);
    assert_eq!(result.len(), 2);

    // Verify t=0 (first eval): gamma * sum lo_a[i] * lo_b[i]
    let mut t0 = Fr::from_u64(0);
    for i in 0..num_pairs {
        let lo_a = Fr::from_u64((i * 2 + 1) as u64);
        let lo_b = Fr::from_u64((i * 2 + 10) as u64);
        t0 += lo_a * lo_b;
    }
    assert_eq!(result[0], gamma * t0, "t=0 mismatch");
}

/// For a ProductSum kernel, verify the Toom-Cook evaluations match the
/// reference computation at each grid point.
#[test]
fn kernel_toom_cook_consistency() {
    let mut rng = ChaCha20Rng::seed_from_u64(9000);
    let d = 4;
    let num_products = 2;

    let formula = product_sum_formula(d, num_products);
    let kernel = compile::<Fr>(&make_spec(&formula));

    let total_inputs = d * num_products;
    let lo: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();
    let hi: Vec<Fr> = (0..total_inputs).map(|_| Fr::random(&mut rng)).collect();

    let evals = eval_kernel(&kernel, &lo, &hi, formula.degree());
    assert_eq!(evals.len(), d);

    // Verify against Toom-Cook grid: {1, 2, ..., D-1, inf}
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

    // Verify P(inf): product of slopes
    let mut expected_inf = Fr::from_u64(0);
    for g in 0..num_products {
        let mut prod = Fr::one();
        for j in 0..d {
            let k = g * d + j;
            prod *= hi[k] - lo[k];
        }
        expected_inf += prod;
    }
    assert_eq!(evals[d - 1], expected_inf, "P(inf) mismatch");
}

fn make_sparse_spec(formula: &Formula) -> KernelSpec {
    KernelSpec {
        num_evals: formula.degree(),
        formula: formula.clone(),
        iteration: Iteration::Sparse,
        binding_order: BindingOrder::LowToHigh,
    }
}

/// Compute Toom-Cook evaluations for a sparse product-sum composition
/// by manually iterating over sorted keys.
fn reference_sparse_toom_cook_reduce(
    value_inputs: &[Vec<Fr>],
    keys: &[u64],
    d: usize,
    num_products: usize,
) -> Vec<Fr> {
    let num_inputs = d * num_products;
    assert_eq!(value_inputs.len(), num_inputs);

    let mut result = vec![Fr::zero(); d];
    let n = keys.len();
    let mut i = 0;

    while i < n {
        let key = keys[i];
        let mut lo = vec![Fr::zero(); num_inputs];
        let mut hi = vec![Fr::zero(); num_inputs];

        if key.is_multiple_of(2) {
            for (k, input) in value_inputs.iter().enumerate() {
                lo[k] = input[i];
            }
            if i + 1 < n && keys[i + 1] == key + 1 {
                for (k, input) in value_inputs.iter().enumerate() {
                    hi[k] = input[i + 1];
                }
                i += 2;
            } else {
                i += 1;
            }
        } else {
            for (k, input) in value_inputs.iter().enumerate() {
                hi[k] = input[i];
            }
            i += 1;
        }

        for (t_idx, res) in result[..(d - 1)].iter_mut().enumerate() {
            let t_f = Fr::from_u64((t_idx + 1) as u64);
            let mut sum = Fr::zero();
            for g in 0..num_products {
                let mut prod = Fr::one();
                for j in 0..d {
                    let k = g * d + j;
                    prod *= lo[k] + t_f * (hi[k] - lo[k]);
                }
                sum += prod;
            }
            *res += sum;
        }

        let mut sum_inf = Fr::zero();
        for g in 0..num_products {
            let mut prod = Fr::one();
            for j in 0..d {
                let k = g * d + j;
                prod *= hi[k] - lo[k];
            }
            sum_inf += prod;
        }
        result[d - 1] += sum_inf;
    }

    result
}

/// Sparse D=4 P=1 through reduce: fully paired entries.
#[test]
fn sparse_product_d4_via_reduce() {
    let b = backend();
    let d = 4;
    let formula = product_sum_formula(d, 1);
    let kernel = compile::<Fr>(&make_sparse_spec(&formula));

    let mut rng = ChaCha20Rng::seed_from_u64(7777);
    let keys = vec![0u64, 1, 6, 7, 10, 11];
    let n = keys.len();

    let value_data: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    let mut bufs: Vec<Buf<CpuBackend, Fr>> = value_data
        .iter()
        .map(|col| DeviceBuffer::Field(b.upload(col)))
        .collect();
    bufs.push(DeviceBuffer::U64(b.upload(&keys)));

    let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
    let result = b.reduce(&kernel, &buf_refs, &[]);
    assert_eq!(result.len(), d);

    let expected = reference_sparse_toom_cook_reduce(&value_data, &keys, d, 1);
    assert_eq!(result, expected, "sparse D=4 Toom-Cook mismatch");
}

/// Sparse D=3 P=1 with mixed pairing: paired, unpaired-even, unpaired-odd.
#[test]
fn sparse_d3_mixed_pairing_via_reduce() {
    let b = backend();
    let d = 3;
    let formula = product_sum_formula(d, 1);
    let kernel = compile::<Fr>(&make_sparse_spec(&formula));

    let mut rng = ChaCha20Rng::seed_from_u64(8888);
    // 3(odd-only), (8,9)(paired), 14(even-only)
    let keys = vec![3u64, 8, 9, 14];
    let n = keys.len();

    let value_data: Vec<Vec<Fr>> = (0..d)
        .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
        .collect();

    let mut bufs: Vec<Buf<CpuBackend, Fr>> = value_data
        .iter()
        .map(|col| DeviceBuffer::Field(b.upload(col)))
        .collect();
    bufs.push(DeviceBuffer::U64(b.upload(&keys)));

    let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
    let result = b.reduce(&kernel, &buf_refs, &[]);
    assert_eq!(result.len(), d);

    let expected = reference_sparse_toom_cook_reduce(&value_data, &keys, d, 1);
    assert_eq!(result, expected, "sparse D=3 mixed-pairing mismatch");
}
