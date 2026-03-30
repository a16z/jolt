//! Parity tests: verify MetalBackend ComputeBackend methods match CpuBackend.
//!
//! Each test runs the same operation on both backends and asserts identical
//! results, ensuring the Metal shaders produce bit-exact output vs CPU.

#![cfg(target_os = "macos")]

use std::sync::LazyLock;

use jolt_compiler::{Factor, Formula, ProductTerm};
use jolt_compute::{BindingOrder, ComputeBackend, EqInput};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_metal::MetalBackend;
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::SeedableRng;

static METAL: LazyLock<MetalBackend> = LazyLock::new(MetalBackend::new_fast_compile);

fn random_elements(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

/// Verify that Fr's in-memory layout matches the Metal shader's expectations.
/// On little-endian ARM64, [u64; 4] and [u32; 8] have identical byte layout.
#[test]
fn fr_memory_layout_compatible() {
    assert_eq!(
        std::mem::size_of::<Fr>(),
        32,
        "Fr must be 32 bytes for Metal"
    );

    let mut rng = StdRng::seed_from_u64(0xABCD);
    for _ in 0..100 {
        let fr = Fr::random(&mut rng);
        let limbs = fr.inner_limbs().0;

        // Check that raw bytes of Fr match the expected [u64; 4] layout.
        // SAFETY: Fr is 32 bytes (verified above) and we read exactly that many.
        let fr_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts((&raw const fr).cast::<u8>(), 32) };
        // SAFETY: limbs is [u64; 4] = 32 bytes, reading the full extent.
        let limb_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(limbs.as_ptr().cast::<u8>(), 32) };
        assert_eq!(fr_bytes, limb_bytes, "Fr memory layout must match [u64; 4]");
    }
}

#[test]
fn upload_download_roundtrip() {
    let metal = &*METAL;
    let mut rng = StdRng::seed_from_u64(0x1111);
    let data = random_elements(&mut rng, 4096);

    let buf = metal.upload(&data);
    let result = metal.download(&buf);

    assert_eq!(data, result);
}

#[test]
fn upload_download_empty() {
    let metal = &*METAL;
    let data: Vec<Fr> = vec![];

    let buf = metal.upload(&data);
    assert_eq!(metal.len(&buf), 0);

    let result = metal.download(&buf);
    assert!(result.is_empty());
}

#[test]
fn alloc_is_zeroed() {
    let metal = &*METAL;
    let buf: <MetalBackend as ComputeBackend>::Buffer<Fr> = metal.alloc(1024);

    let result = metal.download(&buf);
    assert_eq!(result.len(), 1024);
    for (i, val) in result.iter().enumerate() {
        assert!(val.is_zero(), "element {i} should be zero");
    }
}

const INTERP_SIZES: [usize; 3] = [2, 128, 4096];

#[test]
fn interpolate_pairs_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for &n in &INTERP_SIZES {
        let mut rng = StdRng::seed_from_u64(0xCC00 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let cpu_buf = cpu.upload(&data);
        let expected = cpu.download(&cpu.interpolate_pairs(cpu_buf, scalar));

        let mtl_buf = metal.upload(&data);
        let got = metal.download(&metal.interpolate_pairs(mtl_buf, scalar));

        assert_eq!(expected, got, "interpolate_pairs mismatch at n={n}");
    }
}

#[test]
fn interpolate_pairs_inplace_high_to_low_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for &n in &INTERP_SIZES {
        let mut rng = StdRng::seed_from_u64(0xDD00 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&data);
        cpu.interpolate_pairs_inplace(&mut cpu_buf, scalar, BindingOrder::HighToLow);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&data);
        metal.interpolate_pairs_inplace(&mut mtl_buf, scalar, BindingOrder::HighToLow);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected.len(), got.len(), "length mismatch at n={n}");
        assert_eq!(expected, got, "interpolate_inplace H2L mismatch at n={n}");
    }
}

#[test]
fn interpolate_pairs_inplace_low_to_high_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for &n in &INTERP_SIZES {
        let mut rng = StdRng::seed_from_u64(0xEE00 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&data);
        cpu.interpolate_pairs_inplace(&mut cpu_buf, scalar, BindingOrder::LowToHigh);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&data);
        metal.interpolate_pairs_inplace(&mut mtl_buf, scalar, BindingOrder::LowToHigh);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected.len(), got.len(), "length mismatch at n={n}");
        assert_eq!(expected, got, "interpolate_inplace L2H mismatch at n={n}");
    }
}

/// Multiple rounds of inplace interpolation (simulating sumcheck binding).
#[test]
fn interpolate_pairs_inplace_multi_round() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xFF00);
    let n = 1 << 10;
    let data = random_elements(&mut rng, n);

    let mut cpu_buf = cpu.upload(&data);
    let mut mtl_buf = metal.upload(&data);

    // Bind 10 rounds (LowToHigh), halving each time: 1024 -> 512 -> ... -> 1
    for round in 0..10 {
        let scalar = Fr::random(&mut rng);
        cpu.interpolate_pairs_inplace(&mut cpu_buf, scalar, BindingOrder::LowToHigh);
        metal.interpolate_pairs_inplace(&mut mtl_buf, scalar, BindingOrder::LowToHigh);

        let expected = cpu.download(&cpu_buf);
        let got = metal.download(&mtl_buf);
        assert_eq!(expected, got, "multi-round L2H mismatch at round {round}");
    }

    assert_eq!(cpu.len(&cpu_buf), 1);
    assert_eq!(metal.len(&mtl_buf), 1);
}

/// Multiple rounds with HighToLow binding order.
#[test]
fn interpolate_pairs_inplace_multi_round_high_to_low() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xFF01);
    let n = 1 << 10;
    let data = random_elements(&mut rng, n);

    let mut cpu_buf = cpu.upload(&data);
    let mut mtl_buf = metal.upload(&data);

    for round in 0..10 {
        let scalar = Fr::random(&mut rng);
        cpu.interpolate_pairs_inplace(&mut cpu_buf, scalar, BindingOrder::HighToLow);
        metal.interpolate_pairs_inplace(&mut mtl_buf, scalar, BindingOrder::HighToLow);

        let expected = cpu.download(&cpu_buf);
        let got = metal.download(&mtl_buf);
        assert_eq!(expected, got, "multi-round H2L mismatch at round {round}");
    }

    assert_eq!(cpu.len(&cpu_buf), 1);
    assert_eq!(metal.len(&mtl_buf), 1);
}

#[test]
fn eq_table_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for n_vars in [1, 2, 5, 10, 16] {
        let mut rng = StdRng::seed_from_u64(0xAA00 + n_vars as u64);
        let point: Vec<Fr> = (0..n_vars).map(|_| Fr::random(&mut rng)).collect();

        let cpu_table = cpu.download(&cpu.eq_table(&point));
        let mtl_table = metal.download(&metal.eq_table(&point));

        assert_eq!(
            cpu_table.len(),
            mtl_table.len(),
            "table size mismatch for {n_vars} vars"
        );
        assert_eq!(
            cpu_table, mtl_table,
            "eq_table mismatch for {n_vars} vars"
        );
    }
}

/// Eq table with a single variable: should produce [1-r, r].
#[test]
fn eq_table_single_var() {
    let metal = &*METAL;
    let r = Fr::from_u64(7);
    let table = metal.download(&metal.eq_table(&[r]));

    assert_eq!(table.len(), 2);
    assert_eq!(table[0], Fr::from_u64(1) - r); // x=0: (1-r)
    assert_eq!(table[1], r); // x=1: r
}

/// Eq table exercising GPU rounds (2^12 entries).
#[test]
fn eq_table_large() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xCC01);
    let point: Vec<Fr> = (0..12).map(|_| Fr::random(&mut rng)).collect();

    let cpu_table = cpu.download(&cpu.eq_table(&point));
    let mtl_table = metal.download(&metal.eq_table(&point));

    assert_eq!(cpu_table, mtl_table, "large eq_table mismatch");
}

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

fn compile_kernels(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    formula: &Formula,
) -> (
    <CpuBackend as ComputeBackend>::CompiledKernel<Fr>,
    <MetalBackend as ComputeBackend>::CompiledKernel<Fr>,
) {
    let cpu_k = jolt_cpu::compile::<Fr>(formula);
    let mtl_k = metal.compile_kernel::<Fr>(formula);
    let _ = cpu;
    (cpu_k, mtl_k)
}

/// ProductSum D=4, single group, LowToHigh.
#[test]
fn pairwise_reduce_product_sum_d4() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD001);

    let formula = product_sum_formula(4, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_inputs: Vec<Vec<Fr>> = inputs.clone();
    let cpu_refs: Vec<&Vec<Fr>> = cpu_inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected.len(), got.len());
    assert_eq!(expected, got, "pairwise_reduce D=4 mismatch");
}

/// ProductSum D=3, 2 groups.
#[test]
fn pairwise_reduce_product_sum_d3_p2() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD002);

    let formula = product_sum_formula(3, 2);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let k = formula.num_inputs;
    let n = 128;
    let inputs: Vec<Vec<Fr>> = (0..k).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=3 P=2 mismatch");
}

/// ProductSum D=8, single group, large buffer to exercise multiple threadgroups.
#[test]
fn pairwise_reduce_product_sum_d8_large() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD003);

    let formula = product_sum_formula(8, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=8 mismatch");
}

/// ProductSum D=8 with HighToLow binding (exercises split-pass H2L path).
#[test]
fn pairwise_reduce_product_sum_d8_high_to_low() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD010);

    let formula = product_sum_formula(8, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::HighToLow,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::HighToLow,
    );

    assert_eq!(expected, got, "pairwise_reduce D=8 H2L mismatch");
}

/// ProductSum D=8 with varying sizes to find split-pass boundary.
#[test]
fn pairwise_reduce_product_sum_d8_sizes() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    let formula = product_sum_formula(8, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    for n_pairs in [32, 33, 48, 63, 64, 65, 128, 256] {
        let n = n_pairs * 2;
        let mut rng = StdRng::seed_from_u64(0xE000 + n_pairs as u64);
        let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
        let weights = random_elements(&mut rng, n_pairs);

        let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
        let cpu_w = cpu.upload(&weights);
        let expected = cpu.pairwise_reduce(
            &cpu_refs,
            EqInput::Weighted(&cpu_w),
            &cpu_k,
            &[],
            formula.degree(),
            BindingOrder::LowToHigh,
        );

        let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
        let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
        let mtl_w = metal.upload(&weights);
        let got = metal.pairwise_reduce(
            &mtl_refs,
            EqInput::Weighted(&mtl_w),
            &mtl_k,
            &[],
            formula.degree(),
            BindingOrder::LowToHigh,
        );

        assert_eq!(expected, got, "D=8 mismatch at n_pairs={n_pairs}");
    }
}

/// ProductSum D=8 unweighted (exercises split-pass unweighted path).
#[test]
fn pairwise_reduce_product_sum_d8_unweighted() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD011);

    let formula = product_sum_formula(8, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    for n_pairs in [32, 33, 64, 256] {
        let n = n_pairs * 2;
        let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();

        let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
        let expected = cpu.pairwise_reduce(
            &cpu_refs,
            EqInput::Unit,
            &cpu_k,
            &[],
            formula.degree(),
            BindingOrder::LowToHigh,
        );

        let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
        let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
        let got = metal.pairwise_reduce(
            &mtl_refs,
            EqInput::Unit,
            &mtl_k,
            &[],
            formula.degree(),
            BindingOrder::LowToHigh,
        );

        assert_eq!(
            expected, got,
            "pairwise_reduce D=8 unweighted mismatch at n_pairs={n_pairs}"
        );
    }
}

/// HighToLow binding order.
#[test]
fn pairwise_reduce_high_to_low() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD004);

    let formula = product_sum_formula(4, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::HighToLow,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::HighToLow,
    );

    assert_eq!(expected, got, "pairwise_reduce H2L mismatch");
}

/// Custom expression: booleanity h^2 - h.
#[test]
fn pairwise_reduce_custom_booleanity() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD005);

    // h^2 - h (booleanity check)
    let formula = Formula::from_terms(vec![
        ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(0)],
        },
        ProductTerm {
            coefficient: -1,
            factors: vec![Factor::Input(0)],
        },
    ]);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = vec![random_elements(&mut rng, n)];
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce custom booleanity mismatch");
}

/// Custom expression with challenges: gamma * o0 * o1.
#[test]
fn pairwise_reduce_custom_with_challenges() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD006);

    // gamma * a * b
    let formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(1)],
    }]);
    let challenges = vec![Fr::random(&mut rng)];
    let cpu_k = jolt_cpu::compile::<Fr>(&formula);
    let mtl_k = metal.compile_kernel::<Fr>(&formula);

    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &challenges,
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &challenges,
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(
        expected, got,
        "pairwise_reduce custom with challenge mismatch"
    );
}

/// Tensor eq pairwise reduce matches CPU.
#[test]
fn pairwise_reduce_tensor_eq() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD007);

    let formula = product_sum_formula(4, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    // outer_len * inner_len pairs = n/2
    let outer_len = 8;
    let inner_len = 16;
    let n = outer_len * inner_len * 2;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let outer_w = random_elements(&mut rng, outer_len);
    let inner_w = random_elements(&mut rng, inner_len);

    let cpu_inputs: Vec<Vec<Fr>> = inputs.clone();
    let cpu_refs: Vec<&Vec<Fr>> = cpu_inputs.iter().collect();
    let cpu_outer = cpu.upload(&outer_w);
    let cpu_inner = cpu.upload(&inner_w);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Tensor {
            outer: &cpu_outer,
            inner: &cpu_inner,
        },
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_outer = metal.upload(&outer_w);
    let mtl_inner = metal.upload(&inner_w);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Tensor {
            outer: &mtl_outer,
            inner: &mtl_inner,
        },
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "tensor_pairwise_reduce mismatch");
}

/// ProductSum D=2 (smallest nontrivial case).
#[test]
fn pairwise_reduce_product_sum_d2() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD008);

    let formula = product_sum_formula(2, 1);
    let (cpu_k, mtl_k) = compile_kernels(&cpu, metal, &formula);

    let n = 64;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        EqInput::Weighted(&cpu_w),
        &cpu_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=2 mismatch");
}

/// Verify known values for ProductSum D=4 with simple inputs.
#[test]
fn pairwise_reduce_product_sum_known_values() {
    let metal = &*METAL;

    let formula = product_sum_formula(4, 1);
    let mtl_k = metal.compile_kernel::<Fr>(&formula);

    // Single pair: lo = [1,2,3,4], hi = [5,6,7,8]
    // Stored interleaved for LowToHigh: input_k = [lo[k], hi[k]]
    let inputs: Vec<Vec<Fr>> = (0..4)
        .map(|k| {
            let lo = Fr::from_u64(k as u64 + 1);
            let hi = Fr::from_u64(k as u64 + 5);
            vec![lo, hi]
        })
        .collect();
    let weights = vec![Fr::from_u64(1)];

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        EqInput::Weighted(&mtl_w),
        &mtl_k,
        &[],
        formula.degree(),
        BindingOrder::LowToHigh,
    );

    // Toom-Cook grid: P(1) = hi[0]*hi[1]*hi[2]*hi[3] = 5*6*7*8 = 1680
    assert_eq!(got[0], Fr::from_u64(1680), "P(1) mismatch");
    // P(infinity) = diff[0]*diff[1]*diff[2]*diff[3] = 4*4*4*4 = 256
    assert_eq!(got[3], Fr::from_u64(256), "P(infinity) mismatch");
}
