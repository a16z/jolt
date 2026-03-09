//! Parity tests: verify MetalBackend ComputeBackend methods match CpuBackend.
//!
//! Each test runs the same operation on both backends and asserts identical
//! results, ensuring the Metal shaders produce bit-exact output vs CPU.

#![cfg(target_os = "macos")]

use jolt_compute::{BindingOrder, ComputeBackend, CpuBackend};
use jolt_field::{Field, Fr};
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
use jolt_metal::MetalBackend;
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::SeedableRng;

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
    let metal = MetalBackend::new();
    let mut rng = StdRng::seed_from_u64(0x1111);
    let data = random_elements(&mut rng, 4096);

    let buf = metal.upload(&data);
    let result = metal.download(&buf);

    assert_eq!(data, result);
}

#[test]
fn upload_download_empty() {
    let metal = MetalBackend::new();
    let data: Vec<Fr> = vec![];

    let buf = metal.upload(&data);
    assert_eq!(metal.len(&buf), 0);

    let result = metal.download(&buf);
    assert!(result.is_empty());
}

#[test]
fn alloc_is_zeroed() {
    let metal = MetalBackend::new();
    let buf: <MetalBackend as ComputeBackend>::Buffer<Fr> = metal.alloc(1024);

    let result = metal.download(&buf);
    assert_eq!(result.len(), 1024);
    for (i, val) in result.iter().enumerate() {
        assert!(val.is_zero(), "element {i} should be zero");
    }
}

const SIZES: [usize; 4] = [1, 64, 4096, 1 << 16];

#[test]
fn add_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x2222 + n as u64);
        let a_data = random_elements(&mut rng, n);
        let b_data = random_elements(&mut rng, n);

        let a_cpu = cpu.upload(&a_data);
        let b_cpu = cpu.upload(&b_data);
        let expected = cpu.download(&cpu.add(&a_cpu, &b_cpu));

        let a_mtl = metal.upload(&a_data);
        let b_mtl = metal.upload(&b_data);
        let got = metal.download(&metal.add(&a_mtl, &b_mtl));

        assert_eq!(expected, got, "add mismatch at n={n}");
    }
}

#[test]
fn sub_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x3333 + n as u64);
        let a_data = random_elements(&mut rng, n);
        let b_data = random_elements(&mut rng, n);

        let a_cpu = cpu.upload(&a_data);
        let b_cpu = cpu.upload(&b_data);
        let expected = cpu.download(&cpu.sub(&a_cpu, &b_cpu));

        let a_mtl = metal.upload(&a_data);
        let b_mtl = metal.upload(&b_data);
        let got = metal.download(&metal.sub(&a_mtl, &b_mtl));

        assert_eq!(expected, got, "sub mismatch at n={n}");
    }
}

#[test]
fn scale_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x4444 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&data);
        cpu.scale(&mut cpu_buf, scalar);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&data);
        metal.scale(&mut mtl_buf, scalar);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected, got, "scale mismatch at n={n}");
    }
}

#[test]
fn accumulate_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x5555 + n as u64);
        let buf_data = random_elements(&mut rng, n);
        let other_data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&buf_data);
        let cpu_other = cpu.upload(&other_data);
        cpu.accumulate(&mut cpu_buf, scalar, &cpu_other);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&buf_data);
        let mtl_other = metal.upload(&other_data);
        metal.accumulate(&mut mtl_buf, scalar, &mtl_other);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected, got, "accumulate mismatch at n={n}");
    }
}

#[test]
fn sum_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x6666 + n as u64);
        let data = random_elements(&mut rng, n);

        let cpu_buf = cpu.upload(&data);
        let expected: Fr = cpu.sum(&cpu_buf);

        let mtl_buf = metal.upload(&data);
        let got: Fr = metal.sum(&mtl_buf);

        assert_eq!(expected, got, "sum mismatch at n={n}");
    }
}

#[test]
fn sum_empty() {
    let metal = MetalBackend::new();
    let buf = metal.upload::<Fr>(&[]);
    let result: Fr = metal.sum(&buf);
    assert!(result.is_zero());
}

#[test]
fn dot_product_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for &n in &SIZES {
        let mut rng = StdRng::seed_from_u64(0x7777 + n as u64);
        let a_data = random_elements(&mut rng, n);
        let b_data = random_elements(&mut rng, n);

        let a_cpu = cpu.upload(&a_data);
        let b_cpu = cpu.upload(&b_data);
        let expected: Fr = cpu.dot_product(&a_cpu, &b_cpu);

        let a_mtl = metal.upload(&a_data);
        let b_mtl = metal.upload(&b_data);
        let got: Fr = metal.dot_product(&a_mtl, &b_mtl);

        assert_eq!(expected, got, "dot_product mismatch at n={n}");
    }
}

#[test]
fn dot_product_empty() {
    let metal = MetalBackend::new();
    let a = metal.upload::<Fr>(&[]);
    let b = metal.upload::<Fr>(&[]);
    let result: Fr = metal.dot_product(&a, &b);
    assert!(result.is_zero());
}

#[test]
fn sum_single_element() {
    let metal = MetalBackend::new();
    let val = Fr::from_u64(42);
    let buf = metal.upload(&[val]);
    let result: Fr = metal.sum(&buf);
    assert_eq!(result, val);
}

#[test]
fn dot_product_single_element() {
    let metal = MetalBackend::new();
    let a = Fr::from_u64(7);
    let b = Fr::from_u64(6);
    let a_buf = metal.upload(&[a]);
    let b_buf = metal.upload(&[b]);
    let result: Fr = metal.dot_product(&a_buf, &b_buf);
    assert_eq!(result, a * b);
}

#[test]
fn scale_by_zero() {
    let metal = MetalBackend::new();
    let mut rng = StdRng::seed_from_u64(0x8888);
    let data = random_elements(&mut rng, 256);

    let mut buf = metal.upload(&data);
    metal.scale(&mut buf, Fr::zero());
    let result = metal.download(&buf);

    for (i, val) in result.iter().enumerate() {
        assert!(val.is_zero(), "element {i} should be zero after scale by 0");
    }
}

#[test]
fn scale_by_one() {
    let metal = MetalBackend::new();
    let mut rng = StdRng::seed_from_u64(0x9999);
    let data = random_elements(&mut rng, 256);

    let mut buf = metal.upload(&data);
    metal.scale(&mut buf, Fr::from_u64(1));
    let result = metal.download(&buf);

    assert_eq!(data, result);
}

/// Large reduction that exercises multiple threadgroups.
#[test]
fn sum_large() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xAAAA);
    let n = 1 << 18; // 262144 elements — well past MAX_REDUCTION_GROUPS * GROUP_SIZE
    let data = random_elements(&mut rng, n);

    let cpu_buf = cpu.upload(&data);
    let expected: Fr = cpu.sum(&cpu_buf);

    let mtl_buf = metal.upload(&data);
    let got: Fr = metal.sum(&mtl_buf);

    assert_eq!(expected, got, "large sum mismatch at n={n}");
}

/// Large dot product exercising multiple threadgroups.
#[test]
fn dot_product_large() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xBBBB);
    let n = 1 << 18;
    let a_data = random_elements(&mut rng, n);
    let b_data = random_elements(&mut rng, n);

    let a_cpu = cpu.upload(&a_data);
    let b_cpu = cpu.upload(&b_data);
    let expected: Fr = cpu.dot_product(&a_cpu, &b_cpu);

    let a_mtl = metal.upload(&a_data);
    let b_mtl = metal.upload(&b_data);
    let got: Fr = metal.dot_product(&a_mtl, &b_mtl);

    assert_eq!(expected, got, "large dot_product mismatch at n={n}");
}

// ── Phase 3: Interpolation + Product Table ────────────────────────────

const INTERP_SIZES: [usize; 4] = [2, 128, 4096, 1 << 16];

#[test]
fn interpolate_pairs_parity() {
    let metal = MetalBackend::new();
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
    let metal = MetalBackend::new();
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
    let metal = MetalBackend::new();
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
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xFF00);
    let n = 1 << 10;
    let data = random_elements(&mut rng, n);

    let mut cpu_buf = cpu.upload(&data);
    let mut mtl_buf = metal.upload(&data);

    // Bind 10 rounds (LowToHigh), halving each time: 1024 → 512 → ... → 1
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
    let metal = MetalBackend::new();
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
fn product_table_parity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;

    for n_vars in [1, 2, 5, 10, 16] {
        let mut rng = StdRng::seed_from_u64(0xAA00 + n_vars as u64);
        let point: Vec<Fr> = (0..n_vars).map(|_| Fr::random(&mut rng)).collect();

        let cpu_table = cpu.download(&cpu.product_table(&point));
        let mtl_table = metal.download(&metal.product_table(&point));

        assert_eq!(
            cpu_table.len(),
            mtl_table.len(),
            "table size mismatch for {n_vars} vars"
        );
        assert_eq!(
            cpu_table, mtl_table,
            "product_table mismatch for {n_vars} vars"
        );
    }
}

/// Product table with a single variable: should produce [1-r, r].
#[test]
fn product_table_single_var() {
    let metal = MetalBackend::new();
    let r = Fr::from_u64(7);
    let table = metal.download(&metal.product_table(&[r]));

    assert_eq!(table.len(), 2);
    assert_eq!(table[0], Fr::from_u64(1) - r); // x=0: (1-r)
    assert_eq!(table[1], r); // x=1: r
}

/// Product table entries should sum to 1 (property of eq-polynomial).
#[test]
fn product_table_sums_to_one() {
    let metal = MetalBackend::new();
    let mut rng = StdRng::seed_from_u64(0xBB00);
    let point: Vec<Fr> = (0..12).map(|_| Fr::random(&mut rng)).collect();

    let table_buf = metal.product_table(&point);
    let sum: Fr = metal.sum(&table_buf);

    assert_eq!(sum, Fr::from_u64(1), "eq-polynomial table must sum to 1");
}

/// Large product table (2^20 entries).
#[test]
fn product_table_large() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xCC01);
    let point: Vec<Fr> = (0..20).map(|_| Fr::random(&mut rng)).collect();

    let cpu_table = cpu.download(&cpu.product_table(&point));
    let mtl_table = metal.download(&metal.product_table(&point));

    assert_eq!(cpu_table, mtl_table, "large product_table mismatch");
}

// ── Phase 4: Kernel Compilation + Pairwise Reduce ──────────────────────

fn compile_kernels(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    desc: &KernelDescriptor,
) -> (
    <CpuBackend as ComputeBackend>::CompiledKernel<Fr>,
    <MetalBackend as ComputeBackend>::CompiledKernel<Fr>,
) {
    let cpu_k = jolt_cpu_kernels::compile::<Fr>(desc);
    let mtl_k = metal.compile_kernel::<Fr>(desc);
    let _ = cpu;
    (cpu_k, mtl_k)
}

fn compile_kernels_with_challenges(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    desc: &KernelDescriptor,
    challenges: &[Fr],
) -> (
    <CpuBackend as ComputeBackend>::CompiledKernel<Fr>,
    <MetalBackend as ComputeBackend>::CompiledKernel<Fr>,
) {
    let cpu_k = jolt_cpu_kernels::compile_with_challenges::<Fr>(desc, challenges);
    let mtl_k = metal.compile_kernel_with_challenges::<Fr>(desc, challenges);
    let _ = cpu;
    (cpu_k, mtl_k)
}

/// ProductSum D=4, single group, LowToHigh.
#[test]
fn pairwise_reduce_product_sum_d4() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD001);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 1,
        },
        degree: 4,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_inputs: Vec<Vec<Fr>> = inputs.clone();
    let cpu_refs: Vec<&Vec<Fr>> = cpu_inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected.len(), got.len());
    assert_eq!(expected, got, "pairwise_reduce D=4 mismatch");
}

/// ProductSum D=3, 2 groups.
#[test]
fn pairwise_reduce_product_sum_d3_p2() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD002);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 3,
            num_products: 2,
        },
        degree: 3,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let k = desc.num_inputs();
    let n = 128;
    let inputs: Vec<Vec<Fr>> = (0..k).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=3 P=2 mismatch");
}

/// ProductSum D=8, single group, large buffer to exercise multiple threadgroups.
#[test]
fn pairwise_reduce_product_sum_d8_large() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD003);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 8,
            num_products: 1,
        },
        degree: 8,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let n = 1 << 14;
    let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=8 large mismatch");
}

/// HighToLow binding order.
#[test]
fn pairwise_reduce_high_to_low() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD004);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 1,
        },
        degree: 4,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::HighToLow,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::HighToLow,
    );

    assert_eq!(expected, got, "pairwise_reduce H2L mismatch");
}

/// Custom expression: booleanity h^2 - h.
#[test]
fn pairwise_reduce_custom_booleanity() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD005);

    let b = ExprBuilder::new();
    let h = b.opening(0);
    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(h * h - h),
            num_inputs: 1,
        },
        degree: 2,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let n = 512;
    let inputs: Vec<Vec<Fr>> = vec![random_elements(&mut rng, n)];
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce custom booleanity mismatch");
}

/// Custom expression with challenges: gamma * o0 * o1.
#[test]
fn pairwise_reduce_custom_with_challenges() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD006);

    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let gamma = b.challenge(0);
    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(gamma * a * bv),
            num_inputs: 2,
        },
        degree: 3,
        tensor_split: None,
    };
    let challenges = vec![Fr::random(&mut rng)];
    let (cpu_k, mtl_k) = compile_kernels_with_challenges(&cpu, &metal, &desc, &challenges);

    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(
        expected, got,
        "pairwise_reduce custom with challenge mismatch"
    );
}

/// Tensor pairwise reduce matches CPU.
#[test]
fn tensor_pairwise_reduce_product_sum() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD007);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 1,
        },
        degree: 4,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

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
    let expected =
        cpu.tensor_pairwise_reduce(&cpu_refs, &cpu_outer, &cpu_inner, &cpu_k, desc.num_evals());

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_outer = metal.upload(&outer_w);
    let mtl_inner = metal.upload(&inner_w);
    let got =
        metal.tensor_pairwise_reduce(&mtl_refs, &mtl_outer, &mtl_inner, &mtl_k, desc.num_evals());

    assert_eq!(expected, got, "tensor_pairwise_reduce mismatch");
}

/// ProductSum D=2 (smallest nontrivial case).
#[test]
fn pairwise_reduce_product_sum_d2() {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD008);

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 2,
            num_products: 1,
        },
        degree: 2,
        tensor_split: None,
    };
    let (cpu_k, mtl_k) = compile_kernels(&cpu, &metal, &desc);

    let n = 64;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    let weights = random_elements(&mut rng, n / 2);

    let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
    let cpu_w = cpu.upload(&weights);
    let expected = cpu.pairwise_reduce(
        &cpu_refs,
        &cpu_w,
        &cpu_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
    let mtl_w = metal.upload(&weights);
    let got = metal.pairwise_reduce(
        &mtl_refs,
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    assert_eq!(expected, got, "pairwise_reduce D=2 mismatch");
}

/// Verify known values for ProductSum D=4 with simple inputs.
#[test]
fn pairwise_reduce_product_sum_known_values() {
    let metal = MetalBackend::new();

    let desc = KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: 4,
            num_products: 1,
        },
        degree: 4,
        tensor_split: None,
    };
    let mtl_k = metal.compile_kernel::<Fr>(&desc);

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
        &mtl_w,
        &mtl_k,
        desc.num_evals(),
        BindingOrder::LowToHigh,
    );

    // Toom-Cook grid: P(1) = hi[0]*hi[1]*hi[2]*hi[3] = 5*6*7*8 = 1680
    assert_eq!(got[0], Fr::from_u64(1680), "P(1) mismatch");
    // P(∞) = diff[0]*diff[1]*diff[2]*diff[3] = 4*4*4*4 = 256
    assert_eq!(got[3], Fr::from_u64(256), "P(∞) mismatch");
}
