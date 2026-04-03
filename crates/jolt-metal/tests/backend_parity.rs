//! Parity tests: verify MetalBackend ComputeBackend methods match CpuBackend.
//!
//! Each test runs the same operation on both backends and asserts identical
//! results, ensuring the Metal shaders produce bit-exact output vs CPU.

#![cfg(target_os = "macos")]

use std::sync::LazyLock;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
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

fn make_spec(formula: &Formula, order: BindingOrder) -> KernelSpec {
    KernelSpec::new(formula.clone(), Iteration::Dense, order)
}

fn make_tensor_spec(formula: &Formula) -> KernelSpec {
    KernelSpec::new(
        formula.clone(),
        Iteration::DenseTensor,
        BindingOrder::LowToHigh,
    )
}

fn make_sparse_spec(formula: &Formula) -> KernelSpec {
    KernelSpec::new(
        formula.clone(),
        Iteration::Sparse,
        BindingOrder::LowToHigh,
    )
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
fn interpolate_inplace_high_to_low_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for &n in &INTERP_SIZES {
        let mut rng = StdRng::seed_from_u64(0xDD00 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&data);
        cpu.interpolate_inplace(&mut cpu_buf, scalar, BindingOrder::HighToLow);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&data);
        metal.interpolate_inplace(&mut mtl_buf, scalar, BindingOrder::HighToLow);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected.len(), got.len(), "length mismatch at n={n}");
        assert_eq!(expected, got, "interpolate_inplace H2L mismatch at n={n}");
    }
}

#[test]
fn interpolate_inplace_low_to_high_parity() {
    let metal = &*METAL;
    let cpu = CpuBackend;

    for &n in &INTERP_SIZES {
        let mut rng = StdRng::seed_from_u64(0xEE00 + n as u64);
        let data = random_elements(&mut rng, n);
        let scalar = Fr::random(&mut rng);

        let mut cpu_buf = cpu.upload(&data);
        cpu.interpolate_inplace(&mut cpu_buf, scalar, BindingOrder::LowToHigh);
        let expected = cpu.download(&cpu_buf);

        let mut mtl_buf = metal.upload(&data);
        metal.interpolate_inplace(&mut mtl_buf, scalar, BindingOrder::LowToHigh);
        let got = metal.download(&mtl_buf);

        assert_eq!(expected.len(), got.len(), "length mismatch at n={n}");
        assert_eq!(expected, got, "interpolate_inplace L2H mismatch at n={n}");
    }
}

/// Multiple rounds of inplace interpolation (simulating sumcheck binding).
#[test]
fn interpolate_inplace_multi_round() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xFF00);
    let n = 1 << 10;
    let data = random_elements(&mut rng, n);

    let mut cpu_buf = cpu.upload(&data);
    let mut mtl_buf = metal.upload(&data);

    for round in 0..10 {
        let scalar = Fr::random(&mut rng);
        cpu.interpolate_inplace(&mut cpu_buf, scalar, BindingOrder::LowToHigh);
        metal.interpolate_inplace(&mut mtl_buf, scalar, BindingOrder::LowToHigh);

        let expected = cpu.download(&cpu_buf);
        let got = metal.download(&mtl_buf);
        assert_eq!(expected, got, "multi-round L2H mismatch at round {round}");
    }

    assert_eq!(cpu.len(&cpu_buf), 1);
    assert_eq!(metal.len(&mtl_buf), 1);
}

/// Multiple rounds with HighToLow binding order.
#[test]
fn interpolate_inplace_multi_round_high_to_low() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xFF01);
    let n = 1 << 10;
    let data = random_elements(&mut rng, n);

    let mut cpu_buf = cpu.upload(&data);
    let mut mtl_buf = metal.upload(&data);

    for round in 0..10 {
        let scalar = Fr::random(&mut rng);
        cpu.interpolate_inplace(&mut cpu_buf, scalar, BindingOrder::HighToLow);
        metal.interpolate_inplace(&mut mtl_buf, scalar, BindingOrder::HighToLow);

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
        assert_eq!(cpu_table, mtl_table, "eq_table mismatch for {n_vars} vars");
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

// ── Reduce parity tests ────────────────────────────────────────────────

/// Helper: run reduce on both backends and compare results.
fn reduce_parity(
    formula: &Formula,
    order: BindingOrder,
    inputs: &[Vec<Fr>],
    challenges: &[Fr],
    label: &str,
) {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let spec = make_spec(formula, order);

    let cpu_k = cpu.compile::<Fr>(&spec);
    let mtl_k = metal.compile::<Fr>(&spec);

    let cpu_bufs: Vec<Vec<Fr>> = inputs.to_vec();
    let cpu_dev: Vec<DeviceBuffer<Vec<Fr>, Vec<u64>>> = cpu_bufs
        .iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect();
    let cpu_refs: Vec<&Buf<CpuBackend, Fr>> = cpu_dev.iter().collect();
    let expected = cpu.reduce(&cpu_k, &cpu_refs, challenges);

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_dev: Vec<DeviceBuffer<_, _>> = mtl_bufs.into_iter().map(DeviceBuffer::Field).collect();
    let mtl_refs: Vec<&Buf<MetalBackend, Fr>> = mtl_dev.iter().collect();
    let got = metal.reduce(&mtl_k, &mtl_refs, challenges);

    assert_eq!(expected.len(), got.len(), "{label}: length mismatch");
    assert_eq!(expected, got, "{label}: value mismatch");
}

/// ProductSum D=4, single group, LowToHigh.
#[test]
fn reduce_product_sum_d4() {
    let mut rng = StdRng::seed_from_u64(0xD001);
    let formula = product_sum_formula(4, 1);
    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::LowToHigh, &inputs, &[], "D=4 L2H");
}

/// ProductSum D=3, 2 groups.
#[test]
fn reduce_product_sum_d3_p2() {
    let mut rng = StdRng::seed_from_u64(0xD002);
    let formula = product_sum_formula(3, 2);
    let k = formula.num_inputs;
    let n = 128;
    let inputs: Vec<Vec<Fr>> = (0..k).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::LowToHigh, &inputs, &[], "D=3 P=2");
}

/// ProductSum D=8, single group, large buffer.
#[test]
fn reduce_product_sum_d8_large() {
    let mut rng = StdRng::seed_from_u64(0xD003);
    let formula = product_sum_formula(8, 1);
    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::LowToHigh, &inputs, &[], "D=8 large");
}

/// ProductSum D=8 with HighToLow binding.
#[test]
fn reduce_product_sum_d8_high_to_low() {
    let mut rng = StdRng::seed_from_u64(0xD010);
    let formula = product_sum_formula(8, 1);
    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::HighToLow, &inputs, &[], "D=8 H2L");
}

/// ProductSum D=8 with varying sizes.
#[test]
fn reduce_product_sum_d8_sizes() {
    let formula = product_sum_formula(8, 1);
    for n_pairs in [32, 33, 48, 63, 64, 65, 128, 256] {
        let n = n_pairs * 2;
        let mut rng = StdRng::seed_from_u64(0xE000 + n_pairs as u64);
        let inputs: Vec<Vec<Fr>> = (0..8).map(|_| random_elements(&mut rng, n)).collect();
        reduce_parity(
            &formula,
            BindingOrder::LowToHigh,
            &inputs,
            &[],
            &format!("D=8 n_pairs={n_pairs}"),
        );
    }
}

/// HighToLow binding order with D=4.
#[test]
fn reduce_high_to_low() {
    let mut rng = StdRng::seed_from_u64(0xD004);
    let formula = product_sum_formula(4, 1);
    let n = 512;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::HighToLow, &inputs, &[], "D=4 H2L");
}

/// Custom expression with challenges: gamma * a * b.
#[test]
fn reduce_custom_with_challenges() {
    let mut rng = StdRng::seed_from_u64(0xD006);
    let formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(1)],
    }]);
    let challenges = vec![Fr::random(&mut rng)];
    let n = 256;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(
        &formula,
        BindingOrder::LowToHigh,
        &inputs,
        &challenges,
        "challenge",
    );
}

/// Tensor eq reduce matches CPU.
#[test]
fn reduce_tensor_eq() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xD007);

    let formula = product_sum_formula(4, 1);
    let spec = make_tensor_spec(&formula);

    let cpu_k = cpu.compile::<Fr>(&spec);
    let mtl_k = metal.compile::<Fr>(&spec);

    let outer_len = 8;
    let inner_len = 16;
    let n = outer_len * inner_len * 2;
    let inputs: Vec<Vec<Fr>> = (0..4).map(|_| random_elements(&mut rng, n)).collect();
    let outer_w = random_elements(&mut rng, outer_len);
    let inner_w = random_elements(&mut rng, inner_len);

    // CPU: value columns + tensor outer/inner
    let mut cpu_dev: Vec<Buf<CpuBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect();
    cpu_dev.push(DeviceBuffer::Field(outer_w.clone()));
    cpu_dev.push(DeviceBuffer::Field(inner_w.clone()));
    let cpu_refs: Vec<&Buf<CpuBackend, Fr>> = cpu_dev.iter().collect();
    let expected = cpu.reduce(&cpu_k, &cpu_refs, &[]);

    // Metal: same layout
    let mut mtl_dev: Vec<Buf<MetalBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(metal.upload(v)))
        .collect();
    mtl_dev.push(DeviceBuffer::Field(metal.upload(&outer_w)));
    mtl_dev.push(DeviceBuffer::Field(metal.upload(&inner_w)));
    let mtl_refs: Vec<&Buf<MetalBackend, Fr>> = mtl_dev.iter().collect();
    let got = metal.reduce(&mtl_k, &mtl_refs, &[]);

    assert_eq!(expected, got, "tensor reduce mismatch");
}

/// ProductSum D=2 (smallest nontrivial case).
#[test]
fn reduce_product_sum_d2() {
    let mut rng = StdRng::seed_from_u64(0xD008);
    let formula = product_sum_formula(2, 1);
    let n = 64;
    let inputs: Vec<Vec<Fr>> = (0..2).map(|_| random_elements(&mut rng, n)).collect();
    reduce_parity(&formula, BindingOrder::LowToHigh, &inputs, &[], "D=2");
}

/// Verify known values for ProductSum D=4 with simple inputs.
#[test]
fn reduce_product_sum_known_values() {
    let metal = &*METAL;

    let formula = product_sum_formula(4, 1);
    let spec = make_spec(&formula, BindingOrder::LowToHigh);
    let mtl_k = metal.compile::<Fr>(&spec);

    // Single pair: lo = [1,2,3,4], hi = [5,6,7,8]
    // Stored interleaved for LowToHigh: input_k = [lo[k], hi[k]]
    let inputs: Vec<Vec<Fr>> = (0..4)
        .map(|k| {
            let lo = Fr::from_u64(k as u64 + 1);
            let hi = Fr::from_u64(k as u64 + 5);
            vec![lo, hi]
        })
        .collect();

    let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
    let mtl_dev: Vec<Buf<MetalBackend, Fr>> =
        mtl_bufs.into_iter().map(DeviceBuffer::Field).collect();
    let mtl_refs: Vec<&Buf<MetalBackend, Fr>> = mtl_dev.iter().collect();
    let got = metal.reduce(&mtl_k, &mtl_refs, &[]);

    // Toom-Cook grid: P(1) = hi[0]*hi[1]*hi[2]*hi[3] = 5*6*7*8 = 1680
    assert_eq!(got[0], Fr::from_u64(1680), "P(1) mismatch");
    // P(infinity) = diff[0]*diff[1]*diff[2]*diff[3] = 4*4*4*4 = 256
    assert_eq!(got[3], Fr::from_u64(256), "P(infinity) mismatch");
}

// ── Sparse parity tests ────────────────────────────────────────────────

/// Sparse reduce matches CPU for product-sum D=3.
#[test]
fn reduce_sparse_d3() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xE001);

    let formula = product_sum_formula(3, 1);
    let spec = make_sparse_spec(&formula);
    let cpu_k = cpu.compile::<Fr>(&spec);
    let mtl_k = metal.compile::<Fr>(&spec);

    // 8 sparse entries in a 16-position hypercube (4 complete pairs).
    let keys = vec![0u64, 1, 4, 5, 10, 11, 14, 15];
    let n = keys.len();
    let inputs: Vec<Vec<Fr>> = (0..3).map(|_| random_elements(&mut rng, n)).collect();

    let mut cpu_dev: Vec<Buf<CpuBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect();
    cpu_dev.push(DeviceBuffer::U64(keys.clone()));
    let cpu_refs: Vec<_> = cpu_dev.iter().collect();
    let expected = cpu.reduce(&cpu_k, &cpu_refs, &[]);

    let mut mtl_dev: Vec<Buf<MetalBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(metal.upload(v)))
        .collect();
    mtl_dev.push(DeviceBuffer::U64(metal.upload(&keys)));
    let mtl_refs: Vec<_> = mtl_dev.iter().collect();
    let got = metal.reduce(&mtl_k, &mtl_refs, &[]);

    assert_eq!(expected, got, "sparse D=3 reduce mismatch");
}

/// Sparse reduce with incomplete pairs (some entries have only lo or hi).
#[test]
fn reduce_sparse_incomplete_pairs() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xE002);

    let formula = product_sum_formula(3, 1);
    let spec = make_sparse_spec(&formula);
    let cpu_k = cpu.compile::<Fr>(&spec);
    let mtl_k = metal.compile::<Fr>(&spec);

    // Mix of complete and incomplete pairs:
    // key 2 (even, no 3) → lo only
    // key 7 (odd, no 6) → hi only
    // keys 10,11 → complete pair
    let keys = vec![2u64, 7, 10, 11];
    let n = keys.len();
    let inputs: Vec<Vec<Fr>> = (0..3).map(|_| random_elements(&mut rng, n)).collect();

    let mut cpu_dev: Vec<Buf<CpuBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect();
    cpu_dev.push(DeviceBuffer::U64(keys.clone()));
    let cpu_refs: Vec<_> = cpu_dev.iter().collect();
    let expected = cpu.reduce(&cpu_k, &cpu_refs, &[]);

    let mut mtl_dev: Vec<Buf<MetalBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(metal.upload(v)))
        .collect();
    mtl_dev.push(DeviceBuffer::U64(metal.upload(&keys)));
    let mtl_refs: Vec<_> = mtl_dev.iter().collect();
    let got = metal.reduce(&mtl_k, &mtl_refs, &[]);

    assert_eq!(expected, got, "sparse incomplete pairs reduce mismatch");
}

/// Full sparse sumcheck rounds: reduce + bind over multiple rounds.
#[test]
fn sparse_sumcheck_rounds() {
    let metal = &*METAL;
    let cpu = CpuBackend;
    let mut rng = StdRng::seed_from_u64(0xE003);

    let formula = product_sum_formula(3, 1);
    let spec = make_sparse_spec(&formula);
    let cpu_k = cpu.compile::<Fr>(&spec);
    let mtl_k = metal.compile::<Fr>(&spec);

    let keys = vec![0u64, 1, 4, 5, 10, 11, 14, 15];
    let n = keys.len();
    let num_vars = 4;
    let inputs: Vec<Vec<Fr>> = (0..3).map(|_| random_elements(&mut rng, n)).collect();

    let mut cpu_bufs: Vec<Buf<CpuBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect();
    cpu_bufs.push(DeviceBuffer::U64(keys.clone()));

    let mut mtl_bufs: Vec<Buf<MetalBackend, Fr>> = inputs
        .iter()
        .map(|v| DeviceBuffer::Field(metal.upload(v)))
        .collect();
    mtl_bufs.push(DeviceBuffer::U64(metal.upload(&keys)));

    for round in 0..num_vars {
        let cpu_refs: Vec<_> = cpu_bufs.iter().collect();
        let mtl_refs: Vec<_> = mtl_bufs.iter().collect();

        let cpu_evals = cpu.reduce(&cpu_k, &cpu_refs, &[]);
        let mtl_evals = metal.reduce(&mtl_k, &mtl_refs, &[]);
        assert_eq!(
            cpu_evals, mtl_evals,
            "sparse reduce mismatch at round {round}"
        );

        let challenge = Fr::random(&mut rng);
        cpu.bind(&cpu_k, &mut cpu_bufs, challenge);
        metal.bind(&mtl_k, &mut mtl_bufs, challenge);

        // Verify value buffers match after bind.
        for (i, (cb, mb)) in cpu_bufs[..3]
            .iter()
            .zip(mtl_bufs[..3].iter())
            .enumerate()
        {
            let cpu_vals = cb.as_field();
            let mtl_vals: Vec<Fr> = metal.download(mb.as_field());
            assert_eq!(
                *cpu_vals, mtl_vals,
                "sparse bind mismatch at round {round}, input {i}"
            );
        }

        // Verify keys match after bind.
        let cpu_keys = cpu_bufs[3].as_u64();
        let mtl_keys: Vec<u64> = metal.download(mtl_bufs[3].as_u64());
        assert_eq!(
            *cpu_keys, mtl_keys,
            "sparse keys mismatch at round {round}"
        );
    }

    assert_eq!(cpu_bufs[0].as_field().len(), 1);
}
