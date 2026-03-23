//! Parity tests: verify Metal Fr arithmetic matches CPU (jolt-field) bit-for-bit.
//!
//! Each test generates random field elements on the CPU, ships them to Metal
//! via the dispatch layer, runs the operation, and checks the result against
//! the CPU reference.

#![cfg(target_os = "macos")]

use jolt_field::Field;
use jolt_field::Fr;
use jolt_metal::field::{
    dispatch_binary, dispatch_fmadd, dispatch_from_u64, dispatch_unary, FrKernels,
    MetalFieldElement,
};
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::SeedableRng;

type MFr = MetalFieldElement<8>;

fn to_metal(f: Fr) -> MFr {
    MFr::from_u64_limbs(&f.inner_limbs().0)
}

fn to_cpu(g: MFr) -> Fr {
    use jolt_field::Limbs;
    let u64s: [u64; 4] = g.to_u64_limbs().try_into().unwrap();
    Fr::from_bigint_unchecked(Limbs::new(u64s)).unwrap()
}

struct TestCtx {
    device: metal::Device,
    queue: metal::CommandQueue,
    kernels: FrKernels,
}

impl TestCtx {
    fn new() -> Self {
        let device = metal::Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let kernels = FrKernels::new(&device);
        Self {
            device,
            queue,
            kernels,
        }
    }
}

fn random_elements(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

const N: usize = 256;

#[test]
fn mul_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0xdead);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.mul, &a_mtl, &b_mtl);

    for i in 0..N {
        let expected = a_cpu[i] * b_cpu[i];
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "mul mismatch at index {i}");
    }
}

#[test]
fn add_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0xbeef);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.add, &a_mtl, &b_mtl);

    for i in 0..N {
        let expected = a_cpu[i] + b_cpu[i];
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "add mismatch at index {i}");
    }
}

#[test]
fn sub_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0xcafe);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.sub, &a_mtl, &b_mtl);

    for i in 0..N {
        let expected = a_cpu[i] - b_cpu[i];
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "sub mismatch at index {i}");
    }
}

#[test]
fn sqr_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0xf00d);
    let a_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_unary(&ctx.device, &ctx.queue, &ctx.kernels.sqr, &a_mtl);

    for i in 0..N {
        let expected = a_cpu[i].square();
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "sqr mismatch at index {i}");
    }
}

#[test]
fn neg_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0xbad0);
    let a_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_unary(&ctx.device, &ctx.queue, &ctx.kernels.neg, &a_mtl);

    for i in 0..N {
        let expected = -a_cpu[i];
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "neg mismatch at index {i}");
    }
}

#[test]
fn from_u64_parity() {
    let ctx = TestCtx::new();
    let vals: Vec<u64> = (0..N as u64).collect();

    let result: Vec<MFr> = dispatch_from_u64(&ctx.device, &ctx.queue, &ctx.kernels.from_u64, &vals);

    for i in 0..N {
        let expected = Fr::from_u64(vals[i]);
        let got = to_cpu(result[i]);
        assert_eq!(expected, got, "from_u64 mismatch at index {i}");
    }
}

#[test]
fn from_u64_large_values() {
    let ctx = TestCtx::new();
    let vals: Vec<u64> = vec![
        0,
        1,
        u64::MAX,
        u64::MAX - 1,
        0xdead_beef_cafe_babe,
        42,
        1 << 63,
        (1 << 32) - 1,
    ];

    let result: Vec<MFr> = dispatch_from_u64(&ctx.device, &ctx.queue, &ctx.kernels.from_u64, &vals);

    for i in 0..vals.len() {
        let expected = Fr::from_u64(vals[i]);
        let got = to_cpu(result[i]);
        assert_eq!(
            expected, got,
            "from_u64 large mismatch at index {i}, val={}",
            vals[i]
        );
    }
}

#[test]
fn fmadd_parity() {
    let ctx = TestCtx::new();
    let mut rng = StdRng::seed_from_u64(0x1234);
    let stride = 1024;
    let a_cpu = random_elements(&mut rng, stride);
    let b_cpu = random_elements(&mut rng, stride);

    let n_threads = 64;
    let n_fmadd = 256; // must match N_FMADD constant in test_kernels.metal

    let expected: Vec<Fr> = (0..n_threads)
        .map(|tid| {
            let base = tid * n_fmadd;
            let mut acc = Fr::zero();
            for i in 0..n_fmadd {
                let idx = (base + i) % stride;
                acc += a_cpu[idx] * b_cpu[idx];
            }
            acc
        })
        .collect();

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_fmadd(
        &ctx.device,
        &ctx.queue,
        &ctx.kernels.fmadd,
        &a_mtl,
        &b_mtl,
        n_threads,
    );

    for i in 0..n_threads {
        let got = to_cpu(result[i]);
        assert_eq!(expected[i], got, "fmadd mismatch at thread {i}");
    }
}

#[test]
fn edge_cases() {
    let ctx = TestCtx::new();
    let zero = Fr::zero();
    let one = Fr::from_u64(1);

    let z = to_metal(zero);
    let o = to_metal(one);

    // 0 + 0 = 0
    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.add, &[z], &[z]);
    assert_eq!(to_cpu(result[0]), zero);

    // 1 * 0 = 0
    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.mul, &[o], &[z]);
    assert_eq!(to_cpu(result[0]), zero);

    // 0 - 0 = 0
    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.sub, &[z], &[z]);
    assert_eq!(to_cpu(result[0]), zero);

    // 1 * 1 = 1
    let result = dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.mul, &[o], &[o]);
    assert_eq!(to_cpu(result[0]), one);

    // -0 = 0
    let result = dispatch_unary(&ctx.device, &ctx.queue, &ctx.kernels.neg, &[z]);
    assert_eq!(to_cpu(result[0]), zero);

    // a + (-a) = 0
    let mut rng = StdRng::seed_from_u64(0x9999);
    let a = Fr::random(&mut rng);
    let neg_a = -a;
    let result = dispatch_binary(
        &ctx.device,
        &ctx.queue,
        &ctx.kernels.add,
        &[to_metal(a)],
        &[to_metal(neg_a)],
    );
    assert_eq!(to_cpu(result[0]), zero, "a + (-a) should be zero");

    // a * a^{-1} = 1
    let a_inv = a.inverse().unwrap();
    let result = dispatch_binary(
        &ctx.device,
        &ctx.queue,
        &ctx.kernels.mul,
        &[to_metal(a)],
        &[to_metal(a_inv)],
    );
    assert_eq!(to_cpu(result[0]), one, "a * a^-1 should be one");
}
