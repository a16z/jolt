//! GPU parity tests for the 128-bit test field (M127, N=4 limbs).
//!
//! Verifies that the dynamically generated 4-limb MSL arithmetic produces
//! bit-exact results matching the CPU reference in `test_field128.rs`.

#![cfg(target_os = "macos")]

mod test_field128;

use jolt_metal::field::MetalFieldElement;
use jolt_metal::field_config::MslFieldParams;
use jolt_metal::shaders::{build_source_with_preamble, make_pipeline};
use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use test_field128::{F128Config, F128};

type MF128 = MetalFieldElement<4>;

fn to_metal(f: F128) -> MF128 {
    MF128 { limbs: f.limbs }
}

fn to_cpu(g: MF128) -> F128 {
    F128 { limbs: g.limbs }
}

struct F128Kernels {
    mul: ComputePipelineState,
    add: ComputePipelineState,
    sub: ComputePipelineState,
    sqr: ComputePipelineState,
    neg: ComputePipelineState,
    fmadd: ComputePipelineState,
    from_u64: ComputePipelineState,
}

impl F128Kernels {
    fn new(device: &Device) -> Self {
        let field_config = MslFieldParams::new::<F128Config>();
        let source = build_source_with_preamble(
            &field_config.msl_preamble,
            &[&field_config.msl_test_kernels],
        );
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .expect("F128 MSL compilation failed");

        Self {
            mul: make_pipeline(device, &library, "fr_mul_kernel"),
            add: make_pipeline(device, &library, "fr_add_kernel"),
            sub: make_pipeline(device, &library, "fr_sub_kernel"),
            sqr: make_pipeline(device, &library, "fr_sqr_kernel"),
            neg: make_pipeline(device, &library, "fr_neg_kernel"),
            fmadd: make_pipeline(device, &library, "fr_fmadd_kernel"),
            from_u64: make_pipeline(device, &library, "fr_from_u64_kernel"),
        }
    }
}

struct TestCtx {
    device: Device,
    queue: metal::CommandQueue,
    kernels: F128Kernels,
}

impl TestCtx {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let kernels = F128Kernels::new(&device);
        Self {
            device,
            queue,
            kernels,
        }
    }
}

fn upload_slice<T>(device: &Device, data: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_buffer(device: &Device, byte_len: u64) -> metal::Buffer {
    device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
}

fn dispatch_and_wait(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&metal::Buffer],
    n: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    for (i, buf) in buffers.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }
    let tpg = pipeline.max_total_threads_per_threadgroup().min(n as u64);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tpg, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn read_buffer(buf: &metal::Buffer, n: usize) -> Vec<MF128> {
    // SAFETY: buf was allocated with StorageModeShared and has at least n * size_of::<MF128>() bytes.
    // MF128 is repr(C) with u32 limbs, matching the GPU output layout.
    unsafe {
        let ptr = buf.contents().cast::<MF128>();
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

fn dispatch_binary(
    ctx: &TestCtx,
    pipeline: &ComputePipelineState,
    a: &[MF128],
    b: &[MF128],
) -> Vec<MF128> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let buf_a = upload_slice(&ctx.device, a);
    let buf_b = upload_slice(&ctx.device, b);
    let buf_out = alloc_buffer(&ctx.device, std::mem::size_of_val(a) as u64);
    dispatch_and_wait(&ctx.queue, pipeline, &[&buf_a, &buf_b, &buf_out], n);
    read_buffer(&buf_out, n)
}

fn dispatch_unary(ctx: &TestCtx, pipeline: &ComputePipelineState, a: &[MF128]) -> Vec<MF128> {
    let n = a.len();
    let buf_a = upload_slice(&ctx.device, a);
    let buf_out = alloc_buffer(&ctx.device, std::mem::size_of_val(a) as u64);
    dispatch_and_wait(&ctx.queue, pipeline, &[&buf_a, &buf_out], n);
    read_buffer(&buf_out, n)
}

/// Simple deterministic PRNG for test reproducibility (xorshift64).
struct Xorshift64(u64);

impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn random_f128(&mut self) -> F128 {
        let lo = self.next();
        let hi = self.next() & 0x7FFF_FFFF_FFFF_FFFF; // ensure < 2^127
        let limbs = [lo as u32, (lo >> 32) as u32, hi as u32, (hi >> 32) as u32];
        // Already in [0, 2^127), and p = 2^127 - 1, so all values are < p
        // except when all limbs are max. Convert to Montgomery form via R^2 = 4.
        let raw = F128 { limbs };
        raw.mul(F128 {
            limbs: [4, 0, 0, 0],
        })
    }
}

const N: usize = 256;

fn random_elements(rng: &mut Xorshift64, n: usize) -> Vec<F128> {
    (0..n).map(|_| rng.random_f128()).collect()
}

#[test]
fn f128_shader_compiles() {
    let device = Device::system_default().expect("no Metal device");
    let _kernels = F128Kernels::new(&device);
}

#[test]
fn f128_mul_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0xdead_beef);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MF128> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx, &ctx.kernels.mul, &a_mtl, &b_mtl);
    for i in 0..N {
        let expected = a_cpu[i].mul(b_cpu[i]);
        assert_eq!(expected, to_cpu(result[i]), "mul mismatch at {i}");
    }
}

#[test]
fn f128_add_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0xcafe_babe);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MF128> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx, &ctx.kernels.add, &a_mtl, &b_mtl);
    for i in 0..N {
        let expected = a_cpu[i].add(b_cpu[i]);
        assert_eq!(expected, to_cpu(result[i]), "add mismatch at {i}");
    }
}

#[test]
fn f128_sub_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0xf00d_face);
    let a_cpu = random_elements(&mut rng, N);
    let b_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MF128> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_binary(&ctx, &ctx.kernels.sub, &a_mtl, &b_mtl);
    for i in 0..N {
        let expected = a_cpu[i].sub(b_cpu[i]);
        assert_eq!(expected, to_cpu(result[i]), "sub mismatch at {i}");
    }
}

#[test]
fn f128_sqr_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0x1234_5678);
    let a_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_unary(&ctx, &ctx.kernels.sqr, &a_mtl);
    for i in 0..N {
        let expected = a_cpu[i].sqr();
        assert_eq!(expected, to_cpu(result[i]), "sqr mismatch at {i}");
    }
}

#[test]
fn f128_neg_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0xabad_1dea);
    let a_cpu = random_elements(&mut rng, N);

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_unary(&ctx, &ctx.kernels.neg, &a_mtl);
    for i in 0..N {
        let expected = a_cpu[i].neg();
        assert_eq!(expected, to_cpu(result[i]), "neg mismatch at {i}");
    }
}

#[test]
fn f128_from_u64_parity() {
    let ctx = TestCtx::new();
    let vals: Vec<u64> = (0..N as u64).collect();

    let buf_vals = upload_slice(&ctx.device, &vals);
    let buf_out = alloc_buffer(&ctx.device, (N * std::mem::size_of::<MF128>()) as u64);
    dispatch_and_wait(&ctx.queue, &ctx.kernels.from_u64, &[&buf_vals, &buf_out], N);
    let result = read_buffer(&buf_out, N);

    for i in 0..N {
        let expected = F128::from_u64(vals[i]);
        assert_eq!(expected, to_cpu(result[i]), "from_u64 mismatch at {i}");
    }
}

#[test]
fn f128_fmadd_parity() {
    let ctx = TestCtx::new();
    let mut rng = Xorshift64(0x9876_5432);
    let stride = 256;
    let a_cpu = random_elements(&mut rng, stride);
    let b_cpu = random_elements(&mut rng, stride);

    let n_threads = 16;
    let n_fmadd = 256;

    let expected: Vec<F128> = (0..n_threads)
        .map(|tid| {
            let base = tid * n_fmadd;
            let mut acc = F128::ZERO;
            for i in 0..n_fmadd {
                let idx = (base + i) % stride;
                acc = acc.add(a_cpu[idx].mul(b_cpu[idx]));
            }
            acc
        })
        .collect();

    let a_mtl: Vec<MF128> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MF128> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let buf_a = upload_slice(&ctx.device, &a_mtl);
    let buf_b = upload_slice(&ctx.device, &b_mtl);
    let buf_out = alloc_buffer(
        &ctx.device,
        (n_threads * std::mem::size_of::<MF128>()) as u64,
    );
    let buf_params = upload_slice(&ctx.device, &[stride as u32]);

    dispatch_and_wait(
        &ctx.queue,
        &ctx.kernels.fmadd,
        &[&buf_a, &buf_b, &buf_out, &buf_params],
        n_threads,
    );
    let result = read_buffer(&buf_out, n_threads);

    for i in 0..n_threads {
        assert_eq!(
            expected[i],
            to_cpu(result[i]),
            "fmadd mismatch at thread {i}"
        );
    }
}

#[test]
fn f128_edge_cases() {
    let ctx = TestCtx::new();
    let zero = F128::ZERO;
    let one = F128::one();

    let z = to_metal(zero);
    let o = to_metal(one);

    // 0 + 0 = 0
    let r = dispatch_binary(&ctx, &ctx.kernels.add, &[z], &[z]);
    assert_eq!(to_cpu(r[0]), zero);

    // 1 * 0 = 0
    let r = dispatch_binary(&ctx, &ctx.kernels.mul, &[o], &[z]);
    assert_eq!(to_cpu(r[0]), zero);

    // 1 * 1 = 1
    let r = dispatch_binary(&ctx, &ctx.kernels.mul, &[o], &[o]);
    assert_eq!(to_cpu(r[0]), one);

    // a + (-a) = 0
    let five = F128::from_u64(5);
    let neg_five = five.neg();
    let r = dispatch_binary(
        &ctx,
        &ctx.kernels.add,
        &[to_metal(five)],
        &[to_metal(neg_five)],
    );
    assert_eq!(to_cpu(r[0]), zero, "a + (-a) should be zero");
}
