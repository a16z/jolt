//! Parity tests for cooperative field arithmetic.
//!
//! Verifies that the cooperative (8-threads-per-element) fr_mul produces
//! bit-identical results to the standard (1-thread-per-element) fr_mul.

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use jolt_field::{Field, Fr, MontgomeryConstants};
use jolt_metal::field::MetalFieldElement;
use jolt_metal::shaders::build_source_with_preamble;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use num_traits::{One, Zero};
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

fn compile_coop_kernel(device: &Device) -> metal::ComputePipelineState {
    let field_preamble = jolt_metal::msl_field_gen::generate_full_preamble::<Fr>();
    let coop_preamble = jolt_metal::coop_field_gen::generate_coop_preamble(Fr::NUM_U32_LIMBS);
    let test_kernel = jolt_metal::coop_field_gen::generate_coop_test_kernel(Fr::NUM_U32_LIMBS);

    let source = build_source_with_preamble(&field_preamble, &[&coop_preamble, &test_kernel]);

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(&source, &options)
        .unwrap_or_else(|e| panic!("Cooperative MSL compilation failed: {e}"));

    let func = library
        .get_function("coop_fr_mul_kernel", None)
        .expect("coop_fr_mul_kernel not found");
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("pipeline creation failed")
}

fn dispatch_coop_mul(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    a: &[MFr],
    b: &[MFr],
) -> Vec<MFr> {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let buf_a = device.new_buffer_with_data(
        a.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(a) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_b = device.new_buffer_with_data(
        b.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(b) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_out = device.new_buffer(
        std::mem::size_of_val(a) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);

    // Cooperative: 8 threads per element, so total threads = n * 8.
    // Threadgroup size must be a multiple of 8 (and ideally 32 for simdgroup alignment).
    let total_threads = (n * 8) as u64;
    let threads_per_group = 32u64.min(total_threads);

    enc.dispatch_threads(
        MTLSize::new(total_threads, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    unsafe {
        let ptr = buf_out.contents().cast::<MFr>();
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

#[test]
fn coop_mul_parity_random() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let pipeline = compile_coop_kernel(&device);

    let mut rng = StdRng::seed_from_u64(0xcafe);
    let n = 256;
    let a_cpu: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let b_cpu: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    let result = dispatch_coop_mul(&device, &queue, &pipeline, &a_mtl, &b_mtl);

    for i in 0..n {
        let expected = a_cpu[i] * b_cpu[i];
        let got = to_cpu(result[i]);
        assert_eq!(
            expected, got,
            "coop mul mismatch at index {i}: expected {expected:?}, got {got:?}"
        );
    }
}

#[test]
fn coop_mul_edge_cases() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let pipeline = compile_coop_kernel(&device);

    // Test edge cases: zero, one, modulus-1, max field element, small values
    let cases: Vec<(Fr, Fr)> = vec![
        (Fr::zero(), Fr::zero()),
        (Fr::zero(), Fr::one()),
        (Fr::one(), Fr::one()),
        (Fr::one(), Fr::from_u64(2)),
        (Fr::from_u64(u64::MAX), Fr::from_u64(u64::MAX)),
        (Fr::from_u64(0xFFFF_FFFF), Fr::from_u64(0xFFFF_FFFF)),
        // -1 * -1 = 1
        (-Fr::one(), -Fr::one()),
        (-Fr::one(), Fr::one()),
        // Large random pair (repeated for coverage)
        (
            Fr::from_u64(0xDEAD_BEEF_CAFE_BABE),
            Fr::from_u64(0x1234_5678_9ABC_DEF0),
        ),
    ];

    let a_cpu: Vec<Fr> = cases.iter().map(|(a, _)| *a).collect();
    let b_cpu: Vec<Fr> = cases.iter().map(|(_, b)| *b).collect();

    let a_mtl: Vec<MFr> = a_cpu.iter().map(|x| to_metal(*x)).collect();
    let b_mtl: Vec<MFr> = b_cpu.iter().map(|x| to_metal(*x)).collect();

    // Pad to a multiple of 4 (simdgroup alignment)
    let pad_to = ((a_mtl.len() + 3) / 4) * 4;
    let mut a_padded = a_mtl.clone();
    let mut b_padded = b_mtl.clone();
    while a_padded.len() < pad_to {
        a_padded.push(to_metal(Fr::zero()));
        b_padded.push(to_metal(Fr::zero()));
    }

    let result = dispatch_coop_mul(&device, &queue, &pipeline, &a_padded, &b_padded);

    for (i, (a, b)) in cases.iter().enumerate() {
        let expected = *a * *b;
        let got = to_cpu(result[i]);
        assert_eq!(
            expected, got,
            "coop mul edge case {i} failed: {a:?} * {b:?} = expected {expected:?}, got {got:?}"
        );
    }
}
