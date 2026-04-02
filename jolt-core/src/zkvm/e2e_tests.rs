// Force-link inline crates so their `inventory::submit!` entries are retained by the linker.
extern crate jolt_inlines_keccak256;
extern crate jolt_inlines_sha2;

use serial_test::serial;

use crate::host;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
use crate::zkvm::prover::{commit_trusted_advice_preprocessing_only, JoltProverPreprocessing};
use crate::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use crate::zkvm::{RV64IMACProver, RV64IMACVerifier};

fn configure_dory(layout: Option<DoryLayout>) {
    DoryGlobals::reset();
    if let Some(layout) = layout {
        DoryGlobals::set_layout(layout);
    }
}

fn prove_and_verify_from_elf(
    program: &mut host::Program,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    max_trace_length: usize,
) -> tracer::JoltDevice {
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let (_, _, _, io_device) = program.trace(inputs, untrusted_advice, trusted_advice);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        e_entry,
    )
    .unwrap();
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");

    let (trusted_commitment, trusted_hint) = if trusted_advice.is_empty() {
        (None, None)
    } else {
        let (commitment, hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, trusted_advice);
        (Some(commitment), Some(hint))
    };

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        inputs,
        untrusted_advice,
        trusted_advice,
        trusted_commitment,
        trusted_hint,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        trusted_commitment,
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Failed to verify proof");

    io_device
}

#[test]
#[serial]
fn fib_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&100u32).unwrap();
    prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);
}

#[test]
#[serial]
fn small_trace_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        8192,
        e_entry,
    )
    .unwrap();

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");
    let log_chunk = 13;
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );

    assert!(
        prover.padded_trace_len <= (1 << log_chunk),
        "Test requires T <= chunk_size ({}), got T = {}",
        1 << log_chunk,
        prover.padded_trace_len
    );

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Failed to verify proof");
}

#[test]
#[serial]
fn sha3_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("sha3-guest");
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let io_device = prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);

    assert_eq!(
        io_device.inputs, inputs,
        "Inputs mismatch: expected {:?}, got {:?}",
        inputs, io_device.inputs
    );
    let expected_output = &[
        0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88, 0x33,
        0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f, 0x44, 0x72,
        0x4, 0x66,
    ];
    assert_eq!(io_device.outputs, expected_output, "Outputs mismatch");
}

#[test]
#[serial]
fn sha2_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("sha2-guest");
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let io_device = prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);

    let expected_output = &[
        0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73, 0xb1,
        0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6, 0x3b, 0x50,
        0xd2, 0x57,
    ];
    assert_eq!(
        io_device.outputs, expected_output,
        "Outputs mismatch: expected {:?}, got {:?}",
        expected_output, io_device.outputs
    );
}

#[test]
#[serial]
fn sha2_e2e_dory_with_unused_advice() {
    configure_dory(None);

    let mut program = host::Program::new("sha2-guest");
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();
    let io_device = prove_and_verify_from_elf(
        &mut program,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        1 << 16,
    );

    let expected_output = &[
        0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73, 0xb1,
        0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6, 0x3b, 0x50,
        0xd2, 0x57,
    ];
    assert_eq!(io_device.outputs, expected_output);
}

#[test]
#[serial]
fn advice_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("merkle-tree-guest");
    let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
    let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
    trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

    let io_device = prove_and_verify_from_elf(
        &mut program,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        1 << 16,
    );

    let expected_output = &[
        0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
        0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32, 0xbb,
        0x16, 0xd7,
    ];
    assert_eq!(io_device.outputs, expected_output);
}

#[test]
#[serial]
fn memory_ops_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("memory-ops-guest");
    prove_and_verify_from_elf(&mut program, &[], &[], &[], 1 << 16);
}

#[test]
#[serial]
fn btreemap_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("btreemap-guest");
    let inputs = postcard::to_stdvec(&50u32).unwrap();
    prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);
}

#[test]
#[serial]
fn muldiv_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);
}

/// Exercises std mode guest compilation (riscv64imac-zero-linux-musl custom target spec).
/// Catches regressions in target spec JSON generation, e.g. target-pointer-width type errors.
#[test]
#[serial]
fn stdlib_e2e_dory() {
    configure_dory(None);

    let mut program = host::Program::new("stdlib-guest");
    program.set_std(true);
    program.set_func("int_to_string");
    let inputs = postcard::to_stdvec(&81i32).unwrap();
    prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);
}

#[test]
#[serial]
fn fib_e2e_dory_address_major() {
    configure_dory(Some(DoryLayout::AddressMajor));

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&50u32).unwrap();
    prove_and_verify_from_elf(&mut program, &inputs, &[], &[], 1 << 16);
}

#[test]
#[serial]
fn advice_e2e_dory_address_major() {
    configure_dory(Some(DoryLayout::AddressMajor));

    let mut program = host::Program::new("merkle-tree-guest");
    let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
    let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
    trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

    let io_device = prove_and_verify_from_elf(
        &mut program,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        1 << 16,
    );

    let expected_output = &[
        0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
        0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32, 0xbb,
        0x16, 0xd7,
    ];
    assert_eq!(io_device.outputs, expected_output);
}
