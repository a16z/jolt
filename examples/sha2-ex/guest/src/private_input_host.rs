#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
use core::ops::Deref;
pub fn memory_config_sha2() -> jolt::MemoryConfig {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    MemoryConfig {
        max_input_size: 4096,
        max_output_size: 4096,
        max_private_input_size: 4096,
        stack_size: 4096,
        memory_size: 10240,
        program_size: None,
    }
}
pub fn build_prover_sha2(
    program: jolt::host::Program,
    preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
) -> impl Fn(
    &[u8],
    jolt::Private<[u8; 32]>,
) -> ([u8; 32], jolt::RV64IMACJoltProof, jolt::JoltDevice) + Sync + Send {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let program = std::sync::Arc::new(program);
    let preprocessing = std::sync::Arc::new(preprocessing);
    let prove_closure = move |
        public_input: &[u8],
        second_input: jolt::Private<[u8; 32]>|
    {
        let program = (*program).clone();
        let preprocessing = (*preprocessing).clone();
        prove_sha2(program, preprocessing, public_input, second_input)
    };
    prove_closure
}
pub fn build_verifier_sha2(
    preprocessing: jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS>,
) -> impl Fn(&[u8], ([u8; 32]), bool, jolt::RV64IMACJoltProof) -> bool + Sync + Send {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let preprocessing = std::sync::Arc::new(preprocessing);
    let verify_closure = move |
        public_input: &[u8],
        output,
        panic,
        proof: jolt::RV64IMACJoltProof|
    {
        let preprocessing = (*preprocessing).clone();
        let memory_config = MemoryConfig {
            max_input_size: preprocessing.shared.memory_layout.max_input_size,
            max_output_size: preprocessing.shared.memory_layout.max_output_size,
            max_private_input_size: preprocessing
                .shared
                .memory_layout
                .max_private_input_size,
            stack_size: preprocessing.shared.memory_layout.stack_size,
            memory_size: preprocessing.shared.memory_layout.memory_size,
            program_size: Some(preprocessing.shared.memory_layout.program_size),
        };
        let mut io_device = JoltDevice::new(&memory_config);
        io_device.inputs.append(&mut jolt::postcard::to_stdvec(&public_input).unwrap());
        io_device.outputs.append(&mut jolt::postcard::to_stdvec(&output).unwrap());
        io_device.panic = panic;
        JoltRV64IMAC::verify(&preprocessing, proof, io_device, None).is_ok()
    };
    verify_closure
}
pub fn sha2(public_input: &[u8], second_input: jolt::Private<[u8; 32]>) -> [u8; 32] {
    {
        let hash1 = jolt_inlines_sha2::Sha256::digest(public_input);
        let hash2 = jolt_inlines_sha2::Sha256::digest(second_input.deref());
        let mut concatenated = [0u8; 64];
        concatenated[..32].copy_from_slice(&hash1);
        concatenated[32..].copy_from_slice(&hash2);
        jolt_inlines_sha2::Sha256::digest(&concatenated)
    }
}
pub fn compile_sha2(target_dir: &str) -> jolt::host::Program {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let mut program = Program::new("sha2-guest");
    program.set_func("sha2");
    program.set_std(false);
    program.set_memory_size(10240u64);
    program.set_stack_size(4096u64);
    program.set_max_input_size(4096u64);
    program.set_max_output_size(4096u64);
    program.set_max_private_input_size(4096u64);
    program.build_with_channel(target_dir, "stable");
    program
}
pub fn preprocess_prover_sha2(
    program: &mut jolt::host::Program,
) -> jolt::JoltProverPreprocessing<jolt::F, jolt::PCS> {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let (bytecode, memory_init, program_size) = program.decode();
    let memory_config = MemoryConfig {
        max_input_size: 4096,
        max_output_size: 4096,
        max_private_input_size: 4096,
        stack_size: 4096,
        memory_size: 10240,
        program_size: Some(program_size),
    };
    let memory_layout = MemoryLayout::new(&memory_config);
    let preprocessing: JoltProverPreprocessing<jolt::F, jolt::PCS> = JoltRV64IMAC::prover_preprocess(
        bytecode,
        memory_layout,
        memory_init,
        65536,
    );
    preprocessing
}
pub fn preprocess_verifier_sha2(
    program: &mut jolt::host::Program,
) -> jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS> {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let (bytecode, memory_init, program_size) = program.decode();
    let memory_config = MemoryConfig {
        max_input_size: 4096,
        max_output_size: 4096,
        max_private_input_size: 4096,
        stack_size: 4096,
        memory_size: 10240,
        program_size: Some(program_size),
    };
    let memory_layout = MemoryLayout::new(&memory_config);
    let prover_preprocessing: JoltProverPreprocessing<jolt::F, jolt::PCS> = JoltRV64IMAC::prover_preprocess(
        bytecode,
        memory_layout,
        memory_init,
        65536,
    );
    let preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    preprocessing
}
pub fn verifier_preprocessing_from_prover_sha2(
    prover_preprocessing: &jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
) -> jolt::JoltVerifierPreprocessing<jolt::F, jolt::PCS> {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let preprocessing = JoltVerifierPreprocessing::from(prover_preprocessing);
    preprocessing
}
pub fn prove_sha2(
    mut program: jolt::host::Program,
    preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
    public_input: &[u8],
    second_input: jolt::Private<[u8; 32]>,
) -> ([u8; 32], jolt::RV64IMACJoltProof, jolt::JoltDevice) {
    use jolt::{
        Jolt, JoltField, host::Program, JoltProverPreprocessing,
        JoltVerifierPreprocessing, JoltRV64IMAC, RV64IMACJoltProof, MemoryConfig,
        MemoryLayout, JoltDevice,
    };
    let mut input_bytes = ::alloc::vec::Vec::new();
    input_bytes.append(&mut jolt::postcard::to_stdvec(&public_input).unwrap());
    let mut private_input_bytes = ::alloc::vec::Vec::new();
    private_input_bytes.append(&mut jolt::postcard::to_stdvec(&second_input).unwrap());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let (jolt_proof, io_device, _) = JoltRV64IMAC::prove(
        &preprocessing,
        &elf_contents,
        &input_bytes,
        &private_input_bytes,
    );
    let mut outputs = io_device.outputs.clone();
    outputs.resize(preprocessing.shared.memory_layout.max_output_size as usize, 0);
    let ret_val = jolt::postcard::from_bytes::<[u8; 32]>(&outputs).unwrap();
    (ret_val, jolt_proof, io_device)
}
