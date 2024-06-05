use crate::host;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::poly::commitment::hyrax::HyraxScheme;
use ark_bn254::G1Projective;
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    bench_type: BenchType,
    _num_cycles: Option<usize>,
    _memory_size: Option<usize>,
    _bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2chain(),
        BenchType::Fibonacci => fibonacci(),
        _ => panic!("BenchType does not have a mapping"),
    }
}

fn fibonacci() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("fibonacci-guest", &9u32)
}

fn sha2() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha2-guest", &vec![5u8; 2048])
}

fn sha3() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha3-guest", &vec![5u8; 2048])
}

#[allow(dead_code)]
fn serialize_and_print_size(name: &str, item: &impl ark_serialize::CanonicalSerialize) {
    use std::fs::File;
    let mut file = File::create("temp_file").unwrap();
    item.serialize_compressed(&mut file).unwrap();
    let file_size_bytes = file.metadata().unwrap().len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    let file_size_mb = file_size_kb / 1024.0;
    println!("{:<30} : {:.3} MB", name, file_size_mb);
}

fn prove_example<T: Serialize>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    program.set_input(input);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<
            ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>,
            HyraxScheme<ark_ec::short_weierstrass::Projective<ark_bn254::g1::Config>>,
        > = RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::prove(
                io_device,
                trace,
                circuit_flags,
                preprocessing.clone(),
            );

        // println!("Proof sizing:");
        // serialize_and_print_size("jolt_commitments", &jolt_commitments);
        // serialize_and_print_size("jolt_proof", &jolt_proof);
        // serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        // serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        // serialize_and_print_size(" jolt_proof.read_write_memory", &jolt_proof.read_write_memory);
        // serialize_and_print_size(" jolt_proof.instruction_lookups", &jolt_proof.instruction_lookups);

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha2chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");
    program.set_input(&[5u8; 32]);
    program.set_input(&1024u32);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<
            ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>,
            HyraxScheme<ark_ec::short_weierstrass::Projective<ark_bn254::g1::Config>>,
        > = RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::prove(
                io_device,
                trace,
                circuit_flags,
                preprocessing.clone(),
            );
        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}
