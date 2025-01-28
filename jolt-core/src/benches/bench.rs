use crate::field::JoltField;
use crate::host;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::zeromorph::Zeromorph;
use crate::subprotocols::shout::ShoutProof;
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use ark_bn254::{Bn254, Fr};
use ark_std::test_rng;
use rand_core::RngCore;
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Zeromorph,
    HyperKZG,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Shout,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
    _num_cycles: Option<usize>,
    _memory_size: Option<usize>,
    _bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Zeromorph => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha2Chain => {
                sha2chain::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Fibonacci => {
                fibonacci::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Shout => shout::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha2Chain => {
                sha2chain::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Fibonacci => {
                fibonacci::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Shout => shout::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn shout<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const TABLE_SIZE: usize = 64;
    const NUM_LOOKUPS: usize = 1 << 20;

    let mut rng = test_rng();

    let lookup_table: Vec<F> = (0..TABLE_SIZE).map(|_| F::random(&mut rng)).collect();
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        .collect();

    let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
    let r_cycle: Vec<F> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());

    let task = move || {
        let _proof = ShoutProof::prove(
            lookup_table,
            read_addresses,
            &r_cycle,
            &mut prover_transcript,
        );
    };

    tasks.push((
        tracing::info_span!("Shout d=1"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn fibonacci<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<u32, PCS, F, ProofTranscript>("fibonacci-guest", &9u32)
}

fn sha2<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<Vec<u8>, PCS, F, ProofTranscript>("sha2-guest", &vec![5u8; 2048])
}

fn sha3<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<Vec<u8>, PCS, F, ProofTranscript>("sha3-guest", &vec![5u8; 2048])
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

fn prove_example<T: Serialize, PCS, F, ProofTranscript>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    program.set_input(input);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<C, F, PCS, ProofTranscript> =
            RV32IJoltVM::preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 18,
                1 << 18,
                1 << 18,
            );

        let (jolt_proof, jolt_commitments, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::prove(
                io_device,
                trace,
                preprocessing.clone(),
            );

        println!("Proof sizing:");
        serialize_and_print_size("jolt_commitments", &jolt_commitments);
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        serialize_and_print_size(
            " jolt_proof.read_write_memory",
            &jolt_proof.read_write_memory,
        );
        serialize_and_print_size(
            " jolt_proof.instruction_lookups",
            &jolt_proof.instruction_lookups,
        );

        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, None);
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

fn sha2chain<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");
    program.set_input(&[5u8; 32]);
    program.set_input(&1000u32);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<C, F, PCS, ProofTranscript> =
            RV32IJoltVM::preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 20,
                1 << 20,
                1 << 22,
            );

        let (jolt_proof, jolt_commitments, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::prove(
                io_device,
                trace,
                preprocessing.clone(),
            );
        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, None);
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
