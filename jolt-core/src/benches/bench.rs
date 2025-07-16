#![allow(unused_imports)]
#![allow(clippy::extra_unused_type_parameters)]

use crate::dag::{jolt_dag, state_manager};
use crate::field::JoltField;
use crate::host;
use crate::jolt::vm::rv32i_vm::RV32IJoltVM;
use crate::jolt::vm::{Jolt, JoltProverPreprocessing, JoltVerifierPreprocessing};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryCommitmentScheme as Dory, DoryGlobals};
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::subprotocols::twist::{TwistAlgorithm, TwistProof};
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use ark_bn254::{Bn254, Fr};
use ark_std::test_rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Zipf};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use tracer;
use tracer::instruction::RV32IMCycle;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Dory,
    HyperKZG,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    FibonacciDag,
    Sha2,
    Sha3,
    Sha2Chain,
    Shout,
    SparseDenseShout,
    Twist,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Dory => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Dory, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, Dory, KeccakTranscript>(),
            BenchType::Sha2Chain => sha2chain::<Fr, Dory, KeccakTranscript>(),
            BenchType::Fibonacci => fibonacci::<Fr, Dory, KeccakTranscript>(),
            BenchType::FibonacciDag => fibonacci_dag::<Fr, Dory, KeccakTranscript>(),
            BenchType::Shout => shout::<Fr, KeccakTranscript>(),
            BenchType::Twist => twist::<Fr, KeccakTranscript>(),
            BenchType::SparseDenseShout => sparse_dense_shout::<Fr, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254>, KeccakTranscript>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyperKZG<Bn254>, KeccakTranscript>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyperKZG<Bn254>, KeccakTranscript>(),
            BenchType::Shout => shout::<Fr, KeccakTranscript>(),
            BenchType::Twist => twist::<Fr, KeccakTranscript>(),
            BenchType::SparseDenseShout => sparse_dense_shout::<Fr, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn shout<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    todo!()
}

fn sparse_dense_shout<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    todo!()
    // let small_value_lookup_tables = F::compute_lookup_tables();
    // F::initialize_lookup_tables(small_value_lookup_tables);

    // let mut tasks = Vec::new();

    // const WORD_SIZE: usize = 32;
    // const LOG_K: usize = 2 * WORD_SIZE;
    // const LOG_T: usize = 19;
    // const T: u64 = 1 << LOG_T;

    // let mut rng = StdRng::seed_from_u64(12345);

    // let trace: Vec<_> = (0..T)
    //     .map(|_| {
    //         let mut step = JoltTraceStep::no_op();
    //         step.instruction_lookup = Some(LookupTables::random(&mut rng, None));
    //         step
    //     })
    //     .collect();

    // let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    // let r_cycle: Vec<F> = prover_transcript.challenge_vector(LOG_T);

    // let task = move || {
    //     let (proof, rv_claim, ra_claims, flag_claims) =
    //         prove_sparse_dense_shout::<WORD_SIZE, _, _>(&trace, r_cycle, &mut prover_transcript);

    //     let mut verifier_transcript = ProofTranscript::new(b"test_transcript");
    //     let r_cycle: Vec<F> = verifier_transcript.challenge_vector(LOG_T);
    //     let verification_result = verify_sparse_dense_shout::<WORD_SIZE, _, _>(
    //         &proof,
    //         LOG_T,
    //         r_cycle,
    //         rv_claim,
    //         ra_claims,
    //         &flag_claims,
    //         &mut verifier_transcript,
    //     );
    //     assert!(
    //         verification_result.is_ok(),
    //         "Verification failed with error: {:?}",
    //         verification_result.err()
    //     );
    // };

    // tasks.push((
    //     tracing::info_span!("Sparse-dense shout d=4"),
    //     Box::new(task) as Box<dyn FnOnce()>,
    // ));

    // tasks
}

fn twist<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const K: usize = 1 << 10;
    const T: usize = 1 << 20;
    const ZIPF_S: f64 = 0.0;
    let zipf = Zipf::new(K as u64, ZIPF_S).unwrap();

    let mut rng = test_rng();

    let mut registers = [0u32; K];
    let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut read_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    for _ in 0..T {
        // Random read register
        let read_address = zipf.sample(&mut rng) as usize - 1;
        // Random write register
        let write_address = zipf.sample(&mut rng) as usize - 1;
        read_addresses.push(read_address);
        write_addresses.push(write_address);
        // Read the value currently in the read register
        read_values.push(registers[read_address]);
        // Random write value
        let write_value = rng.next_u32();
        write_values.push(write_value);
        // The increment is the difference between the new value and the old value
        let write_increment = (write_value as i64) - (registers[write_address] as i64);
        write_increments.push(write_increment);
        // Write the new value to the write register
        registers[write_address] = write_value;
    }

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    let r: Vec<F> = prover_transcript.challenge_vector(K.log_2());
    let r_prime: Vec<F> = prover_transcript.challenge_vector(T.log_2());

    let task = move || {
        let _proof = TwistProof::prove(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            r.clone(),
            r_prime.clone(),
            &mut prover_transcript,
            TwistAlgorithm::Local,
        );
    };

    tasks.push((
        tracing::info_span!("Twist d=1"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn fibonacci<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<u32, PCS, F, ProofTranscript>("fibonacci-guest", &400000u32)
}

fn fibonacci_dag<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    prove_example_dag::<u32, PCS, F, ProofTranscript>("fibonacci-guest", &40000u32)
}

fn sha2<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    prove_example_dag::<Vec<u8>, PCS, F, ProofTranscript>("sha2-guest", &vec![5u8; 10000])
}

fn sha3<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
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
    println!("{name:<30} : {file_size_mb:.3} MB");
}

fn prove_example<T: Serialize, PCS, F, ProofTranscript>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    let inputs = postcard::to_stdvec(input).unwrap();

    let task = move || {
        let (trace, final_memory_state, io_device) = program.trace(&inputs);
        let (bytecode, init_memory_state) = program.decode();

        let preprocessing: JoltProverPreprocessing<F, PCS> = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 18,
            1 << 18,
            1 << 20,
        );

        let (jolt_proof, program_io, _) = <RV32IJoltVM as Jolt<32, _, PCS, _>>::prove(
            io_device,
            trace,
            final_memory_state,
            preprocessing.clone(),
        );

        let verifier_preprocessing = JoltVerifierPreprocessing::<F, PCS>::from(&preprocessing);

        println!("Proof sizing:");
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.commitments", &jolt_proof.commitments);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        // serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        // serialize_and_print_size(" jolt_proof.ram", &jolt_proof.ram);
        serialize_and_print_size(" jolt_proof.registers", &jolt_proof.registers);
        // serialize_and_print_size(
        //     " jolt_proof.instruction_lookups",
        //     &jolt_proof.instruction_lookups,
        // );
        serialize_and_print_size(" jolt_proof.opening_proof", &jolt_proof.opening_proof);

        let verification_result =
            RV32IJoltVM::verify(verifier_preprocessing, jolt_proof, program_io, None);
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

fn prove_example_dag<T: Serialize, PCS, F, ProofTranscript>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    let inputs = postcard::to_stdvec(input).unwrap();

    let task = move || {
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);
        let (bytecode, init_memory_state) = program.decode();

        let preprocessing: JoltProverPreprocessing<F, PCS> = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 18,
            1 << 18,
            1 << 19,
        );

        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        trace.resize(padded_trace_length, RV32IMCycle::NoOp);

        // Truncate trailing zeros on device outputs
        io_device.outputs.truncate(
            io_device
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // Initialize Dory globals
        // let _guard = DoryGlobals::initialize(1 << 18, 1 << 20);

        // Create state manager components
        let prover_accumulator_pre_wrap =
            crate::poly::opening_proof::ProverOpeningAccumulator::<F>::new();
        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let prover_transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(HashMap::new()));
        let commitments = Rc::new(RefCell::new(None));

        // Create prover state manager
        let mut prover_state_manager = state_manager::StateManager::new_prover(
            prover_accumulator,
            prover_transcript.clone(),
            proofs.clone(),
            commitments.clone(),
        );
        prover_state_manager.set_prover_data(
            &preprocessing,
            trace.clone(),
            io_device.clone(),
            final_memory_state.clone(),
        );

        // We only need the prover state manager for benchmarking
        let verifier_accumulator_pre_wrap =
            crate::poly::opening_proof::VerifierOpeningAccumulator::<F>::new();
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator_pre_wrap));
        let verifier_transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let verifier_state_manager = state_manager::StateManager::new_verifier(
            verifier_accumulator,
            verifier_transcript.clone(),
            proofs,
            commitments,
        );

        let mut dag = jolt_dag::JoltDAG::new(prover_state_manager, verifier_state_manager);

        // Only run the prover
        if let Err(e) = dag.prove() {
            panic!("DAG prove failed: {e}");
        }
    };

    tasks.push((
        tracing::info_span!("DAG_Prover_Only"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha2chain<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");

    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&1000u32).unwrap());

    let task = move || {
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);
        let (bytecode, init_memory_state) = program.decode();

        let preprocessing: JoltProverPreprocessing<F, PCS> = RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 18,
            1 << 18,
            1 << 25,
        );

        // Setup trace length and padding (similar to DAG test)
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        trace.resize(padded_trace_length, RV32IMCycle::NoOp);

        // Truncate trailing zeros on device outputs
        io_device.outputs.truncate(
            io_device
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // Initialize Dory globals
        // let _guard = DoryGlobals::initialize(1 << 18, 1 << 20);

        // Create state manager components
        let prover_accumulator_pre_wrap =
            crate::poly::opening_proof::ProverOpeningAccumulator::<F>::new();
        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let prover_transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(HashMap::new()));
        let commitments = Rc::new(RefCell::new(None));

        // Create prover state manager
        let mut prover_state_manager = state_manager::StateManager::new_prover(
            prover_accumulator,
            prover_transcript.clone(),
            proofs.clone(),
            commitments.clone(),
        );
        prover_state_manager.set_prover_data(
            &preprocessing,
            trace.clone(),
            io_device.clone(),
            final_memory_state.clone(),
        );

        // We only need the prover state manager for benchmarking
        let verifier_accumulator_pre_wrap =
            crate::poly::opening_proof::VerifierOpeningAccumulator::<F>::new();
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator_pre_wrap));
        let verifier_transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let verifier_state_manager = state_manager::StateManager::new_verifier(
            verifier_accumulator,
            verifier_transcript.clone(),
            proofs,
            commitments,
        );

        let mut dag = jolt_dag::JoltDAG::new(prover_state_manager, verifier_state_manager);

        // Only run the prover
        if let Err(e) = dag.prove() {
            panic!("DAG prove failed: {e}");
        }
    };

    tasks.push((
        tracing::info_span!("DAG_Prover_Only"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}
