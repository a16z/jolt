#![allow(unused_imports)]
#![allow(clippy::extra_unused_type_parameters)]

use crate::field::JoltField;
use crate::host;
use crate::jolt::vm::rv32i_vm::RV32IJoltVM;
use crate::jolt::vm::{Jolt, JoltProverPreprocessing, JoltVerifierPreprocessing};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryCommitmentScheme as Dory;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::subprotocols::shout::ShoutProof;
use crate::subprotocols::twist::{TwistAlgorithm, TwistProof};
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use ark_bn254::{Bn254, Fr};
use ark_std::test_rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Zipf};
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Dory,
    HyperKZG,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Shout,
    SparseDenseShout,
    Twist,
}

/// Dynamic Extension Code
use tracer::emulator::cpu::Cpu;
use tracer::instruction::inline::INLINE;
use tracer::instruction::{RISCVInstruction, RV32IMInstruction};
use tracer::{register_inline, list_registered_inlines};

use crate::benches::generator;

/// SHA-256 initial hash values
pub const BLOCK: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 round constants (K)
pub const K: [u64; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn execute_sha256_compression(initial_state: [u32; 8], input: [u32; 16]) -> [u32; 8] {
    let mut a = initial_state[0];
    let mut b = initial_state[1];
    let mut c = initial_state[2];
    let mut d = initial_state[3];
    let mut e = initial_state[4];
    let mut f = initial_state[5];
    let mut g = initial_state[6];
    let mut h = initial_state[7];

    let mut w = [0u32; 64];

    w[..16].copy_from_slice(&input);

    // Calculate word schedule
    for i in 16..64 {
        // σ₁(w[i-2]) + w[i-7] + σ₀(w[i-15]) + w[i-16]
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    // Perform 64 rounds
    for i in 0..64 {
        let ch = (e & f) ^ ((!e) & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);

        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22); // Σ₀(a)
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25); // Σ₁(e)

        let t1 = h
            .wrapping_add(sigma1)
            .wrapping_add(ch)
            .wrapping_add(K[i] as u32)
            .wrapping_add(w[i]);
        let t2 = sigma0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    // Final IV addition
    [
        initial_state[0].wrapping_add(a),
        initial_state[1].wrapping_add(b),
        initial_state[2].wrapping_add(c),
        initial_state[3].wrapping_add(d),
        initial_state[4].wrapping_add(e),
        initial_state[5].wrapping_add(f),
        initial_state[6].wrapping_add(g),
        initial_state[7].wrapping_add(h),
    ]
}

fn sha2_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Load 16 input words from memory at rs1
    let mut input = [0u32; 16];
    for (i, word) in input.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs1].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load input word")
            .0;
    }

    // Load 8 initial state words from memory at rs2
    let mut iv = [0u32; 8];
    for (i, word) in iv.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64)
            .expect("SHA256: Failed to load initial state")
            .0;
    }

    // Execute compression and store result at rs2
    let result = execute_sha256_compression(iv, input);
    for (i, &word) in result.iter().enumerate() {
        cpu.mmu
            .store_word(
                cpu.x[instr.operands.rs2].wrapping_add((i * 4) as i64) as u64,
                word,
            )
            .expect("SHA256: Failed to store result");
    }
}

const VIRTUAL_REGISTER_COUNT: u64 = 32; //  see Section 6.1 of Jolt paper
pub const fn virtual_register_index(index: u64) -> u64 {
    index + VIRTUAL_REGISTER_COUNT
}

// Virtual instructions builder for sha256 - returns empty sequence as XOR is atomic
fn sha2_virtual_sequence_builder(_address: u64, _rs1: usize, _rs2: usize) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let mut vr = [0; 32];
    (0..32).for_each(|i| {
        vr[i] = virtual_register_index(i as u64) as usize;
    });
    let builder = generator::Sha256SequenceBuilder::new(
        _address, vr, _rs1, _rs2, false, // not initial - uses custom IV from rs2
    );
    builder.build()
}

// Initialize and register inlines
pub fn init_inlines() -> Result<(), String> {
    // Register SHA256 with funct7=0x00 (matching the SDK's assembly instruction)
    register_inline(
        0x00,
        0x00,
        "SHA256_INLINE",
        std::boxed::Box::new(sha2_exec),
        std::boxed::Box::new(sha2_virtual_sequence_builder),
    )?;

    // Also register with funct7=0x01 for compatibility
    register_inline(
        0x01,
        0x00,
        "SHA2_INLINE",
        std::boxed::Box::new(sha2_exec),
        std::boxed::Box::new(sha2_virtual_sequence_builder),
    )?;

    Ok(())
}

#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        eprintln!("Failed to register inlines: {}", e);
    }
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Dory => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Dory<KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, Dory<KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha2Chain => sha2chain::<Fr, Dory<KeccakTranscript>, KeccakTranscript>(),
            BenchType::Fibonacci => fibonacci::<Fr, Dory<KeccakTranscript>, KeccakTranscript>(),
            BenchType::Shout => shout::<Fr, KeccakTranscript>(),
            BenchType::Twist => twist::<Fr, KeccakTranscript>(),
            BenchType::SparseDenseShout => sparse_dense_shout::<Fr, KeccakTranscript>(),
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
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const TABLE_SIZE: usize = 1 << 16;
    const NUM_LOOKUPS: usize = 1 << 20;

    let mut rng = test_rng();

    let lookup_table: Vec<F> = (0..TABLE_SIZE).map(|_| F::random(&mut rng)).collect();
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        .collect();

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
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
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<u32, PCS, F, ProofTranscript>("fibonacci-guest", &400000u32)
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
    println!("{name:<30} : {file_size_mb:.3} MB");
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
    let inputs = postcard::to_stdvec(input).unwrap();

    let task = move || {
        let (trace, final_memory_state, io_device) = program.trace(&inputs);
        let (bytecode, init_memory_state) = program.decode();

        println!(
            "Trace Length: {}  ----   Bytecode Length: {}",
            trace.len(),
            bytecode.len()
        );

        let preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                init_memory_state,
                1 << 18,
                1 << 18,
                1 << 20,
            );

        let (jolt_proof, program_io, _) = <RV32IJoltVM as Jolt<32, _, PCS, ProofTranscript>>::prove(
            io_device,
            trace,
            final_memory_state,
            preprocessing.clone(),
        );

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<F, PCS, ProofTranscript>::from(&preprocessing);

        println!("Proof sizing:");
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.commitments", &jolt_proof.commitments);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        serialize_and_print_size(" jolt_proof.ram", &jolt_proof.ram);
        serialize_and_print_size(" jolt_proof.registers", &jolt_proof.registers);
        serialize_and_print_size(
            " jolt_proof.instruction_lookups",
            &jolt_proof.instruction_lookups,
        );
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

fn sha2chain<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");

    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&1500u32).unwrap());

    let task = move || {
        let (trace, final_memory_state, io_device) = program.trace(&inputs);
        let (bytecode, init_memory_state) = program.decode();

        let preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                init_memory_state,
                1 << 20,
                1 << 20,
                1 << 24,
            );

        let (jolt_proof, program_io, _) = <RV32IJoltVM as Jolt<32, _, PCS, ProofTranscript>>::prove(
            io_device,
            trace,
            final_memory_state,
            preprocessing.clone(),
        );

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<F, PCS, ProofTranscript>::from(&preprocessing);

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
