use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use jolt_inlines_sha2 as _;
use jolt_prover_legacy::{
    host,
    zkvm::{
        akita::{
            AkitaLegacyBlake2bTranscript, AkitaNoCurve, AkitaNoopCommitmentScheme,
            AkitaRV64IMACProver,
        },
        preprocessing::JoltSharedPreprocessing,
        program::ProgramPreprocessing,
        proof::verifier_preprocessing_from_prover,
        prover::JoltProverPreprocessing,
        RV64IMACProver,
    },
};
use tracing_subscriber::EnvFilter;

const CYCLES_PER_SHA256: f64 = 3396.0;
const DEFAULT_TARGET_CYCLES: usize = 1 << 20;
const DEFAULT_BYTECODE_CHUNK_COUNT: usize = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PcsMode {
    Dory,
    Akita,
    Both,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = PcsMode::Both)]
    mode: PcsMode,
    #[arg(long, default_value_t = DEFAULT_TARGET_CYCLES)]
    target_cycles: usize,
    #[arg(long)]
    iterations: Option<u32>,
    #[arg(long, default_value_t = DEFAULT_BYTECODE_CHUNK_COUNT)]
    bytecode_chunk_count: usize,
    #[arg(long)]
    skip_verify: bool,
    #[arg(long)]
    trace_akita_phases: bool,
}

#[derive(Clone, Copy, Debug)]
struct TraceTarget {
    iterations: u32,
    trace_len: usize,
    padded_trace_len: usize,
}

#[derive(Debug)]
struct Measurement {
    mode: &'static str,
    iterations: u32,
    trace_len: usize,
    padded_trace_len: usize,
    prove_duration: Duration,
    proof_size_bytes: Option<usize>,
}

fn main() {
    let args = Args::parse();
    if args.trace_akita_phases {
        init_tracing();
    }
    let target = match args.iterations {
        Some(iterations) => trace_target(iterations),
        None => calibrate_trace_target(args.target_cycles),
    };
    let verify = !args.skip_verify;

    println!(
        "sha2-chain target: iterations={}, trace_len={}, padded_trace_len={}",
        target.iterations, target.trace_len, target.padded_trace_len
    );

    let mut measurements = Vec::new();
    if matches!(args.mode, PcsMode::Dory | PcsMode::Both) {
        measurements.push(measure_dory(target, verify));
    }
    if matches!(args.mode, PcsMode::Akita | PcsMode::Both) {
        measurements.push(measure_akita(target, args.bytecode_chunk_count, verify));
    }

    for measurement in measurements {
        let proof_size = measurement
            .proof_size_bytes
            .map_or_else(|| "na".to_string(), |size| size.to_string());
        println!(
            "RESULT mode={} iterations={} trace_len={} padded_trace_len={} prove_seconds={:.6} proof_size_bytes={}",
            measurement.mode,
            measurement.iterations,
            measurement.trace_len,
            measurement.padded_trace_len,
            measurement.prove_duration.as_secs_f64(),
            proof_size,
        );
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_env_filter(filter)
        .try_init();
}

fn calibrate_trace_target(target_cycles: usize) -> TraceTarget {
    let mut iterations = scale_to_target_ops(target_cycles, CYCLES_PER_SHA256);
    let mut best = trace_target(iterations);
    for _ in 0..4 {
        if best.trace_len == 0 {
            break;
        }
        let next = scale_iterations(iterations, target_cycles, best.trace_len);
        if next == iterations {
            break;
        }
        iterations = next;
        let candidate = trace_target(iterations);
        if distance(candidate.trace_len, target_cycles) <= distance(best.trace_len, target_cycles) {
            best = candidate;
        }
    }
    best
}

fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    ((target_cycles as f64 / cycles_per_op) as u32).max(1)
}

fn scale_iterations(iterations: u32, target_cycles: usize, trace_len: usize) -> u32 {
    let scaled = (iterations as u128)
        .saturating_mul(target_cycles as u128)
        .checked_div(trace_len as u128)
        .unwrap_or(iterations as u128);
    (scaled as u32).max(1)
}

fn distance(left: usize, right: usize) -> usize {
    left.abs_diff(right)
}

fn trace_target(iterations: u32) -> TraceTarget {
    let input = sha2_chain_input(iterations);
    let mut program = host::Program::new("sha2-chain-guest");
    let (_, trace, _, _) = program.trace(&input, &[], &[]);
    TraceTarget {
        iterations,
        trace_len: trace.len(),
        padded_trace_len: trace.len().next_power_of_two(),
    }
}

fn sha2_chain_input(iterations: u32) -> Vec<u8> {
    [
        postcard::to_stdvec(&[5u8; 32]).expect("serialize sha2-chain input"),
        postcard::to_stdvec(&iterations).expect("serialize sha2-chain iterations"),
    ]
    .concat()
}

fn measure_dory(target: TraceTarget, verify: bool) -> Measurement {
    let input = sha2_chain_input(target.iterations);
    let mut program = host::Program::new("sha2-chain-guest");
    let (bytecode, init_memory_state, _, entry) = program.decode();
    let (_, _, _, program_io) = program.trace(&input, &[], &[]);
    let program_data =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry).expect("preprocess");
    let shared_preprocessing = JoltSharedPreprocessing::new(
        program_data,
        program_io.memory_layout.clone(),
        target.padded_trace_len,
    );
    let preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .expect("sha2-chain guest should provide ELF contents");
    let prover = RV64IMACProver::gen_from_elf(
        &preprocessing,
        &elf_contents,
        &input,
        &[],
        &[],
        None,
        None,
        None,
    );

    let start = Instant::now();
    let (proof, _) = prover.prove().expect("Dory prover should produce a proof");
    let prove_duration = start.elapsed();
    let proof_size_bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
        .expect("serialize Dory proof")
        .len();

    if verify {
        let verifier_preprocessing = verifier_preprocessing_from_prover(&preprocessing);
        jolt_verifier::verify::<
            jolt_field::Fr,
            jolt_dory::DoryScheme,
            jolt_crypto::Pedersen<jolt_crypto::Bn254G1>,
            jolt_transcript::LegacyBlake2bTranscript<jolt_field::Fr>,
        >(&verifier_preprocessing, &program_io, &proof, None, false)
        .expect("Dory proof should verify");
    }

    Measurement {
        mode: "dory",
        iterations: target.iterations,
        trace_len: target.trace_len,
        padded_trace_len: target.padded_trace_len,
        prove_duration,
        proof_size_bytes: Some(proof_size_bytes),
    }
}

fn measure_akita(target: TraceTarget, bytecode_chunk_count: usize, verify: bool) -> Measurement {
    let input = sha2_chain_input(target.iterations);
    let mut program = host::Program::new("sha2-chain-guest");
    let (bytecode, init_memory_state, _, entry) = program.decode();
    let (_, _, _, program_io) = program.trace(&input, &[], &[]);
    let program_data: ProgramPreprocessing<AkitaNoopCommitmentScheme> =
        ProgramPreprocessing::preprocess(bytecode, init_memory_state, entry).expect("preprocess");
    let (shared, committed_program_prover_data, generators) =
        JoltSharedPreprocessing::new_committed(
            program_data,
            program_io.memory_layout.clone(),
            target.padded_trace_len,
            bytecode_chunk_count,
        );
    let preprocessing = JoltProverPreprocessing::<
        jolt_prover_legacy::field::akita::JoltAkitaField,
        AkitaNoCurve,
        AkitaNoopCommitmentScheme,
    >::new_committed(shared, committed_program_prover_data, generators);
    let elf_contents = program
        .get_elf_contents()
        .expect("sha2-chain guest should provide ELF contents");
    let prover = AkitaRV64IMACProver::gen_from_elf(
        &preprocessing,
        &elf_contents,
        &input,
        &[],
        &[],
        None,
        None,
        None,
    );
    let packed_setup = prover
        .setup_akita_packed_witness()
        .expect("Akita setup should be produced");
    let precommitted = prover
        .commit_akita_precommitted_program_from_setup(&packed_setup)
        .expect("Akita precommitted program artifacts should be produced");

    let start = Instant::now();
    let (verifier_preprocessing, proof, _) = prover
        .prove_akita_with_precomputed(packed_setup, precommitted)
        .expect("Akita prover should produce a proof");
    let prove_duration = start.elapsed();
    if verify {
        jolt_verifier::akita::verify_akita_clear::<AkitaLegacyBlake2bTranscript>(
            &verifier_preprocessing,
            &program_io,
            &proof,
            None,
            &proof.protocol,
        )
        .expect("Akita proof should verify");
    }

    Measurement {
        mode: "akita",
        iterations: target.iterations,
        trace_len: target.trace_len,
        padded_trace_len: target.padded_trace_len,
        prove_duration,
        proof_size_bytes: None,
    }
}
