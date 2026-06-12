#![expect(
    clippy::expect_used,
    reason = "bench setup and e2e verification should fail loudly"
)]

use std::{
    io::Write,
    sync::Once,
    time::{Duration, Instant},
};

use blake2::{
    digest::{consts::U32, Digest},
    Blake2b,
};
use common::{
    constants::ONEHOT_CHUNK_THRESHOLD_LOG_T,
    jolt_device::{JoltDevice, MemoryLayout},
};
use jolt_backends::cpu::CpuBackend;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::CommitmentScheme;
use jolt_program::{
    execution::{JoltProgram, OwnedTrace, RamAccess, TraceOutput, TraceRow},
    preprocess::JoltProgramPreprocessing,
};
use jolt_prover::{JoltProverPreprocessing, ProofParameters, ProverConfig};
use jolt_sdk::host::Program;
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{
    compat, zk_vector_commitment_capacity_requirement, JoltVerifierPreprocessing, NoPcsAssist,
};
use jolt_witness::protocols::jolt_vm::{
    JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
};
use serde::de::DeserializeOwned;
use tracer::TracerBackend;

type Pcs = DoryScheme;
type Vc = Pedersen<Bn254G1>;

static RAYON_INIT: Once = Once::new();

const TARGET_DIR: &str = "/tmp/jolt-prover-microbench-guests";
const SMALL_CASE: BenchCase = BenchCase {
    name: "sha2-chain-2^16",
    target_trace_length: 1 << 16,
};
const LARGE_CASE: BenchCase = BenchCase {
    name: "sha2-chain-2^20",
    target_trace_length: 1 << 20,
};
const TRACE_TARGET_UTILIZATION: f64 = 0.90;
const LARGE_CASE_RATIO_GATE: f64 = 1.15;
const FORCE_LARGE_CASE_ENV: &str = "JOLT_PROVER_MICROBENCH_RUN_2_20";
const BENCH_STACK_SIZE: usize = 128 * 1024 * 1024;
const DEFAULT_SAMPLES: usize = 3;

fn main() {
    initialize_rayon();

    let samples = sample_count();
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "jolt-prover e2e microbench").expect("bench output should write");
    writeln!(stdout, "program: sha2-chain").expect("bench output should write");
    writeln!(stdout, "samples: {samples}").expect("bench output should write");

    let small = run_sha2_case(&mut stdout, SMALL_CASE, samples);

    if should_run_large_case(small.core_ratio) {
        let _ = run_sha2_case(&mut stdout, LARGE_CASE, samples);
    } else {
        writeln!(
            stdout,
            "{}: skipped; requires default transparent {} ratio <= {:.2} or {FORCE_LARGE_CASE_ENV}=1",
            LARGE_CASE.name, SMALL_CASE.name, LARGE_CASE_RATIO_GATE
        )
        .expect("bench output should write");
    }
}

fn sample_count() -> usize {
    std::env::var("JOLT_PROVER_MICROBENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|samples| *samples > 0)
        .unwrap_or(DEFAULT_SAMPLES)
}

fn should_run_large_case(core_ratio: Option<f64>) -> bool {
    force_large_case() || core_ratio.is_some_and(|ratio| ratio <= LARGE_CASE_RATIO_GATE)
}

fn force_large_case() -> bool {
    std::env::var(FORCE_LARGE_CASE_ENV)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn run_sha2_case(writer: &mut impl Write, case: BenchCase, samples: usize) -> BenchCaseOutcome {
    writeln!(writer, "case: {}", case.name).expect("bench output should write");
    writeln!(writer, "target_trace_length: {}", case.target_trace_length)
        .expect("bench output should write");

    let (modular_stats, run_input) =
        bench_modular_sha2_chain(writer, case, modular_mode_label(), samples);
    write_stats(writer, &modular_stats);

    #[cfg(not(any(feature = "zk", feature = "field-inline")))]
    {
        let core_stats = bench_core_sha2_chain(&run_input, samples);
        write_stats(writer, &core_stats);
        let ratio = modular_stats.mean_seconds() / core_stats.mean_seconds();
        writeln!(writer, "{}: modular/core mean ratio={ratio:.3}", case.name)
            .expect("bench output should write");
        BenchCaseOutcome {
            core_ratio: Some(ratio),
        }
    }

    #[cfg(any(feature = "zk", feature = "field-inline"))]
    {
        let _ = run_input;
        BenchCaseOutcome { core_ratio: None }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchCase {
    name: &'static str,
    target_trace_length: usize,
}

#[derive(Clone, Debug)]
struct Sha2RunInput {
    input: [u8; 32],
    iters: u32,
    expected_output: [u8; 32],
}

#[derive(Clone, Debug)]
struct Sha2TraceStats {
    iters: u32,
    target_unpadded_cycles: usize,
    calibration_base_cycles: f64,
    calibration_cycles_per_input: f64,
    unpadded_cycles: usize,
    padded_trace_length: usize,
}

#[derive(Clone, Copy, Debug)]
struct BenchCaseOutcome {
    core_ratio: Option<f64>,
}

struct SelectedSha2Input {
    run_input: Sha2RunInput,
    trace_stats: Sha2TraceStats,
    trace: TraceOutput<OwnedTrace>,
}

struct Sha2CycleCalibration {
    base_cycles: f64,
    cycles_per_input: f64,
}

struct TracedSha2Input {
    run_input: Sha2RunInput,
    trace: TraceOutput<OwnedTrace>,
    unpadded_cycles: usize,
    padded_trace_length: usize,
}

fn select_sha2_chain_input(program: &mut Program, target_trace_length: usize) -> SelectedSha2Input {
    let input = [5u8; 32];
    let calibration = calibrate_sha2_chain_cycles(program, input);
    let target_unpadded_cycles = target_unpadded_cycles(target_trace_length);
    let estimated_iters = ((target_unpadded_cycles as f64 - calibration.base_cycles)
        / calibration.cycles_per_input)
        .floor()
        .clamp(1.0, f64::from(u32::MAX)) as u32;

    let selected = trace_scaled_sha2_input(
        program,
        input,
        estimated_iters,
        target_trace_length,
        target_unpadded_cycles,
    );

    SelectedSha2Input {
        trace_stats: Sha2TraceStats {
            iters: selected.run_input.iters,
            target_unpadded_cycles,
            calibration_base_cycles: calibration.base_cycles,
            calibration_cycles_per_input: calibration.cycles_per_input,
            unpadded_cycles: selected.unpadded_cycles,
            padded_trace_length: selected.padded_trace_length,
        },
        run_input: selected.run_input,
        trace: selected.trace,
    }
}

fn calibrate_sha2_chain_cycles(program: &mut Program, input: [u8; 32]) -> Sha2CycleCalibration {
    let low_iters = 1;
    let high_iters = 8;
    let low_cycles = trace_sha2_chain_cycles(program, input, low_iters);
    let high_cycles = trace_sha2_chain_cycles(program, input, high_iters);
    let cycles_per_input = (high_cycles.saturating_sub(low_cycles) as f64
        / f64::from(high_iters - low_iters))
    .max(1.0);
    let base_cycles = low_cycles as f64 - cycles_per_input * f64::from(low_iters);

    Sha2CycleCalibration {
        base_cycles,
        cycles_per_input,
    }
}

fn target_unpadded_cycles(target_trace_length: usize) -> usize {
    let target = (target_trace_length as f64 * TRACE_TARGET_UTILIZATION).floor() as usize;
    target
        .max(target_trace_length / 2)
        .min(target_trace_length.saturating_sub(1))
}

fn trace_scaled_sha2_input(
    program: &mut Program,
    input: [u8; 32],
    estimated_iters: u32,
    target_trace_length: usize,
    target_unpadded_cycles: usize,
) -> TracedSha2Input {
    let mut best = trace_sha2_chain_candidate(program, input, 1);
    assert!(
        best.padded_trace_length <= target_trace_length,
        "sha2-chain trace exceeds target length at one iteration"
    );

    let mut upper_iters = estimated_iters.max(2);
    loop {
        let candidate = trace_sha2_chain_candidate(program, input, upper_iters);
        if !candidate_fits(&candidate, target_trace_length, target_unpadded_cycles) {
            break;
        }

        best = candidate;
        if upper_iters == u32::MAX {
            break;
        }

        let scale = target_unpadded_cycles as f64 / best.unpadded_cycles.max(1) as f64;
        let next_iters = ((f64::from(upper_iters) * scale * 1.02).ceil()).clamp(
            f64::from(upper_iters.saturating_add(1)),
            f64::from(u32::MAX),
        ) as u32;
        if next_iters == upper_iters {
            break;
        }
        upper_iters = next_iters;
    }

    let mut low_iters = best.run_input.iters;
    while low_iters.saturating_add(1) < upper_iters {
        let mid_iters = low_iters + (upper_iters - low_iters) / 2;
        let candidate = trace_sha2_chain_candidate(program, input, mid_iters);
        if candidate_fits(&candidate, target_trace_length, target_unpadded_cycles) {
            low_iters = mid_iters;
            best = candidate;
        } else {
            upper_iters = mid_iters;
        }
    }

    assert!(
        best.padded_trace_length == target_trace_length,
        "scaled sha2-chain padded trace length {} did not hit target {target_trace_length}",
        best.padded_trace_length
    );
    best
}

fn candidate_fits(
    candidate: &TracedSha2Input,
    target_trace_length: usize,
    target_unpadded_cycles: usize,
) -> bool {
    candidate.padded_trace_length <= target_trace_length
        && candidate.unpadded_cycles <= target_unpadded_cycles
}

fn trace_sha2_chain_candidate(
    program: &mut Program,
    input: [u8; 32],
    iters: u32,
) -> TracedSha2Input {
    let run_input = sha2_run_input(input, iters);
    let trace = trace_sha2_chain(program, &run_input);
    let unpadded_cycles = trace.trace.rows().len();
    let padded_trace_length = padded_trace_length_from_unpadded(unpadded_cycles);

    TracedSha2Input {
        run_input,
        trace,
        unpadded_cycles,
        padded_trace_length,
    }
}

fn sha2_run_input(input: [u8; 32], iters: u32) -> Sha2RunInput {
    Sha2RunInput {
        input,
        iters,
        expected_output: sha2_chain_guest::sha2_chain(input, iters),
    }
}

fn trace_sha2_chain_cycles(program: &mut Program, input: [u8; 32], iters: u32) -> usize {
    let run_input = sha2_run_input(input, iters);
    trace_sha2_chain(program, &run_input).trace.rows().len()
}

fn trace_sha2_chain(program: &mut Program, run_input: &Sha2RunInput) -> TraceOutput<OwnedTrace> {
    let mut tracer = TracerBackend::new();
    program
        .trace_with_backend(&mut tracer, &sha2_input_bytes(run_input), &[], &[])
        .expect("guest should execute through the tracer backend")
}

fn sha2_input_bytes(run_input: &Sha2RunInput) -> Vec<u8> {
    let mut input_bytes = Vec::new();
    append_input(&mut input_bytes, &run_input.input);
    append_input(&mut input_bytes, &run_input.iters);
    input_bytes
}

fn bench_modular_sha2_chain(
    writer: &mut impl Write,
    case: BenchCase,
    bench_id: &'static str,
    samples: usize,
) -> (BenchStats, Sha2RunInput) {
    let mut host_program = sha2_chain_guest::compile_sha2_chain(TARGET_DIR);
    let program = host_program
        .jolt_program()
        .expect("guest should compile into a modular Jolt program");
    let selected = select_sha2_chain_input(&mut host_program, case.target_trace_length);
    write_trace_stats(writer, &selected.trace_stats);
    let run_input = selected.run_input.clone();
    let trace = selected.trace;
    let public_io = trace.device.clone();
    let program_preprocessing =
        program_preprocessing(&program, &public_io, case.target_trace_length);
    let proof_parameters = proof_parameters(&program_preprocessing, &trace);
    let preprocessing =
        prover_preprocessing(program_preprocessing, &program, case.target_trace_length);
    let witness = trace_witness(
        &program,
        &preprocessing.verifier.program,
        trace,
        proof_parameters,
    );
    let config = ProverConfig::default().with_proof_parameters(proof_parameters);

    #[cfg(feature = "field-inline")]
    let field_inline_witness = witness
        .field_inline_witness()
        .expect("field-inline witness should build from the traced program");

    let stats = measure(bench_id, samples, || {
        let mut backend = CpuBackend::default();
        let expected_output = run_input.expected_output;
        let preprocessing_ref = &preprocessing;
        let public_io_ref = &public_io;
        let witness_ref = &witness;
        let config_ref = &config;

        #[cfg(feature = "field-inline")]
        let field_inline_witness_ref = &field_inline_witness;

        run_on_bench_stack(move || {
            #[cfg(feature = "field-inline")]
            let prover_output = jolt_prover::prove_with_components(
                preprocessing_ref,
                public_io_ref,
                witness_ref,
                field_inline_witness_ref,
                config_ref.clone(),
                &mut backend,
            )
            .expect("modular prover should produce a proof");

            #[cfg(not(feature = "field-inline"))]
            let prover_output = jolt_prover::prove_with_components(
                preprocessing_ref,
                public_io_ref,
                witness_ref,
                config_ref.clone(),
                &mut backend,
            )
            .expect("modular prover should produce a proof");

            jolt_verifier::verify::<Fr, Pcs, Vc, Blake2bTranscript, NoPcsAssist>(
                &preprocessing_ref.verifier,
                public_io_ref,
                &prover_output.proof,
                prover_output.trusted_advice_commitment.as_ref(),
                cfg!(feature = "zk"),
            )
            .expect("modular verifier should accept the modular proof");

            let actual_output: [u8; 32] = decode_output(public_io_ref);
            assert_eq!(actual_output, expected_output, "sha2-chain output mismatch");
            let _ = std::hint::black_box(prover_output);
        });
    });

    (stats, run_input)
}

#[cfg(not(any(feature = "zk", feature = "field-inline")))]
fn bench_core_sha2_chain(run_input: &Sha2RunInput, samples: usize) -> BenchStats {
    let mut program = sha2_chain_guest::compile_sha2_chain(TARGET_DIR);
    let shared_preprocessing = sha2_chain_guest::preprocess_shared_sha2_chain(&mut program)
        .expect("jolt-core shared preprocessing should build");
    let prover_preprocessing =
        sha2_chain_guest::preprocess_prover_sha2_chain(shared_preprocessing.clone());
    let verifier_preprocessing = sha2_chain_guest::preprocess_verifier_sha2_chain(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );
    let prove_sha2_chain = sha2_chain_guest::build_prover_sha2_chain(program, prover_preprocessing);
    let verify_sha2_chain = sha2_chain_guest::build_verifier_sha2_chain(verifier_preprocessing);

    measure("jolt-core-transparent", samples, || {
        run_on_bench_stack(|| {
            let (output, proof, program_io) = prove_sha2_chain(run_input.input, run_input.iters);
            assert_eq!(
                output, run_input.expected_output,
                "jolt-core sha2-chain output mismatch"
            );
            let accepted = verify_sha2_chain(
                run_input.input,
                run_input.iters,
                output,
                program_io.panic,
                proof,
            );
            assert!(accepted, "jolt-core verifier should accept the proof");
            let _ = std::hint::black_box(accepted);
        });
    })
}

fn modular_mode_label() -> &'static str {
    match (cfg!(feature = "zk"), cfg!(feature = "field-inline")) {
        (false, false) => "jolt-prover-transparent",
        (true, false) => "jolt-prover-zk",
        (false, true) => "jolt-prover-field-inline",
        (true, true) => "jolt-prover-zk-field-inline",
    }
}

fn run_on_bench_stack<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("jolt-prover-microbench".to_string())
            .stack_size(BENCH_STACK_SIZE)
            .spawn_scoped(scope, f)
            .expect("bench thread should spawn")
            .join()
            .expect("bench thread should not panic")
    })
}

fn initialize_rayon() {
    RAYON_INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(BENCH_STACK_SIZE)
            .build_global()
            .expect("global rayon pool should initialize for e2e benches");
    });
}

#[derive(Debug)]
struct BenchStats {
    label: &'static str,
    samples: usize,
    min: Duration,
    median: Duration,
    mean: Duration,
}

fn measure(label: &'static str, samples: usize, mut run: impl FnMut()) -> BenchStats {
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        run();
        durations.push(started.elapsed());
    }
    BenchStats::from_durations(label, durations)
}

impl BenchStats {
    fn from_durations(label: &'static str, mut durations: Vec<Duration>) -> Self {
        durations.sort_unstable();
        let samples = durations.len();
        let min = durations[0];
        let median = durations[samples / 2];
        let total_nanos = durations.iter().map(Duration::as_nanos).sum::<u128>();
        let mean_nanos = (total_nanos / samples as u128).min(u128::from(u64::MAX));
        let mean = Duration::from_nanos(
            u64::try_from(mean_nanos).expect("mean duration should fit after saturation"),
        );

        Self {
            label,
            samples,
            min,
            median,
            mean,
        }
    }

    #[cfg(not(any(feature = "zk", feature = "field-inline")))]
    fn mean_seconds(&self) -> f64 {
        self.mean.as_secs_f64()
    }
}

fn write_trace_stats(writer: &mut impl Write, stats: &Sha2TraceStats) {
    writeln!(
        writer,
        "input: sha2_chain_iterations={} target_unpadded_cycles={} calibration_base_cycles={:.1} calibration_cycles_per_input={:.1} unpadded_cycles={} padded_trace_length={}",
        stats.iters,
        stats.target_unpadded_cycles,
        stats.calibration_base_cycles,
        stats.calibration_cycles_per_input,
        stats.unpadded_cycles,
        stats.padded_trace_length
    )
    .expect("bench output should write");
}

fn write_stats(writer: &mut impl Write, stats: &BenchStats) {
    writeln!(
        writer,
        "{}: samples={} min={} median={} mean={}",
        stats.label,
        stats.samples,
        format_duration(stats.min),
        format_duration(stats.median),
        format_duration(stats.mean)
    )
    .expect("bench output should write");
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

fn append_input<T>(bytes: &mut Vec<u8>, value: &T)
where
    T: serde::Serialize + ?Sized,
{
    bytes.append(
        &mut jolt_sdk::postcard::to_stdvec(value)
            .expect("guest input should serialize with postcard"),
    );
}

fn decode_output<Output>(public_io: &JoltDevice) -> Output
where
    Output: DeserializeOwned,
{
    let mut outputs = public_io.outputs.clone();
    outputs.resize(public_io.memory_layout.max_output_size as usize, 0);
    jolt_sdk::postcard::from_bytes(&outputs).expect("guest output should deserialize with postcard")
}

fn program_preprocessing(
    program: &JoltProgram,
    public_io: &JoltDevice,
    max_trace_length: usize,
) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        public_io.memory_layout.clone(),
        program.entry_address,
        max_trace_length,
        program.profile,
    )
    .expect("program preprocessing should be valid")
}

fn prover_preprocessing(
    program: JoltProgramPreprocessing,
    jolt_program: &JoltProgram,
    max_trace_length: usize,
) -> JoltProverPreprocessing<Pcs, Vc> {
    let max_log_t = max_trace_length.next_power_of_two().trailing_zeros() as usize;
    let max_log_k_chunk = if max_log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        4
    } else {
        8
    };
    let (pcs_setup, _) = Pcs::setup(max_log_k_chunk + max_log_t);
    let preprocessing_digest = preprocessing_digest(&program);
    let verifier = JoltVerifierPreprocessing::<Pcs, Vc>::from_pcs_prover_setup(
        program,
        preprocessing_digest,
        &pcs_setup,
        zk_vector_commitment_capacity_requirement(),
    );

    #[cfg(feature = "field-inline")]
    let verifier = {
        let code_size = verifier.program.bytecode.code_size;
        verifier.with_field_inline_bytecode(field_inline_bytecode(jolt_program, code_size))
    };

    #[cfg(not(feature = "field-inline"))]
    let _ = jolt_program;

    JoltProverPreprocessing::new(verifier, pcs_setup)
}

fn preprocessing_digest(program: &JoltProgramPreprocessing) -> [u8; 32] {
    let bytes = bincode::serde::encode_to_vec(program, bincode::config::standard())
        .expect("program preprocessing should serialize into memory");
    Blake2b::<U32>::digest(bytes).into()
}

fn trace_witness<'a>(
    program: &'a JoltProgram,
    preprocessing: &'a JoltProgramPreprocessing,
    trace: TraceOutput<OwnedTrace>,
    proof_parameters: ProofParameters,
) -> TraceBackedJoltVmWitness<'a, OwnedTrace> {
    let config = JoltVmWitnessConfig::new(
        proof_parameters.trace_length.trailing_zeros() as usize,
        proof_parameters.ram_k,
        proof_parameters.one_hot_config,
    )
    .retain_trace_rows(true)
    .include_trusted_advice(!trace.device.trusted_advice.is_empty())
    .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());

    TraceBackedJoltVmWitness::new(
        config,
        JoltVmWitnessInputs::new(program, preprocessing, trace),
    )
}

fn proof_parameters(
    preprocessing: &JoltProgramPreprocessing,
    trace: &TraceOutput<OwnedTrace>,
) -> ProofParameters {
    let trace_length = padded_trace_length(trace.trace.rows().len(), preprocessing);
    let log_t = trace_length.trailing_zeros() as usize;
    let ram_k = ram_k(
        preprocessing,
        trace.trace.rows(),
        &trace.device.memory_layout,
    );
    let ram_log_k = ram_k.trailing_zeros() as usize;

    let rw_config = compat::config::ReadWriteConfig::try_from((log_t, ram_log_k))
        .expect("read-write config should be valid");
    let one_hot_config = compat::config::OneHotConfig::from(log_t);

    ProofParameters::new(
        trace_length,
        ram_k,
        JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: rw_config.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: rw_config.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: rw_config.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: rw_config.registers_rw_phase2_num_rounds,
        },
        JoltOneHotConfig {
            log_k_chunk: one_hot_config.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: one_hot_config.lookups_ra_virtual_log_k_chunk,
        },
        TracePolynomialOrder::CycleMajor,
    )
}

fn padded_trace_length(
    unpadded_trace_length: usize,
    preprocessing: &JoltProgramPreprocessing,
) -> usize {
    let trace_length = padded_trace_length_from_unpadded(unpadded_trace_length);
    assert!(
        trace_length <= preprocessing.max_padded_trace_length,
        "trace length {trace_length} exceeds max {}",
        preprocessing.max_padded_trace_length
    );
    trace_length
}

fn padded_trace_length_from_unpadded(unpadded_trace_length: usize) -> usize {
    if unpadded_trace_length < 256 {
        256
    } else {
        (unpadded_trace_length + 1).next_power_of_two()
    }
}

fn ram_k(
    preprocessing: &JoltProgramPreprocessing,
    rows: &[TraceRow],
    layout: &MemoryLayout,
) -> usize {
    let trace_max = rows
        .iter()
        .filter_map(|row| {
            ram_access_address(row).and_then(|address| remap_address(address, layout))
        })
        .max()
        .unwrap_or(0);
    let bytecode_end = remap_address(preprocessing.ram.min_bytecode_address, layout).unwrap_or(0)
        + preprocessing.ram.bytecode_words.len() as u64
        + 1;
    trace_max.max(bytecode_end).next_power_of_two() as usize
}

fn ram_access_address(row: &TraceRow) -> Option<u64> {
    match row.ram_access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

fn remap_address(address: u64, layout: &MemoryLayout) -> Option<u64> {
    if address == 0 || address < layout.get_lowest_address() {
        None
    } else {
        Some((address - layout.get_lowest_address()) / 8)
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_bytecode(
    program: &JoltProgram,
    padded_len: usize,
) -> Vec<jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow> {
    use jolt_claims::protocols::field_inline::formulas::bytecode as field_bytecode;

    let mut rows = program
        .expanded_bytecode
        .iter()
        .map(field_bytecode::field_inline_bytecode_row)
        .collect::<Vec<_>>();
    rows.resize(padded_len, Default::default());
    rows
}
