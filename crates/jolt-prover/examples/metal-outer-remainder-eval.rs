//! Fixed production-fixture evaluator for the Spartan outer remainder.
//!
//! The binary runs complete clear Akita proofs so the real stage-1 carry and
//! resident-row lifecycle are exercised. `scripts/metal_outer_remainder_eval.py`
//! extracts only `OuterRemainder::complete_member` from the Chrome trace.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "the evaluator fails loudly and emits one machine-readable record"
)]

use std::{fs, path::PathBuf, time::Instant};

use clap::Parser;
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_akita::AkitaSetupParams;
use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_openings::CommitmentScheme;
use jolt_profiling::{setup_tracing_with_trace_path, TracingFormat};
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput,
};
use jolt_prover::{akita, JoltProverPreprocessing, ProverConfig};
use jolt_prover_legacy::host;
use jolt_prover_legacy::zkvm::packed::{
    akita_verifier_preprocessing, AkitaField, AkitaPackedScheme, AkitaScheme, AkitaTranscript,
    AkitaVc,
};
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_verifier::proof::JoltProof;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend};
use serde_json::{json, Value};
use tracer::execution_backend::TracerBackend;

const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9;
const REQUIRED_RAYON_THREADS: usize = 16;

type EvalBackend = akita::JoltAkitaBackend<AkitaField, AkitaScheme>;
type EvalPreprocessing = JoltProverPreprocessing<AkitaScheme, AkitaVc>;
type EvalProof = JoltProof<AkitaScheme, AkitaVc>;

#[derive(Parser, Debug)]
struct Cli {
    #[clap(long, default_value_t = 26)]
    log_n: usize,

    #[clap(long, default_value_t = 5)]
    pairs: usize,

    #[clap(long)]
    trace_path: PathBuf,

    #[clap(long, default_value_t = 256)]
    materialize_threads: usize,

    #[clap(long, default_value_t = 128)]
    transition_threads: usize,

    #[clap(long, default_value_t = 256)]
    output_threads: usize,

    #[clap(long, default_value_t = 16)]
    cutoff_log2: u32,

    #[clap(long, default_value_t = 18)]
    trace_cutoff_log2: u32,
}

struct Fixture<W> {
    preprocessing: EvalPreprocessing,
    config: ProverConfig,
    witness: W,
    public_io: JoltDevice,
    trace_rows: usize,
}

fn main() {
    let cli = Cli::parse();
    assert_eq!(cli.log_n, 26, "the frozen evaluator targets log_n=26");
    assert_eq!(
        cli.pairs, 5,
        "the frozen evaluator requires five timed pairs"
    );
    assert_eq!(
        rayon::current_num_threads(),
        REQUIRED_RAYON_THREADS,
        "RAYON_NUM_THREADS must be 16"
    );
    if let Some(parent) = cli.trace_path.parent() {
        fs::create_dir_all(parent).expect("create trace directory");
    }
    let tracing_guards = setup_tracing_with_trace_path(&[TracingFormat::Chrome], &cli.trace_path);

    let fixture_span = tracing::info_span!("OuterRemainderEval::fixture_setup").entered();
    let fixture = fibonacci_fixture(cli.log_n);
    assert_eq!(fixture.config.trace_length, 1usize << cli.log_n);

    let optimized = EvalBackend::optimized();
    let mut metal_config = jolt_kernels::metal::MetalConfig::default();
    metal_config
        .spartan_outer_remainder
        .dispatch
        .materialize_threads_per_threadgroup = Some(cli.materialize_threads);
    metal_config
        .spartan_outer_remainder
        .dispatch
        .stream_bind_threads_per_threadgroup = Some(cli.transition_threads);
    metal_config
        .spartan_outer_remainder
        .dispatch
        .transition_threads_per_threadgroup = Some(cli.transition_threads);
    metal_config
        .spartan_outer_remainder
        .dispatch
        .opening_threads_per_threadgroup = Some(cli.output_threads);
    metal_config
        .spartan_outer_remainder
        .dispatch
        .cpu_tail_elements = 1usize
        .checked_shl(cli.cutoff_log2)
        .expect("outer remainder cutoff log2 fits usize");
    metal_config.spartan_outer_remainder.trace_cutoff_elements = 1usize
        .checked_shl(cli.trace_cutoff_log2)
        .expect("outer remainder trace cutoff log2 fits usize");
    let metal = EvalBackend::metal(metal_config).expect("initialize Metal backend and pipelines");
    drop(fixture_span);

    let orders = (0..cli.pairs)
        .map(|pair| {
            if pair % 2 == 0 {
                ["optimized", "metal"]
            } else {
                ["metal", "optimized"]
            }
        })
        .collect::<Vec<_>>();

    let warmup = run_pair(
        -1,
        true,
        ["optimized", "metal"],
        &optimized,
        &metal,
        &fixture,
    );
    let samples = orders
        .iter()
        .enumerate()
        .map(|(pair, order)| run_pair(pair as i64, false, *order, &optimized, &metal, &fixture))
        .collect::<Vec<_>>();

    drop(tracing_guards);
    println!(
        "{}",
        json!({
            "schema": "outer_remainder_runner_v1",
            "schema_version": 1,
            "fixture": "real-fibonacci-akita-proof",
            "log_n": cli.log_n,
            "trace_rows": fixture.trace_rows,
            "padded_trace_rows": fixture.config.trace_length,
            "pairs": cli.pairs,
            "excluded_warmup_pairs": 1,
            "rayon_threads": rayon::current_num_threads(),
            "orders": orders,
            "parameters": {
                "materialize_threads": cli.materialize_threads,
                "transition_threads": cli.transition_threads,
                "output_threads": cli.output_threads,
                "cutoff_log2": cli.cutoff_log2,
                "trace_cutoff_log2": cli.trace_cutoff_log2,
            },
            "warmup": warmup,
            "samples": samples,
        })
    );
}

fn run_pair<W: JoltWitnessPlane<AkitaField>>(
    pair: i64,
    excluded_warmup: bool,
    order: [&'static str; 2],
    optimized: &EvalBackend,
    metal: &EvalBackend,
    fixture: &Fixture<W>,
) -> Value {
    let mut cpu: Option<(EvalProof, u128)> = None;
    let mut gpu: Option<(EvalProof, u128)> = None;

    for (order_position, backend_label) in order.into_iter().enumerate() {
        let backend = match backend_label {
            "optimized" => optimized,
            "metal" => metal,
            _ => unreachable!("the order is constructed from two fixed labels"),
        };
        let sample = prove_arm(
            pair,
            excluded_warmup,
            order_position,
            backend_label,
            backend,
            fixture,
        );
        match backend_label {
            "optimized" => cpu = Some(sample),
            "metal" => gpu = Some(sample),
            _ => unreachable!("the order is constructed from two fixed labels"),
        }
    }

    let (cpu_proof, cpu_full_prove_ns) = cpu.expect("pair contains optimized arm");
    let (metal_proof, metal_full_prove_ns) = gpu.expect("pair contains Metal arm");
    let proofs_exact = cpu_proof == metal_proof;
    assert!(proofs_exact, "optimized and Metal proofs differ");
    json!({
        "pair": pair,
        "excluded_warmup": excluded_warmup,
        "order": order,
        "optimized": {
            "full_prove_ns": cpu_full_prove_ns,
            "proof_verified": true,
        },
        "metal": {
            "full_prove_ns": metal_full_prove_ns,
            "proof_verified": true,
        },
        "proofs_exact": proofs_exact,
    })
}

fn prove_arm<W: JoltWitnessPlane<AkitaField>>(
    pair: i64,
    excluded_warmup: bool,
    order_position: usize,
    backend_label: &'static str,
    backend: &EvalBackend,
    fixture: &Fixture<W>,
) -> (EvalProof, u128) {
    let arm_span = tracing::info_span!(
        "OuterRemainderEval::arm",
        backend = backend_label,
        sample_index = pair,
        pair,
        order_position,
        excluded_warmup,
        trace_rows = fixture.trace_rows,
        padded_trace_rows = fixture.config.trace_length,
    )
    .entered();
    let started = Instant::now();
    let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
        backend,
        &fixture.preprocessing,
        &fixture.config,
        None,
        None,
        &fixture.witness,
        &fixture.public_io,
    )
    .expect("modular Akita proof");
    let full_prove_ns = started.elapsed().as_nanos();
    drop(arm_span);

    jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
        &fixture.preprocessing.verifier,
        &fixture.public_io,
        &proof,
        None,
    )
    .expect("modular Akita proof verifies");
    (proof, full_prove_ns)
}

fn fibonacci_fixture(log_n: usize) -> Fixture<TraceBackend<'static, OwnedTrace>> {
    let max_trace_length = 1usize << log_n;
    let target_cycles = (max_trace_length as f64 * SAFETY_MARGIN) as usize;
    let input_units = std::cmp::max(1, (target_cycles as f64 / CYCLES_PER_FIBONACCI_UNIT) as u32);
    let input = postcard::to_stdvec(&input_units).expect("serialize Fibonacci input");

    let mut program = host::Program::new("fibonacci-guest");
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(legacy_trace.len().next_power_of_two() <= max_trace_length);
    drop(legacy_trace);
    let elf_contents = program.get_elf_contents().expect("Fibonacci ELF contents");
    let memory_layout = io_device.memory_layout.clone();
    drop(program);

    let program_data =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy program preprocessing");
    let program_meta = program_data.meta();
    let shared_preprocessing: JoltSharedPreprocessing<AkitaPackedScheme> =
        JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
    let legacy_preprocessing = LegacyProverPreprocessing::new(shared_preprocessing);
    // The one-shot evaluator keeps the immutable program owners alive for all
    // twelve proof replays. Leaking them avoids a self-referential fixture.
    let jolt_program = Box::leak(Box::new(JoltProgram::from_elf_bytes(elf_contents)));

    let trace_output = trace_modular(jolt_program, &memory_layout, &input);
    let trace_rows = trace_output.trace.rows().len();
    let config = ProverConfig::derive::<AkitaField>(
        trace_output.trace.rows(),
        &memory_layout,
        program_meta.min_bytecode_address,
        program_meta.program_image_len_words,
        max_trace_length,
    )
    .expect("derive prover config");
    let log_t = config.trace_length.ilog2() as usize;
    let log_k_chunk = config.one_hot_config.committed_chunk_bits();
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_meta.bytecode_len,
        config.ram_K,
        JoltRelationId::HammingWeightClaimReduction,
    )
    .expect("derive packed formula dimensions");
    let one_hot_shape = OneHotTraceShape {
        ra_layout: dimensions.ra_layout,
        log_t,
        log_k_chunk,
    };
    let setup_shape = ONE_HOT_TRACE_LAYOUT
        .setup_shape(&one_hot_shape)
        .expect("derive packed setup shape");
    let layout_digest = ONE_HOT_TRACE_LAYOUT
        .layout_digest(&one_hot_shape)
        .expect("derive packed layout digest");
    let setup_params = AkitaSetupParams::one_hot_only(
        setup_shape.num_vars,
        setup_shape.num_polys,
        layout_digest,
        1usize << log_k_chunk,
    );
    let (pcs_setup, verifier_setup) =
        AkitaScheme::setup(setup_params).expect("derive transparent Akita setup");
    let verifier_preprocessing =
        akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);
    drop(legacy_preprocessing);

    let program_preprocessing = Box::leak(Box::new(
        verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone(),
    ));
    let public_io = trace_output.device.clone();
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(jolt_program, program_preprocessing, trace_output),
    );
    let preprocessing = EvalPreprocessing {
        verifier: verifier_preprocessing,
        pcs_setup,
        committed_program: None,
    };
    Fixture {
        preprocessing,
        config,
        witness,
        public_io,
        trace_rows,
    }
}

fn trace_modular(
    program: &JoltProgram,
    memory_layout: &common::jolt_device::MemoryLayout,
    inputs: &[u8],
) -> TraceOutput<OwnedTrace> {
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    TracerBackend::new()
        .trace(
            program,
            TraceInputs {
                inputs: inputs.to_vec(),
                untrusted_advice: Vec::new(),
                trusted_advice: Vec::new(),
                memory_config,
            },
        )
        .expect("modular Fibonacci trace")
}
