#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "the evaluator exits on invalid evidence and emits one JSON record"
)]

use std::{error::Error, path::PathBuf, time::Duration};

use clap::{Parser, ValueEnum};
use common::jolt_device::MemoryConfig;
use jolt_akita::AkitaSetupParams;
use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_kernels::metal::{
    ProductRemainderCpuEvalSample, ProductRemainderCpuMetalEvalFixture, ProductRemainderEvalResult,
    ProductRemainderMetalEvalSample,
};
use jolt_openings::CommitmentScheme;
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput,
};
use jolt_prover::ProverConfig;
use jolt_prover_legacy::host;
use jolt_prover_legacy::zkvm::packed::{
    akita_verifier_preprocessing, AkitaField, AkitaPackedScheme, AkitaScheme,
};
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use serde_json::{json, Value};
use tracer::execution_backend::TracerBackend;

type EvalResult<T> = Result<T, Box<dyn Error>>;

const SEED: u64 = 0x243f_6a88_85a3_08d3;
const REQUIRED_RAYON_THREADS: usize = 16;
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchName {
    Fibonacci,
    Sha2Chain,
    #[value(name = "btreemap")]
    BTreeMap,
}

impl BenchName {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Fibonacci => "fibonacci",
            Self::Sha2Chain => "sha2-chain",
            Self::BTreeMap => "btreemap",
        }
    }

    fn input(self, target: usize) -> Vec<u8> {
        match self {
            Self::Fibonacci => postcard::to_stdvec(&target_ops(target, CYCLES_PER_FIBONACCI_UNIT))
                .expect("serialize Fibonacci input"),
            Self::Sha2Chain => [
                postcard::to_stdvec(&[5u8; 32]).expect("serialize SHA-2 seed"),
                postcard::to_stdvec(&target_ops(target, CYCLES_PER_SHA256))
                    .expect("serialize SHA-2 repetitions"),
            ]
            .concat(),
            Self::BTreeMap => postcard::to_stdvec(&target_ops(target, CYCLES_PER_BTREEMAP_OP))
                .expect("serialize BTreeMap input"),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Arm {
    Cpu,
    Metal,
    Both,
}

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "optimized_cpu",
            Self::Metal => "metal",
            Self::Both => "counterbalanced",
        }
    }
}

#[derive(Parser, Debug)]
struct Cli {
    #[clap(long, value_enum, default_value = "btreemap")]
    name: BenchName,

    #[clap(long, default_value_t = 20)]
    scale: usize,

    #[clap(long)]
    target_trace_size: Option<usize>,

    #[clap(long, default_value_t = 3)]
    samples: usize,

    #[clap(long, value_enum)]
    arm: Arm,

    #[clap(long)]
    materialize_threads: Option<usize>,

    #[clap(long)]
    transition_threads: Option<usize>,

    #[clap(long)]
    openings_threads: Option<usize>,

    #[clap(long)]
    output: Option<PathBuf>,
}

struct Fixture {
    witness: TraceBackend<'static, OwnedTrace>,
    trace_rows: usize,
    padded_rows: usize,
    log_t: usize,
}

fn main() -> EvalResult<()> {
    let cli = Cli::parse();
    if !(16..=28).contains(&cli.scale) || !(1..=7).contains(&cli.samples) {
        return Err(failure("scale must be 16..=28 and samples must be 1..=7"));
    }
    if rayon::current_num_threads() != REQUIRED_RAYON_THREADS {
        return Err(failure("RAYON_NUM_THREADS must be 16"));
    }
    let target = cli
        .target_trace_size
        .unwrap_or((SAFETY_MARGIN * (1usize << cli.scale) as f64) as usize);
    let fixture = build_fixture(cli.name, cli.scale, target);
    let evaluator = ProductRemainderCpuMetalEvalFixture::new(
        &fixture.witness,
        fixture.log_t,
        SEED,
        cli.materialize_threads,
        cli.transition_threads,
        cli.openings_threads,
    )?;
    let mut oracle = None;
    let (warmup, measurements) = match cli.arm {
        Arm::Both => {
            let cpu = run_arm(Arm::Cpu, &evaluator, &fixture.witness, &mut oracle)?;
            let metal = run_arm(Arm::Metal, &evaluator, &fixture.witness, &mut oracle)?;
            let mut cpu_samples = Vec::with_capacity(cli.samples);
            let mut metal_samples = Vec::with_capacity(cli.samples);
            for pair in 0..cli.samples {
                let order = if pair.is_multiple_of(2) {
                    [Arm::Cpu, Arm::Metal]
                } else {
                    [Arm::Metal, Arm::Cpu]
                };
                for arm in order {
                    let mut sample = run_arm(arm, &evaluator, &fixture.witness, &mut oracle)?;
                    sample["pair"] = json!(pair);
                    sample["order"] = json!(match order {
                        [Arm::Cpu, Arm::Metal] => "cpu_metal",
                        _ => "metal_cpu",
                    });
                    match arm {
                        Arm::Cpu => cpu_samples.push(sample),
                        Arm::Metal => metal_samples.push(sample),
                        Arm::Both => return Err(failure("invalid nested counterbalanced arm")),
                    }
                }
            }
            (
                json!({"optimized_cpu": cpu, "metal": metal}),
                json!({"optimized_cpu": cpu_samples, "metal": metal_samples}),
            )
        }
        arm => {
            let warmup = run_arm(arm, &evaluator, &fixture.witness, &mut oracle)?;
            let mut samples = Vec::with_capacity(cli.samples);
            for sample_index in 0..cli.samples {
                let mut sample = run_arm(arm, &evaluator, &fixture.witness, &mut oracle)?;
                sample["sample"] = json!(sample_index);
                samples.push(sample);
            }
            (warmup, json!(samples))
        }
    };

    let device = evaluator.device_info();
    let shape = evaluator.shape();
    let numeric_widths = evaluator.numeric_widths();
    let result = oracle
        .as_ref()
        .ok_or_else(|| failure("evaluator produced no parity oracle"))?;
    let report = json!({
        "schema": "product_remainder_cpu_metal_v1",
        "schema_version": 1,
        "workload": cli.name.as_str(),
        "scale": cli.scale,
        "target_trace_size": target,
        "execution": "isolated",
        "arm": cli.arm.label(),
        "metal_route": match cli.arm {
            Arm::Metal | Arm::Both => Some(evaluator.metal_route()),
            Arm::Cpu => None,
        },
        "trace_rows": fixture.trace_rows,
        "padded_rows": fixture.padded_rows,
        "log_t": evaluator.log_t(),
        "cycles": evaluator.cycles(),
        "samples": cli.samples,
        "excluded_warmups": 1,
        "rayon_threads": rayon::current_num_threads(),
        "fixture": {
            "wall_ns": duration_ns(evaluator.fixture_wall())?,
            "source_bytes": shape.source_bytes,
            "source_row_bytes": shape.source_row_bytes,
        },
        "shape": {
            "state_a_bytes": shape.state_a_bytes,
            "state_b_bytes": shape.state_b_bytes,
            "workspace_bytes": shape.workspace_bytes,
            "cpu_tail_elements": shape.cpu_tail_elements,
        },
        "numeric_widths": {
            "samples": numeric_widths.samples,
            "left_zero": numeric_widths.left_zero,
            "left_u32": numeric_widths.left_u32,
            "left_lookup_zero": numeric_widths.left_lookup_zero,
            "left_lookup_u16": numeric_widths.left_lookup_u16,
            "left_lookup_u32": numeric_widths.left_lookup_u32,
            "lookup_zero": numeric_widths.lookup_zero,
            "lookup_u16": numeric_widths.lookup_u16,
            "lookup_u32": numeric_widths.lookup_u32,
            "right_zero": numeric_widths.right_zero,
            "right_u32": numeric_widths.right_u32,
            "right_u64": numeric_widths.right_u64,
        },
        "dispatch": {
            "materialize_threads_per_threadgroup": evaluator.materialize_threads_per_threadgroup(),
            "transition_threads_per_threadgroup": evaluator.transition_threads_per_threadgroup(),
            "openings_threads_per_threadgroup": evaluator.openings_threads_per_threadgroup(),
        },
        "device": {
            "name": device.name,
            "max_buffer_length": device.max_buffer_length,
            "max_threadgroup_memory_length": device.max_threadgroup_memory_length,
            "recommended_max_working_set_size": device.recommended_max_working_set_size,
            "offset": device.offset,
        },
        "boundary": {
            "charged_metal": "sequence setup plus member; uniskip excluded",
            "charged_cpu": "member; uniskip excluded",
            "upstream_outer_storage": "excluded; Product borrows production Outer state A",
            "prefetch": "disabled so initial materialization is charged in prepare",
            "joint_instruction_work": "disabled",
        },
        "parity": {
            "exact": true,
            "scope": "within arm; compare checksums across isolated arms",
            "checksum_fnv1a64": format!("{:016x}", result.checksum()),
            "rounds": result.rounds(),
            "output_claims": result.output_claims(),
        },
        "warmup": warmup,
        "measurements": measurements,
    });
    if let Some(path) = &cli.output {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{report}");
    Ok(())
}

fn run_arm(
    arm: Arm,
    evaluator: &ProductRemainderCpuMetalEvalFixture,
    witness: &TraceBackend<'static, OwnedTrace>,
    oracle: &mut Option<ProductRemainderEvalResult>,
) -> EvalResult<Value> {
    match arm {
        Arm::Cpu => {
            let sample = evaluator.run_cpu(witness)?;
            require_exact(oracle, &sample.result)?;
            cpu_sample_record(&sample)
        }
        Arm::Metal => {
            let sample = evaluator.run_metal(witness)?;
            require_exact(oracle, &sample.result)?;
            metal_sample_record(&sample)
        }
        Arm::Both => Err(failure(
            "counterbalanced arm must be expanded before execution",
        )),
    }
}

fn require_exact(
    oracle: &mut Option<ProductRemainderEvalResult>,
    result: &ProductRemainderEvalResult,
) -> EvalResult<()> {
    match oracle {
        Some(expected) if expected != result => Err(failure(
            "Product remainder evaluator output changed between samples",
        )),
        Some(_) => Ok(()),
        None => {
            *oracle = Some(result.clone());
            Ok(())
        }
    }
}

fn cpu_sample_record(sample: &ProductRemainderCpuEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "charged_wall_ns": duration_ns(sample.member_wall)?,
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "uniskip_setup_wall_ns_excluded": duration_ns(sample.uniskip_setup_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "rounds": round_records(&sample.round_timings)?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn metal_sample_record(sample: &ProductRemainderMetalEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "charged_wall_ns": duration_ns(sample.charged_wall())?,
        "upstream_storage_wall_ns_excluded": duration_ns(sample.upstream_storage_wall)?,
        "sequence_setup_wall_ns": duration_ns(sample.sequence_setup_wall)?,
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "uniskip_setup_wall_ns_excluded": duration_ns(sample.uniskip_setup_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "rounds": round_records(&sample.round_timings)?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn round_records(
    rounds: &[jolt_kernels::metal::ProductRemainderRoundTiming],
) -> EvalResult<Vec<Value>> {
    rounds
        .iter()
        .map(|round| {
            Ok(json!({
                "round": round.round,
                "wall_ns": duration_ns(round.wall)?,
            }))
        })
        .collect()
}

fn build_fixture(bench: BenchName, scale: usize, target: usize) -> Fixture {
    let max_trace_length = 1usize << scale;
    let input = bench.input(target);
    let mut program = host::Program::new(&format!("{}-guest", bench.as_str()));
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(legacy_trace.len().next_power_of_two() <= max_trace_length);
    drop(legacy_trace);
    let elf_contents = program.get_elf_contents().expect("guest ELF contents");
    let memory_layout = io_device.memory_layout.clone();
    drop(program);

    let program_data =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy program preprocessing");
    let program_meta = program_data.meta();
    let shared_preprocessing: JoltSharedPreprocessing<AkitaPackedScheme> =
        JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
    let legacy_preprocessing = LegacyProverPreprocessing::new(shared_preprocessing);
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
    .expect("derive packed dimensions");
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
    let (_, verifier_setup) = AkitaScheme::setup(setup_params).expect("derive Akita setup");
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
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(jolt_program, program_preprocessing, trace_output),
    );
    Fixture {
        witness,
        trace_rows,
        padded_rows: config.trace_length,
        log_t,
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
                advice_tape: None,
                memory_config,
            },
        )
        .expect("modular trace")
}

fn target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
}

fn duration_ns(duration: Duration) -> EvalResult<u64> {
    Ok(u64::try_from(duration.as_nanos())?)
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::other(message.into()))
}
