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
    RamHammingBooleanityCpuEvalFixture, RamHammingBooleanityEvalResult,
    RamHammingBooleanityEvalSample,
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

const SEED: u64 = 0x6a09_e667_f3bc_c909;
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
    let evaluator = RamHammingBooleanityCpuEvalFixture::new(&fixture.witness, fixture.log_t, SEED)?;
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

    let shape = evaluator.shape();
    let result = oracle
        .as_ref()
        .ok_or_else(|| failure("evaluator produced no parity oracle"))?;
    let report = json!({
        "schema": "ram_hamming_booleanity_cpu_metal_v1",
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
            "census_wall_ns": duration_ns(shape.census_wall)?,
            "address_source_bytes": shape.address_source_bytes,
        },
        "shape": {
            "accesses": shape.accesses,
            "no_access_cycles": shape.no_access_cycles,
            "access_runs": shape.access_runs,
            "mixed_parents_by_child_width": shape.mixed_parents_by_child_width,
            "packed_access_bytes": shape.packed_access_bytes,
            "dense_h_field_bytes": shape.dense_h_field_bytes,
            "width_32_dense_a_bytes": shape.width_32_dense_a_bytes,
            "width_32_dense_b_bytes": shape.width_32_dense_b_bytes,
        },
        "parity": {
            "exact": true,
            "scope": "exact across every measured arm",
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
    evaluator: &RamHammingBooleanityCpuEvalFixture,
    witness: &TraceBackend<'static, OwnedTrace>,
    oracle: &mut Option<RamHammingBooleanityEvalResult>,
) -> EvalResult<Value> {
    let sample = match arm {
        Arm::Cpu => evaluator.run_cpu(witness)?,
        Arm::Metal => evaluator.run_metal()?,
        Arm::Both => {
            return Err(failure(
                "counterbalanced arm must be expanded before execution",
            ))
        }
    };
    require_exact(oracle, &sample.result)?;
    sample_record(&sample)
}

fn require_exact(
    oracle: &mut Option<RamHammingBooleanityEvalResult>,
    result: &RamHammingBooleanityEvalResult,
) -> EvalResult<()> {
    match oracle {
        Some(expected) if expected != result => Err(failure(
            "RAM Hamming booleanity evaluator output changed between samples",
        )),
        Some(_) => Ok(()),
        None => {
            *oracle = Some(result.clone());
            Ok(())
        }
    }
}

fn sample_record(sample: &RamHammingBooleanityEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "rounds": sample.round_timings.iter().map(|round| Ok(json!({
            "round": round.round,
            "wall_ns": duration_ns(round.wall)?,
        }))).collect::<EvalResult<Vec<_>>>()?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
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
    let committed_chunk_bits = config.one_hot_config.committed_chunk_bits();
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
        log_k_chunk: committed_chunk_bits,
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
        1usize << committed_chunk_bits,
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
