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
    RegistersReadWriteCpuEvalSample, RegistersReadWriteCpuMetalEvalFixture,
    RegistersReadWriteEvalResult,
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
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetalSource {
    Packed,
    Stage1,
    Stage1Primed,
}

impl MetalSource {
    const fn label(self) -> &'static str {
        match self {
            Self::Packed => "packed_rows_v1",
            Self::Stage1 => "stage1_simd_segmented_predecessor_v1",
            Self::Stage1Primed => "stage1_resident_primed_v1",
        }
    }
}

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "optimized_cpu",
            Self::Metal => "metal",
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

    #[clap(long, value_enum, default_value = "packed")]
    metal_source: MetalSource,

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
    let evaluator = match cli.metal_source {
        MetalSource::Packed => {
            RegistersReadWriteCpuMetalEvalFixture::new(&fixture.witness, fixture.log_t, SEED)?
        }
        MetalSource::Stage1 | MetalSource::Stage1Primed => {
            RegistersReadWriteCpuMetalEvalFixture::new_stage1(
                &fixture.witness,
                fixture.log_t,
                SEED,
            )?
        }
    };
    let mut oracle = None;
    let warmup = run_arm(cli.arm, cli.metal_source, &evaluator, &mut oracle)?;
    let mut samples = Vec::with_capacity(cli.samples);
    for sample_index in 0..cli.samples {
        let mut sample = run_arm(cli.arm, cli.metal_source, &evaluator, &mut oracle)?;
        sample["sample"] = json!(sample_index);
        samples.push(sample);
    }

    let device = evaluator.device_info();
    let shape = evaluator.shape();
    let result = oracle
        .as_ref()
        .ok_or_else(|| failure("evaluator produced no parity oracle"))?;
    let report = json!({
        "schema": "registers_read_write_cpu_metal_v1",
        "schema_version": 1,
        "workload": cli.name.as_str(),
        "scale": cli.scale,
        "target_trace_size": target,
        "execution": "isolated",
        "arm": cli.arm.label(),
        "metal_source": cli.metal_source.label(),
        "metal_route": match cli.arm {
            Arm::Metal if evaluator.log_t() >= 25 => Some(match cli.metal_source {
                MetalSource::Packed => "device_packed_cycle_sequence_and_operand_claims_then_cpu_address",
                MetalSource::Stage1 => "device_stage1_cycle_sequence_and_operand_claims_then_cpu_address",
                MetalSource::Stage1Primed => "device_primed_stage1_cycle_sequence_and_operand_claims_then_cpu_address",
            }),
            Arm::Metal if evaluator.log_t() >= 16 => Some(match cli.metal_source {
                MetalSource::Packed => "device_packed_cycle_sequence_then_cpu_address",
                MetalSource::Stage1 => "device_stage1_cycle_sequence_then_cpu_address",
                MetalSource::Stage1Primed => "device_primed_stage1_cycle_sequence_then_cpu_address",
            }),
            Arm::Metal => Some("optimized_cpu_host"),
            Arm::Cpu => None,
        },
        "trace_rows": fixture.trace_rows,
        "padded_rows": fixture.padded_rows,
        "log_t": evaluator.log_t(),
        "log_k": evaluator.log_k(),
        "cycles": evaluator.cycles(),
        "physical_rows": evaluator.physical_rows(),
        "samples": cli.samples,
        "excluded_warmups": 1,
        "rayon_threads": rayon::current_num_threads(),
        "fixture": {
            "wall_ns": duration_ns(evaluator.fixture_wall())?,
            "source_bytes": evaluator.source_bytes(),
        },
        "shape": {
            "rs1_reads": shape.rs1_reads,
            "rs2_reads": shape.rs2_reads,
            "writes": shape.writes,
            "rs1_rs2_same_register": shape.rs1_rs2_same_register,
            "rd_same_as_read_register": shape.rd_same_as_read_register,
            "rd_distinct_from_reads": shape.rd_distinct_from_reads,
            "rd_distinct_signed_39_overflow": shape.rd_distinct_signed_39_overflow,
            "active_registers": shape.active_registers,
            "entries_by_cycle_level": shape.entries_by_cycle_level,
            "read_entries_by_cycle_level": shape.read_entries_by_cycle_level,
            "write_entries_by_cycle_level": shape.write_entries_by_cycle_level,
            "value_change_entries_by_cycle_level": shape.value_change_entries_by_cycle_level,
            "packed_source_row_bytes": shape.packed_source_row_bytes,
            "indexed_entry_bytes": shape.indexed_entry_bytes,
            "direct_entry_bytes": shape.direct_entry_bytes,
        },
        "device": {
            "name": device.name,
            "max_buffer_length": device.max_buffer_length,
            "max_threadgroup_memory_length": device.max_threadgroup_memory_length,
            "recommended_max_working_set_size": device.recommended_max_working_set_size,
            "offset": device.offset,
        },
        "parity": {
            "exact": true,
            "scope": "within_arm; compare checksums across isolated arms",
            "checksum_fnv1a64": format!("{:016x}", result.checksum()),
            "rounds": result.rounds(),
            "output_claims": result.output_claims(),
        },
        "warmup": warmup,
        "measurements": samples,
    });
    if let Some(path) = &cli.output {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{report}");
    Ok(())
}

fn run_arm(
    arm: Arm,
    metal_source: MetalSource,
    evaluator: &RegistersReadWriteCpuMetalEvalFixture,
    oracle: &mut Option<RegistersReadWriteEvalResult>,
) -> EvalResult<Value> {
    let sample = match arm {
        Arm::Cpu => evaluator.run_cpu()?,
        Arm::Metal if matches!(metal_source, MetalSource::Stage1Primed) => {
            evaluator.run_metal_primed()?
        }
        Arm::Metal => evaluator.run_metal()?,
    };
    require_exact(oracle, &sample.result)?;
    sample_record(&sample)
}

fn require_exact(
    oracle: &mut Option<RegistersReadWriteEvalResult>,
    result: &RegistersReadWriteEvalResult,
) -> EvalResult<()> {
    match oracle {
        Some(expected) if expected != result => Err(failure(
            "registers read/write evaluator output changed between samples",
        )),
        Some(_) => Ok(()),
        None => {
            *oracle = Some(result.clone());
            Ok(())
        }
    }
}

fn sample_record(sample: &RegistersReadWriteCpuEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "source_to_state_wall_ns": duration_ns(sample.source_to_state_wall)?,
        "kernel_setup_wall_ns": duration_ns(sample.kernel_setup_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "rounds": sample.round_timings.iter().map(|round| Ok(json!({
            "round": round.round,
            "wall_ns": duration_ns(round.wall)?,
        }))).collect::<EvalResult<Vec<_>>>()?,
        "metal_first_message": match (
            sample.metal_first_message_prepare_wall,
            sample.metal_first_message_wall,
            sample.metal_first_message_gpu_active,
        ) {
            (Some(prepare), Some(wall), Some(gpu_active)) => Some(json!({
                "prepare_wall_ns": duration_ns(prepare)?,
                "wall_ns": duration_ns(wall)?,
                "gpu_active_ns": duration_ns(gpu_active)?,
                "charged_wall_ns": duration_ns(prepare + wall)?,
                "threads": sample.metal_first_message_threads,
                "execution_width": sample.metal_first_message_execution_width,
                "static_threadgroup_bytes": sample.metal_first_message_static_threadgroup_bytes,
                "resident_bytes": sample.metal_first_message_resident_bytes,
                "source_zero_copy": sample.metal_first_message_source_zero_copy,
            })),
            _ => None,
        },
        "metal_cycle_sequence": sample.metal_cycle_sequence_prepare_wall.map(|prepare| -> EvalResult<Value> {
            Ok(json!({
                "prepare_wall_ns": duration_ns(prepare)?,
                "rounds": sample.metal_cycle_timings.iter().map(|round| Ok(json!({
                    "round": round.round,
                    "allocation_ns": duration_ns(round.allocation)?,
                    "wall_ns": duration_ns(round.wall)?,
                    "gpu_active_ns": duration_ns(round.gpu_active)?,
                    "prefill_gpu_active_ns": duration_ns(round.prefill_gpu_active)?,
                    "live_entries": round.live_entries,
                    "resident_bytes": round.resident_bytes,
                    "peak_transition_bytes": round.peak_transition_bytes,
                }))).collect::<EvalResult<Vec<_>>>()?,
                "finish": match (
                    sample.metal_cycle_finish_allocation,
                    sample.metal_cycle_finish_wall,
                    sample.metal_cycle_finish_gpu_active,
                    sample.metal_cycle_finish_resident_bytes,
                ) {
                    (Some(allocation), Some(wall), Some(gpu_active), Some(resident_bytes)) => Some(json!({
                        "allocation_ns": duration_ns(allocation)?,
                        "wall_ns": duration_ns(wall)?,
                        "gpu_active_ns": duration_ns(gpu_active)?,
                        "resident_bytes": resident_bytes,
                    })),
                    _ => None,
                },
                "peak_transition_bytes": sample.metal_cycle_peak_transition_bytes,
            }))
        }).transpose()?,
        "metal_operand_claims": match (
            sample.metal_operand_claims_prepare_wall,
            sample.metal_operand_claims_wall,
            sample.metal_operand_claims_gpu_active,
        ) {
            (Some(prepare), Some(wall), Some(gpu_active)) => Some(json!({
                "prepare_wall_ns": duration_ns(prepare)?,
                "wall_ns": duration_ns(wall)?,
                "gpu_active_ns": duration_ns(gpu_active)?,
                "charged_wall_ns": duration_ns(prepare + wall)?,
            })),
            _ => None,
        },
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
