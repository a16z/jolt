#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "the evaluator exits on invalid evidence and emits one JSON record"
)]

use std::{error::Error, time::Duration};

use clap::{Parser, ValueEnum};
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_kernels::metal::solinas::OuterRemainderStorageInitialization;
use jolt_kernels::metal::{
    OuterRemainderCpuEvalSample, OuterRemainderCpuMetalEvalFixture, OuterRemainderEvalResult,
    OuterRemainderMetalEvalSample, OuterRemainderPipelineSnapshot,
};
use jolt_program::execution::OwnedTrace;
use jolt_witness::TraceBackend;
use serde_json::{json, Value};

#[path = "metal_eval_support/mod.rs"]
mod metal_eval_support;

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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StorageInitialization {
    Full,
    Lazy,
}

impl StorageInitialization {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Lazy => "lazy",
        }
    }
}

impl From<StorageInitialization> for OuterRemainderStorageInitialization {
    fn from(value: StorageInitialization) -> Self {
        match value {
            StorageInitialization::Full => Self::Full,
            StorageInitialization::Lazy => Self::Lazy,
        }
    }
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

#[derive(Parser, Debug)]
struct Cli {
    #[clap(long, value_enum, default_value = "btreemap")]
    name: BenchName,

    #[clap(long, default_value_t = 20)]
    scale: usize,

    #[clap(long)]
    target_trace_size: Option<usize>,

    #[clap(long, default_value_t = 3)]
    pairs: usize,

    #[clap(long, value_enum, default_value = "full")]
    storage_initialization: StorageInitialization,

    #[clap(long)]
    borrow_product_state_b: bool,

    #[clap(long, value_enum)]
    arm: Option<Arm>,
}

struct Fixture {
    witness: TraceBackend<OwnedTrace>,
    trace_rows: usize,
    padded_rows: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Arm {
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "metal-base")]
    MetalBase,
    #[value(name = "metal-carrier")]
    MetalCarrier,
}

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "optimized_cpu",
            Self::MetalBase => "metal_carrier_off",
            Self::MetalCarrier => "metal_carrier_on",
        }
    }
}

fn main() -> EvalResult<()> {
    let cli = Cli::parse();
    if !(16..=28).contains(&cli.scale) || !(1..=7).contains(&cli.pairs) {
        return Err(failure("scale must be 16..=28 and pairs must be 1..=7"));
    }
    if rayon::current_num_threads() != REQUIRED_RAYON_THREADS {
        return Err(failure("RAYON_NUM_THREADS must be 16"));
    }
    let target = cli
        .target_trace_size
        .unwrap_or((SAFETY_MARGIN * (1usize << cli.scale) as f64) as usize);
    let fixture = build_fixture(cli.name, cli.scale, target);
    let evaluator = OuterRemainderCpuMetalEvalFixture::new(
        &fixture.witness,
        cli.scale,
        SEED,
        cli.borrow_product_state_b,
    )?;
    let mut oracle = None;

    let storage_initialization = cli.storage_initialization.into();
    let (warmup, samples) = if let Some(arm) = cli.arm {
        let warmup = run_arm(
            arm,
            &evaluator,
            &fixture.witness,
            storage_initialization,
            &mut oracle,
        )?;
        let mut samples = Vec::with_capacity(cli.pairs);
        for sample_index in 0..cli.pairs {
            let mut sample = run_arm(
                arm,
                &evaluator,
                &fixture.witness,
                storage_initialization,
                &mut oracle,
            )?;
            sample["sample"] = json!(sample_index);
            samples.push(sample);
        }
        (warmup, samples)
    } else {
        let warmup_order = [Arm::Cpu, Arm::MetalBase, Arm::MetalCarrier];
        let warmup = run_triplet(
            warmup_order,
            &evaluator,
            &fixture.witness,
            storage_initialization,
            &mut oracle,
        )?;
        let mut samples = Vec::with_capacity(cli.pairs);
        for pair in 0..cli.pairs {
            let order = match pair % 3 {
                0 => [Arm::Cpu, Arm::MetalBase, Arm::MetalCarrier],
                1 => [Arm::MetalBase, Arm::MetalCarrier, Arm::Cpu],
                _ => [Arm::MetalCarrier, Arm::Cpu, Arm::MetalBase],
            };
            let mut sample = run_triplet(
                order,
                &evaluator,
                &fixture.witness,
                storage_initialization,
                &mut oracle,
            )?;
            sample["pair"] = json!(pair);
            samples.push(sample);
        }
        (warmup, samples)
    };

    let device = evaluator.device_info();
    let checksum = oracle
        .as_ref()
        .map(OuterRemainderEvalResult::checksum)
        .ok_or_else(|| failure("evaluator produced no parity oracle"))?;
    println!(
        "{}",
        json!({
            "schema": "outer_remainder_cpu_metal_v2",
            "schema_version": 2,
            "workload": cli.name.as_str(),
            "scale": cli.scale,
            "target_trace_size": target,
            "storage_initialization": cli.storage_initialization.as_str(),
            "borrow_product_state_b": cli.borrow_product_state_b,
            "execution": if cli.arm.is_some() { "isolated" } else { "mixed" },
            "arm": cli.arm.map(Arm::label),
            "trace_rows": fixture.trace_rows,
            "padded_rows": fixture.padded_rows,
            "pairs": cli.pairs,
            "excluded_warmup_triplets": 1,
            "rayon_threads": rayon::current_num_threads(),
            "fixture": {
                "wall_ns": duration_ns(evaluator.fixture_wall())?,
                "resident_row_bytes": evaluator.resident_row_bytes(),
                "producer_state_b_bytes": evaluator.producer_state_b_bytes(),
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
                "scope": if cli.arm.is_some() { "within_arm" } else { "cross_arm" },
                "checksum_fnv1a64": format!("{checksum:016x}"),
                "rounds": oracle.as_ref().map_or(0, OuterRemainderEvalResult::rounds),
                "output_claims": oracle.as_ref().map_or(0, OuterRemainderEvalResult::output_claims),
            },
            "warmup": warmup,
            "samples": samples,
        })
    );
    Ok(())
}

fn run_arm(
    arm: Arm,
    evaluator: &OuterRemainderCpuMetalEvalFixture,
    witness: &TraceBackend<OwnedTrace>,
    storage_initialization: OuterRemainderStorageInitialization,
    oracle: &mut Option<OuterRemainderEvalResult>,
) -> EvalResult<Value> {
    match arm {
        Arm::Cpu => {
            let sample = evaluator.run_cpu(witness)?;
            require_exact(oracle, &sample.result)?;
            cpu_record(&sample)
        }
        Arm::MetalBase => {
            let sample = evaluator.run_metal(false, storage_initialization)?;
            require_exact(oracle, &sample.result)?;
            metal_record(&sample)
        }
        Arm::MetalCarrier => {
            let sample = evaluator.run_metal(true, storage_initialization)?;
            require_exact(oracle, &sample.result)?;
            metal_record(&sample)
        }
    }
}

fn run_triplet(
    order: [Arm; 3],
    evaluator: &OuterRemainderCpuMetalEvalFixture,
    witness: &TraceBackend<OwnedTrace>,
    storage_initialization: OuterRemainderStorageInitialization,
    oracle: &mut Option<OuterRemainderEvalResult>,
) -> EvalResult<Value> {
    let mut cpu = None;
    let mut metal_base = None;
    let mut metal_carrier = None;
    for arm in order {
        match arm {
            Arm::Cpu => {
                let sample = evaluator.run_cpu(witness)?;
                require_exact(oracle, &sample.result)?;
                cpu = Some(cpu_record(&sample)?);
            }
            Arm::MetalBase => {
                let sample = evaluator.run_metal(false, storage_initialization)?;
                require_exact(oracle, &sample.result)?;
                metal_base = Some(metal_record(&sample)?);
            }
            Arm::MetalCarrier => {
                let sample = evaluator.run_metal(true, storage_initialization)?;
                require_exact(oracle, &sample.result)?;
                metal_carrier = Some(metal_record(&sample)?);
            }
        }
    }
    Ok(json!({
        "order": order.map(Arm::label),
        "optimized_cpu": cpu.ok_or_else(|| failure("triplet missed CPU arm"))?,
        "metal_carrier_off": metal_base.ok_or_else(|| failure("triplet missed base Metal arm"))?,
        "metal_carrier_on": metal_carrier.ok_or_else(|| failure("triplet missed carrier Metal arm"))?,
    }))
}

fn require_exact(
    oracle: &mut Option<OuterRemainderEvalResult>,
    result: &OuterRemainderEvalResult,
) -> EvalResult<()> {
    match oracle {
        Some(expected) if expected != result => Err(failure("CPU/Metal Outer outputs differ")),
        Some(_) => Ok(()),
        None => {
            *oracle = Some(result.clone());
            Ok(())
        }
    }
}

fn cpu_record(sample: &OuterRemainderCpuEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn metal_record(sample: &OuterRemainderMetalEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "registers_claim_carrier": sample.registers_claim_carrier,
        "borrowed_state_b": sample.borrowed_state_b,
        "storage_initialization": sample.storage_initialization.as_str(),
        "charged_wall_ns": duration_ns(sample.charged_wall())?,
        "setup_wall_ns": duration_ns(sample.setup_wall)?,
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "materialize_wall_ns": duration_ns(sample.materialize_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "member_gpu_active_ns": duration_ns(sample.member_gpu_active)?,
        "phase_gpu_active_ns": {
            "materialize": duration_ns(sample.phase_gpu_active.materialize)?,
            "first_bind": duration_ns(sample.phase_gpu_active.first_bind)?,
            "dense_rounds": duration_ns(sample.phase_gpu_active.dense_rounds)?,
            "openings": duration_ns(sample.phase_gpu_active.openings)?,
        },
        "storage": {
            "owned_bytes": sample.storage_owned_bytes,
            "initialized_bytes": sample.initialized_bytes,
            "initialization_device_buffers": sample.initialization_device_buffers,
            "initialization_gpu_active_ns": duration_ns(sample.initialization_gpu_active)?,
        },
        "pipelines": pipeline_record(&sample.pipelines),
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn pipeline_record(pipelines: &OuterRemainderPipelineSnapshot) -> Value {
    let limit = |value: jolt_kernels::metal::solinas::PipelineLimits| {
        json!({
            "thread_execution_width": value.thread_execution_width,
            "max_total_threads_per_threadgroup": value.max_total_threads_per_threadgroup,
            "static_threadgroup_memory_length": value.static_threadgroup_memory_length,
        })
    };
    json!({
        "materialize": limit(pipelines.materialize),
        "stream_bind": limit(pipelines.stream_bind),
        "transition": limit(pipelines.transition),
        "opening": limit(pipelines.opening),
        "reduction": limit(pipelines.reduction),
        "registers_claim_build": pipelines.registers_claim_build.map(limit),
        "registers_claim_reduce": pipelines.registers_claim_reduce.map(limit),
        "registers_claim_dot": pipelines.registers_claim_dot.map(limit),
        "threads": {
            "materialize": pipelines.threads.materialize,
            "stream_bind": pipelines.threads.stream_bind,
            "transition": pipelines.threads.transition,
            "opening": pipelines.threads.opening,
            "reduction": pipelines.threads.reduction,
            "registers_claim_build": pipelines.threads.registers_claim_build,
            "registers_claim_reduce": pipelines.threads.registers_claim_reduce,
            "registers_claim_dot": pipelines.threads.registers_claim_dot,
        },
        "opening_dynamic_threadgroup_bytes": pipelines.opening_dynamic_threadgroup_bytes,
    })
}

fn build_fixture(bench: BenchName, scale: usize, target: usize) -> Fixture {
    let built =
        metal_eval_support::build_witness(bench.as_str(), &bench.input(target), 1usize << scale);
    Fixture {
        witness: built.witness,
        trace_rows: built.trace_rows,
        padded_rows: built.padded_rows,
    }
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
