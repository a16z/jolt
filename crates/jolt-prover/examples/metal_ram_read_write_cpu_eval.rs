#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "the evaluator exits on invalid evidence and emits one JSON record"
)]

use std::{error::Error, path::PathBuf, time::Duration};

use clap::{Parser, ValueEnum};
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_kernels::metal::{
    RamReadWriteCpuEvalSample, RamReadWriteCpuMetalEvalFixture, RamReadWriteEvalResult,
    RamReadWriteMetalEvalSample,
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

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "optimized_cpu",
            Self::Metal => "metal_base",
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

    #[clap(long)]
    dispatch_timing: bool,

    #[clap(long)]
    hot_threshold: Option<usize>,
}

struct Fixture {
    witness: TraceBackend<OwnedTrace>,
    trace_rows: usize,
    padded_rows: usize,
    log_t: usize,
    log_k: usize,
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
    let evaluator =
        RamReadWriteCpuMetalEvalFixture::new(&fixture.witness, fixture.log_t, fixture.log_k, SEED)?;
    let mut oracle = None;
    if (cli.dispatch_timing || cli.hot_threshold.is_some()) && !matches!(cli.arm, Arm::Metal) {
        return Err(failure(
            "--dispatch-timing and --hot-threshold require --arm metal",
        ));
    }
    let warmup = run_arm(
        cli.arm,
        &evaluator,
        &fixture.witness,
        &mut oracle,
        cli.dispatch_timing,
        cli.hot_threshold,
    )?;
    let mut samples = Vec::with_capacity(cli.samples);
    for sample_index in 0..cli.samples {
        let mut sample = run_arm(
            cli.arm,
            &evaluator,
            &fixture.witness,
            &mut oracle,
            cli.dispatch_timing,
            cli.hot_threshold,
        )?;
        sample["sample"] = json!(sample_index);
        samples.push(sample);
    }

    let device = evaluator.device_info();
    let result = oracle
        .as_ref()
        .ok_or_else(|| failure("evaluator produced no parity oracle"))?;
    let report = json!({
        "schema": "ram_read_write_cpu_metal_v1",
        "schema_version": 1,
        "workload": cli.name.as_str(),
        "scale": cli.scale,
        "target_trace_size": target,
        "execution": "isolated",
        "arm": cli.arm.label(),
        "trace_rows": fixture.trace_rows,
        "padded_rows": fixture.padded_rows,
        "log_t": evaluator.log_t(),
        "log_k": evaluator.log_k(),
        "cycles": evaluator.cycles(),
        "addresses": evaluator.addresses(),
        "access_count": evaluator.access_count(),
        "samples": cli.samples,
        "excluded_warmups": 1,
        "dispatch_timing": cli.dispatch_timing,
        "hot_threshold": cli.hot_threshold,
        "rayon_threads": rayon::current_num_threads(),
        "fixture": {
            "wall_ns": duration_ns(evaluator.fixture_wall())?,
            "source_bytes": evaluator.source_bytes(),
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
    evaluator: &RamReadWriteCpuMetalEvalFixture,
    witness: &TraceBackend<OwnedTrace>,
    oracle: &mut Option<RamReadWriteEvalResult>,
    dispatch_timing: bool,
    hot_threshold: Option<usize>,
) -> EvalResult<Value> {
    match arm {
        Arm::Cpu => {
            let sample = evaluator.run_cpu(witness)?;
            require_exact(oracle, &sample.result)?;
            cpu_record(&sample)
        }
        Arm::Metal => {
            let sample = if let Some(threshold) = hot_threshold {
                evaluator.run_metal_with_hot_threshold(witness, threshold, dispatch_timing)?
            } else if dispatch_timing {
                evaluator.run_metal_profiled(witness)?
            } else {
                evaluator.run_metal(witness)?
            };
            require_exact(oracle, &sample.result)?;
            metal_record(&sample)
        }
    }
}

fn require_exact(
    oracle: &mut Option<RamReadWriteEvalResult>,
    result: &RamReadWriteEvalResult,
) -> EvalResult<()> {
    match oracle {
        Some(expected) if expected != result => Err(failure(
            "RAM read/write evaluator output changed between samples",
        )),
        Some(_) => Ok(()),
        None => {
            *oracle = Some(result.clone());
            Ok(())
        }
    }
}

fn cpu_record(sample: &RamReadWriteCpuEvalSample) -> EvalResult<Value> {
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn metal_record(sample: &RamReadWriteMetalEvalSample) -> EvalResult<Value> {
    let preparation = sample.preparation;
    let buckets = sample.buckets;
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "prepare_wall_ns": duration_ns(sample.prepare_wall)?,
        "final_memory_wall_ns": duration_ns(sample.final_memory_wall)?,
        "rounds_wall_ns": duration_ns(sample.rounds_wall)?,
        "finish_wall_ns": duration_ns(sample.finish_wall)?,
        "output_wall_ns": duration_ns(sample.output_wall)?,
        "cycle_sequence_wall_ns": duration_ns(sample.cycle_sequence_wall)?,
        "cycle_sequence_gpu_active_ns": duration_ns(sample.cycle_sequence_gpu_active)?,
        "dispatch_ns": sample.dispatch.map(dispatch_record).transpose()?,
        "preparation_ns": {
            "bucket_plan": duration_ns(preparation.bucket_plan)?,
            "allocation": duration_ns(preparation.allocation)?,
            "initialization_and_scatter": duration_ns(preparation.initialization_and_scatter)?,
            "pipeline_setup": duration_ns(preparation.pipeline_setup)?,
            "sequence_total": duration_ns(preparation.sequence_total)?,
        },
        "buckets": {
            "accesses": buckets.accesses,
            "active_addresses": buckets.active_addresses,
            "maximum_segment": buckets.maximum_segment,
            "p50_segment": buckets.p50_segment,
            "p95_segment": buckets.p95_segment,
            "p99_segment": buckets.p99_segment,
            "hot_addresses": buckets.hot_addresses,
            "hot_message_chunks": buckets.hot_message_chunks,
            "hot_state_entries": buckets.hot_state_entries,
            "hot_compaction_threads": buckets.hot_compaction_threads,
            "hot_compaction_threadgroup_bytes": buckets.hot_compaction_threadgroup_bytes,
            "hot_auxiliary_bytes": buckets.hot_auxiliary_bytes,
            "address_bytes": buckets.address_bytes,
            "cycle_bytes": buckets.cycle_bytes,
            "resident_bytes": buckets.resident_bytes,
        },
        "rounds": sample.round_timings.iter().map(|round| Ok(json!({
            "round": round.round,
            "wall_ns": duration_ns(round.wall)?,
            "sequence_wall_ns": duration_ns(round.sequence_wall)?,
            "gpu_active_ns": duration_ns(round.gpu_active)?,
            "dispatch_ns": round.dispatch.map(dispatch_record).transpose()?,
        }))).collect::<EvalResult<Vec<_>>>()?,
        "checksum_fnv1a64": format!("{:016x}", sample.result.checksum()),
    }))
}

fn dispatch_record(timing: jolt_kernels::metal::RamReadWriteDispatchSnapshot) -> EvalResult<Value> {
    Ok(json!({
        "address": duration_ns(timing.address)?,
        "hot_count": duration_ns(timing.hot_count)?,
        "hot_prefix": duration_ns(timing.hot_prefix)?,
        "hot_scatter": duration_ns(timing.hot_scatter)?,
        "hot_message": duration_ns(timing.hot_message)?,
        "cycle": duration_ns(timing.cycle)?,
        "reductions": duration_ns(timing.reductions)?,
    }))
}

fn build_fixture(bench: BenchName, scale: usize, target: usize) -> Fixture {
    let built =
        metal_eval_support::build_witness(bench.as_str(), &bench.input(target), 1usize << scale);
    Fixture {
        witness: built.witness,
        trace_rows: built.trace_rows,
        padded_rows: built.padded_rows,
        log_t: built.log_t,
        log_k: built.log_k,
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
