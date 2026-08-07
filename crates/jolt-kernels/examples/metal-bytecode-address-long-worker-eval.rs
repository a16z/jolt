#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{
    env,
    error::Error,
    hint::black_box,
    io,
    time::{Duration, Instant},
};

use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_kernels::metal::solinas::bytecode_read_raf::{
    build_long_worker_slice_topology_from_booleanity_rows, direct_pushforward_oracle,
    split_stage_eq_tables, BytecodeReadRafConfig, BytecodeReadRafCsrInvocation,
    BytecodeReadRafCsrTelemetry, BytecodeReadRafFusedProductPath,
    BytecodeReadRafLongWorkerSliceInvocation, BytecodeReadRafRowWords, BytecodeReadRafShape,
    BytecodeReadRafSliceRuntimeError, BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_DOMAIN,
    BYTECODE_ADDRESS_STAGES,
};
use jolt_kernels::metal::solinas::{BooleanityRow, MetalError, PipelineLimits, SolinasMetal};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;

const PROMOTE_ACTIVE_NS: u128 = 12_850_000;
const CSR_PROMOTE_ACTIVE_NS: u128 = 15_000_000;
const HARD_KILL_ACTIVE_NS: u128 = 30_124_000;

#[derive(Clone, Copy)]
enum EvalMode {
    Worker,
    Csr,
}

#[derive(Clone, Copy)]
struct Args {
    log_n: usize,
    samples: usize,
    support: usize,
    product_path: BytecodeReadRafFusedProductPath,
    mode: EvalMode,
}

struct EvalExecution {
    gpu_active: Duration,
    telemetry: Option<BytecodeReadRafCsrTelemetry>,
}

struct CsrOnlyResult {
    gpu_active: Vec<Duration>,
    median: Duration,
    telemetry: BytecodeReadRafCsrTelemetry,
}

enum Invocation {
    Worker(BytecodeReadRafLongWorkerSliceInvocation),
    Csr(BytecodeReadRafCsrInvocation),
}

fn main() -> EvalResult<()> {
    let args = parse_args()?;
    let rows_count = 1usize
        .checked_shl(args.log_n as u32)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "log-n is too large"))?;
    let shape = BytecodeReadRafShape::new(rows_count, BYTECODE_ADDRESS_DOMAIN)?;
    let config = BytecodeReadRafConfig::default();
    if args.support == 0
        || args.support > shape.addresses()
        || shape.inner_length() / args.support <= config.short_threshold
        || (matches!(args.mode, EvalMode::Worker) && args.support != 1)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "support must produce long runs; worker mode requires support=1",
        )
        .into());
    }

    let fixture_started = Instant::now();
    let rows = fixture_rows(shape, args.support)?;
    let topology = match args.mode {
        EvalMode::Worker => Some(build_long_worker_slice_topology_from_booleanity_rows(
            &rows,
            shape,
            config.short_threshold,
        )?),
        EvalMode::Csr => None,
    };
    let points = stage_points(args.log_n);
    let tables = split_stage_eq_tables(&points, shape)?;
    let expected = fixture_expected(shape, &tables.e_lo, &tables.e_hi, args.support);
    let fixture_wall = fixture_started.elapsed();

    let direct_oracle = if args.log_n <= 20 {
        let direct_started = Instant::now();
        let oracle_rows = rows
            .iter()
            .copied()
            .map(|row| BytecodeReadRafRowWords::from_words(row.words()))
            .collect::<Vec<_>>();
        let direct = direct_pushforward_oracle(&oracle_rows, &tables.e_lo, &tables.e_hi, shape)?;
        if direct != expected {
            return Err(io::Error::other("analytic fixture disagrees with direct oracle").into());
        }
        Some(direct_started.elapsed())
    } else {
        None
    };

    let compile_started = Instant::now();
    let context = SolinasMetal::for_akita_bytecode_read_raf_probe()?;
    let compile_wall = compile_started.elapsed();
    let compilation = context.compilation_stats().clone();
    let device = context.device_info();

    let upload_started = Instant::now();
    let resident_rows = context.prepare_booleanity_rows(&rows)?;
    let resident_rows_storage_id = resident_rows.allocation_identity();
    let row_upload_wall = upload_started.elapsed();
    drop(rows);

    let prepare_started = Instant::now();
    let invocation = match args.mode {
        EvalMode::Worker => Invocation::Worker(
            context.prepare_bytecode_read_raf_long_worker_slice(
                resident_rows,
                topology
                    .as_ref()
                    .ok_or_else(|| io::Error::other("worker mode is missing its host topology"))?,
                &tables,
                config,
                args.product_path,
            )?,
        ),
        EvalMode::Csr => Invocation::Csr(context.prepare_bytecode_read_raf_csr(
            resident_rows,
            &tables,
            config,
            args.product_path,
        )?),
    };
    let prepare_wall = prepare_started.elapsed();
    let identities = invocation.static_buffer_identities();
    let rows_reused = invocation.source_rows_storage_id() == resident_rows_storage_id;
    drop(topology);

    let csr_only = sample_csr_only(&invocation, shape, args.support, args.samples)?;

    let warmup = invocation.execute_timed()?;
    if let Some(telemetry) = warmup.telemetry {
        validate_fixture_telemetry(shape, args.support, telemetry)?;
    }
    if invocation.read_output()? != expected {
        return Err(io::Error::other("warmup output disagrees with the exact oracle").into());
    }

    let mut active = Vec::with_capacity(args.samples);
    let mut dispatch = Vec::with_capacity(args.samples);
    let mut readback = Vec::with_capacity(args.samples);
    let mut complete = Vec::with_capacity(args.samples);
    let mut buffers_stable = true;
    let mut retained_telemetry = None;
    for _ in 0..args.samples {
        let complete_started = Instant::now();
        let dispatch_started = Instant::now();
        let execution = invocation.execute_timed()?;
        let dispatch_wall = dispatch_started.elapsed();
        let readback_started = Instant::now();
        let output = invocation.read_output()?;
        let readback_wall = readback_started.elapsed();
        if black_box(output) != expected {
            return Err(io::Error::other("sample output disagrees with the exact oracle").into());
        }
        if let Some(telemetry) = execution.telemetry {
            validate_fixture_telemetry(shape, args.support, telemetry)?;
            if retained_telemetry.is_some_and(|retained| retained != telemetry) {
                return Err(io::Error::other("CSR telemetry changed between samples").into());
            }
            retained_telemetry = Some(telemetry);
        }
        buffers_stable &= invocation.static_buffer_identities() == identities;
        active.push(execution.gpu_active);
        dispatch.push(dispatch_wall);
        readback.push(readback_wall);
        complete.push(complete_started.elapsed());
    }

    let active_median = median(&mut active);
    let dispatch_median = median(&mut dispatch);
    let readback_median = median(&mut readback);
    let complete_median = median(&mut complete);
    let useful_products =
        4u128 * rows_count as u128 + BYTECODE_ADDRESS_STAGES as u128 * shape.outer_length() as u128;
    let useful_products_per_second = useful_products as f64 / active_median.as_secs_f64();
    let log26_gate_applies = args.log_n == 26;
    let promote_active_ns = match args.mode {
        EvalMode::Worker => PROMOTE_ACTIVE_NS,
        EvalMode::Csr => CSR_PROMOTE_ACTIVE_NS,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "kernel": "bytecode_read_raf_address_pushforward",
            "run_class": "screening_only",
            "mode": mode_name(args.mode),
            "machine": device.name,
            "log_n": args.log_n,
            "rows": rows_count,
            "outer_blocks": shape.outer_length(),
            "active_addresses_per_outer": args.support,
            "samples": args.samples,
            "product_path": product_path_name(args.product_path),
            "proof_boundary": {
                "stages": BYTECODE_ADDRESS_STAGES,
                "base_stages": BYTECODE_ADDRESS_BASE_STAGES,
                "addresses": BYTECODE_ADDRESS_DOMAIN,
                "host_fiat_shamir": true,
                "csr_included": matches!(args.mode, EvalMode::Csr),
                "host_rounds_included": false
            },
            "correctness": {
                "exact_output": true,
                "direct_oracle_checked": direct_oracle.is_some(),
                "direct_oracle_ms": direct_oracle.map(ms),
                "resident_rows_reused": rows_reused,
                "static_buffers_stable": buffers_stable
            },
            "setup_ms": {
                "fixture": ms(fixture_wall),
                "source_assembly": ms(compilation.source_assembly_wall),
                "library_compile": ms(compilation.library_compile_wall),
                "constructor_total": ms(compile_wall),
                "row_upload": ms(row_upload_wall),
                "invocation_prepare": ms(prepare_wall)
            },
            "median_ms": {
                "csr_only_gpu_active": csr_only.as_ref().map(|result| ms(result.median)),
                "gpu_active": ms(active_median),
                "dispatch_wait": ms(dispatch_median),
                "readback_validate": ms(readback_median),
                "complete_slice": ms(complete_median)
            },
            "samples_ms": {
                "csr_only_gpu_active": csr_only.as_ref().map(|result| result.gpu_active.iter().copied().map(ms).collect::<Vec<_>>()),
                "gpu_active": active.iter().copied().map(ms).collect::<Vec<_>>(),
                "dispatch_wait": dispatch.iter().copied().map(ms).collect::<Vec<_>>(),
                "readback_validate": readback.iter().copied().map(ms).collect::<Vec<_>>(),
                "complete_slice": complete.iter().copied().map(ms).collect::<Vec<_>>()
            },
            "throughput": {
                "useful_products": useful_products,
                "useful_products_per_second": useful_products_per_second,
                "gproducts_per_second": useful_products_per_second / 1e9
            },
            "resources": {
                "owned_bytes": invocation.owned_bytes(),
                "source_rows_bytes": rows_count * 40,
                "source_bytes": compilation.source_bytes,
                "csr_pipeline": invocation.csr_pipeline_limits().map(pipeline_json),
                "short_pipeline": invocation.short_pipeline_limits().map(pipeline_json),
                "long_pipeline": pipeline_json(invocation.long_pipeline_limits()),
                "finalize_pipeline": pipeline_json(invocation.finalize_pipeline_limits())
            },
            "csr_telemetry": retained_telemetry.or_else(|| csr_only.as_ref().map(|result| result.telemetry)).map(telemetry_json),
            "log26_screen": {
                "applies": log26_gate_applies,
                "promote_active_ns": promote_active_ns,
                "hard_kill_active_ns": HARD_KILL_ACTIVE_NS,
                "clears_promote": log26_gate_applies && active_median.as_nanos() <= promote_active_ns,
                "hits_hard_kill": log26_gate_applies && active_median.as_nanos() > HARD_KILL_ACTIVE_NS
            }
        }))?
    );
    Ok(())
}

impl Invocation {
    fn execute_csr_only_timed(
        &self,
    ) -> Result<Option<(Duration, BytecodeReadRafCsrTelemetry)>, BytecodeReadRafSliceRuntimeError>
    {
        match self {
            Self::Worker(_) => Ok(None),
            Self::Csr(invocation) => {
                let execution = invocation.execute_csr_only_timed()?;
                Ok(Some((execution.gpu_active, execution.telemetry)))
            }
        }
    }

    fn execute_timed(&self) -> Result<EvalExecution, BytecodeReadRafSliceRuntimeError> {
        match self {
            Self::Worker(invocation) => Ok(EvalExecution {
                gpu_active: invocation.execute_timed()?,
                telemetry: None,
            }),
            Self::Csr(invocation) => {
                let execution = invocation.execute_timed()?;
                Ok(EvalExecution {
                    gpu_active: execution.gpu_active,
                    telemetry: Some(execution.telemetry),
                })
            }
        }
    }

    fn read_output(&self) -> Result<Vec<AkitaField>, BytecodeReadRafSliceRuntimeError> {
        match self {
            Self::Worker(invocation) => invocation.read_output(),
            Self::Csr(invocation) => invocation.read_output(),
        }
    }

    fn source_rows_storage_id(&self) -> usize {
        match self {
            Self::Worker(invocation) => invocation.source_rows_storage_id(),
            Self::Csr(invocation) => invocation.source_rows_storage_id(),
        }
    }

    fn static_buffer_identities(&self) -> Vec<usize> {
        match self {
            Self::Worker(invocation) => invocation.static_buffer_identities().to_vec(),
            Self::Csr(invocation) => invocation.static_buffer_identities().to_vec(),
        }
    }

    fn owned_bytes(&self) -> usize {
        match self {
            Self::Worker(invocation) => invocation.owned_bytes(),
            Self::Csr(invocation) => invocation.owned_bytes(),
        }
    }

    fn csr_pipeline_limits(&self) -> Option<PipelineLimits> {
        match self {
            Self::Worker(_) => None,
            Self::Csr(invocation) => Some(invocation.csr_pipeline_limits()),
        }
    }

    fn short_pipeline_limits(&self) -> Option<PipelineLimits> {
        match self {
            Self::Worker(_) => None,
            Self::Csr(invocation) => Some(invocation.short_pipeline_limits()),
        }
    }

    fn long_pipeline_limits(&self) -> PipelineLimits {
        match self {
            Self::Worker(invocation) => invocation.long_pipeline_limits(),
            Self::Csr(invocation) => invocation.long_pipeline_limits(),
        }
    }

    fn finalize_pipeline_limits(&self) -> PipelineLimits {
        match self {
            Self::Worker(invocation) => invocation.finalize_pipeline_limits(),
            Self::Csr(invocation) => invocation.finalize_pipeline_limits(),
        }
    }
}

fn sample_csr_only(
    invocation: &Invocation,
    shape: BytecodeReadRafShape,
    support: usize,
    samples: usize,
) -> EvalResult<Option<CsrOnlyResult>> {
    let Some((_, warmup_telemetry)) = invocation.execute_csr_only_timed()? else {
        return Ok(None);
    };
    validate_fixture_telemetry(shape, support, warmup_telemetry)?;
    let mut gpu_active = Vec::with_capacity(samples);
    let mut retained_telemetry = None;
    for _ in 0..samples {
        let (duration, telemetry) = invocation
            .execute_csr_only_timed()?
            .ok_or_else(|| io::Error::other("CSR mode lost its topology invocation"))?;
        validate_fixture_telemetry(shape, support, telemetry)?;
        if retained_telemetry.is_some_and(|retained| retained != telemetry) {
            return Err(io::Error::other("CSR-only telemetry changed between samples").into());
        }
        retained_telemetry = Some(telemetry);
        gpu_active.push(duration);
    }
    let telemetry = retained_telemetry
        .ok_or_else(|| io::Error::other("CSR-only sampling produced no telemetry"))?;
    let median = median(&mut gpu_active);
    Ok(Some(CsrOnlyResult {
        gpu_active,
        median,
        telemetry,
    }))
}

fn validate_fixture_telemetry(
    shape: BytecodeReadRafShape,
    support: usize,
    telemetry: BytecodeReadRafCsrTelemetry,
) -> EvalResult<()> {
    let status = telemetry.status;
    let diagnostics = telemetry.diagnostics;
    let expected_runs = shape.outer_length() * support;
    let maximum_run = shape.inner_length().div_ceil(support);
    let mut expected_histogram = [0u32; 16];
    for residue in 0..support {
        let count = (shape.inner_length() + support - 1 - residue) / support;
        expected_histogram[count.ilog2() as usize] += shape.outer_length() as u32;
    }
    if status.short_runs != 0
        || status.long_runs as usize != expected_runs
        || status.invalid_rows != 0
        || status.completed_groups as usize != shape.outer_length()
        || status.occurrence_rows as usize != shape.rows()
        || diagnostics.short_occurrences != 0
        || diagnostics.long_occurrences as usize != shape.rows()
        || diagnostics.maximum_run as usize != maximum_run
        || diagnostics.run_histogram != expected_histogram
    {
        return Err(
            io::Error::other("CSR telemetry disagrees with the uniform-run fixture").into(),
        );
    }
    Ok(())
}

fn telemetry_json(telemetry: BytecodeReadRafCsrTelemetry) -> serde_json::Value {
    json!({
        "short_runs": telemetry.status.short_runs,
        "long_runs": telemetry.status.long_runs,
        "invalid_rows": telemetry.status.invalid_rows,
        "completed_groups": telemetry.status.completed_groups,
        "occurrence_rows": telemetry.status.occurrence_rows,
        "short_occurrences": telemetry.diagnostics.short_occurrences,
        "long_occurrences": telemetry.diagnostics.long_occurrences,
        "maximum_run": telemetry.diagnostics.maximum_run,
        "run_histogram": telemetry.diagnostics.run_histogram
    })
}

fn fixture_rows(
    shape: BytecodeReadRafShape,
    support: usize,
) -> Result<Vec<BooleanityRow>, MetalError> {
    (0..shape.rows())
        .into_par_iter()
        .map(|row| {
            let outer = row / shape.inner_length();
            BooleanityRow::new(
                row as u128,
                Some(((row % shape.inner_length()) % support) as u64),
                None,
                fixture_increment(outer),
            )
        })
        .collect()
}

fn pipeline_json(limits: PipelineLimits) -> serde_json::Value {
    json!({
        "thread_execution_width": limits.thread_execution_width,
        "max_total_threads_per_threadgroup": limits.max_total_threads_per_threadgroup,
        "static_threadgroup_memory_length": limits.static_threadgroup_memory_length
    })
}

fn fixture_increment(outer: usize) -> i128 {
    match outer % 4 {
        0 => u64::MAX as i128,
        1 => -(u64::MAX as i128),
        2 => (1i128 << 63) + outer as i128,
        _ => -((1i128 << 63) + outer as i128),
    }
}

fn stage_points(log_n: usize) -> Vec<Vec<AkitaField>> {
    (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..log_n)
                .map(|coordinate| {
                    AkitaField::from_u64(splitmix(
                        0x6279_7465_636f_6465 ^ ((stage as u64) << 32) ^ coordinate as u64,
                    ))
                })
                .collect()
        })
        .collect()
}

fn fixture_expected(
    shape: BytecodeReadRafShape,
    e_lo: &[Vec<AkitaField>],
    e_hi: &[Vec<AkitaField>],
    support: usize,
) -> Vec<AkitaField> {
    let mut output = vec![AkitaField::zero(); BYTECODE_ADDRESS_STAGES * shape.addresses()];
    let residue_sums = e_lo
        .iter()
        .map(|table| {
            let mut sums = vec![AkitaField::zero(); support];
            for (inner, &weight) in table.iter().enumerate() {
                sums[inner % support] += weight;
            }
            sums
        })
        .collect::<Vec<_>>();
    for (stage, table) in e_hi.iter().enumerate() {
        for (outer, &weight) in table.iter().enumerate() {
            for address in 0..support {
                let mut value = weight * residue_sums[stage][address];
                if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                    value *= AkitaField::from_i128(fixture_increment(outer));
                }
                output[stage * shape.addresses() + address] += value;
            }
        }
    }
    output
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn product_path_name(path: BytecodeReadRafFusedProductPath) -> &'static str {
    match path {
        BytecodeReadRafFusedProductPath::FullWidth => "full",
        BytecodeReadRafFusedProductPath::ExactU64 => "u64",
    }
}

fn mode_name(mode: EvalMode) -> &'static str {
    match mode {
        EvalMode::Worker => "worker",
        EvalMode::Csr => "csr",
    }
}

fn parse_args() -> EvalResult<Args> {
    let mut args = Args {
        log_n: 20,
        samples: 5,
        support: 1,
        product_path: BytecodeReadRafFusedProductPath::FullWidth,
        mode: EvalMode::Worker,
    };
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--log-n" => args.log_n = parse_value(&flag, values.next())?,
            "--samples" => args.samples = parse_value(&flag, values.next())?,
            "--support" => args.support = parse_value(&flag, values.next())?,
            "--mode" => {
                args.mode = match values.next().as_deref() {
                    Some("worker") => EvalMode::Worker,
                    Some("csr") => EvalMode::Csr,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "--mode must be `worker` or `csr`",
                        )
                        .into())
                    }
                }
            }
            "--path" => {
                args.product_path = match values.next().as_deref() {
                    Some("full") => BytecodeReadRafFusedProductPath::FullWidth,
                    Some("u64") => BytecodeReadRafFusedProductPath::ExactU64,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "--path must be `full` or `u64`",
                        )
                        .into())
                    }
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument `{flag}`"),
                )
                .into())
            }
        }
    }
    if args.samples == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "samples must be nonzero").into());
    }
    Ok(args)
}

fn parse_value<T: std::str::FromStr>(flag: &str, value: Option<String>) -> EvalResult<T>
where
    T::Err: Error + 'static,
{
    value
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing {flag}")))?
        .parse()
        .map_err(Into::into)
}
