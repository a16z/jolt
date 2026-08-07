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
    split_stage_eq_tables, BytecodeReadRafConfig, BytecodeReadRafFusedProductPath,
    BytecodeReadRafRowWords, BytecodeReadRafShape, BYTECODE_ADDRESS_BASE_STAGES,
    BYTECODE_ADDRESS_DOMAIN, BYTECODE_ADDRESS_STAGES,
};
use jolt_kernels::metal::solinas::{BooleanityRow, MetalError, SolinasMetal};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;

const PROMOTE_ACTIVE_NS: u128 = 12_850_000;
const HARD_KILL_ACTIVE_NS: u128 = 30_124_000;

#[derive(Clone, Copy)]
struct Args {
    log_n: usize,
    samples: usize,
    product_path: BytecodeReadRafFusedProductPath,
}

fn main() -> EvalResult<()> {
    let args = parse_args()?;
    let rows_count = 1usize
        .checked_shl(args.log_n as u32)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "log-n is too large"))?;
    let shape = BytecodeReadRafShape::new(rows_count, BYTECODE_ADDRESS_DOMAIN)?;
    let config = BytecodeReadRafConfig::default();

    let fixture_started = Instant::now();
    let rows = fixture_rows(shape)?;
    let topology = build_long_worker_slice_topology_from_booleanity_rows(
        &rows,
        shape,
        config.short_threshold,
    )?;
    let points = stage_points(args.log_n);
    let tables = split_stage_eq_tables(&points, shape)?;
    let expected = fixture_expected(shape, &tables.e_hi);
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
    let invocation = context.prepare_bytecode_read_raf_long_worker_slice(
        resident_rows,
        &topology,
        &tables,
        config,
        args.product_path,
    )?;
    let prepare_wall = prepare_started.elapsed();
    let identities = invocation.static_buffer_identities();
    let rows_reused = invocation.source_rows_storage_id() == resident_rows_storage_id;

    let _ = invocation.execute_timed()?;
    if invocation.read_output()? != expected {
        return Err(io::Error::other("warmup output disagrees with the exact oracle").into());
    }

    let mut active = Vec::with_capacity(args.samples);
    let mut dispatch = Vec::with_capacity(args.samples);
    let mut readback = Vec::with_capacity(args.samples);
    let mut complete = Vec::with_capacity(args.samples);
    let mut buffers_stable = true;
    for _ in 0..args.samples {
        let complete_started = Instant::now();
        let dispatch_started = Instant::now();
        let gpu_active = invocation.execute_timed()?;
        let dispatch_wall = dispatch_started.elapsed();
        let readback_started = Instant::now();
        let output = invocation.read_output()?;
        let readback_wall = readback_started.elapsed();
        if black_box(output) != expected {
            return Err(io::Error::other("sample output disagrees with the exact oracle").into());
        }
        buffers_stable &= invocation.static_buffer_identities() == identities;
        active.push(gpu_active);
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

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "kernel": "bytecode_read_raf_address_long_worker_slice",
            "run_class": "screening_only",
            "machine": device.name,
            "log_n": args.log_n,
            "rows": rows_count,
            "outer_blocks": shape.outer_length(),
            "samples": args.samples,
            "product_path": product_path_name(args.product_path),
            "proof_boundary": {
                "stages": BYTECODE_ADDRESS_STAGES,
                "base_stages": BYTECODE_ADDRESS_BASE_STAGES,
                "addresses": BYTECODE_ADDRESS_DOMAIN,
                "host_fiat_shamir": true,
                "csr_included": false,
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
                "gpu_active": ms(active_median),
                "dispatch_wait": ms(dispatch_median),
                "readback_validate": ms(readback_median),
                "complete_slice": ms(complete_median)
            },
            "samples_ms": {
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
                "long_pipeline": pipeline_json(invocation.long_pipeline_limits()),
                "finalize_pipeline": pipeline_json(invocation.finalize_pipeline_limits())
            },
            "log26_screen": {
                "applies": log26_gate_applies,
                "promote_active_ns": PROMOTE_ACTIVE_NS,
                "hard_kill_active_ns": HARD_KILL_ACTIVE_NS,
                "clears_promote": log26_gate_applies && active_median.as_nanos() <= PROMOTE_ACTIVE_NS,
                "hits_hard_kill": log26_gate_applies && active_median.as_nanos() > HARD_KILL_ACTIVE_NS
            }
        }))?
    );
    Ok(())
}

fn fixture_rows(shape: BytecodeReadRafShape) -> Result<Vec<BooleanityRow>, MetalError> {
    (0..shape.rows())
        .into_par_iter()
        .map(|row| {
            let outer = row / shape.inner_length();
            BooleanityRow::new(
                row as u128,
                Some((outer % shape.addresses()) as u64),
                None,
                fixture_increment(outer),
            )
        })
        .collect()
}

fn pipeline_json(limits: jolt_kernels::metal::solinas::PipelineLimits) -> serde_json::Value {
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

fn fixture_expected(shape: BytecodeReadRafShape, e_hi: &[Vec<AkitaField>]) -> Vec<AkitaField> {
    let mut output = vec![AkitaField::zero(); BYTECODE_ADDRESS_STAGES * shape.addresses()];
    for (stage, table) in e_hi.iter().enumerate() {
        for (outer, &weight) in table.iter().enumerate() {
            let address = outer % shape.addresses();
            let mut value = weight;
            if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                value *= AkitaField::from_i128(fixture_increment(outer));
            }
            output[stage * shape.addresses() + address] += value;
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

fn parse_args() -> EvalResult<Args> {
    let mut args = Args {
        log_n: 20,
        samples: 5,
        product_path: BytecodeReadRafFusedProductPath::FullWidth,
    };
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--log-n" => args.log_n = parse_value(&flag, values.next())?,
            "--samples" => args.samples = parse_value(&flag, values.next())?,
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
