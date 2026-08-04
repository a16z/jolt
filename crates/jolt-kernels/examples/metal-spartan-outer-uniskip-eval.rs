#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{
    env,
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    evaluate_spartan_outer_uniskip_cpu, SolinasMetal, SpartanOuterUniskipConfig,
    SpartanOuterUniskipRow, SpartanOuterUniskipRows, SPARTAN_OUTER_EXTENDED_NODES,
};
use jolt_poly::lagrange::{centered_lagrange_evals, interpolate_to_coeffs, poly_mul};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{LabeledRoundPoly, RoundMessage};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;
type EvalTranscript = Blake2bTranscript<AkitaField>;

const DOMAIN: usize = 10;
const EXTENDED_SIZE: usize = 19;
const DOMAIN_START: i64 = -4;
const EXTENDED_START: i64 = -9;

#[derive(Clone, Debug, Eq, PartialEq)]
struct HostResult {
    extended: [AkitaField; SPARTAN_OUTER_EXTENDED_NODES],
    coefficients: Vec<AkitaField>,
    challenge: AkitaField,
    output_claim: AkitaField,
    transcript_state: [u8; 32],
}

struct GpuSample {
    result: HostResult,
    complete: Duration,
    prepare: Duration,
    dispatch_wall: Duration,
    gpu_active: Duration,
    readback_and_host: Duration,
    no_execute_allocations: bool,
}

struct StandaloneSample {
    complete: Duration,
    row_copy: Duration,
    gpu_active: Duration,
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn synthetic_row(index: usize, seed: u64) -> SpartanOuterUniskipRow {
    let mut words = [0u64; 20];
    for (word, value) in words[..19].iter_mut().enumerate() {
        *value = splitmix(seed ^ index as u64 ^ (word as u64).wrapping_mul(0x1000_0001));
    }
    words[2] &= (1 << 24) - 1;
    words[4] &= (1 << 24) - 1;
    words[8] = 0;
    words[15] &= (1 << 24) - 1;
    let selector = splitmix(seed ^ index as u64 ^ 0xa5a5_5a5a);
    let mut flags = 0u64;
    match selector % 3 {
        1 => flags |= 1 << 0,
        2 => flags |= 1 << 1,
        _ => {}
    }
    match (selector >> 2) % 4 {
        1 => flags |= 1 << 2,
        2 => flags |= 1 << 3,
        3 => flags |= 1 << 4,
        _ => {}
    }
    for bit in 5..=16 {
        flags |= ((selector >> (bit + 7)) & 1) << bit;
    }
    flags |= ((selector >> 40) & 1) << 17;
    flags |= ((selector >> 41) & 1) << 18;
    flags |= ((selector >> 42) & 1) << 19;
    words[19] = flags;
    SpartanOuterUniskipRow::from_words(words)
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64) & ((1u64 << 56) - 1)))
        .collect()
}

fn host_result(
    extended: [AkitaField; SPARTAN_OUTER_EXTENDED_NODES],
    tau_high: AkitaField,
) -> EvalResult<HostResult> {
    let mut t1_values = vec![AkitaField::zero(); EXTENDED_SIZE];
    t1_values[..5].copy_from_slice(&extended[..5]);
    t1_values[15..].copy_from_slice(&extended[5..]);
    let kernel_values = centered_lagrange_evals::<AkitaField>(DOMAIN, tau_high)?;
    let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
    let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &t1_values);
    let poly = UnivariatePoly::new(poly_mul(&kernel_coefficients, &t1_coefficients));
    let mut transcript = EvalTranscript::new(b"metal-spartan-outer-uniskip-eval");
    LabeledRoundPoly::uniskip(&poly).append_to_transcript(&mut transcript);
    let challenge = transcript.challenge();
    let output_claim = poly.evaluate(challenge);
    transcript.append_labeled(b"opening_claim", &output_claim);
    Ok(HostResult {
        extended,
        coefficients: poly.coefficients().to_vec(),
        challenge,
        output_claim,
        transcript_state: transcript.state(),
    })
}

fn run_gpu(
    context: &SolinasMetal,
    rows: &SpartanOuterUniskipRows,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
    tau_high: AkitaField,
    threads: usize,
) -> EvalResult<GpuSample> {
    let complete_started = Instant::now();
    let prepare_started = Instant::now();
    let invocation = context.prepare_spartan_outer_uniskip_with_rows(
        rows,
        e_in,
        e_out,
        SpartanOuterUniskipConfig {
            threads_per_threadgroup: Some(threads),
        },
    )?;
    let no_execute_allocations = invocation.execution_device_buffer_allocations() == 0;
    let prepare = prepare_started.elapsed();
    let dispatch_started = Instant::now();
    let gpu_active = invocation.execute_timed()?;
    let dispatch_wall = dispatch_started.elapsed();
    let host_started = Instant::now();
    let result = host_result(invocation.read_output()?, tau_high)?;
    let readback_and_host = host_started.elapsed();
    Ok(GpuSample {
        result,
        complete: complete_started.elapsed(),
        prepare,
        dispatch_wall,
        gpu_active,
        readback_and_host,
        no_execute_allocations,
    })
}

fn run_standalone(
    context: &SolinasMetal,
    rows: &[SpartanOuterUniskipRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
    tau_high: AkitaField,
    threads: usize,
) -> EvalResult<StandaloneSample> {
    let complete_started = Instant::now();
    let row_copy_started = Instant::now();
    let resident = context.prepare_spartan_outer_uniskip_rows(rows)?;
    let row_copy = row_copy_started.elapsed();
    let sample = run_gpu(context, &resident, e_in, e_out, tau_high, threads)?;
    Ok(StandaloneSample {
        complete: complete_started.elapsed(),
        row_copy,
        gpu_active: sample.gpu_active,
    })
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    Ok(env::var(name).map_or(Ok(default), |value| value.parse())?)
}

fn env_u64(name: &str, default: u64) -> EvalResult<u64> {
    Ok(env::var(name).map_or(Ok(default), |value| value.parse())?)
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 22)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 3)?;
    let threads = env_usize("JOLT_METAL_SPARTAN_OUTER_THREADS", 256)?;
    let seed = env_u64("JOLT_METAL_EVAL_SEED", 1)?;
    if !(10..=28).contains(&log_n) || repeats == 0 {
        return Err("log_n or repeats is outside the evaluator domain".into());
    }
    let elements = 1usize << log_n;
    let rows = (0..elements)
        .into_par_iter()
        .map(|index| synthetic_row(index, seed))
        .collect::<Vec<_>>();
    let tau_low = values(log_n + 1, seed ^ 0x6a09_e667_f3bc_c909);
    let tau_high = values(1, seed ^ 0xbb67_ae85_84ca_a73b)[0];
    let split = tau_low.len() / 2;
    let e_out = EqPolynomial::<AkitaField>::evals(&tau_low[..split], None);
    let e_in = EqPolynomial::<AkitaField>::evals(&tau_low[split..], None);
    let context = SolinasMetal::for_akita()?;
    let resident_prepare_started = Instant::now();
    let resident_rows = context.prepare_spartan_outer_uniskip_rows(&rows)?;
    let resident_prepare_once = resident_prepare_started.elapsed();

    let cpu_extended = evaluate_spartan_outer_uniskip_cpu(&rows, &e_in, &e_out)?;
    let cpu_reference = host_result(cpu_extended, tau_high)?;
    let exact_gpu = run_gpu(&context, &resident_rows, &e_in, &e_out, tau_high, threads)?;
    let exact_extended_nodes = cpu_reference.extended == exact_gpu.result.extended;
    let exact_round_poly = cpu_reference.coefficients == exact_gpu.result.coefficients;
    let exact_challenge = cpu_reference.challenge == exact_gpu.result.challenge;
    let exact_output_claim = cpu_reference.output_claim == exact_gpu.result.output_claim
        && cpu_reference.transcript_state == exact_gpu.result.transcript_state;
    if !exact_extended_nodes || !exact_round_poly || !exact_challenge || !exact_output_claim {
        return Err("Metal Spartan outer uni-skip differs from the optimized CPU oracle".into());
    }

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut complete_times = Vec::with_capacity(repeats);
    let mut prepare_times = Vec::with_capacity(repeats);
    let mut dispatch_times = Vec::with_capacity(repeats);
    let mut gpu_active_times = Vec::with_capacity(repeats);
    let mut host_times = Vec::with_capacity(repeats);
    let mut standalone_times = Vec::with_capacity(repeats);
    let mut row_copy_times = Vec::with_capacity(repeats);
    let mut standalone_gpu_times = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let cpu_sample = || -> EvalResult<Duration> {
            let started = Instant::now();
            let extended = black_box(evaluate_spartan_outer_uniskip_cpu(&rows, &e_in, &e_out)?);
            let _ = black_box(host_result(extended, tau_high)?);
            Ok(started.elapsed())
        };
        let mut record_gpu = |sample: GpuSample| {
            let _ = black_box(&sample.result);
            complete_times.push(sample.complete);
            prepare_times.push(sample.prepare);
            dispatch_times.push(sample.dispatch_wall);
            gpu_active_times.push(sample.gpu_active);
            host_times.push(sample.readback_and_host);
        };
        if repeat.is_multiple_of(2) {
            cpu_times.push(cpu_sample()?);
            record_gpu(run_gpu(
                &context,
                &resident_rows,
                &e_in,
                &e_out,
                tau_high,
                threads,
            )?);
        } else {
            record_gpu(run_gpu(
                &context,
                &resident_rows,
                &e_in,
                &e_out,
                tau_high,
                threads,
            )?);
            cpu_times.push(cpu_sample()?);
        }
        let standalone = run_standalone(&context, &rows, &e_in, &e_out, tau_high, threads)?;
        standalone_times.push(standalone.complete);
        row_copy_times.push(standalone.row_copy);
        standalone_gpu_times.push(standalone.gpu_active);
    }
    let cpu_median = median(&mut cpu_times);
    let complete_median = median(&mut complete_times);
    let prepare_median = median(&mut prepare_times);
    let dispatch_median = median(&mut dispatch_times);
    let gpu_active_median = median(&mut gpu_active_times);
    let host_median = median(&mut host_times);
    let standalone_median = median(&mut standalone_times);
    let row_copy_median = median(&mut row_copy_times);
    let hybrid_speedup = cpu_median.as_secs_f64() / complete_median.as_secs_f64();
    let standalone_speedup = cpu_median.as_secs_f64() / standalone_median.as_secs_f64();
    let useful_field_multiplications = 18u128 * elements as u128;
    let standalone_bytes = 192u128 * elements as u128;
    let direct_handoff_bytes = 160u128 * elements as u128;
    let no_execute_allocations = exact_gpu.no_execute_allocations;
    let limits = exact_gpu.result.extended.len();
    let device = context.device_info();
    let output = json!({
        "schema_version": 1,
        "kernel": "spartan_outer_uniskip",
        "metrics": {
            "gpu_dispatch_seconds": dispatch_median.as_secs_f64(),
            "hybrid_speedup": hybrid_speedup,
            "standalone_copy_speedup": standalone_speedup,
            "resident_gpu_speedup": cpu_median.as_secs_f64() / dispatch_median.as_secs_f64(),
            "useful_gpu_gmul_per_second": useful_field_multiplications as f64 / dispatch_median.as_secs_f64() / 1e9
        },
        "timings": {
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "complete_hybrid_median_seconds": complete_median.as_secs_f64(),
            "prepare_median_seconds": prepare_median.as_secs_f64(),
            "gpu_dispatch_wall_median_seconds": dispatch_median.as_secs_f64(),
            "gpu_active_median_seconds": gpu_active_median.as_secs_f64(),
            "readback_and_host_median_seconds": host_median.as_secs_f64(),
            "standalone_copy_complete_median_seconds": standalone_median.as_secs_f64(),
            "standalone_row_copy_median_seconds": row_copy_median.as_secs_f64(),
            "resident_row_prepare_once_seconds": resident_prepare_once.as_secs_f64(),
            "repeats": repeats
        },
        "guards": {
            "exact_extended_nodes": exact_extended_nodes,
            "exact_round_poly": exact_round_poly,
            "exact_challenge": exact_challenge,
            "exact_output_claim": exact_output_claim,
            "host_fiat_shamir": true,
            "no_execute_allocations": no_execute_allocations
        },
        "analytical": {
            "useful_field_multiplications": useful_field_multiplications.to_string(),
            "standalone_logical_bytes": standalone_bytes.to_string(),
            "direct_handoff_logical_bytes": direct_handoff_bytes.to_string(),
            "compute_floor_seconds_at_16_42_gmul_s": useful_field_multiplications as f64 / 16.42e9,
            "standalone_memory_floor_seconds_at_420_68_gib_s": standalone_bytes as f64 / (420.68 * 1024.0_f64.powi(3)),
            "direct_handoff_memory_floor_seconds_at_420_68_gib_s": direct_handoff_bytes as f64 / (420.68 * 1024.0_f64.powi(3))
        },
        "resources": {
            "gpu_seconds": gpu_active_times.iter().map(Duration::as_secs_f64).sum::<f64>(),
            "standalone_gpu_seconds": standalone_gpu_times.iter().map(Duration::as_secs_f64).sum::<f64>(),
            "extended_outputs": limits
        },
        "workload": {
            "log_n": log_n,
            "elements": elements,
            "threads": threads,
            "host_fiat_shamir": true,
            "standalone_packed_row_bytes": 160,
            "primary_metric_uses_resident_direct_handoff": true,
            "standalone_control_includes_row_buffer_copy": true
        },
        "fingerprint": {
            "device": device.name,
            "max_buffer_length": device.max_buffer_length,
            "max_threadgroup_memory_length": device.max_threadgroup_memory_length,
            "cpu_threads": std::thread::available_parallelism()?.get()
        }
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
