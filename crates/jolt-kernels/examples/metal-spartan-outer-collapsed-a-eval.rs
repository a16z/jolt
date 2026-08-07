#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error, hint::black_box, time::Instant};

use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    spartan_outer_successor::{
        SpartanOuterCollapsedAProbeConfig, SpartanOuterCollapsedAProbeStats,
        SpartanOuterDeferredBProbeConfig, SOURCE,
    },
    OuterBindingPlan, OuterKernelArtifact, SolinasMetal,
};
use jolt_poly::EqPolynomial;
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let log_t = env_usize("JOLT_SPARTAN_OUTER_COLLAPSED_A_LOG_T", 8)?;
    let samples = env_usize("JOLT_SPARTAN_OUTER_COLLAPSED_A_SAMPLES", 5)?;
    let threads = env_usize("JOLT_SPARTAN_OUTER_COLLAPSED_A_THREADS", 128)?;
    if !(2..=28).contains(&log_t) || samples == 0 {
        return Err("log_t must be in 2..=28 and samples must be nonzero".into());
    }
    let rows = 1usize << log_t;
    let seed = 0x510e_527f_ade6_82d1;
    let artifact = OuterKernelArtifact::new(SOURCE.to_owned(), OuterBindingPlan::BOnlyV1)?;
    let context = SolinasMetal::for_akita_with_outer_artifact(&artifact)?;
    let setup_started = Instant::now();
    let resident =
        context.prepare_spartan_outer_deferred_b_synthetic_rows(rows, seed ^ 0x1000_0001)?;
    let materialize_point = values(log_t, seed ^ 0xbb67_ae85_84ca_a73b);
    let materialize_split = log_t.div_ceil(2);
    let materialize_e_out =
        EqPolynomial::<AkitaField>::evals(&materialize_point[..materialize_split], None);
    let materialize_e_in =
        EqPolynomial::<AkitaField>::evals(&materialize_point[materialize_split..], None);
    let lagrange = std::array::from_fn(|index| {
        AkitaField::from_u64(splitmix(seed ^ 0x3c6e_f372 ^ index as u64))
    });
    let mut materialize = context.prepare_spartan_outer_deferred_b_probe(
        resident,
        &lagrange,
        &materialize_e_in,
        &materialize_e_out,
        SpartanOuterDeferredBProbeConfig::default(),
    )?;
    let _ = materialize.run_parent()?;

    let stream_point = values(log_t - 1, seed ^ 0x1f83_d9ab_fb41_bd6b);
    let stream_split = (log_t - 1).div_ceil(2);
    let stream_e_out = EqPolynomial::<AkitaField>::evals(&stream_point[..stream_split], None);
    let stream_e_in = EqPolynomial::<AkitaField>::evals(&stream_point[stream_split..], None);
    let challenge = AkitaField::from_u64(splitmix(seed ^ 0x5be0_cd19_137e_2179));
    let mut probe = materialize.into_collapsed_a_probe(
        challenge,
        &stream_e_in,
        &stream_e_out,
        SpartanOuterCollapsedAProbeConfig {
            threads_per_threadgroup: Some(threads),
            ..SpartanOuterCollapsedAProbeConfig::default()
        },
    )?;
    let setup_wall = setup_started.elapsed();

    let parent_warmup = probe.run_parent()?;
    let parent_state = (log_t <= 16)
        .then(|| probe.read_dense_state())
        .transpose()?;
    let candidate_warmup = probe.run_candidate()?;
    let candidate_state = (log_t <= 16)
        .then(|| probe.read_dense_state())
        .transpose()?;
    if parent_warmup.message != candidate_warmup.message || parent_state != candidate_state {
        return Err("collapsed-A stream parity failed".into());
    }

    let mut parent = Vec::with_capacity(samples);
    let mut candidate = Vec::with_capacity(samples);
    for sample in 0..samples {
        let (parent_sample, candidate_sample) = if sample.is_multiple_of(2) {
            (probe.run_parent()?, probe.run_candidate()?)
        } else {
            let candidate_sample = probe.run_candidate()?;
            let parent_sample = probe.run_parent()?;
            (parent_sample, candidate_sample)
        };
        if parent_sample.message != candidate_sample.message {
            return Err("alternating collapsed-A messages differ".into());
        }
        parent.push(parent_sample);
        candidate.push(candidate_sample);
    }
    let _ = black_box((&parent, &candidate));

    let parent_wall_ns = median_ns(&parent, |sample| sample.wall.as_nanos());
    let candidate_wall_ns = median_ns(&candidate, |sample| sample.wall.as_nanos());
    let parent_active_ns = median_ns(&parent, |sample| sample.gpu_active.as_nanos());
    let candidate_active_ns = median_ns(&candidate, |sample| sample.gpu_active.as_nanos());
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema": "spartan_outer_collapsed_a_probe_v1",
            "log_t": log_t,
            "rows": rows,
            "samples": samples,
            "threads_per_threadgroup": threads,
            "full_state_parity": log_t <= 16,
            "message_parity": true,
            "setup_wall_ns": setup_wall.as_nanos(),
            "library_compile_wall_ns": context.compilation_stats().library_compile_wall.as_nanos(),
            "parent": sample_json(parent_wall_ns, parent_active_ns, parent_warmup),
            "candidate": sample_json(candidate_wall_ns, candidate_active_ns, candidate_warmup),
            "candidate_speedup": {
                "wall": ratio(parent_wall_ns, candidate_wall_ns),
                "gpu_active": ratio(parent_active_ns, candidate_active_ns),
            },
            "projected_stream_plus_dense_prefix_ms": candidate_active_ns as f64 / 1e6 + 15.002_958,
            "stream_plus_prefix_gate_ms": 32.0,
        }))?
    );
    Ok(())
}

fn sample_json(
    wall_ns: u128,
    active_ns: u128,
    warmup: SpartanOuterCollapsedAProbeStats,
) -> serde_json::Value {
    json!({
        "wall_median_ns": wall_ns,
        "gpu_active_median_ns": active_ns,
        "pipeline_limits": {
            "thread_execution_width": warmup.pipeline_limits.thread_execution_width,
            "max_total_threads_per_threadgroup": warmup.pipeline_limits.max_total_threads_per_threadgroup,
            "static_threadgroup_memory_length": warmup.pipeline_limits.static_threadgroup_memory_length,
        },
    })
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64)))
        .collect()
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn median_ns(
    samples: &[SpartanOuterCollapsedAProbeStats],
    projection: impl Fn(&SpartanOuterCollapsedAProbeStats) -> u128,
) -> u128 {
    let mut values = samples.iter().map(projection).collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

fn ratio(numerator: u128, denominator: u128) -> f64 {
    numerator as f64 / denominator as f64
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
