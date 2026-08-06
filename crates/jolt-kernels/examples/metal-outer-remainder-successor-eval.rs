#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error, fmt::Write as _, fs, path::PathBuf, time::Duration};

use jolt_field::FixedBytes;
use jolt_kernels::metal::solinas::{
    OuterRemainderDispatchCounts, SealedOuterArtifact, SolinasMetal, SolinasMetalCompilationStats,
};
use jolt_kernels::metal::{
    OuterRemainderEvalFixture, OuterRemainderEvalResult, OuterRemainderEvalSample,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type EvalResult<T> = Result<T, Box<dyn Error>>;

const FIXTURE: &str = "resident-outer-remainder-v2";
const SEED: u64 = 0x243f_6a88_85a3_08d3;

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N")?;
    let pairs = env_usize("JOLT_METAL_EVAL_REPEATS")?;
    if !(16..=27).contains(&log_n) || pairs != 4 {
        return Err(failure(
            "successor evaluator requires log_n 16..=27 and four pairs",
        ));
    }
    let expected_binary = required_env("JOLT_AUTORESEARCH_RUNNER_SHA256")?;
    let binary_path = env::current_exe()?;
    let binary_before = file_sha256(&binary_path)?;
    if expected_binary != binary_before {
        return Err(failure("running binary does not match the sealed digest"));
    }

    let parent_path = PathBuf::from(required_env("JOLT_AUTORESEARCH_PARENT_ARTIFACT")?);
    let candidate_path = PathBuf::from(required_env("JOLT_AUTORESEARCH_CANDIDATE_ARTIFACT")?);
    let parent_artifact = SealedOuterArtifact::load(&parent_path)?;
    let candidate_artifact = SealedOuterArtifact::load(&candidate_path)?;
    let cycles = 1usize << log_n;
    for artifact in [&parent_artifact, &candidate_artifact] {
        if cycles < artifact.trace_cutoff_elements()
            || cycles < artifact.sequence_config().cpu_tail_elements
        {
            return Err(failure("artifact dispatch does not admit the target trace"));
        }
    }

    let parent_context = parent_artifact.compile_akita()?;
    let candidate_context = candidate_artifact.compile_akita()?;
    if parent_context.device_info().name != candidate_context.device_info().name {
        return Err(failure(
            "parent and candidate compiled on different devices",
        ));
    }
    let fixture = OuterRemainderEvalFixture::new(&parent_context, log_n, SEED)?;
    let parent_config = parent_artifact.sequence_config();
    let candidate_config = candidate_artifact.sequence_config();
    if parent_config.cpu_tail_elements != candidate_config.cpu_tail_elements
        || parent_artifact.trace_cutoff_elements() != candidate_artifact.trace_cutoff_elements()
    {
        return Err(failure(
            "successor comparison requires identical CPU and trace cutoffs",
        ));
    }

    let mut oracle: Option<OuterRemainderEvalResult> = None;
    let (warmup_parent, warmup_candidate) = run_pair(
        &fixture,
        &parent_context,
        parent_config,
        &candidate_context,
        candidate_config,
        true,
    )?;
    require_exact(
        &mut oracle,
        &warmup_parent,
        &warmup_candidate,
        cycles,
        parent_config.cpu_tail_elements,
    )?;
    let mut resource_gpu_active = arm_resource_gpu_active(&warmup_parent)?
        .checked_add(arm_resource_gpu_active(&warmup_candidate)?)
        .ok_or_else(|| failure("warmup GPU charge overflowed"))?;
    let excluded_warmup = pair_record(["parent", "candidate"], &warmup_parent, &warmup_candidate)?;
    let mut parent_pipeline_compile_ns = vec![duration_ns(warmup_parent.pipeline_compile_wall)?];
    let mut candidate_pipeline_compile_ns =
        vec![duration_ns(warmup_candidate.pipeline_compile_wall)?];

    let mut samples = Vec::with_capacity(pairs);
    let mut speedups = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let parent_first = pair % 2 == 0;
        let (parent, candidate) = run_pair(
            &fixture,
            &parent_context,
            parent_config,
            &candidate_context,
            candidate_config,
            parent_first,
        )?;
        require_exact(
            &mut oracle,
            &parent,
            &candidate,
            cycles,
            parent_config.cpu_tail_elements,
        )?;
        resource_gpu_active = resource_gpu_active
            .checked_add(arm_resource_gpu_active(&parent)?)
            .and_then(|total| total.checked_add(arm_resource_gpu_active(&candidate).ok()?))
            .ok_or_else(|| failure("timed GPU charge overflowed"))?;
        let parent_ns = duration_ns(parent.member_gpu_active)?;
        let candidate_ns = duration_ns(candidate.member_gpu_active)?;
        parent_pipeline_compile_ns.push(duration_ns(parent.pipeline_compile_wall)?);
        candidate_pipeline_compile_ns.push(duration_ns(candidate.pipeline_compile_wall)?);
        speedups.push(parent_ns as f64 / candidate_ns as f64);
        let order = if parent_first {
            ["parent", "candidate"]
        } else {
            ["candidate", "parent"]
        };
        let mut record = pair_record(order, &parent, &candidate)?;
        record["pair"] = json!(pair);
        samples.push(record);
    }
    speedups.sort_by(f64::total_cmp);
    let successor_speedup = f64::midpoint(
        speedups[speedups.len() / 2 - 1],
        speedups[speedups.len() / 2],
    );
    let orders = (0..pairs)
        .map(|pair| {
            if pair % 2 == 0 {
                json!(["parent", "candidate"])
            } else {
                json!(["candidate", "parent"])
            }
        })
        .collect::<Vec<_>>();

    let binary_after = file_sha256(&binary_path)?;
    if binary_after != binary_before {
        return Err(failure("sealed runner changed during evaluation"));
    }
    let device = parent_context.device_info();
    let result = json!({
        "schema": "outer_remainder_successor_v2",
        "schema_version": 2,
        "kernel": "OuterRemainder",
        "fingerprint": {
            "fixture": FIXTURE,
            "log_n": log_n,
            "pairs": pairs,
            "excluded_warmup_pairs": 1,
            "orders": orders,
            "parent_artifact_sha256": parent_artifact.artifact_sha256(),
            "candidate_artifact_sha256": candidate_artifact.artifact_sha256(),
            "runner_binary_sha256": binary_before,
        },
        "metrics": {
            "successor_speedup": successor_speedup,
            "paired_speedups": speedups,
        },
        "excluded_warmup": excluded_warmup,
        "samples": samples,
        "guards": {
            "all_exact": true,
            "correctness_exact": true,
            "target_scale": fixture.log_t() == log_n && fixture.cycles() == cycles,
            "runtime_artifacts_exact": true,
            "resident_row_handle_lifecycle_exact": true,
            "metal_phase_schedule_exact": true,
            "gpu_timestamps_exact": true,
        },
        "all_exact": true,
        "resources": {
            "gpu_active_total_ns": duration_ns(resource_gpu_active)?,
            "gpu_seconds": resource_gpu_active.as_secs_f64(),
        },
        "telemetry": {
            "device_name": device.name,
            "device_registry_shared": true,
            "cycles": cycles,
            "parent_binding_plan": parent_artifact.binding_plan().as_str(),
            "candidate_binding_plan": candidate_artifact.binding_plan().as_str(),
            "parent_source_sha256": parent_artifact.outer_source_sha256(),
            "candidate_source_sha256": candidate_artifact.outer_source_sha256(),
            "production_last_owner_release_deferred": true,
            "compilation": {
                "context_order": ["parent", "candidate"],
                "parent": compilation_record(
                    parent_context.compilation_stats(),
                    &parent_pipeline_compile_ns,
                )?,
                "candidate": compilation_record(
                    candidate_context.compilation_stats(),
                    &candidate_pipeline_compile_ns,
                )?,
            },
        },
    });
    println!("{result}");
    Ok(())
}

fn run_pair(
    fixture: &OuterRemainderEvalFixture,
    parent_context: &SolinasMetal,
    parent_config: jolt_kernels::metal::solinas::OuterRemainderSequenceConfig,
    candidate_context: &SolinasMetal,
    candidate_config: jolt_kernels::metal::solinas::OuterRemainderSequenceConfig,
    parent_first: bool,
) -> EvalResult<(OuterRemainderEvalSample, OuterRemainderEvalSample)> {
    if parent_first {
        Ok((
            fixture.run(parent_context, parent_config)?,
            fixture.run(candidate_context, candidate_config)?,
        ))
    } else {
        let candidate = fixture.run(candidate_context, candidate_config)?;
        let parent = fixture.run(parent_context, parent_config)?;
        Ok((parent, candidate))
    }
}

fn require_exact(
    oracle: &mut Option<OuterRemainderEvalResult>,
    parent: &OuterRemainderEvalSample,
    candidate: &OuterRemainderEvalSample,
    cycles: usize,
    expected_tail: usize,
) -> EvalResult<()> {
    if parent.result != candidate.result {
        return Err(failure("candidate result differs from the accepted parent"));
    }
    if let Some(expected) = oracle {
        if parent.result != *expected {
            return Err(failure("accepted parent result is not deterministic"));
        }
    } else {
        *oracle = Some(parent.result.clone());
    }
    validate_lifecycle(parent, cycles, expected_tail)?;
    validate_lifecycle(candidate, cycles, expected_tail)
}

fn validate_lifecycle(
    sample: &OuterRemainderEvalSample,
    cycles: usize,
    expected_tail: usize,
) -> EvalResult<()> {
    let counts = sample.dispatch_counts;
    let dense_transitions = (cycles / expected_tail).ilog2() as usize;
    if counts.materializations != 1
        || counts.stream_transitions != 1
        || counts.dense_transitions != dense_transitions
        || counts.cpu_tail_exports != 1
        || counts.opening_scans != 1
        || counts.command_buffers
            != counts.materializations
                + counts.stream_transitions
                + counts.dense_transitions
                + counts.opening_scans
        || sample.result.round_polynomials.is_empty()
        || sample.result.round_polynomials.len() != sample.result.challenges.len()
        || sample.result.opening_point
            != sample
                .result
                .challenges
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>()
        || sample.result.member_claims.len() != 1
        || sample.result.expected_final_claim != sample.result.final_claim
        || sample.result.output_claims.len() != 35
        || sample.tail_elements != expected_tail
        || !expected_tail.is_power_of_two()
        || !cycles.is_multiple_of(expected_tail)
        || sample.initialized_bytes == 0
        || sample.initialized_bytes != sample.storage_owned_bytes
        || sample.round_device_buffer_allocations != 0
    {
        return Err(failure("OuterRemainder lifecycle contract is invalid"));
    }
    Ok(())
}

fn pair_record(
    order: [&str; 2],
    parent: &OuterRemainderEvalSample,
    candidate: &OuterRemainderEvalSample,
) -> EvalResult<Value> {
    Ok(json!({
        "order": order,
        "parent": arm_record(parent)?,
        "candidate": arm_record(candidate)?,
    }))
}

fn arm_record(sample: &OuterRemainderEvalSample) -> EvalResult<Value> {
    let member_gpu_active_ns = duration_ns(sample.member_gpu_active)?;
    let member_wall_ns = duration_ns(sample.member_wall)?;
    let setup_gpu_active_ns = duration_ns(sample.setup_gpu_active)?;
    let setup_wall_ns = duration_ns(sample.setup_wall)?;
    let pipeline_compile_ns = duration_ns(sample.pipeline_compile_wall)?;
    let materialize_ns = duration_ns(sample.phase_gpu_active.materialize)?;
    let first_bind_ns = duration_ns(sample.phase_gpu_active.first_bind)?;
    let dense_rounds_ns = duration_ns(sample.phase_gpu_active.dense_rounds)?;
    let openings_ns = duration_ns(sample.phase_gpu_active.openings)?;
    let phase_gpu_active_ns = [materialize_ns, first_bind_ns, dense_rounds_ns, openings_ns]
        .into_iter()
        .try_fold(0u64, |total, value| total.checked_add(value))
        .ok_or_else(|| failure("phase GPU time overflowed"))?;
    if member_gpu_active_ns == 0
        || member_gpu_active_ns > member_wall_ns
        || materialize_ns == 0
        || first_bind_ns == 0
        || (dense_rounds_ns == 0 && sample.dispatch_counts.dense_transitions != 0)
        || openings_ns == 0
        || phase_gpu_active_ns != member_gpu_active_ns
        || setup_gpu_active_ns == 0
        || setup_gpu_active_ns > setup_wall_ns
        || pipeline_compile_ns > setup_wall_ns
    {
        return Err(failure("GPU timestamps are not nested in wall timing"));
    }
    let resource_gpu_active_ns = member_gpu_active_ns
        .checked_add(setup_gpu_active_ns)
        .ok_or_else(|| failure("arm GPU charge overflowed"))?;
    Ok(json!({
        "gpu_active_ns": member_gpu_active_ns,
        "wall_ns": member_wall_ns,
        "resource_gpu_active_ns": resource_gpu_active_ns,
        "setup_gpu_active_ns": setup_gpu_active_ns,
        "setup_wall_ns": setup_wall_ns,
        "phase_gpu_active_ns": {
            "materialize": materialize_ns,
            "first_bind": first_bind_ns,
            "dense_rounds": dense_rounds_ns,
            "openings": openings_ns,
        },
        "tail_elements": sample.tail_elements,
        "initialized_bytes": sample.initialized_bytes,
        "storage_owned_bytes": sample.storage_owned_bytes,
        "round_device_buffer_allocations": sample.round_device_buffer_allocations,
        "output_sha256": output_digest(&sample.result),
        "dispatch_counts": dispatch_counts(sample.dispatch_counts),
    }))
}

fn compilation_record(
    stats: &SolinasMetalCompilationStats,
    pipeline_set_ns: &[u64],
) -> EvalResult<Value> {
    let pipeline_set_total_ns = pipeline_set_ns.iter().try_fold(0u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| failure("pipeline compilation wall overflowed"))
    })?;
    Ok(json!({
        "source_assembly_ns": duration_ns(stats.source_assembly_wall)?,
        "library_compile_ns": duration_ns(stats.library_compile_wall)?,
        "source_bytes": stats.source_bytes,
        "assembled_source_sha256": hex_digest(&stats.assembled_source_sha256),
        "pipeline_set_ns": pipeline_set_ns,
        "pipeline_set_total_ns": pipeline_set_total_ns,
    }))
}

fn dispatch_counts(counts: OuterRemainderDispatchCounts) -> Value {
    json!({
        "materializations": counts.materializations,
        "stream_transitions": counts.stream_transitions,
        "dense_transitions": counts.dense_transitions,
        "cpu_tail_exports": counts.cpu_tail_exports,
        "opening_scans": counts.opening_scans,
        "command_buffers": counts.command_buffers,
    })
}

fn output_digest(result: &OuterRemainderEvalResult) -> String {
    let mut digest = Sha256::new();
    for value in [
        result.input_claim,
        result.coefficient,
        result.final_claim,
        result.expected_final_claim,
    ] {
        digest.update(value.to_bytes_array());
    }
    for value in result
        .round_polynomials
        .iter()
        .flatten()
        .chain(&result.challenges)
        .chain(&result.member_claims)
        .chain(&result.output_claims)
        .chain(&result.opening_point)
    {
        digest.update(value.to_bytes_array());
    }
    digest.update(result.transcript_state);
    hex_digest(digest.finalize().as_slice())
}

fn arm_resource_gpu_active(sample: &OuterRemainderEvalSample) -> EvalResult<Duration> {
    sample
        .member_gpu_active
        .checked_add(sample.setup_gpu_active)
        .ok_or_else(|| failure("arm GPU charge overflowed"))
}

fn env_usize(name: &str) -> EvalResult<usize> {
    Ok(required_env(name)?.parse()?)
}

fn required_env(name: &str) -> EvalResult<String> {
    env::var(name).map_err(|_| failure(format!("{name} is required")))
}

fn file_sha256(path: &PathBuf) -> EvalResult<String> {
    Ok(hex_digest(Sha256::digest(fs::read(path)?).as_slice()))
}

fn duration_ns(duration: Duration) -> EvalResult<u64> {
    Ok(u64::try_from(duration.as_nanos())?)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(2 * bytes.len());
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::other(message.into()))
}
