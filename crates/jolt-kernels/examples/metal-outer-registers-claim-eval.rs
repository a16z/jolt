#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::{env, error::Error, time::Duration};

use jolt_kernels::metal::solinas::{OuterRemainderSequenceConfig, SolinasMetal};
use jolt_kernels::metal::{OuterRemainderEvalFixture, OuterRemainderEvalSample};
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;

const SEED: u64 = 0x243f_6a88_85a3_08d3;

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N")?;
    let pairs = env_usize("JOLT_METAL_EVAL_REPEATS")?;
    if !(16..=28).contains(&log_n) || pairs == 0 {
        return Err(failure(
            "outer registers-claim evaluator requires log_n 16..=28 and at least one pair",
        ));
    }

    let context = SolinasMetal::for_akita()?;
    let fixture = OuterRemainderEvalFixture::new(&context, log_n, SEED)?;
    let baseline = OuterRemainderSequenceConfig {
        cpu_tail_elements: 1 << 16,
        product_uniskip_carrier: true,
        ..Default::default()
    };
    let candidate = OuterRemainderSequenceConfig {
        registers_claim_carrier: true,
        ..baseline
    };

    let warm_baseline = fixture.run(&context, baseline)?;
    let warm_candidate = fixture.run(&context, candidate)?;
    require_exact(&warm_baseline, &warm_candidate)?;

    let mut samples = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let baseline_first = pair % 2 == 0;
        let (baseline_sample, candidate_sample) = if baseline_first {
            (
                fixture.run(&context, baseline)?,
                fixture.run(&context, candidate)?,
            )
        } else {
            let candidate_sample = fixture.run(&context, candidate)?;
            let baseline_sample = fixture.run(&context, baseline)?;
            (baseline_sample, candidate_sample)
        };
        require_exact(&baseline_sample, &candidate_sample)?;
        samples.push(json!({
            "pair": pair,
            "order": if baseline_first {
                ["baseline", "candidate"]
            } else {
                ["candidate", "baseline"]
            },
            "baseline": sample_record(&baseline_sample)?,
            "candidate": sample_record(&candidate_sample)?,
        }));
    }

    let device = context.device_info();
    println!(
        "{}",
        json!({
            "schema": "outer_registers_claim_screen_v1",
            "schema_version": 1,
            "fingerprint": {
                "device": device.name,
                "log_n": log_n,
                "cycles": fixture.cycles(),
                "pairs": pairs,
                "seed": SEED,
                "cpu_tail_elements": baseline.cpu_tail_elements,
                "product_uniskip_carrier": true,
            },
            "guards": {
                "exact_complete_member": true,
                "baseline_carrier_count": 0,
                "candidate_carrier_count": 1,
                "same_command_buffer_count": true,
                "zero_round_allocations": true,
            },
            "excluded_warmup": {
                "baseline": sample_record(&warm_baseline)?,
                "candidate": sample_record(&warm_candidate)?,
            },
            "samples": samples,
        })
    );
    Ok(())
}

fn require_exact(
    baseline: &OuterRemainderEvalSample,
    candidate: &OuterRemainderEvalSample,
) -> EvalResult<()> {
    if baseline.result != candidate.result {
        return Err(failure(
            "registers-claim carrier changed the OuterRemainder result",
        ));
    }
    if baseline.dispatch_counts.registers_claim_carriers != 0
        || candidate.dispatch_counts.registers_claim_carriers != 1
        || baseline.dispatch_counts.command_buffers != candidate.dispatch_counts.command_buffers
        || baseline.round_device_buffer_allocations != 0
        || candidate.round_device_buffer_allocations != 0
    {
        return Err(failure(
            "registers-claim carrier dispatch lifecycle is inconsistent",
        ));
    }
    Ok(())
}

fn sample_record(sample: &OuterRemainderEvalSample) -> EvalResult<serde_json::Value> {
    Ok(json!({
        "member_wall_ns": duration_ns(sample.member_wall)?,
        "member_gpu_active_ns": duration_ns(sample.member_gpu_active)?,
        "opening_gpu_active_ns": duration_ns(sample.phase_gpu_active.openings)?,
        "storage_owned_bytes": sample.storage_owned_bytes,
        "initialized_bytes": sample.initialized_bytes,
        "round_device_buffer_allocations": sample.round_device_buffer_allocations,
        "dispatch_counts": {
            "materializations": sample.dispatch_counts.materializations,
            "stream_transitions": sample.dispatch_counts.stream_transitions,
            "dense_transitions": sample.dispatch_counts.dense_transitions,
            "cpu_tail_exports": sample.dispatch_counts.cpu_tail_exports,
            "opening_scans": sample.dispatch_counts.opening_scans,
            "registers_claim_carriers": sample.dispatch_counts.registers_claim_carriers,
            "command_buffers": sample.dispatch_counts.command_buffers,
        },
    }))
}

fn env_usize(name: &str) -> EvalResult<usize> {
    Ok(env::var(name)?.parse()?)
}

fn duration_ns(duration: Duration) -> EvalResult<u64> {
    Ok(u64::try_from(duration.as_nanos())?)
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::other(message.into()))
}
