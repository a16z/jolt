#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]
#![recursion_limit = "256"]

use std::collections::HashSet;
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use jolt_kernels::metal::solinas::{
    InstructionInputPrimerStats, InstructionInputStorageInitialization, SolinasMetal,
};
use serde_json::json;

#[expect(
    dead_code,
    reason = "the shared support also exposes steady-state evaluator controls"
)]
#[path = "support/instruction_input.rs"]
mod instruction_input;

use instruction_input::{
    derived_eq_cycle_is_exact, expected_states, final_relation_is_exact, run_cpu, run_hybrid,
    Capture, EvalResult, SequenceDispatch, Workload, TABLES,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Lazy,
    Minimal,
    ComputeControl,
    NativePrimer,
}

impl Arm {
    fn from_env() -> EvalResult<Self> {
        match env::var("JOLT_METAL_INSTRUCTION_INPUT_COLD_ARM")
            .as_deref()
            .unwrap_or("lazy")
        {
            "lazy" => Ok(Self::Lazy),
            "minimal" => Ok(Self::Minimal),
            "compute_control" => Ok(Self::ComputeControl),
            "native_primer" => Ok(Self::NativePrimer),
            _ => Err(
                "cold arm must be `lazy`, `minimal`, `compute_control`, or `native_primer`".into(),
            ),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Lazy => "lazy",
            Self::Minimal => "minimal",
            Self::ComputeControl => "compute_control",
            Self::NativePrimer => "native_primer",
        }
    }

    const fn storage_initialization(self) -> InstructionInputStorageInitialization {
        match self {
            Self::Lazy => InstructionInputStorageInitialization::Lazy,
            Self::Minimal | Self::ComputeControl | Self::NativePrimer => {
                InstructionInputStorageInitialization::Minimal
            }
        }
    }
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn duration_ns(value: Duration) -> EvalResult<u64> {
    Ok(u64::try_from(value.as_nanos())?)
}

fn duration_ns_samples(values: &[Duration]) -> EvalResult<Vec<u64>> {
    values.iter().copied().map(duration_ns).collect()
}

fn sum_durations(values: &[Duration]) -> Duration {
    values.iter().copied().sum()
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 26)?;
    let cutoff_log2 = env_usize("JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2", 16)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let arm = Arm::from_env()?;
    let dispatch = SequenceDispatch {
        native_message: env_usize("JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS", 256)?,
        native_transition: env_usize(
            "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS",
            128,
        )?,
        dense_transition: env_usize("JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS", 128)?,
    };
    if !(6..=28).contains(&log_n) || cutoff_log2 < 1 || cutoff_log2 + 2 > log_n {
        return Err("log size or cutoff is outside the evaluator domain".into());
    }
    let cutoff = 1usize << cutoff_log2;
    let context = SolinasMetal::for_akita()?;
    let noop = context.prepare_noop()?;
    let mut workload = Workload::new(log_n, seed)?;
    let cpu = black_box(run_cpu(&workload, cutoff, Capture::TARGET)?);

    let sequence_preparation_started = Instant::now();
    let mut sequence = workload.prepare_sequence_with_storage_initialization(
        &context,
        dispatch,
        arm.storage_initialization(),
    )?;
    let sequence_preparation = sequence_preparation_started.elapsed();
    let initialization = sequence.storage_initialization();
    let storage_bytes = sequence.owned_storage_bytes();
    let initial_buffer_identities = sequence.static_buffer_identity();
    let initial_resident_row_identity = sequence.resident_row_identity();

    let mut primer: Option<InstructionInputPrimerStats> = None;
    let (control_wall, control_active) = match arm {
        Arm::Lazy | Arm::Minimal => (Duration::ZERO, Duration::ZERO),
        Arm::ComputeControl => {
            let started = Instant::now();
            let active = noop.execute_timed()?;
            (started.elapsed(), active)
        }
        Arm::NativePrimer => {
            let stats = sequence.prime_native_pipeline()?;
            primer = Some(stats);
            (stats.wall, stats.gpu_active)
        }
    };
    let hybrid = black_box(run_hybrid(
        &mut sequence,
        &workload,
        cutoff,
        Capture::TARGET,
    )?);

    let command_count = log_n - cutoff_log2 + 1;
    let round_0_wall = hybrid.gpu_command_wall[0];
    let round_0_active = hybrid.gpu_command_active[0];
    let first_three_wall = sum_durations(&hybrid.gpu_command_wall[..3]);
    let first_three_active = sum_durations(&hybrid.gpu_command_active[..3]);
    let later_wall = sum_durations(&hybrid.gpu_command_wall[3..]);
    let later_active = sum_durations(&hybrid.gpu_command_active[3..]);
    let expected_initialization_bytes = match arm.storage_initialization() {
        InstructionInputStorageInitialization::Lazy => 0,
        InstructionInputStorageInitialization::Minimal => 6 * 16,
        InstructionInputStorageInitialization::Full => unreachable!(),
    };
    let expected_initialization_buffers = usize::from(expected_initialization_bytes != 0) * 6;
    let expects_control = matches!(arm, Arm::ComputeControl | Arm::NativePrimer);
    let expects_primer = arm == Arm::NativePrimer;
    let primer_geometry_exact = primer.is_some_and(|stats| {
        stats.source_elements == 64
            && stats.e_in_elements == 1
            && stats.e_out_elements == 32
            && stats.resident_row_identity == initial_resident_row_identity
            && stats.storage_buffer_identities == initial_buffer_identities
    });
    let trace = &hybrid.trace;
    let cpu_trace = &cpu.trace;
    let guards = json!({
        "exact_four_sample_q_evals": cpu_trace.q_evals == trace.q_evals,
        "exact_round_polynomials": cpu_trace.round_polys == trace.round_polys,
        "exact_host_fiat_shamir_challenges": cpu_trace.challenges == trace.challenges,
        "exact_round_schedule": cpu_trace.states == expected_states(log_n) && trace.states == expected_states(log_n),
        "exact_final_eight_claims": cpu_trace.final_claims == trace.final_claims,
        "exact_final_sumcheck_claim": cpu_trace.final_sumcheck_claim == trace.final_sumcheck_claim,
        "exact_transcript_state": cpu_trace.transcript_state == trace.transcript_state,
        "exact_derived_eq_cycle": cpu_trace.derived_eq_cycle == trace.derived_eq_cycle && derived_eq_cycle_is_exact(&workload, cpu_trace) && derived_eq_cycle_is_exact(&workload, trace),
        "exact_final_relation": final_relation_is_exact(&workload, cpu_trace) && final_relation_is_exact(&workload, trace),
        "storage_initialization_mode_exact": initialization.mode == arm.storage_initialization(),
        "storage_initialization_bytes_exact": initialization.bytes == expected_initialization_bytes,
        "storage_initialization_buffer_count_exact": initialization.device_buffers == expected_initialization_buffers,
        "storage_initialization_completed_before_member": sequence_preparation >= initialization.wall,
        "storage_initialization_timestamps_exact": if expected_initialization_bytes == 0 { initialization.gpu_active == Duration::ZERO } else { initialization.gpu_active > Duration::ZERO && initialization.gpu_active <= initialization.wall },
        "control_command_presence_exact": expects_control == (control_wall > Duration::ZERO && control_active > Duration::ZERO),
        "control_command_timestamps_valid": !expects_control || control_active <= control_wall,
        "native_primer_geometry_exact": expects_primer == primer_geometry_exact,
        "static_device_buffer_identities_stable": hybrid.static_device_buffers_stable && initial_buffer_identities == initialization.buffer_identities && sequence.static_buffer_identity() == initial_buffer_identities,
        "static_device_buffer_identities_distinct": initialization.buffer_identities.into_iter().collect::<HashSet<_>>().len() == 6,
        "resident_rows_stable": hybrid.resident_rows_stable && sequence.resident_row_identity() == initial_resident_row_identity,
        "exactly_one_dense_readback": hybrid.readbacks == 1,
        "readback_bytes_exact": hybrid.preallocated_readback_bytes == TABLES * cutoff * 16,
        "round_device_buffer_allocations_zero": sequence.round_device_buffer_allocations() == 0,
        "gpu_command_count_exact": hybrid.gpu_command_wall.len() == command_count && hybrid.gpu_command_active.len() == command_count,
        "gpu_wall_reconciled": sum_durations(&hybrid.gpu_command_wall) == hybrid.gpu_wall,
        "gpu_active_reconciled": sum_durations(&hybrid.gpu_command_active) == hybrid.gpu_active,
        "gpu_command_timestamps_valid": hybrid.gpu_command_active.iter().zip(&hybrid.gpu_command_wall).all(|(active, wall)| *active > Duration::ZERO && active <= wall),
        "host_fiat_shamir": true,
        "no_excluded_target_warmup": true,
        "one_first_use_target_sequence": true,
    });
    let all_exact = guards
        .as_object()
        .is_some_and(|values| values.values().all(|value| value == &json!(true)));
    if !all_exact {
        return Err("InstructionInput cold-mechanism evaluator guard failed".into());
    }

    let primer_stats = primer.unwrap_or(InstructionInputPrimerStats {
        source_elements: 0,
        e_in_elements: 0,
        e_out_elements: 0,
        wall: Duration::ZERO,
        gpu_active: Duration::ZERO,
        resident_row_identity: 0,
        storage_buffer_identities: [0; 6],
    });
    let device = context.device_info();
    let output = json!({
        "schema": "instruction_input_cold_mechanism_v1",
        "schema_version": 1,
        "kernel": "instruction_input",
        "arm": arm.as_str(),
        "metrics": {
            "member_wall_ns": duration_ns(hybrid.wall)?,
            "round_0_nonactive_ns": duration_ns(round_0_wall.saturating_sub(round_0_active))?,
            "control_plus_member_ns": duration_ns(control_wall + hybrid.wall)?,
        },
        "timings": {
            "cpu_control_ns": duration_ns(cpu.wall)?,
            "sequence_preparation_ns": duration_ns(sequence_preparation)?,
            "storage_initialization_wall_ns": duration_ns(initialization.wall)?,
            "storage_initialization_gpu_active_ns": duration_ns(initialization.gpu_active)?,
            "control_wall_ns": duration_ns(control_wall)?,
            "control_gpu_active_ns": duration_ns(control_active)?,
            "member_wall_ns": duration_ns(hybrid.wall)?,
            "gpu_dispatch_wall_ns": duration_ns(hybrid.gpu_wall)?,
            "gpu_active_ns": duration_ns(hybrid.gpu_active)?,
            "host_round_ns": duration_ns(hybrid.host_rounds)?,
            "readback_ns": duration_ns(hybrid.readback)?,
            "cpu_tail_ns": duration_ns(hybrid.cpu_tail)?,
            "round_0_gpu_command_wall_ns": duration_ns(round_0_wall)?,
            "round_0_gpu_command_active_ns": duration_ns(round_0_active)?,
            "round_0_nonactive_ns": duration_ns(round_0_wall.saturating_sub(round_0_active))?,
            "first_three_gpu_command_wall_ns": duration_ns(first_three_wall)?,
            "first_three_gpu_command_active_ns": duration_ns(first_three_active)?,
            "later_gpu_command_wall_ns": duration_ns(later_wall)?,
            "later_gpu_command_active_ns": duration_ns(later_active)?,
            "gpu_command_wall_ns": duration_ns_samples(&hybrid.gpu_command_wall)?,
            "gpu_command_active_ns": duration_ns_samples(&hybrid.gpu_command_active)?,
        },
        "guards": guards,
        "all_exact": all_exact,
        "resources": {
            "sequence_owned_storage_bytes": storage_bytes,
            "storage_initialization_bytes": initialization.bytes,
            "storage_initialization_device_buffers": initialization.device_buffers,
            "storage_buffer_identities": initialization.buffer_identities,
            "resident_row_identity": initial_resident_row_identity,
            "primer_source_elements": primer_stats.source_elements,
            "primer_e_in_elements": primer_stats.e_in_elements,
            "primer_e_out_elements": primer_stats.e_out_elements,
            "primer_resident_row_identity": primer_stats.resident_row_identity,
            "primer_storage_buffer_identities": primer_stats.storage_buffer_identities,
            "cutoff_readback_bytes": hybrid.preallocated_readback_bytes,
            "persistent_device_buffers": 6,
            "round_device_buffer_allocations": sequence.round_device_buffer_allocations(),
        },
        "workload": {
            "log_n": log_n,
            "rows": workload.rows(),
            "cutoff_log2": cutoff_log2,
            "cutoff_elements": cutoff,
            "tables": TABLES,
            "host_fiat_shamir": true,
            "target_sequences": 1,
            "excluded_target_warmups": 0,
            "cpu_control_before_sequence_preparation": true,
            "storage_initialization_outside_member_timer": true,
            "control_outside_member_timer": true,
        },
        "fingerprint": {
            "device": device.name,
            "max_buffer_length": device.max_buffer_length,
            "recommended_max_working_set_size": device.recommended_max_working_set_size,
            "cpu_threads": std::thread::available_parallelism()?.get(),
            "seed": seed,
            "log_n": log_n,
            "cutoff_log2": cutoff_log2,
            "native_message_threads": dispatch.native_message,
            "native_transition_threads": dispatch.native_transition,
            "dense_transition_threads": dispatch.dense_transition,
            "storage_initialization": arm.storage_initialization().as_str(),
            "control": arm.as_str(),
            "gpu_command_count": command_count,
            "process_model": "one_cold_target_sequence_per_process",
        },
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
