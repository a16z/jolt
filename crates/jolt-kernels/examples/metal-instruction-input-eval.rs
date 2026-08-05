#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]
#![recursion_limit = "256"]

use std::collections::HashSet;
use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use jolt_kernels::metal::solinas::{SolinasMetal, SpartanOuterUniskipRow};
use serde_json::json;

#[expect(
    dead_code,
    reason = "the shared support also exposes cold-residency timing fields"
)]
#[path = "support/instruction_input.rs"]
mod instruction_input;

use instruction_input::{
    derived_eq_cycle_is_exact, expected_states, final_relation_is_exact, median,
    run_actual_optimized, run_cpu, run_hybrid, Capture, EvalResult, SequenceDispatch, TimedTrace,
    Trace, Workload, TABLES,
};

const FROZEN_CPU_REFERENCE_NS: u64 = 814_395_125;
const FROZEN_CPU_REFERENCE_PROVENANCE: &str =
    "median of 25 CPU ns samples from immutable instruction-input-a2-2f87d8b6a8 at 2f87d8b6a81f1bb253c27795badc7da7baa3d0d8; compact-JSON sample SHA256 59f9946b7d1a3c05d3094528e853d2228ae5ec0d94a5dae2c63d5713a560a966";

#[derive(Clone, Copy, Debug)]
struct Guards {
    exact_q_evals: bool,
    exact_round_polys: bool,
    exact_challenges: bool,
    exact_round_schedule: bool,
    exact_cutoff_tables: bool,
    exact_final_claims: bool,
    exact_final_sumcheck_claim: bool,
    exact_transcript: bool,
    exact_derived_eq_cycle: bool,
    exact_final_relation: bool,
    actual_optimized_cpu_parity: bool,
    resident_rows_stable: bool,
    static_device_buffers_stable: bool,
    one_dense_readback: bool,
    preallocated_host_readback: bool,
    distinct_protocol_tapes: bool,
    protocol_retarget_reuses_cpu_rows: bool,
    production_trace_cutoff_admits_target: bool,
    raw_timing_relations: bool,
}

impl Guards {
    fn from_pair(
        workload: &Workload,
        cpu: &Trace,
        hybrid: &Trace,
        hybrid_run: &TimedTrace,
        cutoff: usize,
    ) -> Self {
        let schedule = expected_states(workload.log_n);
        Self {
            exact_q_evals: cpu.q_evals == hybrid.q_evals,
            exact_round_polys: cpu.round_polys == hybrid.round_polys,
            exact_challenges: cpu.challenges == hybrid.challenges,
            exact_round_schedule: cpu.states == schedule && hybrid.states == schedule,
            exact_cutoff_tables: cpu.cutoff_tables == hybrid.cutoff_tables,
            exact_final_claims: cpu.final_claims == hybrid.final_claims,
            exact_final_sumcheck_claim: cpu.final_sumcheck_claim == hybrid.final_sumcheck_claim,
            exact_transcript: cpu.transcript_state == hybrid.transcript_state,
            exact_derived_eq_cycle: cpu.derived_eq_cycle == hybrid.derived_eq_cycle
                && derived_eq_cycle_is_exact(workload, cpu)
                && derived_eq_cycle_is_exact(workload, hybrid),
            exact_final_relation: final_relation_is_exact(workload, cpu)
                && final_relation_is_exact(workload, hybrid),
            actual_optimized_cpu_parity: true,
            resident_rows_stable: hybrid_run.resident_rows_stable,
            static_device_buffers_stable: hybrid_run.static_device_buffers_stable,
            one_dense_readback: hybrid_run.readbacks == 1,
            preallocated_host_readback: hybrid_run.preallocated_readback_bytes
                == TABLES * cutoff * 16,
            distinct_protocol_tapes: true,
            protocol_retarget_reuses_cpu_rows: true,
            production_trace_cutoff_admits_target: true,
            raw_timing_relations: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.exact_q_evals &= other.exact_q_evals;
        self.exact_round_polys &= other.exact_round_polys;
        self.exact_challenges &= other.exact_challenges;
        self.exact_round_schedule &= other.exact_round_schedule;
        self.exact_cutoff_tables &= other.exact_cutoff_tables;
        self.exact_final_claims &= other.exact_final_claims;
        self.exact_final_sumcheck_claim &= other.exact_final_sumcheck_claim;
        self.exact_transcript &= other.exact_transcript;
        self.exact_derived_eq_cycle &= other.exact_derived_eq_cycle;
        self.exact_final_relation &= other.exact_final_relation;
        self.actual_optimized_cpu_parity &= other.actual_optimized_cpu_parity;
        self.resident_rows_stable &= other.resident_rows_stable;
        self.static_device_buffers_stable &= other.static_device_buffers_stable;
        self.one_dense_readback &= other.one_dense_readback;
        self.preallocated_host_readback &= other.preallocated_host_readback;
        self.distinct_protocol_tapes &= other.distinct_protocol_tapes;
        self.protocol_retarget_reuses_cpu_rows &= other.protocol_retarget_reuses_cpu_rows;
        self.production_trace_cutoff_admits_target &= other.production_trace_cutoff_admits_target;
        self.raw_timing_relations &= other.raw_timing_relations;
    }

    fn all_exact(self) -> bool {
        self.exact_q_evals
            && self.exact_round_polys
            && self.exact_challenges
            && self.exact_round_schedule
            && self.exact_cutoff_tables
            && self.exact_final_claims
            && self.exact_final_sumcheck_claim
            && self.exact_transcript
            && self.exact_derived_eq_cycle
            && self.exact_final_relation
            && self.actual_optimized_cpu_parity
            && self.resident_rows_stable
            && self.static_device_buffers_stable
            && self.one_dense_readback
            && self.preallocated_host_readback
            && self.distinct_protocol_tapes
            && self.protocol_retarget_reuses_cpu_rows
            && self.production_trace_cutoff_admits_target
            && self.raw_timing_relations
    }
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn protocol_seed(seed: u64, repeat: usize) -> u64 {
    seed ^ 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(repeat as u64 + 1)
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 26)?;
    let validation_log_n = env_usize("JOLT_METAL_EVAL_VALIDATE_LOG_N", 12)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 5)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let frozen_cpu_reference_ns = env_usize(
        "JOLT_METAL_EVAL_CPU_REFERENCE_NS",
        FROZEN_CPU_REFERENCE_NS as usize,
    )? as u64;
    let cutoff_log2 = env_usize("JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2", 16)?;
    let trace_cutoff_log2 = env_usize("JOLT_METAL_INSTRUCTION_INPUT_TRACE_CUTOFF_LOG2", 25)?;
    let native_message_threads =
        env_usize("JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS", 256)?;
    let native_transition_threads = env_usize(
        "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS",
        128,
    )?;
    let dense_transition_threads =
        env_usize("JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS", 128)?;
    if !(3..=28).contains(&log_n)
        || !(3..=20).contains(&validation_log_n)
        || cutoff_log2 < 1
        || cutoff_log2 >= log_n
        || trace_cutoff_log2 > log_n
        || repeats < 5
        || repeats.is_multiple_of(2)
        || frozen_cpu_reference_ns != FROZEN_CPU_REFERENCE_NS
    {
        return Err("log sizes, cutoff, or repeats are outside the evaluator domain".into());
    }
    let cutoff = 1usize << cutoff_log2;
    let frozen_cpu_reference = Duration::from_nanos(frozen_cpu_reference_ns);
    let validation_cutoff_log2 = cutoff_log2.min(validation_log_n - 1);
    let validation_cutoff = 1usize << validation_cutoff_log2;
    let dispatch = SequenceDispatch {
        native_message: native_message_threads,
        native_transition: native_transition_threads,
        dense_transition: dense_transition_threads,
    };
    let context = SolinasMetal::for_akita()?;

    let mut validation = Workload::new(validation_log_n, seed ^ 0xbb67_ae85_84ca_a73b)?;
    let mut validation_sequence = validation.prepare_sequence(&context, dispatch)?;
    let validation_cpu = run_cpu(&validation, validation_cutoff, Capture::VALIDATION)?;
    let validation_actual = run_actual_optimized(&validation)?;
    let mut validation_full_sequence_metal_runs = 0usize;
    let validation_hybrid = run_hybrid(
        &mut validation_sequence,
        &validation,
        validation_cutoff,
        Capture::VALIDATION,
    )?;
    validation_full_sequence_metal_runs += 1;
    let mut guards = Guards::from_pair(
        &validation,
        &validation_cpu.trace,
        &validation_hybrid.trace,
        &validation_hybrid,
        validation_cutoff,
    );
    guards.exact_cutoff_tables &= validation_cpu.trace.cutoff_tables.is_some()
        && validation_hybrid.trace.cutoff_tables.is_some();
    guards.actual_optimized_cpu_parity &= validation_cpu.trace.round_polys
        == validation_actual.round_polys
        && validation_cpu.trace.challenges == validation_actual.challenges
        && validation_cpu.trace.final_claims == validation_actual.final_claims
        && validation_cpu.trace.final_sumcheck_claim == validation_actual.final_sumcheck_claim
        && validation_cpu.trace.transcript_state == validation_actual.transcript_state;
    let validation_gpu_active = validation_hybrid.gpu_active;
    drop(validation_sequence);
    drop(validation);

    let workload_preparation_started = Instant::now();
    let mut workload = Workload::new(log_n, seed)?;
    let protocol_seeds = (0..repeats)
        .map(|repeat| protocol_seed(seed, repeat))
        .collect::<Vec<_>>();
    let trial_workloads = protocol_seeds
        .iter()
        .map(|trial_seed| workload.retarget(*trial_seed))
        .collect::<EvalResult<Vec<_>>>()?;
    let cpu_rows_identity = workload.cpu_rows_identity();
    guards.protocol_retarget_reuses_cpu_rows &= trial_workloads
        .iter()
        .all(|trial| trial.cpu_rows_identity() == cpu_rows_identity);
    let workload_preparation = workload_preparation_started.elapsed();

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut hybrid_times = Vec::with_capacity(repeats);
    let mut resident_times = Vec::with_capacity(repeats);
    let mut reset_times = Vec::with_capacity(repeats);
    let mut gpu_wall_times = Vec::with_capacity(repeats);
    let mut host_round_times = Vec::with_capacity(repeats);
    let mut readback_times = Vec::with_capacity(repeats);
    let mut cpu_tail_times = Vec::with_capacity(repeats);
    let mut gpu_active_times = Vec::with_capacity(repeats);
    let mut paired_hybrid_speedups = Vec::with_capacity(repeats);
    let mut paired_resident_speedups = Vec::with_capacity(repeats);
    let mut paired_frozen_cpu_reference_ratios = Vec::with_capacity(repeats);
    let mut timed_gpu_active_total = Duration::ZERO;
    let mut protocol_tapes = HashSet::with_capacity(repeats);
    let mut protocol_transcript_states = Vec::with_capacity(repeats);

    let cpu_results = trial_workloads
        .iter()
        .map(|trial_workload| black_box(run_cpu(trial_workload, cutoff, Capture::TARGET)))
        .collect::<EvalResult<Vec<_>>>()?;
    let sequence_preparation_started = Instant::now();
    let mut sequence = workload.prepare_sequence(&context, dispatch)?;
    let sequence_preparation = sequence_preparation_started.elapsed();
    let sequence_owned_bytes = sequence.owned_storage_bytes();
    let native_message_limits = sequence.native_message_pipeline_limits();
    let native_transition_limits = sequence.native_transition_pipeline_limits();
    let dense_transition_limits = sequence.dense_transition_pipeline_limits();
    // A target-size CPU pass adds 170--250 ms before the next GPU timestamp. Warm
    // the exact sequence once so the search metric ranks steady-state shader work.
    let mut residency_warmup_runs = 0usize;
    let residency_warmup = black_box(run_hybrid(
        &mut sequence,
        &trial_workloads[0],
        cutoff,
        Capture::TARGET,
    )?);
    residency_warmup_runs += 1;
    guards.merge(Guards::from_pair(
        &trial_workloads[0],
        &cpu_results[0].trace,
        &residency_warmup.trace,
        &residency_warmup,
        cutoff,
    ));
    let mut hybrid_results = Vec::with_capacity(repeats);
    let mut timed_full_sequence_metal_runs = 0usize;
    for trial_workload in &trial_workloads {
        let result = black_box(run_hybrid(
            &mut sequence,
            trial_workload,
            cutoff,
            Capture::TARGET,
        )?);
        timed_full_sequence_metal_runs += 1;
        hybrid_results.push(result);
    }

    for ((trial_workload, cpu), hybrid) in
        trial_workloads.iter().zip(cpu_results).zip(hybrid_results)
    {
        guards.merge(Guards::from_pair(
            trial_workload,
            &cpu.trace,
            &hybrid.trace,
            &hybrid,
            cutoff,
        ));
        let _ = protocol_tapes.insert(cpu.trace.transcript_state);
        protocol_transcript_states.push(cpu.trace.transcript_state);
        let resident_wall = hybrid.wall.saturating_sub(hybrid.reset);
        paired_hybrid_speedups.push(cpu.wall.as_secs_f64() / hybrid.wall.as_secs_f64());
        paired_resident_speedups.push(cpu.wall.as_secs_f64() / resident_wall.as_secs_f64());
        paired_frozen_cpu_reference_ratios
            .push(frozen_cpu_reference.as_secs_f64() / hybrid.wall.as_secs_f64());
        cpu_times.push(cpu.wall);
        hybrid_times.push(hybrid.wall);
        resident_times.push(resident_wall);
        reset_times.push(hybrid.reset);
        gpu_wall_times.push(hybrid.gpu_wall);
        host_round_times.push(hybrid.host_rounds);
        readback_times.push(hybrid.readback);
        cpu_tail_times.push(hybrid.cpu_tail);
        gpu_active_times.push(hybrid.gpu_active);
        timed_gpu_active_total += hybrid.gpu_active;
    }
    guards.distinct_protocol_tapes &= protocol_tapes.len() == repeats
        && protocol_seeds.iter().copied().collect::<HashSet<_>>().len() == repeats;
    guards.production_trace_cutoff_admits_target &=
        (1usize << log_n) >= (1usize << trace_cutoff_log2);
    guards.raw_timing_relations &= (0..repeats).all(|index| {
        let accounted = gpu_wall_times[index]
            .checked_add(host_round_times[index])
            .and_then(|value| value.checked_add(readback_times[index]))
            .and_then(|value| value.checked_add(cpu_tail_times[index]));
        hybrid_times[index].checked_sub(reset_times[index]) == Some(resident_times[index])
            && accounted.is_some_and(|value| value <= resident_times[index])
            && gpu_active_times[index] <= gpu_wall_times[index]
    });
    let residency_warmup_resident_wall = residency_warmup
        .wall
        .checked_sub(residency_warmup.reset)
        .ok_or("InstructionInput warmup reset exceeds wall time")?;
    let residency_warmup_accounted = residency_warmup
        .gpu_wall
        .checked_add(residency_warmup.host_rounds)
        .and_then(|value| value.checked_add(residency_warmup.readback))
        .and_then(|value| value.checked_add(residency_warmup.cpu_tail));
    guards.raw_timing_relations &= residency_warmup_accounted
        .is_some_and(|value| value <= residency_warmup_resident_wall)
        && residency_warmup.gpu_active <= residency_warmup.gpu_wall;

    let cpu_median = median(&cpu_times);
    let hybrid_median = median(&hybrid_times);
    let resident_median = median(&resident_times);
    let reset_median = median(&reset_times);
    let gpu_wall_median = median(&gpu_wall_times);
    let host_round_median = median(&host_round_times);
    let readback_median = median(&readback_times);
    let cpu_tail_median = median(&cpu_tail_times);
    let hybrid_speedup = median_f64(&paired_hybrid_speedups);
    let resident_speedup = median_f64(&paired_resident_speedups);
    let frozen_cpu_reference_ratio = median_f64(&paired_frozen_cpu_reference_ratios);
    let rows = workload.rows();
    let cpu_row_bytes = size_of::<instruction_input::CpuInstructionInputRow>() * rows;
    let resident_row_bytes = size_of::<SpartanOuterUniskipRow>() * rows;
    let metal_phase_persistent_modeled_bytes = cpu_row_bytes
        .checked_add(resident_row_bytes)
        .and_then(|value| value.checked_add(sequence_owned_bytes as usize))
        .ok_or("InstructionInput resource accounting overflow")?;
    let cpu_first_dense_bytes = TABLES * (rows / 2) * 16;
    let cpu_bind_scratch_capacity_bytes = (rows / 4) * 16;
    let cpu_trial_peak_modeled_bytes = cpu_row_bytes
        .checked_add(cpu_first_dense_bytes)
        .and_then(|value| value.checked_add(cpu_bind_scratch_capacity_bytes))
        .ok_or("InstructionInput CPU peak accounting overflow")?;
    let hybrid_tail_allocated_bytes = 2 * TABLES * cutoff * 16;
    let hybrid_cpu_tail_bind_scratch_capacity_bytes = (cutoff / 2) * 16;
    let metal_warmup_and_trial_peak_modeled_bytes = metal_phase_persistent_modeled_bytes
        .checked_add(hybrid_tail_allocated_bytes)
        .and_then(|value| value.checked_add(hybrid_cpu_tail_bind_scratch_capacity_bytes))
        .ok_or("InstructionInput hybrid peak accounting overflow")?;
    let sequence_setup_peak_modeled_bytes = metal_phase_persistent_modeled_bytes
        .checked_add(resident_row_bytes)
        .ok_or("InstructionInput setup peak accounting overflow")?;
    let evaluator_peak_modeled_bytes = cpu_trial_peak_modeled_bytes
        .max(sequence_setup_peak_modeled_bytes)
        .max(metal_warmup_and_trial_peak_modeled_bytes);
    let device_info = context.device_info();
    let cpu_ns_samples = duration_ns_samples(&cpu_times)?;
    let hybrid_ns_samples = duration_ns_samples(&hybrid_times)?;
    let resident_ns_samples = duration_ns_samples(&resident_times)?;
    let reset_ns_samples = duration_ns_samples(&reset_times)?;
    let gpu_wall_ns_samples = duration_ns_samples(&gpu_wall_times)?;
    let host_round_ns_samples = duration_ns_samples(&host_round_times)?;
    let readback_ns_samples = duration_ns_samples(&readback_times)?;
    let cpu_tail_ns_samples = duration_ns_samples(&cpu_tail_times)?;
    let gpu_active_ns_samples = duration_ns_samples(&gpu_active_times)?;
    let validation_gpu_active_ns = duration_ns(validation_gpu_active)?;
    let residency_warmup_wall_ns = duration_ns(residency_warmup.wall)?;
    let residency_warmup_resident_ns = duration_ns(residency_warmup_resident_wall)?;
    let residency_warmup_reset_ns = duration_ns(residency_warmup.reset)?;
    let residency_warmup_gpu_wall_ns = duration_ns(residency_warmup.gpu_wall)?;
    let residency_warmup_host_round_ns = duration_ns(residency_warmup.host_rounds)?;
    let residency_warmup_readback_ns = duration_ns(residency_warmup.readback)?;
    let residency_warmup_cpu_tail_ns = duration_ns(residency_warmup.cpu_tail)?;
    let residency_warmup_gpu_active_ns = duration_ns(residency_warmup.gpu_active)?;
    let evaluator_gpu_active_total = validation_gpu_active
        .checked_add(residency_warmup.gpu_active)
        .and_then(|value| value.checked_add(timed_gpu_active_total))
        .ok_or("InstructionInput evaluator GPU time overflow")?;
    let evaluator_full_sequence_metal_runs = validation_full_sequence_metal_runs
        + residency_warmup_runs
        + timed_full_sequence_metal_runs;
    let round_device_buffer_allocations_zero = sequence.round_device_buffer_allocations() == 0;
    let exactly_one_excluded_residency_warmup = validation_full_sequence_metal_runs == 1
        && residency_warmup_runs == 1
        && timed_full_sequence_metal_runs == repeats
        && evaluator_full_sequence_metal_runs == repeats + 2;
    let all_exact = guards.all_exact()
        && round_device_buffer_allocations_zero
        && exactly_one_excluded_residency_warmup;
    if !all_exact {
        return Err("InstructionInput evaluator correctness guard failed".into());
    }

    // `instruction_input_v3` is a closed schema: every emitted field is declared here,
    // with no extension/property map whose keys vary between runs.
    let output = json!({
        "schema": "instruction_input_v3",
        "schema_version": 3,
        "kernel": "instruction_input",
        "metrics": {
            "hybrid_speedup": hybrid_speedup,
            "resident_speedup": resident_speedup,
            "frozen_cpu_reference_ratio": frozen_cpu_reference_ratio,
            "paired_hybrid_speedups": paired_hybrid_speedups,
            "paired_resident_speedups": paired_resident_speedups,
            "paired_frozen_cpu_reference_ratios": paired_frozen_cpu_reference_ratios,
            "cpu_ns_samples": cpu_ns_samples,
            "hybrid_ns_samples": hybrid_ns_samples,
            "resident_ns_samples": resident_ns_samples,
            "cpu_million_rows_per_second": rows as f64 / cpu_median.as_secs_f64() / 1e6,
            "hybrid_million_rows_per_second": rows as f64 / hybrid_median.as_secs_f64() / 1e6
        },
        "timings": {
            "workload_and_protocol_preparation_seconds": workload_preparation.as_secs_f64(),
            "resident_source_sequence_upload_and_storage_preparation_seconds": sequence_preparation.as_secs_f64(),
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "hybrid_median_seconds": hybrid_median.as_secs_f64(),
            "resident_median_seconds": resident_median.as_secs_f64(),
            "sequence_reset_median_seconds": reset_median.as_secs_f64(),
            "gpu_dispatch_wall_median_seconds": gpu_wall_median.as_secs_f64(),
            "host_round_median_seconds": host_round_median.as_secs_f64(),
            "readback_median_seconds": readback_median.as_secs_f64(),
            "cpu_tail_median_seconds": cpu_tail_median.as_secs_f64(),
            "timed_gpu_active_total_seconds": timed_gpu_active_total.as_secs_f64(),
            "evaluator_gpu_active_total_seconds": evaluator_gpu_active_total.as_secs_f64(),
            "validation_gpu_active_ns": validation_gpu_active_ns,
            "residency_warmup_wall_ns": residency_warmup_wall_ns,
            "residency_warmup_resident_ns": residency_warmup_resident_ns,
            "residency_warmup_reset_ns": residency_warmup_reset_ns,
            "residency_warmup_gpu_dispatch_wall_ns": residency_warmup_gpu_wall_ns,
            "residency_warmup_host_round_ns": residency_warmup_host_round_ns,
            "residency_warmup_readback_ns": residency_warmup_readback_ns,
            "residency_warmup_cpu_tail_ns": residency_warmup_cpu_tail_ns,
            "residency_warmup_gpu_active_ns": residency_warmup_gpu_active_ns,
            "residency_warmup_to_timed_gpu_active_ratio": residency_warmup.gpu_active.as_secs_f64() / median(&gpu_active_times).as_secs_f64(),
            "sequence_reset_ns_samples": reset_ns_samples,
            "gpu_dispatch_wall_ns_samples": gpu_wall_ns_samples,
            "host_round_ns_samples": host_round_ns_samples,
            "readback_ns_samples": readback_ns_samples,
            "cpu_tail_ns_samples": cpu_tail_ns_samples,
            "gpu_active_ns_samples": gpu_active_ns_samples,
            "repeats": repeats
        },
        "guards": {
            "exact_four_sample_q_evals": guards.exact_q_evals,
            "exact_round_polynomials": guards.exact_round_polys,
            "exact_host_fiat_shamir_challenges": guards.exact_challenges,
            "exact_round_schedule": guards.exact_round_schedule,
            "exact_cutoff_tables": guards.exact_cutoff_tables,
            "exact_final_eight_claims": guards.exact_final_claims,
            "exact_final_sumcheck_claim": guards.exact_final_sumcheck_claim,
            "exact_transcript_state": guards.exact_transcript,
            "exact_derived_eq_cycle": guards.exact_derived_eq_cycle,
            "exact_final_relation": guards.exact_final_relation,
            "actual_optimized_cpu_validation_parity": guards.actual_optimized_cpu_parity,
            "resident_rows_stable_across_reset": guards.resident_rows_stable,
            "static_device_buffer_identities_stable": guards.static_device_buffers_stable,
            "exactly_one_dense_readback": guards.one_dense_readback,
            "host_readback_preallocated_before_primary_timer": guards.preallocated_host_readback,
            "distinct_protocol_tapes": guards.distinct_protocol_tapes,
            "protocol_retarget_reuses_cpu_rows": guards.protocol_retarget_reuses_cpu_rows,
            "production_trace_cutoff_admits_target": guards.production_trace_cutoff_admits_target,
            "raw_timing_relations": guards.raw_timing_relations,
            "round_device_buffer_allocations_zero": round_device_buffer_allocations_zero,
            "host_fiat_shamir": true,
            "cpu_tail_uses_exact_four_samples": true,
            "exactly_one_excluded_residency_warmup": exactly_one_excluded_residency_warmup,
            "all_exact": all_exact
        },
        "resources": {
            "gpu_seconds": evaluator_gpu_active_total.as_secs_f64(),
            "cpu_native_rows_bytes": cpu_row_bytes,
            "resident_stage1_rows_bytes": resident_row_bytes,
            "sequence_owned_working_storage_bytes": sequence_owned_bytes,
            "cpu_phase_persistent_modeled_bytes": cpu_row_bytes,
            "cpu_first_dense_table_bytes": cpu_first_dense_bytes,
            "cpu_bind_scratch_capacity_bytes": cpu_bind_scratch_capacity_bytes,
            "cpu_trial_peak_modeled_bytes": cpu_trial_peak_modeled_bytes,
            "metal_phase_persistent_modeled_bytes": metal_phase_persistent_modeled_bytes,
            "hybrid_readback_plus_tail_table_capacity_bytes": hybrid_tail_allocated_bytes,
            "hybrid_cpu_tail_bind_scratch_capacity_bytes": hybrid_cpu_tail_bind_scratch_capacity_bytes,
            "metal_warmup_and_trial_peak_modeled_bytes": metal_warmup_and_trial_peak_modeled_bytes,
            "sequence_setup_peak_modeled_bytes": sequence_setup_peak_modeled_bytes,
            "evaluator_peak_modeled_bytes": evaluator_peak_modeled_bytes,
            "resident_source_host_copy_bytes_dropped_before_metal_trials": resident_row_bytes,
            "setup_peak_increment_from_resident_source_copy_bytes": resident_row_bytes,
            "cutoff_readback_bytes": TABLES * cutoff * 16,
            "unified_memory_no_per_round_row_upload": true,
            "sequence_owned_storage_includes_dense_ping_pong_weights_and_reductions": true
        },
        "workload": {
            "log_n": log_n,
            "rows": rows,
            "validation_log_n": validation_log_n,
            "tables": TABLES,
            "samples_per_round": 4,
            "descriptor_fields_returned_by_gpu": 3,
            "cpu_native_row_bytes": size_of::<instruction_input::CpuInstructionInputRow>(),
            "resident_stage1_row_bytes": size_of::<SpartanOuterUniskipRow>(),
            "cutoff_log2": cutoff_log2,
            "cutoff_elements": cutoff,
            "trace_cutoff_log2": trace_cutoff_log2,
            "trace_cutoff_elements": 1usize << trace_cutoff_log2,
            "native_message_threads": native_message_threads,
            "native_transition_threads": native_transition_threads,
            "dense_transition_threads": dense_transition_threads,
            "host_fiat_shamir": true,
            "primary_timing": "after one excluded full-sequence residency warmup: resident sequence reset plus Metal rounds, host Fiat-Shamir, one dense readback, and exact four-sample CPU tail",
            "primary_metric": "timed complete-member throughput normalized by a frozen CPU reference",
            "frozen_cpu_reference_ns": frozen_cpu_reference_ns,
            "frozen_cpu_reference_provenance": FROZEN_CPU_REFERENCE_PROVENANCE,
            "live_cpu_controls_in_primary_metric": false,
            "workload_preparation_in_primary_metric": false,
            "sequence_preparation_in_primary_metric": false,
            "resident_source_materialization_in_primary_metric": false,
            "residency_warmup_in_primary_metric": false,
            "residency_warmup_reuses_first_protocol_tape": true,
            "residency_warmup_runs": 1,
            "host_readback_allocation_in_primary_metric": false,
            "protocol_tape_preparation_in_primary_metric": false,
            "protocol_tapes_per_process": repeats,
            "protocol_tape_derivation": "base_seed xor ((repeat + 1) * 0x9e3779b97f4a7c15 modulo 2^64)",
            "cpu_trials_run_while_resident_metal_sequence_is_allocated": false,
            "cpu_trials_run_before_resident_source_materialization": true,
            "cpu_control": "standalone row-stride and arithmetic mirror of OptimizedInstructionInputKernel",
            "metal_control": "public InstructionInputSequence over resident SpartanOuterUniskipRow storage"
        },
        "pipelines": {
            "native_message_execution_width": native_message_limits.thread_execution_width,
            "native_message_max_threads": native_message_limits.max_total_threads_per_threadgroup,
            "native_transition_execution_width": native_transition_limits.thread_execution_width,
            "native_transition_max_threads": native_transition_limits.max_total_threads_per_threadgroup,
            "dense_transition_execution_width": dense_transition_limits.thread_execution_width,
            "dense_transition_max_threads": dense_transition_limits.max_total_threads_per_threadgroup
        },
        "fingerprint": {
            "device": device_info.name,
            "max_buffer_length": device_info.max_buffer_length,
            "recommended_max_working_set_size": device_info.recommended_max_working_set_size,
            "current_allocated_size": device_info.current_allocated_size,
            "cpu_threads": std::thread::available_parallelism()?.get(),
            "log_n": log_n,
            "validation_log_n": validation_log_n,
            "repeats": repeats,
            "seed": seed,
            "frozen_cpu_reference_ns": frozen_cpu_reference_ns,
            "protocol_seeds": protocol_seeds,
            "protocol_transcript_states": protocol_transcript_states,
            "cutoff_log2": cutoff_log2,
            "trace_cutoff_log2": trace_cutoff_log2,
            "native_message_threads": native_message_threads,
            "native_transition_threads": native_transition_threads,
            "dense_transition_threads": dense_transition_threads,
            "arm_schedule": ["cpu_batch", "excluded_full_metal_warmup", "metal_timed_batch"],
            "process_model": "single_process_steady_state_search_proxy",
            "warmup_tape_index": 0,
            "validation_full_sequence_metal_runs": validation_full_sequence_metal_runs,
            "residency_warmup_runs": residency_warmup_runs,
            "timed_full_sequence_metal_runs": timed_full_sequence_metal_runs,
            "evaluator_full_sequence_metal_runs": evaluator_full_sequence_metal_runs
        }
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn median_f64(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable_by(f64::total_cmp);
    ordered[ordered.len() / 2]
}

fn duration_ns_samples(values: &[Duration]) -> EvalResult<Vec<u64>> {
    values.iter().copied().map(duration_ns).collect()
}

fn duration_ns(value: Duration) -> EvalResult<u64> {
    u64::try_from(value.as_nanos())
        .map_err(|_| "InstructionInput duration exceeds u64 nanoseconds".into())
}
