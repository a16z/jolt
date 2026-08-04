#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use jolt_kernels::metal::solinas::{InstructionRaMaterializeWidth, SolinasMetal};
use serde_json::json;

#[path = "support/instruction_ra.rs"]
mod instruction_ra;

use instruction_ra::{
    derived_eq_cycle_is_exact, expected_cpu_states, expected_hybrid_states,
    final_relation_is_exact, first_factor_only_gamma_unscale, median, run_cpu, run_hybrid, Capture,
    EvalResult, SequenceDispatch, Trace, Workload, FACTORS,
};

#[derive(Default)]
struct Guards {
    exact_q_evals: bool,
    exact_round_polys: bool,
    exact_challenges: bool,
    cpu_schedule: bool,
    hybrid_schedule: bool,
    exact_scheduled_tables: bool,
    exact_cutoff_tables: bool,
    exact_final_claims: bool,
    exact_transcript: bool,
    first_factor_only_gamma_unscale: bool,
    exact_derived_eq_cycle: bool,
    exact_final_relation: bool,
    resident_plane_zero_copy: bool,
    static_device_buffers_stable: bool,
    inverse_dense_b_handoff_exact: bool,
    preallocated_host_readback: bool,
}

impl Guards {
    fn from_pair(
        workload: &Workload,
        cpu: &Trace,
        hybrid: &Trace,
        materialize_width: usize,
        cutoff: usize,
    ) -> Self {
        Self {
            exact_q_evals: cpu.q_evals == hybrid.q_evals,
            exact_round_polys: cpu.round_polys == hybrid.round_polys,
            exact_challenges: cpu.challenges == hybrid.challenges,
            cpu_schedule: cpu.states == expected_cpu_states(workload.log_n),
            hybrid_schedule: hybrid.states
                == expected_hybrid_states(workload.log_n, materialize_width, cutoff),
            exact_scheduled_tables: cpu.scheduled_tables == hybrid.scheduled_tables,
            exact_cutoff_tables: cpu.cutoff_tables == hybrid.cutoff_tables,
            exact_final_claims: cpu.raw_final_claims == hybrid.raw_final_claims
                && cpu.final_claims == hybrid.final_claims
                && cpu.final_claims.len() == FACTORS
                && cpu.final_sumcheck_claim == hybrid.final_sumcheck_claim,
            exact_transcript: cpu.transcript_state == hybrid.transcript_state,
            first_factor_only_gamma_unscale: first_factor_only_gamma_unscale(
                hybrid,
                workload.gamma,
            ),
            exact_derived_eq_cycle: cpu.derived_eq_cycle == hybrid.derived_eq_cycle
                && derived_eq_cycle_is_exact(workload, hybrid),
            exact_final_relation: final_relation_is_exact(cpu) && final_relation_is_exact(hybrid),
            resident_plane_zero_copy: true,
            static_device_buffers_stable: true,
            inverse_dense_b_handoff_exact: true,
            preallocated_host_readback: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.exact_q_evals &= other.exact_q_evals;
        self.exact_round_polys &= other.exact_round_polys;
        self.exact_challenges &= other.exact_challenges;
        self.cpu_schedule &= other.cpu_schedule;
        self.hybrid_schedule &= other.hybrid_schedule;
        self.exact_scheduled_tables &= other.exact_scheduled_tables;
        self.exact_cutoff_tables &= other.exact_cutoff_tables;
        self.exact_final_claims &= other.exact_final_claims;
        self.exact_transcript &= other.exact_transcript;
        self.first_factor_only_gamma_unscale &= other.first_factor_only_gamma_unscale;
        self.exact_derived_eq_cycle &= other.exact_derived_eq_cycle;
        self.exact_final_relation &= other.exact_final_relation;
        self.resident_plane_zero_copy &= other.resident_plane_zero_copy;
        self.static_device_buffers_stable &= other.static_device_buffers_stable;
        self.inverse_dense_b_handoff_exact &= other.inverse_dense_b_handoff_exact;
        self.preallocated_host_readback &= other.preallocated_host_readback;
    }

    fn all_exact(&self) -> bool {
        self.exact_q_evals
            && self.exact_round_polys
            && self.exact_challenges
            && self.cpu_schedule
            && self.hybrid_schedule
            && self.exact_scheduled_tables
            && self.exact_cutoff_tables
            && self.exact_final_claims
            && self.exact_transcript
            && self.first_factor_only_gamma_unscale
            && self.exact_derived_eq_cycle
            && self.exact_final_relation
            && self.resident_plane_zero_copy
            && self.static_device_buffers_stable
            && self.inverse_dense_b_handoff_exact
            && self.preallocated_host_readback
    }
}

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn materialize_width(value: usize) -> EvalResult<InstructionRaMaterializeWidth> {
    match value {
        16 => Ok(InstructionRaMaterializeWidth::W16),
        32 => Ok(InstructionRaMaterializeWidth::W32),
        64 => Ok(InstructionRaMaterializeWidth::W64),
        128 => Ok(InstructionRaMaterializeWidth::W128),
        256 => Ok(InstructionRaMaterializeWidth::W256),
        512 => Ok(InstructionRaMaterializeWidth::W512),
        _ => {
            Err("Instruction RA materialization width must be 16, 32, 64, 128, 256, or 512".into())
        }
    }
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 26)?;
    let validation_log_n = env_usize("JOLT_METAL_EVAL_VALIDATE_LOG_N", 12)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 3)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let cutoff_log2 = env_usize("JOLT_METAL_INSTRUCTION_RA_CUTOFF_LOG2", 10)?;
    let message_threads = env_usize("JOLT_METAL_INSTRUCTION_RA_THREADS", 128)?;
    let materialize_threads = env_usize("JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_THREADS", 64)?;
    let materialize_width = materialize_width(env_usize(
        "JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_WIDTH",
        16,
    )?)?;
    let materialize_width_elements = materialize_width.elements();
    let reuse_inverse = env_usize("JOLT_METAL_INSTRUCTION_RA_REUSE_INVERSE", 0)?;
    let materialize_log2 = materialize_width_elements.ilog2() as usize;
    if !(5..=28).contains(&log_n)
        || !(5..=20).contains(&validation_log_n)
        || validation_log_n < materialize_log2 + 1
        || cutoff_log2 < 1
        || cutoff_log2 > log_n - materialize_log2
        || repeats < 3
        || repeats.is_multiple_of(2)
        || reuse_inverse > 1
        || (reuse_inverse == 1 && materialize_width_elements == 16)
    {
        return Err("log sizes, materialization, cutoff, reuse, or repeats are outside the evaluator domain".into());
    }
    let cutoff = 1usize << cutoff_log2;
    let dispatch = SequenceDispatch {
        message_threads,
        materialize_threads,
        materialize_width,
        reuse_inverse_for_dense: reuse_inverse == 1,
    };
    let context = SolinasMetal::for_akita()?;

    let validation = Workload::new(validation_log_n, seed ^ 0xbb67_ae85_84ca_a73b)?;
    let validation_setup_plane = validation.prepare_plane(&context)?;
    let validation_run_plane = if dispatch.reuse_inverse_for_dense {
        validation.prepare_plane(&context)?
    } else {
        validation_setup_plane.clone()
    };
    let mut validation_sequence =
        validation.prepare_sequence(&context, validation_setup_plane, dispatch)?;
    let validation_dense_log2 = validation_log_n - materialize_log2;
    let validation_cutoff_log2 = cutoff_log2.min(validation_dense_log2.saturating_sub(4).max(1));
    let validation_cutoff = 1usize << validation_cutoff_log2;
    let validation_capture = Capture::validation(validation.rows(), materialize_width_elements);
    let validation_cpu = run_cpu(&validation, validation_cutoff, validation_capture)?;
    let validation_hybrid = run_hybrid(
        &mut validation_sequence,
        validation_run_plane,
        &validation,
        validation_cutoff,
        validation_capture,
    )?;
    let mut guards = Guards::from_pair(
        &validation,
        &validation_cpu.trace,
        &validation_hybrid.trace,
        materialize_width_elements,
        validation_cutoff,
    );
    guards.exact_scheduled_tables &= validation_cpu.trace.scheduled_tables.is_some();
    guards.exact_cutoff_tables &= validation_cpu.trace.cutoff_tables.is_some();
    guards.resident_plane_zero_copy &= validation_hybrid.resident_plane_zero_copy;
    guards.static_device_buffers_stable &= validation_hybrid.static_device_buffers_stable;
    guards.inverse_dense_b_handoff_exact &= validation_hybrid.inverse_dense_b_handoff_exact;
    guards.preallocated_host_readback &= validation_hybrid.preallocated_readback_bytes
        == FACTORS * (validation_cutoff + validation.rows() / materialize_width_elements) * 16;

    let mut workload = Workload::new(log_n, seed)?;
    let plane_preparation_started = Instant::now();
    let setup_plane = workload.prepare_plane(&context)?;
    let plane_preparation = plane_preparation_started.elapsed();
    let persistent_plane = (!dispatch.reuse_inverse_for_dense).then(|| setup_plane.clone());
    let sequence_preparation_started = Instant::now();
    let mut sequence = workload.prepare_sequence(&context, setup_plane, dispatch)?;
    let sequence_preparation = sequence_preparation_started.elapsed();
    if !dispatch.reuse_inverse_for_dense {
        workload.release_table_major_layout();
    }

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut hybrid_times = Vec::with_capacity(repeats);
    let mut resident_times = Vec::with_capacity(repeats);
    let mut paired_hybrid_speedups = Vec::with_capacity(repeats);
    let mut paired_resident_speedups = Vec::with_capacity(repeats);
    let mut reset_times = Vec::with_capacity(repeats);
    let mut trial_plane_times = Vec::with_capacity(repeats);
    let mut gpu_wall_times = Vec::with_capacity(repeats);
    let mut host_round_times = Vec::with_capacity(repeats);
    let mut readback_times = Vec::with_capacity(repeats);
    let mut cpu_tail_times = Vec::with_capacity(repeats);
    let mut gpu_active_total = Duration::ZERO;

    for repeat in 0..repeats {
        let trial_plane_started = Instant::now();
        let trial_plane = if dispatch.reuse_inverse_for_dense {
            workload.prepare_plane(&context)?
        } else {
            persistent_plane
                .as_ref()
                .ok_or("persistent Instruction RA plane is missing")?
                .clone()
        };
        trial_plane_times.push(trial_plane_started.elapsed());

        let (cpu, hybrid) = if repeat.is_multiple_of(2) {
            let cpu = black_box(run_cpu(&workload, cutoff, Capture::TARGET)?);
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                trial_plane,
                &workload,
                cutoff,
                Capture::TARGET,
            )?);
            (cpu, hybrid)
        } else {
            let hybrid = black_box(run_hybrid(
                &mut sequence,
                trial_plane,
                &workload,
                cutoff,
                Capture::TARGET,
            )?);
            let cpu = black_box(run_cpu(&workload, cutoff, Capture::TARGET)?);
            (cpu, hybrid)
        };

        let mut run_guards = Guards::from_pair(
            &workload,
            &cpu.trace,
            &hybrid.trace,
            materialize_width_elements,
            cutoff,
        );
        run_guards.resident_plane_zero_copy &= hybrid.resident_plane_zero_copy;
        run_guards.static_device_buffers_stable &= hybrid.static_device_buffers_stable;
        run_guards.inverse_dense_b_handoff_exact &= hybrid.inverse_dense_b_handoff_exact;
        run_guards.preallocated_host_readback &=
            hybrid.preallocated_readback_bytes == FACTORS * cutoff * 16;
        guards.merge(run_guards);
        let resident_wall = hybrid.wall.saturating_sub(hybrid.reset);
        paired_hybrid_speedups.push(cpu.wall.as_secs_f64() / hybrid.wall.as_secs_f64());
        paired_resident_speedups.push(cpu.wall.as_secs_f64() / resident_wall.as_secs_f64());
        cpu_times.push(cpu.wall);
        hybrid_times.push(hybrid.wall);
        resident_times.push(resident_wall);
        reset_times.push(hybrid.reset);
        gpu_wall_times.push(hybrid.gpu_wall);
        host_round_times.push(hybrid.host_rounds);
        readback_times.push(hybrid.readback);
        cpu_tail_times.push(hybrid.cpu_tail);
        gpu_active_total += hybrid.gpu_active;
    }

    let cpu_median = median(&mut cpu_times);
    let hybrid_median = median(&mut hybrid_times);
    let resident_median = median(&mut resident_times);
    let reset_median = median(&mut reset_times);
    let trial_plane_median = median(&mut trial_plane_times);
    let gpu_wall_median = median(&mut gpu_wall_times);
    let host_round_median = median(&mut host_round_times);
    let readback_median = median(&mut readback_times);
    let cpu_tail_median = median(&mut cpu_tail_times);
    let hybrid_speedup = median_f64(&paired_hybrid_speedups);
    let resident_speedup = median_f64(&paired_resident_speedups);
    let rows = workload.rows();
    let scratch = dispatch.config().scratch_layout(rows)?;
    let output = json!({
        "schema_version": 1,
        "kernel": "instruction_ra_virtualization",
        "metrics": {
            "hybrid_speedup": hybrid_speedup,
            "resident_speedup": resident_speedup,
            "paired_hybrid_speedups": paired_hybrid_speedups,
            "paired_resident_speedups": paired_resident_speedups,
            "cpu_million_rows_per_second": rows as f64 / cpu_median.as_secs_f64() / 1e6,
            "hybrid_million_rows_per_second": rows as f64 / hybrid_median.as_secs_f64() / 1e6
        },
        "timings": {
            "setup_lookup_plane_seconds": plane_preparation.as_secs_f64(),
            "per_trial_lookup_plane_median_seconds": trial_plane_median.as_secs_f64(),
            "sequence_storage_preparation_seconds": sequence_preparation.as_secs_f64(),
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "hybrid_median_seconds": hybrid_median.as_secs_f64(),
            "resident_median_seconds": resident_median.as_secs_f64(),
            "sequence_reset_median_seconds": reset_median.as_secs_f64(),
            "gpu_dispatch_wall_median_seconds": gpu_wall_median.as_secs_f64(),
            "host_round_median_seconds": host_round_median.as_secs_f64(),
            "readback_median_seconds": readback_median.as_secs_f64(),
            "cpu_tail_median_seconds": cpu_tail_median.as_secs_f64(),
            "gpu_active_total_seconds": gpu_active_total.as_secs_f64(),
            "repeats": repeats
        },
        "guards": {
            "exact_q_evals": guards.exact_q_evals,
            "exact_round_polys": guards.exact_round_polys,
            "exact_challenges": guards.exact_challenges,
            "cpu_w16_schedule": guards.cpu_schedule,
            "configured_hybrid_schedule": guards.hybrid_schedule,
            "exact_same_bind_scheduled_tables": guards.exact_scheduled_tables,
            "exact_cutoff_tables": guards.exact_cutoff_tables,
            "exact_final_claims": guards.exact_final_claims,
            "exact_transcript_state": guards.exact_transcript,
            "first_factor_only_gamma_unscale": guards.first_factor_only_gamma_unscale,
            "exact_derived_eq_cycle": guards.exact_derived_eq_cycle,
            "exact_final_relation": guards.exact_final_relation,
            "host_fiat_shamir": true,
            "resident_plane_zero_copy": guards.resident_plane_zero_copy,
            "static_device_buffer_identities_stable": guards.static_device_buffers_stable,
            "inverse_dense_b_handoff_exact": guards.inverse_dense_b_handoff_exact,
            "host_readback_preallocated_before_primary_timer": guards.preallocated_host_readback,
            "all_exact": guards.all_exact()
        },
        "resources": {
            "lookup_plane_bytes": 20 * rows,
            "branch_a_bytes": scratch.branch_a_bytes,
            "branch_b_bytes": scratch.branch_b_bytes,
            "dense_a_bytes": scratch.dense_a_bytes,
            "dense_b_active_bytes": scratch.dense_b_active_bytes,
            "dense_b_owned_bytes": scratch.dense_b_owned_bytes,
            "dense_b_physical_bytes": scratch.dense_b_physical_bytes,
            "modeled_branch_and_dense_owned_bytes": scratch.owned_bytes(),
            "modeled_branch_and_dense_resident_bytes_after_inverse_handoff": scratch.resident_bytes_after_handoff(),
            "scratch_layout_excludes_weight_and_reduction_buffers": true,
            "target_cutoff_readback_bytes": FACTORS * cutoff * 16,
            "gpu_seconds": gpu_active_total.as_secs_f64()
        },
        "workload": {
            "log_n": log_n,
            "rows": rows,
            "validation_log_n": validation_log_n,
            "groups": 4,
            "factors_per_group": 4,
            "chunk_bits": 8,
            "cpu_row_bytes": 40,
            "materialize_width": materialize_width_elements,
            "reuse_inverse_for_dense": dispatch.reuse_inverse_for_dense,
            "inverse_plane_lifetime": if dispatch.reuse_inverse_for_dense { "one shot; freshly prepared outside each primary trial" } else { "immutable; explicitly cloned for each reset" },
            "cutoff_log2": cutoff_log2,
            "message_threads": message_threads,
            "materialize_threads": materialize_threads,
            "host_fiat_shamir": true,
            "layout": "bit-reversed table-major with cycle inverse",
            "primary_timing": "sequence reset plus resident rounds, host Fiat-Shamir, readback, and CPU tail",
            "lookup_plane_preparation_in_primary_metric": false,
            "sequence_storage_preparation_in_primary_metric": false,
            "host_readback_allocation_in_primary_metric": false,
            "cpu_control": "standalone mirror of production LazyFoldedRa arithmetic over 40-byte InstructionCycleRow-compatible stride",
            "production_cpu_wall_match": "algorithm and row stride matched; empirical wall cross-check not run by this artifact"
        },
        "fingerprint": {
            "device": context.device_info().name,
            "max_buffer_length": context.device_info().max_buffer_length,
            "max_threadgroup_memory_length": context.device_info().max_threadgroup_memory_length,
            "cpu_threads": std::thread::available_parallelism()?.get()
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
