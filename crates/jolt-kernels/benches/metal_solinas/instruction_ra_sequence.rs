use std::{env, hint::black_box, mem::size_of, time::Duration};

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{InstructionRaMaterializeWidth, MetalError, SolinasMetal};

#[path = "../../examples/support/instruction_ra.rs"]
#[expect(
    dead_code,
    reason = "shared evaluator support includes entry points not used by this benchmark"
)]
mod support;

use support::{
    derived_eq_cycle_is_exact, expected_cpu_states, expected_hybrid_states,
    final_relation_is_exact, first_factor_only_gamma_unscale, run_cpu, run_hybrid, Capture,
    SequenceDispatch, Workload, FACTORS,
};

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const DEFAULT_CUTOFF_LOG2: usize = 10;
const DEFAULT_VALIDATION_LOG_N: usize = 12;

#[derive(Clone, Copy)]
struct DispatchCase {
    name: &'static str,
    dispatch: SequenceDispatch,
}

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let message_threads = env_usize("JOLT_METAL_INSTRUCTION_RA_THREADS", 128);
    let materialize_threads = env_usize("JOLT_METAL_INSTRUCTION_RA_MATERIALIZE_THREADS", 64);
    let cutoff_log2 = env_usize("JOLT_METAL_INSTRUCTION_RA_CUTOFF_LOG2", DEFAULT_CUTOFF_LOG2);
    let validation_log_n = env_usize(
        "JOLT_SOLINAS_BENCH_VALIDATE_LOG_N",
        DEFAULT_VALIDATION_LOG_N,
    );
    assert!((10..=20).contains(&validation_log_n));
    assert!((1..usize::BITS as usize).contains(&cutoff_log2));
    assert!(message_threads > 0 && materialize_threads > 0);

    let dispatches = dispatch_cases(message_threads, materialize_threads);
    validate_dispatches(context, &dispatches, validation_log_n, cutoff_log2);

    let mut group = c.benchmark_group("metal_sumcheck/instruction_ra_sequence_complete_rows");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");

    for rows in cases(cutoff_log2) {
        let log_n = rows.ilog2() as usize;
        let cutoff = 1usize << cutoff_log2;
        let workload = Workload::new(log_n, 1).expect("Instruction RA workload should build");
        let _ = group.throughput(Throughput::Elements(rows as u64));
        if cpu_first {
            bench_cpu(&mut group, &workload, cutoff);
        }
        for case in dispatches {
            bench_hybrid(&mut group, context, &workload, cutoff, case);
        }
        if !cpu_first {
            bench_cpu(&mut group, &workload, cutoff);
        }
    }
    group.finish();
}

fn bench_cpu(group: &mut BenchmarkGroup<'_, WallTime>, workload: &Workload, cutoff: usize) {
    let id = BenchmarkId::new(
        "cpu_w16_mirror",
        format!("n{}_cutoff{cutoff}", workload.rows()),
    );
    let _ = group.bench_function(id, |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                let output = black_box(
                    run_cpu(workload, cutoff, Capture::TARGET)
                        .expect("optimized Instruction RA CPU sequence should complete"),
                );
                measured += output.wall;
                let _ = black_box(output.trace.final_sumcheck_claim);
            }
            measured
        });
    });
}

fn bench_hybrid(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &SolinasMetal,
    workload: &Workload,
    cutoff: usize,
    case: DispatchCase,
) {
    let setup_plane = workload
        .prepare_plane(context)
        .expect("resident Instruction RA plane should prepare");
    let persistent_plane = (!case.dispatch.reuse_inverse_for_dense).then(|| setup_plane.clone());
    let mut sequence = workload
        .prepare_sequence(context, setup_plane, case.dispatch)
        .expect("Instruction RA Metal sequence should prepare");
    let id = BenchmarkId::new(
        case.name,
        format!(
            "n{}_cutoff{cutoff}_msg{}_mat{}",
            workload.rows(),
            case.dispatch.message_threads,
            case.dispatch.materialize_threads,
        ),
    );
    let _ = group.bench_function(id, |bench| {
        bench.iter_custom(|iterations| {
            // The stage-5 plane and reusable storage are resident; reset and all rounds are charged.
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                let plane = if case.dispatch.reuse_inverse_for_dense {
                    workload
                        .prepare_plane(context)
                        .expect("one-shot Instruction RA plane should prepare")
                } else {
                    persistent_plane
                        .as_ref()
                        .expect("persistent Instruction RA plane should exist")
                        .clone()
                };
                let output = black_box(
                    run_hybrid(&mut sequence, plane, workload, cutoff, Capture::TARGET)
                        .expect("Instruction RA Metal hybrid sequence should complete"),
                );
                measured += output.wall;
                let _ = black_box(output.trace.final_sumcheck_claim);
            }
            measured
        });
    });
}

fn validate_dispatches(
    context: &SolinasMetal,
    dispatches: &[DispatchCase],
    validation_log_n: usize,
    cutoff_log2: usize,
) {
    let workload = Workload::new(validation_log_n, 0xbb67_ae85_84ca_a73b)
        .expect("validation Instruction RA workload should build");
    let invalid_w16_reuse = SequenceDispatch {
        reuse_inverse_for_dense: true,
        ..dispatches[0].dispatch
    };
    assert!(matches!(
        invalid_w16_reuse.config().scratch_layout(workload.rows()),
        Err(MetalError::InvalidInstructionRaState(_))
    ));
    eprintln!(
        "instruction_ra_sequence gate=w16_reuse rows={} exact=rejected_insufficient_inverse_bytes",
        workload.rows()
    );

    for case in dispatches {
        let width = case.dispatch.materialize_width.elements();
        let dense_log_n = validation_log_n - width.ilog2() as usize;
        let validation_cutoff_log2 = cutoff_log2.min(dense_log_n.saturating_sub(4).max(1));
        let cutoff = 1usize << validation_cutoff_log2;
        let setup_plane = workload
            .prepare_plane(context)
            .expect("validation resident plane should prepare");
        let run_plane = if case.dispatch.reuse_inverse_for_dense {
            workload
                .prepare_plane(context)
                .expect("validation one-shot plane should prepare")
        } else {
            setup_plane.clone()
        };
        let mut sequence = workload
            .prepare_sequence(context, setup_plane, case.dispatch)
            .expect("validation Metal sequence should prepare");
        let capture = Capture::validation(workload.rows(), width);
        let cpu = run_cpu(&workload, cutoff, capture)
            .expect("validation CPU Instruction RA should complete");
        let hybrid = run_hybrid(&mut sequence, run_plane, &workload, cutoff, capture)
            .expect("validation Metal Instruction RA should complete");

        assert_eq!(hybrid.trace.q_evals, cpu.trace.q_evals);
        assert_eq!(hybrid.trace.round_polys, cpu.trace.round_polys);
        assert_eq!(hybrid.trace.challenges, cpu.trace.challenges);
        assert_eq!(cpu.trace.states, expected_cpu_states(validation_log_n));
        assert_eq!(
            hybrid.trace.states,
            expected_hybrid_states(validation_log_n, width, cutoff)
        );
        assert_eq!(hybrid.trace.scheduled_tables, cpu.trace.scheduled_tables);
        assert!(hybrid.trace.scheduled_tables.is_some());
        assert_eq!(hybrid.trace.cutoff_tables, cpu.trace.cutoff_tables);
        assert!(hybrid.trace.cutoff_tables.is_some());
        assert_eq!(hybrid.trace.raw_final_claims, cpu.trace.raw_final_claims);
        assert_eq!(hybrid.trace.final_claims, cpu.trace.final_claims);
        assert_eq!(hybrid.trace.final_claims.len(), FACTORS);
        assert_eq!(
            hybrid.trace.final_sumcheck_claim,
            cpu.trace.final_sumcheck_claim
        );
        assert_eq!(hybrid.trace.derived_eq_cycle, cpu.trace.derived_eq_cycle);
        assert_eq!(hybrid.trace.transcript_state, cpu.trace.transcript_state);
        assert!(first_factor_only_gamma_unscale(
            &hybrid.trace,
            workload.gamma
        ));
        assert!(derived_eq_cycle_is_exact(&workload, &hybrid.trace));
        assert!(final_relation_is_exact(&cpu.trace));
        assert!(final_relation_is_exact(&hybrid.trace));
        assert!(hybrid.resident_plane_zero_copy);
        assert!(hybrid.static_device_buffers_stable);
        assert!(hybrid.inverse_dense_b_handoff_exact);
        assert_eq!(
            hybrid.preallocated_readback_bytes,
            FACTORS * (cutoff + workload.rows() / width) * size_of::<AkitaField>()
        );
        eprintln!(
            "instruction_ra_sequence gate={} rows={} cutoff={} exact=true",
            case.name,
            workload.rows(),
            cutoff
        );
    }
}

fn dispatch_cases(message_threads: usize, materialize_threads: usize) -> [DispatchCase; 3] {
    [
        DispatchCase {
            name: "metal_w16_owned_dense_b",
            dispatch: SequenceDispatch {
                message_threads,
                materialize_threads,
                materialize_width: InstructionRaMaterializeWidth::W16,
                reuse_inverse_for_dense: false,
            },
        },
        DispatchCase {
            name: "metal_w32_owned_dense_b",
            dispatch: SequenceDispatch {
                message_threads,
                materialize_threads,
                materialize_width: InstructionRaMaterializeWidth::W32,
                reuse_inverse_for_dense: false,
            },
        },
        DispatchCase {
            name: "metal_w32_reuse_inverse",
            dispatch: SequenceDispatch {
                message_threads,
                materialize_threads,
                materialize_width: InstructionRaMaterializeWidth::W32,
                reuse_inverse_for_dense: true,
            },
        },
    ]
}

fn cases(cutoff_log2: usize) -> Vec<usize> {
    let rows = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    );
    assert!(
        rows.iter()
            .all(|rows| rows.is_power_of_two() && *rows >= (1usize << cutoff_log2) * 32),
        "Instruction RA rows must cover the W32 dense cutoff"
    );
    rows
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}
