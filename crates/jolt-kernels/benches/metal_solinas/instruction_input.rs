use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    InstructionInputSequence, SolinasMetal, INSTRUCTION_INPUT_TABLES,
};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

#[path = "../../examples/support/instruction_input.rs"]
#[expect(
    dead_code,
    reason = "shared evaluator support includes entry points not used by this benchmark"
)]
mod support;

use support::{
    cpu_native_bind_and_message_preallocated, cpu_native_q_evals, descriptor_grid,
    run_actual_optimized, run_cpu, run_hybrid, run_hybrid_with_readback, Capture, SequenceDispatch,
    Workload,
};

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const DEFAULT_CUTOFF_LOG2: usize = 16;
const DEFAULT_VALIDATION_LOG_N: usize = 12;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);
    bench_message_cases(c, context);
    bench_transition_cases(c, context);
    bench_service_cases(c, context);
}

pub fn bench_message(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);
    bench_message_cases(c, context);
}

pub fn bench_transition(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);
    bench_transition_cases(c, context);
}

pub fn bench_service(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);
    bench_service_cases(c, context);
}

fn bench_message_cases(c: &mut Criterion, context: &SolinasMetal) {
    let dispatch = dispatch();
    let mut group = comparison_group(c, "metal_sumcheck/instruction_input_native_message");
    for rows in cases() {
        let (workload, mut sequence) = prepare_case(context, rows, dispatch, 1);
        let gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
        let e_in = gruen.e_in_current().to_vec();
        let e_out = gruen.e_out_current().to_vec();
        let expected = cpu_native_q_evals(&workload.cpu_rows, &gruen, workload.gamma);
        assert_eq!(
            descriptor_grid(
                sequence
                    .message(workload.gamma, &e_in, &e_out)
                    .expect("InstructionInput validation message should run")
            ),
            expected
        );
        sequence.reset();

        let _ = group.throughput(Throughput::Elements(message_useful_muls(rows, e_out.len())));
        if cpu_first() {
            bench_message_cpu(&mut group, &workload, &gruen, rows);
            bench_message_metal(
                &mut group,
                &mut sequence,
                &workload,
                &e_in,
                &e_out,
                rows,
                dispatch.native_message,
            );
        } else {
            bench_message_metal(
                &mut group,
                &mut sequence,
                &workload,
                &e_in,
                &e_out,
                rows,
                dispatch.native_message,
            );
            bench_message_cpu(&mut group, &workload, &gruen, rows);
        }
    }
    group.finish();
}

fn bench_message_cpu(
    group: &mut BenchmarkGroup<'_, WallTime>,
    workload: &Workload,
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    rows: usize,
) {
    let _ = group.bench_function(
        BenchmarkId::new("cpu_preallocated", format!("n{rows}")),
        |bench| {
            bench.iter(|| {
                black_box(cpu_native_q_evals(
                    black_box(&workload.cpu_rows),
                    gruen,
                    workload.gamma,
                ))
            });
        },
    );
}

fn bench_message_metal(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sequence: &mut InstructionInputSequence,
    workload: &Workload,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
    rows: usize,
    threads: usize,
) {
    let suffix = format!("n{rows}_tg{threads}");
    let _ = group.bench_function(BenchmarkId::new("metal_wall_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                sequence.reset();
                let started = Instant::now();
                let output = sequence
                    .message(workload.gamma, e_in, e_out)
                    .expect("InstructionInput Metal message should run");
                measured += started.elapsed();
                let _ = black_box(output);
            }
            measured
        });
    });
    let _ = group.bench_function(BenchmarkId::new("metal_active_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                sequence.reset();
                let before = sequence.gpu_active_time();
                let output = sequence
                    .message(workload.gamma, e_in, e_out)
                    .expect("InstructionInput Metal message should run");
                measured += sequence
                    .gpu_active_time()
                    .checked_sub(before)
                    .expect("InstructionInput GPU-active time should be monotonic");
                let _ = black_box(output);
            }
            measured
        });
    });
}

fn bench_transition_cases(c: &mut Criterion, context: &SolinasMetal) {
    let dispatch = dispatch();
    let mut group = comparison_group(c, "metal_sumcheck/instruction_input_native_transition");
    for rows in cases() {
        let (workload, mut sequence) = prepare_case(context, rows, dispatch, 2);
        let challenge = -AkitaField::from_u64(0x9e37_79b9);
        let initial_gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
        let initial_e_in = initial_gruen.e_in_current().to_vec();
        let initial_e_out = initial_gruen.e_out_current().to_vec();
        let mut bound_gruen = initial_gruen.clone();
        bound_gruen.bind(challenge);
        let bound_e_in = bound_gruen.e_in_current().to_vec();
        let bound_e_out = bound_gruen.e_out_current().to_vec();
        let mut cpu_tables = vec![AkitaField::zero(); INSTRUCTION_INPUT_TABLES * rows / 2];
        let expected = cpu_native_bind_and_message_preallocated(
            &workload.cpu_rows,
            challenge,
            &bound_gruen,
            workload.gamma,
            &mut cpu_tables,
        );
        let _ = sequence
            .message(workload.gamma, &initial_e_in, &initial_e_out)
            .expect("InstructionInput setup message should run");
        assert_eq!(
            descriptor_grid(
                sequence
                    .bind_and_message(challenge, workload.gamma, &bound_e_in, &bound_e_out)
                    .expect("InstructionInput validation transition should run")
            ),
            expected
        );
        sequence.reset();

        let _ = group.throughput(Throughput::Elements(transition_useful_muls(
            rows,
            bound_e_out.len(),
        )));
        if cpu_first() {
            bench_transition_cpu(
                &mut group,
                &workload,
                challenge,
                &bound_gruen,
                &mut cpu_tables,
                rows,
            );
            bench_transition_metal(
                &mut group,
                &mut sequence,
                &workload,
                challenge,
                &initial_e_in,
                &initial_e_out,
                &bound_e_in,
                &bound_e_out,
                rows,
                dispatch.native_transition,
            );
        } else {
            bench_transition_metal(
                &mut group,
                &mut sequence,
                &workload,
                challenge,
                &initial_e_in,
                &initial_e_out,
                &bound_e_in,
                &bound_e_out,
                rows,
                dispatch.native_transition,
            );
            bench_transition_cpu(
                &mut group,
                &workload,
                challenge,
                &bound_gruen,
                &mut cpu_tables,
                rows,
            );
        }
    }
    group.finish();
}

fn bench_transition_cpu(
    group: &mut BenchmarkGroup<'_, WallTime>,
    workload: &Workload,
    challenge: AkitaField,
    gruen: &GruenSplitEqPolynomial<AkitaField>,
    tables: &mut [AkitaField],
    rows: usize,
) {
    let _ = group.bench_function(
        BenchmarkId::new("cpu_preallocated", format!("n{rows}")),
        |bench| {
            bench.iter(|| {
                black_box(cpu_native_bind_and_message_preallocated(
                    black_box(&workload.cpu_rows),
                    challenge,
                    gruen,
                    workload.gamma,
                    tables,
                ))
            });
        },
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the benchmark keeps initial and bound protocol weights explicit"
)]
fn bench_transition_metal(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sequence: &mut InstructionInputSequence,
    workload: &Workload,
    challenge: AkitaField,
    initial_e_in: &[AkitaField],
    initial_e_out: &[AkitaField],
    bound_e_in: &[AkitaField],
    bound_e_out: &[AkitaField],
    rows: usize,
    threads: usize,
) {
    let suffix = format!("n{rows}_tg{threads}");
    let _ = group.bench_function(BenchmarkId::new("metal_wall_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                sequence.reset();
                let _ = sequence
                    .message(workload.gamma, initial_e_in, initial_e_out)
                    .expect("InstructionInput setup message should run");
                let started = Instant::now();
                let output = sequence
                    .bind_and_message(challenge, workload.gamma, bound_e_in, bound_e_out)
                    .expect("InstructionInput Metal transition should run");
                measured += started.elapsed();
                let _ = black_box(output);
            }
            measured
        });
    });
    let _ = group.bench_function(BenchmarkId::new("metal_active_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                sequence.reset();
                let _ = sequence
                    .message(workload.gamma, initial_e_in, initial_e_out)
                    .expect("InstructionInput setup message should run");
                let before = sequence.gpu_active_time();
                let output = sequence
                    .bind_and_message(challenge, workload.gamma, bound_e_in, bound_e_out)
                    .expect("InstructionInput Metal transition should run");
                measured += sequence
                    .gpu_active_time()
                    .checked_sub(before)
                    .expect("InstructionInput GPU-active time should be monotonic");
                let _ = black_box(output);
            }
            measured
        });
    });
}

fn bench_service_cases(c: &mut Criterion, context: &SolinasMetal) {
    let dispatch = dispatch();
    let cutoff_log2 = env_usize(
        "JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2",
        DEFAULT_CUTOFF_LOG2,
    );
    assert!(cutoff_log2 > 0);
    let mut group = comparison_group(c, "metal_sumcheck/instruction_input_service_resident");
    for rows in cases() {
        let cutoff = 1usize << cutoff_log2.min(rows.ilog2() as usize - 1);
        let (workload, mut sequence) = prepare_case(context, rows, dispatch, 3);
        let cpu_warm = run_cpu(&workload, cutoff, Capture::TARGET)
            .expect("InstructionInput CPU warmup should run");
        let metal_warm = run_hybrid(&mut sequence, &workload, cutoff, Capture::TARGET)
            .expect("InstructionInput Metal warmup should run");
        assert_eq!(metal_warm.trace, cpu_warm.trace);
        assert!(metal_warm.resident_rows_stable);
        assert!(metal_warm.static_device_buffers_stable);
        assert_eq!(metal_warm.readbacks, 1);

        let _ = group.throughput(Throughput::Elements(rows as u64));
        if cpu_first() {
            bench_service_cpu(&mut group, &workload, cutoff, rows);
            bench_service_metal(&mut group, &mut sequence, &workload, cutoff, rows, dispatch);
        } else {
            bench_service_metal(&mut group, &mut sequence, &workload, cutoff, rows, dispatch);
            bench_service_cpu(&mut group, &workload, cutoff, rows);
        }
    }
    group.finish();
}

fn bench_service_cpu(
    group: &mut BenchmarkGroup<'_, WallTime>,
    workload: &Workload,
    cutoff: usize,
    rows: usize,
) {
    let _ = group.bench_function(
        BenchmarkId::new("cpu_optimized_mirror", format!("n{rows}_cutoff{cutoff}")),
        |bench| {
            bench.iter_custom(|iterations| {
                let mut measured = Duration::ZERO;
                for _ in 0..iterations {
                    let output = run_cpu(workload, cutoff, Capture::TARGET)
                        .expect("InstructionInput optimized CPU sequence should run");
                    measured += output.wall;
                    let _ = black_box(output.trace.final_sumcheck_claim);
                }
                measured
            });
        },
    );
}

fn bench_service_metal(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sequence: &mut InstructionInputSequence,
    workload: &Workload,
    cutoff: usize,
    rows: usize,
    dispatch: SequenceDispatch,
) {
    let suffix = format!(
        "n{rows}_cutoff{cutoff}_tg{}-{}-{}",
        dispatch.native_message, dispatch.native_transition, dispatch.dense_transition
    );
    let mut cutoff_readback = vec![AkitaField::zero(); INSTRUCTION_INPUT_TABLES * cutoff];
    let _ = group.bench_function(BenchmarkId::new("metal_wall_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                let output = run_hybrid_with_readback(
                    sequence,
                    workload,
                    cutoff,
                    Capture::TARGET,
                    &mut cutoff_readback,
                )
                .expect("InstructionInput Metal hybrid sequence should run");
                measured += output.wall;
                let _ = black_box(output.trace.final_sumcheck_claim);
            }
            measured
        });
    });
    let _ = group.bench_function(BenchmarkId::new("metal_active_compact", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                let output = run_hybrid_with_readback(
                    sequence,
                    workload,
                    cutoff,
                    Capture::TARGET,
                    &mut cutoff_readback,
                )
                .expect("InstructionInput Metal hybrid sequence should run");
                measured += output.gpu_active;
                let _ = black_box(output.trace.final_sumcheck_claim);
            }
            measured
        });
    });
}

fn validate(context: &SolinasMetal) {
    let log_n = env_usize(
        "JOLT_SOLINAS_BENCH_VALIDATE_LOG_N",
        DEFAULT_VALIDATION_LOG_N,
    );
    assert!((3..=20).contains(&log_n));
    let rows = 1usize << log_n;
    let cutoff_log2 = env_usize(
        "JOLT_METAL_INSTRUCTION_INPUT_CUTOFF_LOG2",
        DEFAULT_CUTOFF_LOG2,
    );
    let cutoff = 1usize << cutoff_log2.min(log_n - 1);
    let dispatch = dispatch();
    let (workload, mut sequence) = prepare_case(context, rows, dispatch, 0xbb67_ae85);
    let cpu = run_cpu(&workload, cutoff, Capture::VALIDATION)
        .expect("InstructionInput validation CPU sequence should run");
    let actual = run_actual_optimized(&workload)
        .expect("actual optimized InstructionInput kernel should run");
    assert_eq!(actual.round_polys, cpu.trace.round_polys);
    assert_eq!(actual.challenges, cpu.trace.challenges);
    assert_eq!(actual.final_claims, cpu.trace.final_claims);
    assert_eq!(actual.final_sumcheck_claim, cpu.trace.final_sumcheck_claim);
    assert_eq!(actual.transcript_state, cpu.trace.transcript_state);

    let mut gruen = GruenSplitEqPolynomial::new(&workload.point, BindingOrder::LowToHigh);
    let expected_message = cpu_native_q_evals(&workload.cpu_rows, &gruen, workload.gamma);
    assert_eq!(
        descriptor_grid(
            sequence
                .message(workload.gamma, gruen.e_in_current(), gruen.e_out_current())
                .expect("InstructionInput validation message should run")
        ),
        expected_message
    );
    let challenge = cpu.trace.challenges[0];
    gruen.bind(challenge);
    let mut expected_tables = vec![AkitaField::zero(); INSTRUCTION_INPUT_TABLES * rows / 2];
    let expected_transition = cpu_native_bind_and_message_preallocated(
        &workload.cpu_rows,
        challenge,
        &gruen,
        workload.gamma,
        &mut expected_tables,
    );
    assert_eq!(
        descriptor_grid(
            sequence
                .bind_and_message(
                    challenge,
                    workload.gamma,
                    gruen.e_in_current(),
                    gruen.e_out_current(),
                )
                .expect("InstructionInput validation transition should run")
        ),
        expected_transition
    );
    let mut actual_tables = vec![AkitaField::zero(); expected_tables.len()];
    sequence
        .read_current_tables(&mut actual_tables)
        .expect("InstructionInput validation tables should read back");
    assert_eq!(actual_tables, expected_tables);

    let hybrid = run_hybrid(&mut sequence, &workload, cutoff, Capture::VALIDATION)
        .expect("InstructionInput validation Metal sequence should run");
    assert_eq!(hybrid.trace, cpu.trace);
    assert!(hybrid.resident_rows_stable);
    assert!(hybrid.static_device_buffers_stable);
    assert_eq!(hybrid.readbacks, 1);
    eprintln!("instruction_input gate=exact rows={rows} cutoff={cutoff} compact_row_bytes=48");
}

fn prepare_case(
    context: &SolinasMetal,
    rows: usize,
    dispatch: SequenceDispatch,
    seed: u64,
) -> (Workload, InstructionInputSequence) {
    let mut workload = Workload::new(rows.ilog2() as usize, seed)
        .expect("InstructionInput benchmark workload should build");
    let mut sequence = workload
        .prepare_sequence(context, dispatch)
        .expect("InstructionInput resident sequence should prepare");
    let primer = sequence
        .prime_native_pipeline()
        .expect("InstructionInput native pipeline should prime");
    eprintln!(
        "instruction_input primer_rows={} wall_ns={} gpu_active_ns={}",
        primer.source_elements,
        primer.wall.as_nanos(),
        primer.gpu_active.as_nanos()
    );
    (workload, sequence)
}

fn dispatch() -> SequenceDispatch {
    let dispatch = SequenceDispatch {
        native_message: env_usize("JOLT_METAL_INSTRUCTION_INPUT_NATIVE_MESSAGE_THREADS", 256),
        native_transition: env_usize(
            "JOLT_METAL_INSTRUCTION_INPUT_NATIVE_TRANSITION_THREADS",
            128,
        ),
        dense_transition: env_usize("JOLT_METAL_INSTRUCTION_INPUT_DENSE_TRANSITION_THREADS", 128),
    };
    assert!(
        dispatch.native_message > 0
            && dispatch.native_transition > 0
            && dispatch.dense_transition > 0
    );
    dispatch
}

fn cases() -> Vec<usize> {
    let cases = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    );
    assert!(cases
        .iter()
        .all(|rows| { rows.is_power_of_two() && (3..=28).contains(&(rows.ilog2() as usize)) }));
    cases
}

fn message_useful_muls(rows: usize, e_out_elements: usize) -> u64 {
    3 * rows as u64 + 3 * e_out_elements as u64
}

fn transition_useful_muls(rows: usize, e_out_elements: usize) -> u64 {
    17 * rows as u64 / 2 + 3 * e_out_elements as u64
}

fn comparison_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    group
}

fn cpu_first() -> bool {
    env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first")
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}
