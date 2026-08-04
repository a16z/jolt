use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_kernels::metal::solinas::{
    Fp128, Product5Config, Product5Invocation, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
    PRODUCT5_FACTORS,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

use super::{
    cpu,
    reference::{
        product5_fused_transition as oracle_transition, product5_message as oracle_message, values,
    },
};

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const THREADGROUP_SWEEP_ELEMENTS: usize = 1 << 22;
const VALIDATION_ELEMENTS: usize = 256;

pub fn bench_message(c: &mut Criterion, context: &SolinasMetal) {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("product5 CPU pool should build");
    let (validation_tables, validation_e_in, validation_e_out) = inputs(VALIDATION_ELEMENTS, false);
    let validation = context
        .prepare_product5_message(
            &validation_tables,
            VALIDATION_ELEMENTS,
            &validation_e_in,
            &validation_e_out,
            Product5Config::default(),
        )
        .expect("product5 message validation pipeline should compile");
    validate_message(
        &validation,
        &validation_tables,
        &validation_e_in,
        &validation_e_out,
    );
    let expected = oracle_message(
        &validation_tables,
        VALIDATION_ELEMENTS,
        &validation_e_in,
        &validation_e_out,
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        cpu_message(
            &pool,
            &validation_tables,
            VALIDATION_ELEMENTS,
            &validation_e_in,
            &validation_e_out,
        ),
        expected
    );
    report_pipeline(&validation);

    let cpu_first = cpu_first();
    let mut group = comparison_group(c, "metal_solinas/product5_message");
    for elements in cases() {
        let (tables, e_in, e_out) = inputs(elements, false);
        let invocation = context
            .prepare_product5_message(&tables, elements, &e_in, &e_out, configured_dispatch())
            .expect("product5 message pipeline should compile");
        let _ = group.throughput(Throughput::Elements(invocation.useful_multiplications()));

        let add_cpu = |group: &mut criterion::BenchmarkGroup<
            '_,
            criterion::measurement::WallTime,
        >| {
            let _ = group.bench_function(
                BenchmarkId::new("cpu_jolt_field_fused", format!("n{elements}_t{threads}")),
                |bench| {
                    bench.iter(|| black_box(cpu_message(&pool, &tables, elements, &e_in, &e_out)));
                },
            );
        };
        let add_gpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(gpu_id("gpu_wall_fused", &invocation), |bench| {
                    bench.iter(|| {
                        invocation
                            .execute()
                            .expect("product5 message should execute");
                    });
                });
                let _ = group.bench_function(gpu_id("gpu_active_fused", &invocation), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed product5 message should execute");
                        }
                        active
                    });
                });
            };
        if cpu_first {
            add_cpu(&mut group);
            add_gpu(&mut group);
        } else {
            add_gpu(&mut group);
            add_cpu(&mut group);
        }
    }
    group.finish();
}

pub fn bench_transition(c: &mut Criterion, context: &SolinasMetal) {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("product5 CPU pool should build");
    let (validation_tables, validation_e_in, validation_e_out) = inputs(VALIDATION_ELEMENTS, true);
    let challenge = challenge();
    let validation = context
        .prepare_product5_fused_transition(
            &validation_tables,
            VALIDATION_ELEMENTS,
            challenge,
            &validation_e_in,
            &validation_e_out,
            Product5Config::default(),
        )
        .expect("product5 transition validation pipeline should compile");
    validate_transition(
        &validation,
        &validation_tables,
        challenge,
        &validation_e_in,
        &validation_e_out,
    );
    let (expected_bound, expected_message) = oracle_transition(
        &validation_tables,
        VALIDATION_ELEMENTS,
        challenge,
        &validation_e_in,
        &validation_e_out,
        AKITA_OFFSET_FFFFA7F7,
    );
    let mut validation_output = vec![Fp128::ZERO; PRODUCT5_FACTORS * VALIDATION_ELEMENTS / 2];
    assert_eq!(
        cpu_transition(
            &pool,
            &validation_tables,
            VALIDATION_ELEMENTS,
            challenge,
            &validation_e_in,
            &validation_e_out,
            &mut validation_output,
        ),
        expected_message
    );
    assert_eq!(validation_output, expected_bound);
    report_pipeline(&validation);

    let cpu_first = cpu_first();
    let mut group = comparison_group(c, "metal_solinas/product5_transition");
    for elements in cases() {
        let (tables, e_in, e_out) = inputs(elements, true);
        let invocation = context
            .prepare_product5_fused_transition(
                &tables,
                elements,
                challenge,
                &e_in,
                &e_out,
                configured_dispatch(),
            )
            .expect("product5 transition pipeline should compile");
        let mut cpu_output = vec![Fp128::ZERO; PRODUCT5_FACTORS * elements / 2];
        let _ = group.throughput(Throughput::Elements(invocation.useful_multiplications()));

        let mut add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(
                    BenchmarkId::new("cpu_jolt_field_fused", format!("n{elements}_t{threads}")),
                    |bench| {
                        bench.iter(|| {
                            black_box(cpu_transition(
                                &pool,
                                &tables,
                                elements,
                                challenge,
                                &e_in,
                                &e_out,
                                &mut cpu_output,
                            ))
                        });
                    },
                );
            };
        let add_gpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(gpu_id("gpu_wall_fused", &invocation), |bench| {
                    bench.iter(|| {
                        invocation
                            .execute()
                            .expect("product5 transition should execute");
                    });
                });
                let _ = group.bench_function(gpu_id("gpu_active_fused", &invocation), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed product5 transition should execute");
                        }
                        active
                    });
                });
            };
        if cpu_first {
            add_cpu(&mut group);
            add_gpu(&mut group);
        } else {
            add_gpu(&mut group);
            add_cpu(&mut group);
        }
    }
    group.finish();
}

pub fn bench_threadgroups(c: &mut Criterion, context: &SolinasMetal) {
    validate_threadgroup_widths(context);
    let elements = requested_elements(THREADGROUP_SWEEP_ELEMENTS);
    let tables = values(PRODUCT5_FACTORS * elements);

    let (e_in, e_out) = weights(elements, false);
    let limits = context
        .prepare_product5_message(&tables, elements, &e_in, &e_out, Product5Config::default())
        .expect("product5 message pipeline should compile")
        .pipeline_limits();
    let mut message_group = comparison_group(c, "metal_solinas/product5_message_threadgroups");
    for width in threadgroup_widths(limits) {
        let invocation = context
            .prepare_product5_message(
                &tables,
                elements,
                &e_in,
                &e_out,
                Product5Config {
                    threads_per_threadgroup: Some(width),
                },
            )
            .expect("product5 message threadgroup should compile");
        let _ = message_group.throughput(Throughput::Elements(invocation.useful_multiplications()));
        let _ = message_group.bench_function(gpu_id("gpu_active", &invocation), |bench| {
            bench.iter_custom(|iterations| active_duration(&invocation, iterations));
        });
    }
    message_group.finish();

    let (e_in, e_out) = weights(elements, true);
    let mut transition_group =
        comparison_group(c, "metal_solinas/product5_transition_threadgroups");
    for width in threadgroup_widths(limits) {
        let invocation = context
            .prepare_product5_fused_transition(
                &tables,
                elements,
                challenge(),
                &e_in,
                &e_out,
                Product5Config {
                    threads_per_threadgroup: Some(width),
                },
            )
            .expect("product5 transition threadgroup should compile");
        let _ =
            transition_group.throughput(Throughput::Elements(invocation.useful_multiplications()));
        let _ = transition_group.bench_function(gpu_id("gpu_active", &invocation), |bench| {
            bench.iter_custom(|iterations| active_duration(&invocation, iterations));
        });
    }
    transition_group.finish();
}

fn validate_threadgroup_widths(context: &SolinasMetal) {
    let (tables, e_in, e_out) = inputs(VALIDATION_ELEMENTS, false);
    let expected = oracle_message(
        &tables,
        VALIDATION_ELEMENTS,
        &e_in,
        &e_out,
        AKITA_OFFSET_FFFFA7F7,
    );
    let limits = context
        .prepare_product5_message(
            &tables,
            VALIDATION_ELEMENTS,
            &e_in,
            &e_out,
            Product5Config::default(),
        )
        .expect("product5 validation pipeline should compile")
        .pipeline_limits();
    for width in threadgroup_widths(limits) {
        let invocation = context
            .prepare_product5_message(
                &tables,
                VALIDATION_ELEMENTS,
                &e_in,
                &e_out,
                Product5Config {
                    threads_per_threadgroup: Some(width),
                },
            )
            .expect("product5 validation threadgroup should compile");
        invocation
            .execute()
            .expect("product5 validation threadgroup should execute");
        assert_eq!(
            invocation
                .read_message()
                .expect("product5 validation message should read"),
            expected
        );
    }
}

fn validate_message(
    invocation: &Product5Invocation<'_>,
    tables: &[Fp128],
    e_in: &[Fp128],
    e_out: &[Fp128],
) {
    invocation
        .execute()
        .expect("product5 message validation should execute");
    let expected = oracle_message(
        tables,
        invocation.source_elements(),
        e_in,
        e_out,
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        invocation
            .read_message()
            .expect("product5 validation message should read"),
        expected
    );
}

fn validate_transition(
    invocation: &Product5Invocation<'_>,
    tables: &[Fp128],
    challenge: Fp128,
    e_in: &[Fp128],
    e_out: &[Fp128],
) {
    invocation
        .execute()
        .expect("product5 transition validation should execute");
    let (expected_bound, expected_message) = oracle_transition(
        tables,
        invocation.source_elements(),
        challenge,
        e_in,
        e_out,
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        invocation
            .read_message()
            .expect("product5 validation message should read"),
        expected_message
    );
    assert_eq!(
        invocation
            .read_bound_tables()
            .expect("product5 validation output should read")
            .expect("transition should have bound output"),
        expected_bound
    );
}

fn cpu_message(
    pool: &ThreadPool,
    tables: &[Fp128],
    elements: usize,
    e_in: &[Fp128],
    e_out: &[Fp128],
) -> [Fp128; PRODUCT5_FACTORS] {
    let tables = split_five(tables, elements);
    pool.install(|| {
        (0..e_out.len())
            .into_par_iter()
            .map(|x_out| message_block(&tables, e_in, e_out[x_out], x_out))
            .reduce(zero_lanes, add_lanes)
    })
}

fn cpu_transition(
    pool: &ThreadPool,
    tables: &[Fp128],
    elements: usize,
    challenge: Fp128,
    e_in: &[Fp128],
    e_out: &[Fp128],
    output: &mut [Fp128],
) -> [Fp128; PRODUCT5_FACTORS] {
    let tables = split_five(tables, elements);
    let bound_elements = elements / 2;
    let [out0, out1, out2, out3, out4] = split_five_mut(output, bound_elements);
    let block_elements = 2 * e_in.len();
    pool.install(|| {
        (0..e_out.len())
            .into_par_iter()
            .zip(out0.par_chunks_mut(block_elements))
            .zip(out1.par_chunks_mut(block_elements))
            .zip(out2.par_chunks_mut(block_elements))
            .zip(out3.par_chunks_mut(block_elements))
            .zip(out4.par_chunks_mut(block_elements))
            .map(|(((((x_out, out0), out1), out2), out3), out4)| {
                transition_block(
                    &tables,
                    [out0, out1, out2, out3, out4],
                    challenge,
                    e_in,
                    e_out[x_out],
                    x_out,
                )
            })
            .reduce(zero_lanes, add_lanes)
    })
}

fn message_block(
    tables: &[&[Fp128]; PRODUCT5_FACTORS],
    e_in: &[Fp128],
    e_out: Fp128,
    x_out: usize,
) -> [Fp128; PRODUCT5_FACTORS] {
    let mut lanes = zero_lanes();
    for (x_in, &weight) in e_in.iter().enumerate() {
        let pair = x_out * e_in.len() + x_in;
        let endpoints =
            std::array::from_fn(|factor| (tables[factor][2 * pair], tables[factor][2 * pair + 1]));
        accumulate_pair(&mut lanes, endpoints, weight);
    }
    lanes.map(|lane| cpu::mul(e_out, lane))
}

fn transition_block(
    tables: &[&[Fp128]; PRODUCT5_FACTORS],
    mut output: [&mut [Fp128]; PRODUCT5_FACTORS],
    challenge: Fp128,
    e_in: &[Fp128],
    e_out: Fp128,
    x_out: usize,
) -> [Fp128; PRODUCT5_FACTORS] {
    let mut lanes = zero_lanes();
    for (x_in, &weight) in e_in.iter().enumerate() {
        let pair = x_out * e_in.len() + x_in;
        let endpoints = std::array::from_fn(|factor| {
            let source = 4 * pair;
            let lo = cpu::bind(
                tables[factor][source],
                tables[factor][source + 1],
                challenge,
            );
            let hi = cpu::bind(
                tables[factor][source + 2],
                tables[factor][source + 3],
                challenge,
            );
            output[factor][2 * x_in] = lo;
            output[factor][2 * x_in + 1] = hi;
            (lo, hi)
        });
        accumulate_pair(&mut lanes, endpoints, weight);
    }
    lanes.map(|lane| cpu::mul(e_out, lane))
}

fn accumulate_pair(
    lanes: &mut [Fp128; PRODUCT5_FACTORS],
    mut endpoints: [(Fp128, Fp128); PRODUCT5_FACTORS],
    inner_weight: Fp128,
) {
    endpoints[0].0 = cpu::mul(inner_weight, endpoints[0].0);
    endpoints[0].1 = cpu::mul(inner_weight, endpoints[0].1);
    let mut evals = endpoints.map(|(_, hi)| hi);
    let steps = endpoints.map(|(lo, hi)| cpu::sub(hi, lo));
    for (sample, lane) in lanes[..PRODUCT5_FACTORS - 1].iter_mut().enumerate() {
        *lane = cpu::add(*lane, product(evals));
        if sample + 1 < PRODUCT5_FACTORS - 1 {
            for (eval, step) in evals.iter_mut().zip(steps) {
                *eval = cpu::add(*eval, step);
            }
        }
    }
    lanes[PRODUCT5_FACTORS - 1] = cpu::add(lanes[PRODUCT5_FACTORS - 1], product(steps));
}

fn product(values: [Fp128; PRODUCT5_FACTORS]) -> Fp128 {
    values[1..]
        .iter()
        .fold(values[0], |product, &factor| cpu::mul(product, factor))
}

fn zero_lanes() -> [Fp128; PRODUCT5_FACTORS] {
    [Fp128::ZERO; PRODUCT5_FACTORS]
}

fn add_lanes(
    mut lhs: [Fp128; PRODUCT5_FACTORS],
    rhs: [Fp128; PRODUCT5_FACTORS],
) -> [Fp128; PRODUCT5_FACTORS] {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs = cpu::add(*lhs, rhs);
    }
    lhs
}

fn split_five(tables: &[Fp128], elements: usize) -> [&[Fp128]; PRODUCT5_FACTORS] {
    std::array::from_fn(|factor| &tables[factor * elements..(factor + 1) * elements])
}

fn split_five_mut(tables: &mut [Fp128], elements: usize) -> [&mut [Fp128]; PRODUCT5_FACTORS] {
    let (table0, rest) = tables.split_at_mut(elements);
    let (table1, rest) = rest.split_at_mut(elements);
    let (table2, rest) = rest.split_at_mut(elements);
    let (table3, table4) = rest.split_at_mut(elements);
    [table0, table1, table2, table3, table4]
}

fn inputs(elements: usize, transition: bool) -> (Vec<Fp128>, Vec<Fp128>, Vec<Fp128>) {
    let (e_in, e_out) = weights(elements, transition);
    (values(PRODUCT5_FACTORS * elements), e_in, e_out)
}

fn weights(elements: usize, transition: bool) -> (Vec<Fp128>, Vec<Fp128>) {
    let pairs = if transition {
        elements / 4
    } else {
        elements / 2
    };
    let log_pairs = pairs.trailing_zeros() as usize;
    let e_in_length = 1usize << (log_pairs / 2);
    let e_out_length = pairs / e_in_length;
    (values(e_in_length), values(e_out_length))
}

fn challenge() -> Fp128 {
    Fp128::from_u128(0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344)
}

fn cases() -> Vec<usize> {
    if env::var_os("JOLT_SOLINAS_BENCH_ELEMENTS").is_some() {
        return vec![requested_elements(4)];
    }
    DEFAULT_ELEMENTS.to_vec()
}

fn requested_elements(default: usize) -> usize {
    let elements = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or(default, |value| {
        value
            .parse::<usize>()
            .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")
    });
    assert!(
        elements >= 4 && elements.is_power_of_two(),
        "product5 element count must be a power of two of at least four"
    );
    elements
}

fn configured_dispatch() -> Product5Config {
    Product5Config {
        threads_per_threadgroup: env::var("JOLT_SOLINAS_PRODUCT5_THREADS").ok().map(|value| {
            value
                .parse::<usize>()
                .expect("JOLT_SOLINAS_PRODUCT5_THREADS should be a positive integer")
        }),
    }
}

fn threadgroup_widths(limits: jolt_kernels::metal::solinas::PipelineLimits) -> Vec<usize> {
    let mut widths = Vec::new();
    let mut width = limits.thread_execution_width;
    while width <= limits.max_total_threads_per_threadgroup {
        widths.push(width);
        let Some(next) = width.checked_mul(2) else {
            break;
        };
        width = next;
    }
    widths
}

fn active_duration(invocation: &Product5Invocation<'_>, iterations: u64) -> Duration {
    let mut active = Duration::ZERO;
    for _ in 0..iterations {
        active += invocation
            .execute_timed()
            .expect("timed product5 dispatch should execute");
    }
    active
}

fn gpu_id(label: &str, invocation: &Product5Invocation<'_>) -> BenchmarkId {
    let limits = invocation.pipeline_limits();
    BenchmarkId::new(
        label,
        format!(
            "{}_n{}_tg{}_tew{}_max{}_static_smem{}_dynamic_smem{}",
            invocation.name(),
            invocation.source_elements(),
            invocation.threads_per_threadgroup(),
            limits.thread_execution_width,
            limits.max_total_threads_per_threadgroup,
            limits.static_threadgroup_memory_length,
            invocation.dynamic_threadgroup_memory_bytes(),
        ),
    )
}

fn comparison_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    let _ = group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(4));
    group
}

fn cpu_first() -> bool {
    env::var("JOLT_SOLINAS_BENCH_ORDER").is_ok_and(|order| order == "cpu-first")
}

fn report_pipeline(invocation: &Product5Invocation<'_>) {
    let main = invocation.pipeline_limits();
    let reduction = invocation.reduction_pipeline_limits();
    eprintln!(
        "metal-solinas pipeline={} tew={} max_threads={} static_tgmem={} dynamic_tgmem={} reduction_tew={} reduction_max_threads={}",
        invocation.name(),
        main.thread_execution_width,
        main.max_total_threads_per_threadgroup,
        main.static_threadgroup_memory_length,
        invocation.dynamic_threadgroup_memory_bytes(),
        reduction.thread_execution_width,
        reduction.max_total_threads_per_threadgroup,
    );
}
