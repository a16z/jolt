#![cfg_attr(
    target_os = "macos",
    expect(
        clippy::expect_used,
        clippy::panic,
        clippy::print_stderr,
        reason = "a benchmark must fail on setup errors and emit its hardware metadata"
    )
)]

#[cfg(target_os = "macos")]
#[path = "metal_solinas/cpu.rs"]
mod cpu;

#[cfg(target_os = "macos")]
#[path = "metal_solinas/product5.rs"]
mod product5;

#[cfg(target_os = "macos")]
#[path = "../tests/support/mod.rs"]
mod reference;

#[cfg(target_os = "macos")]
mod macos {
    use std::{env, hint::black_box, process::Command, thread, time::Duration};

    use criterion::{
        criterion_group, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
    };
    use jolt_kernels::metal::solinas::{
        DispatchConfig, Fp128, Invocation, Probe, SolinasMetal, OFFSET_275,
    };
    use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};

    use super::{
        cpu, product5,
        reference::{expected_field_for_offset, expected_u32_mad, inputs},
    };

    const STREAM_ELEMENTS: usize = 1 << 20;
    const COMPARISON_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
    const CPU_CHUNK_ELEMENTS: [usize; 3] = [1 << 11, 1 << 13, 1 << 14];
    const CHAIN_ELEMENTS: usize = 1 << 15;
    const CHAIN_ITERATIONS: u32 = 64;
    const RAW_ELEMENTS: usize = 1 << 18;
    const RAW_ITERATIONS: u32 = 128;
    const VALIDATION_ELEMENTS: usize = 256;

    fn metal_solinas(c: &mut Criterion) {
        let context = SolinasMetal::for_offset_275().expect("Solinas Metal library should compile");
        report_environment(&context);

        let (validation_lhs, validation_rhs) = inputs(VALIDATION_ELEMENTS);
        let family = env::var("JOLT_SOLINAS_BENCH_FAMILY").ok();
        assert!(
            env::var_os("JOLT_SOLINAS_BENCH_ELEMENTS").is_none() || family.is_some(),
            "JOLT_SOLINAS_BENCH_ELEMENTS requires JOLT_SOLINAS_BENCH_FAMILY"
        );
        if let Some(family) = family {
            match family.as_str() {
                "info" => {}
                "gpu-wall" => {
                    bench_gpu_wall_multiply(c, &context, &validation_lhs, &validation_rhs);
                }
                "gpu-active-copy" => {
                    bench_gpu_active_copy(c, &context, &validation_lhs, &validation_rhs);
                }
                "gpu-active-mul" => {
                    bench_gpu_active_multiply(c, &context, &validation_lhs, &validation_rhs);
                }
                "cpu-gpu-wall" => {
                    bench_cpu_gpu_multiply(c, &context, &validation_lhs, &validation_rhs);
                }
                "product5-message" => product5::bench_message(c, &context),
                "product5-transition" => product5::bench_transition(c, &context),
                "product5-threadgroups" => product5::bench_threadgroups(c, &context),
                "product5" => {
                    product5::bench_message(c, &context);
                    product5::bench_transition(c, &context);
                }
                _ => panic!("unknown JOLT_SOLINAS_BENCH_FAMILY `{family}`"),
            }
            return;
        }
        bench_command_latency(c, &context);
        bench_copy_bandwidth(c, &context, &validation_lhs, &validation_rhs);
        bench_streaming_field_ops(c, &context, &validation_lhs, &validation_rhs);
        bench_cpu_gpu_multiply(c, &context, &validation_lhs, &validation_rhs);
        bench_gpu_active_copy(c, &context, &validation_lhs, &validation_rhs);
        bench_gpu_active_multiply(c, &context, &validation_lhs, &validation_rhs);
        bench_multiply_threadgroups(c, &context, &validation_lhs, &validation_rhs);
        bench_dependency_latency(c, &context, &validation_lhs, &validation_rhs);
        bench_chains(c, &context, &validation_lhs, &validation_rhs);
        bench_raw_integer(c, &context, &validation_lhs, &validation_rhs);
        product5::bench_message(c, &context);
        product5::bench_transition(c, &context);
    }

    fn bench_gpu_wall_multiply(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let limits = context
            .pipeline_limits(Probe::MulWide)
            .expect("multiply pipeline should compile");
        let gpu_width = limits.max_total_threads_per_threadgroup;
        let mut group = comparison_group(c, "metal_solinas/gpu_only_mul_wall");
        for (elements, _) in comparison_cases() {
            validate_case_size(context, elements);
            let (lhs, rhs) = inputs(elements);
            let invocation = prepare_validated(
                context,
                Probe::MulWide,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig {
                    iterations: 1,
                    threads_per_threadgroup: Some(gpu_width),
                },
            );
            let _ = group.throughput(Throughput::Elements(elements as u64));
            let _ = group.bench_function(
                BenchmarkId::new("gpu_wide", format!("n{elements}_tg{gpu_width}")),
                |bench| {
                    bench.iter(|| invocation.execute().expect("GPU multiply should execute"));
                },
            );
        }
        group.finish();
    }

    fn bench_command_latency(c: &mut Criterion, context: &SolinasMetal) {
        let invocation = context
            .prepare_noop()
            .expect("noop pipeline should compile");
        invocation
            .execute()
            .expect("noop validation should execute");
        let limits = invocation.pipeline_limits();
        let id = BenchmarkId::new(
            "noop",
            format!(
                "tg{}_tew{}_max{}",
                invocation.threads_per_threadgroup(),
                limits.thread_execution_width,
                limits.max_total_threads_per_threadgroup
            ),
        );
        let mut group = configured_group(c, "metal_solinas/command_wall");
        let _ = group.bench_function(id, |bench| {
            bench.iter(|| invocation.execute().expect("noop should execute"));
        });
        group.finish();
    }

    fn bench_gpu_active_copy(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let limits = context
            .pipeline_limits(Probe::Copy)
            .expect("copy pipeline should compile");
        let gpu_width = limits.max_total_threads_per_threadgroup;
        let mut group = comparison_group(c, "metal_solinas/gpu_active_copy");
        for (elements, _) in comparison_cases() {
            validate_case_size(context, elements);
            let (lhs, rhs) = inputs(elements);
            let invocation = prepare_validated(
                context,
                Probe::Copy,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig {
                    iterations: 1,
                    threads_per_threadgroup: Some(gpu_width),
                },
            );
            let _ = group.throughput(Throughput::Bytes(invocation.logical_bytes()));
            let _ = group.bench_function(
                BenchmarkId::new("copy", format!("n{elements}_tg{gpu_width}")),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed GPU copy should execute");
                        }
                        active
                    });
                },
            );
        }
        group.finish();
    }

    fn bench_gpu_active_multiply(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let limits = context
            .pipeline_limits(Probe::MulWide)
            .expect("multiply pipeline should compile");
        let gpu_width = limits.max_total_threads_per_threadgroup;
        let mut group = comparison_group(c, "metal_solinas/gpu_active_mul");
        for (elements, _) in comparison_cases() {
            validate_case_size(context, elements);
            let (lhs, rhs) = inputs(elements);
            let invocation = prepare_validated(
                context,
                Probe::MulWide,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig {
                    iterations: 1,
                    threads_per_threadgroup: Some(gpu_width),
                },
            );
            let _ = group.throughput(Throughput::Elements(elements as u64));
            let _ = group.bench_function(
                BenchmarkId::new("gpu_wide", format!("n{elements}_tg{gpu_width}")),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed GPU multiply should execute");
                        }
                        active
                    });
                },
            );
        }
        group.finish();
    }

    fn bench_cpu_gpu_multiply(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        validate_cpu_mul(validation_lhs, validation_rhs);
        let limits = context
            .pipeline_limits(Probe::MulWide)
            .expect("multiply pipeline should compile");
        let gpu_width = limits.max_total_threads_per_threadgroup;
        let threads = thread::available_parallelism().map_or(1, |count| count.get());
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("CPU comparison pool should build");
        let cpu_first =
            env::var("JOLT_SOLINAS_BENCH_ORDER").is_ok_and(|order| order == "cpu-first");
        let mut group = comparison_group(c, "metal_solinas/cpu_gpu_mul_wall");

        for (elements, chunk) in comparison_cases() {
            validate_case_size(context, elements);
            let (lhs, rhs) = inputs(elements);
            let invocation = prepare_validated(
                context,
                Probe::MulWide,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig {
                    iterations: 1,
                    threads_per_threadgroup: Some(gpu_width),
                },
            );
            let mut seq_output = vec![Fp128::ZERO; elements];
            let mut par_output = vec![Fp128::ZERO; elements];
            let _ = group.throughput(Throughput::Elements(elements as u64));

            if cpu_first {
                add_cpu_seq_case(&mut group, &lhs, &rhs, &mut seq_output);
                add_cpu_parallel_case(
                    &mut group,
                    &pool,
                    threads,
                    chunk,
                    &lhs,
                    &rhs,
                    &mut par_output,
                );
                add_gpu_case(&mut group, &invocation, elements);
            } else {
                add_gpu_case(&mut group, &invocation, elements);
                add_cpu_parallel_case(
                    &mut group,
                    &pool,
                    threads,
                    chunk,
                    &lhs,
                    &rhs,
                    &mut par_output,
                );
                add_cpu_seq_case(&mut group, &lhs, &rhs, &mut seq_output);
            }
        }
        group.finish();
    }

    fn add_gpu_case(
        group: &mut BenchmarkGroup<'_, WallTime>,
        invocation: &Invocation<'_>,
        elements: usize,
    ) {
        let _ = group.bench_function(
            BenchmarkId::new(
                "gpu_wide",
                format!("n{elements}_tg{}", invocation.threads_per_threadgroup()),
            ),
            |bench| bench.iter(|| invocation.execute().expect("GPU multiply should execute")),
        );
    }

    fn add_cpu_seq_case(
        group: &mut BenchmarkGroup<'_, WallTime>,
        lhs: &[Fp128],
        rhs: &[Fp128],
        output: &mut [Fp128],
    ) {
        let _ = group.bench_function(
            BenchmarkId::new("cpu_portable_seq", format!("n{}", lhs.len())),
            |bench| {
                bench.iter(|| {
                    let lhs = black_box(lhs);
                    let rhs = black_box(rhs);
                    for index in 0..output.len() {
                        output[index] = cpu::mul(lhs[index], rhs[index]);
                    }
                    black_box(output[0])
                });
            },
        );
    }

    fn add_cpu_parallel_case(
        group: &mut BenchmarkGroup<'_, WallTime>,
        pool: &ThreadPool,
        threads: usize,
        chunk: usize,
        lhs: &[Fp128],
        rhs: &[Fp128],
        output: &mut [Fp128],
    ) {
        let _ = group.bench_function(
            BenchmarkId::new(
                "cpu_portable_rayon",
                format!("n{}_t{threads}_chunk{chunk}", lhs.len()),
            ),
            |bench| {
                bench.iter(|| {
                    let lhs = black_box(lhs);
                    let rhs = black_box(rhs);
                    pool.install(|| {
                        output.par_chunks_mut(chunk).enumerate().for_each(
                            |(chunk_index, output_chunk)| {
                                let start = chunk_index * chunk;
                                for (offset, value) in output_chunk.iter_mut().enumerate() {
                                    let index = start + offset;
                                    *value = cpu::mul(lhs[index], rhs[index]);
                                }
                            },
                        );
                    });
                    black_box(output[0])
                });
            },
        );
    }

    fn validate_cpu_mul(lhs: &[Fp128], rhs: &[Fp128]) {
        let actual = lhs
            .iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| cpu::mul(lhs, rhs))
            .collect::<Vec<_>>();
        let expected = lhs
            .iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| expected_field_for_offset(Probe::MulWide, lhs, rhs, 1, OFFSET_275))
            .collect::<Result<Vec<_>, _>>()
            .expect("CPU comparison should have an oracle");
        assert_eq!(actual, expected, "portable CPU comparison failed preflight");
    }

    fn bench_copy_bandwidth(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let mut group = configured_group(c, "metal_solinas/copy_size_wall");
        for elements in [1 << 10, 1 << 16, 1 << 20, 1 << 22] {
            let (lhs, rhs) = inputs(elements);
            let invocation = prepare_validated(
                context,
                Probe::Copy,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig::default(),
            );
            let _ = group.throughput(Throughput::Bytes(invocation.logical_bytes()));
            let _ = group.bench_function(invocation_id(&invocation, elements), |bench| {
                bench.iter(|| invocation.execute().expect("copy should execute"));
            });
        }
        group.finish();
    }

    fn bench_streaming_field_ops(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let (lhs, rhs) = inputs(STREAM_ELEMENTS);
        let mut group = configured_group(c, "metal_solinas/streaming_field_wall");
        for probe in [Probe::Add, Probe::Sub, Probe::MulWide] {
            let invocation = prepare_validated(
                context,
                probe,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig::default(),
            );
            let _ = group.throughput(Throughput::Elements(invocation.field_operation_count()));
            let _ = group.bench_function(invocation_id(&invocation, STREAM_ELEMENTS), |bench| {
                bench.iter(|| invocation.execute().expect("field probe should execute"));
            });
        }
        group.finish();
    }

    fn bench_multiply_threadgroups(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let (lhs, rhs) = inputs(STREAM_ELEMENTS);
        let mut group = configured_group(c, "metal_solinas/mul_threadgroup_wall");
        for probe in [Probe::MulWide] {
            let limits = context
                .pipeline_limits(probe)
                .expect("multiply pipeline should compile");
            for width in threadgroup_widths(
                limits.thread_execution_width,
                limits.max_total_threads_per_threadgroup,
            ) {
                let invocation = prepare_validated(
                    context,
                    probe,
                    &lhs,
                    &rhs,
                    validation_lhs,
                    validation_rhs,
                    DispatchConfig {
                        iterations: 1,
                        threads_per_threadgroup: Some(width),
                    },
                );
                let _ = group.throughput(Throughput::Elements(invocation.field_operation_count()));
                let _ =
                    group.bench_function(invocation_id(&invocation, STREAM_ELEMENTS), |bench| {
                        bench.iter(|| invocation.execute().expect("multiply should execute"));
                    });
            }
        }
        group.finish();
    }

    fn bench_chains(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let (lhs, rhs) = inputs(CHAIN_ELEMENTS);
        let mut group = configured_group(c, "metal_solinas/chain_wall");
        for probe in [
            Probe::ChainWide1,
            Probe::ChainWide2,
            Probe::ChainWide4,
            Probe::ChainWide8,
        ] {
            let invocation = prepare_validated(
                context,
                probe,
                &lhs,
                &rhs,
                validation_lhs,
                validation_rhs,
                DispatchConfig {
                    iterations: CHAIN_ITERATIONS,
                    threads_per_threadgroup: None,
                },
            );
            let _ = group.throughput(Throughput::Elements(invocation.field_operation_count()));
            let _ = group.bench_function(invocation_id(&invocation, CHAIN_ELEMENTS), |bench| {
                bench.iter(|| invocation.execute().expect("chain should execute"));
            });
        }
        group.finish();
    }

    fn bench_dependency_latency(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let mut group = configured_group(c, "metal_solinas/dependency_chain_wall");
        for probe in [Probe::ChainWide1] {
            let limits = context
                .pipeline_limits(probe)
                .expect("dependency pipeline should compile");
            let elements = limits.thread_execution_width;
            let (lhs, rhs) = inputs(elements);
            for iterations in [8, 16, 32, 64, 128, 256, 512] {
                let invocation = prepare_validated(
                    context,
                    probe,
                    &lhs,
                    &rhs,
                    validation_lhs,
                    validation_rhs,
                    DispatchConfig {
                        iterations,
                        threads_per_threadgroup: Some(limits.thread_execution_width),
                    },
                );
                let _ = group.throughput(Throughput::Elements(invocation.field_operation_count()));
                let _ = group.bench_function(invocation_id(&invocation, elements), |bench| {
                    bench.iter(|| {
                        invocation
                            .execute()
                            .expect("dependency chain should execute");
                    });
                });
            }
        }
        group.finish();
    }

    fn bench_raw_integer(
        c: &mut Criterion,
        context: &SolinasMetal,
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
    ) {
        let (lhs, rhs) = inputs(RAW_ELEMENTS);
        let invocation = prepare_validated(
            context,
            Probe::U32MadIlp8,
            &lhs,
            &rhs,
            validation_lhs,
            validation_rhs,
            DispatchConfig {
                iterations: RAW_ITERATIONS,
                threads_per_threadgroup: None,
            },
        );
        let lane_updates = RAW_ELEMENTS as u64 * RAW_ITERATIONS as u64 * 8;
        let mut group = configured_group(c, "metal_solinas/u32_mad_wall");
        let _ = group.throughput(Throughput::Elements(lane_updates));
        let _ = group.bench_function(invocation_id(&invocation, RAW_ELEMENTS), |bench| {
            bench.iter(|| invocation.execute().expect("raw probe should execute"));
        });
        group.finish();
    }

    fn prepare_validated<'a>(
        context: &'a SolinasMetal,
        probe: Probe,
        lhs: &[Fp128],
        rhs: &[Fp128],
        validation_lhs: &[Fp128],
        validation_rhs: &[Fp128],
        config: DispatchConfig,
    ) -> Invocation<'a> {
        let validation = context
            .prepare(probe, validation_lhs, validation_rhs, config)
            .expect("validation pipeline should compile");
        validation
            .execute()
            .expect("validation dispatch should execute");
        let actual = validation
            .read_output()
            .expect("validation output should read");
        let expected = if probe == Probe::U32MadIlp8 {
            validation_lhs
                .iter()
                .zip(validation_rhs)
                .map(|(&lhs, &rhs)| expected_u32_mad(lhs, rhs, config.iterations))
                .collect()
        } else {
            validation_lhs
                .iter()
                .zip(validation_rhs)
                .map(|(&lhs, &rhs)| {
                    expected_field_for_offset(probe, lhs, rhs, config.iterations, OFFSET_275)
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("timed field probe should have an oracle")
        };
        assert_eq!(actual, expected, "{} failed preflight", probe.name());

        context
            .prepare(probe, lhs, rhs, config)
            .expect("timed pipeline should compile")
    }

    fn configured_group<'a>(
        c: &'a mut Criterion,
        name: &str,
    ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
        let mut group = c.benchmark_group(name);
        let _ = group
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2));
        group
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

    fn comparison_cases() -> Vec<(usize, usize)> {
        if let Ok(value) = env::var("JOLT_SOLINAS_BENCH_ELEMENTS") {
            let elements = value
                .parse::<usize>()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer");
            assert!(elements > 0, "JOLT_SOLINAS_BENCH_ELEMENTS must be nonzero");
            return vec![(elements, 1 << 14)];
        }
        COMPARISON_ELEMENTS
            .into_iter()
            .zip(CPU_CHUNK_ELEMENTS)
            .collect()
    }

    fn validate_case_size(context: &SolinasMetal, elements: usize) {
        let bytes = elements
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("comparison buffer size should fit u64");
        let maximum = context.device_info().max_buffer_length;
        assert!(
            bytes <= maximum,
            "comparison needs {bytes} bytes per buffer but Metal allows {maximum}"
        );
    }

    fn invocation_id(invocation: &Invocation<'_>, elements: usize) -> BenchmarkId {
        let limits = invocation.pipeline_limits();
        BenchmarkId::new(
            invocation.probe().name(),
            format!(
                "n{elements}_it{}_tg{}_tew{}_max{}_smem{}",
                invocation.iterations(),
                invocation.threads_per_threadgroup(),
                limits.thread_execution_width,
                limits.max_total_threads_per_threadgroup,
                limits.static_threadgroup_memory_length
            ),
        )
    }

    fn threadgroup_widths(execution_width: usize, maximum: usize) -> Vec<usize> {
        [1, 2, 4, 8, 16, 32]
            .into_iter()
            .map(|factor| execution_width * factor)
            .take_while(|width| *width <= maximum)
            .collect()
    }

    fn report_environment(context: &SolinasMetal) {
        let device = context.device_info();
        eprintln!(
            "metal-solinas device={:?} macos={} offset={} max_buffer_length={} max_threadgroup_memory={}",
            device.name,
            macos_version(),
            device.offset,
            device.max_buffer_length,
            device.max_threadgroup_memory_length
        );
        for probe in [
            Probe::Noop,
            Probe::Copy,
            Probe::Add,
            Probe::Sub,
            Probe::MulWide,
            Probe::ChainWide1,
            Probe::ChainWide2,
            Probe::ChainWide4,
            Probe::ChainWide8,
            Probe::U32MadIlp8,
        ] {
            let limits = context
                .pipeline_limits(probe)
                .expect("reported pipeline should compile");
            eprintln!(
                "metal-solinas pipeline={} tew={} max_threads={} static_tgmem={}",
                probe.name(),
                limits.thread_execution_width,
                limits.max_total_threads_per_threadgroup,
                limits.static_threadgroup_memory_length
            );
        }
    }

    fn macos_version() -> String {
        Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map_or_else(
                || "unknown".to_owned(),
                |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            )
    }

    criterion_group!(benches, metal_solinas);
}

#[cfg(target_os = "macos")]
criterion::criterion_main!(macos::benches);

#[cfg(not(target_os = "macos"))]
fn main() {}
