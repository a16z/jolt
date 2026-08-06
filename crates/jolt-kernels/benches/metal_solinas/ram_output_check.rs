use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    ram_output_check_fold_u64_low_prefix, ram_output_check_low_binding_weights,
    RamOutputCheckHybridPlan, SolinasMetal, RAM_OUTPUT_CHECK_FIVE_X_CAP_NS,
    RAM_OUTPUT_CHECK_TARGET_ADDRESSES, RAM_OUTPUT_CHECK_TARGET_CPU_NS,
    RAM_OUTPUT_CHECK_TARGET_MASK_END, RAM_OUTPUT_CHECK_TARGET_MASK_START,
};
use jolt_kernels::optimized::ram_output_check::evaluate_deferred_output_check_cpu;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let threads = setting("JOLT_METAL_RAM_OUTPUT_THREADS", 128);
    let warmup_ms = setting("JOLT_METAL_RAM_OUTPUT_WARMUP_MS", 1_000);
    let measurement_ms = setting("JOLT_METAL_RAM_OUTPUT_MEASUREMENT_MS", 3_000);
    let observe_only = env::var_os("JOLT_METAL_RAM_OUTPUT_OBSERVE_ONLY").is_some();
    let plan = RamOutputCheckHybridPlan::new(
        RAM_OUTPUT_CHECK_TARGET_ADDRESSES,
        RAM_OUTPUT_CHECK_TARGET_MASK_START,
        RAM_OUTPUT_CHECK_TARGET_MASK_END,
        threads,
    )
    .expect("RAM output-check target plan should be valid");
    let source = native_values(plan.addresses());
    let challenges = challenge_values(plan.zero_rounds());
    let tail_challenges = challenge_values(plan.log_k());
    let output_address = output_address_values(plan.log_k());
    let weights = ram_output_check_low_binding_weights(&challenges);
    let val_final = source
        .iter()
        .map(|&value| AkitaField::from_u64(value))
        .collect::<Vec<_>>();
    let mut val_io = vec![AkitaField::from_u64(0); plan.addresses()];
    val_io[plan.mask_start()..plan.mask_end()]
        .copy_from_slice(&val_final[plan.mask_start()..plan.mask_end()]);

    let upload_start = Instant::now();
    let resident = context
        .prepare_ram_output_check_values(&source, true, plan)
        .expect("certified RAM final values should prepare");
    let upload_wall = upload_start.elapsed();
    let setup_start = Instant::now();
    let fold = context
        .prepare_ram_output_check_fold(&resident, &challenges, plan)
        .expect("RAM output-check fold should prepare");
    let setup_wall = setup_start.elapsed();
    assert_eq!(
        fold.source_allocation_identity(),
        resident.allocation_identity()
    );
    assert_eq!(fold.execute_device_buffer_allocations(), 0);

    let cpu_start = Instant::now();
    let expected =
        ram_output_check_fold_u64_low_prefix::<AkitaField>(resident.as_slice(), &weights)
            .expect("resident CPU prefix fold should execute");
    let cpu_wall = cpu_start.elapsed();
    let complete_cpu_start = Instant::now();
    let deferred_output = evaluate_deferred_output_check_cpu(
        &output_address,
        plan.mask_start(),
        plan.mask_end(),
        &val_io,
        &val_final,
        &tail_challenges,
    )
    .expect("complete deferred CPU member should execute");
    let complete_cpu_wall = complete_cpu_start.elapsed();
    let _ = black_box(deferred_output);
    let cold_start = Instant::now();
    let (cold, cold_active) = fold
        .execute_timed()
        .expect("cold RAM output-check fold should execute");
    let cold_wall = cold_start.elapsed();
    let warm_start = Instant::now();
    let (warm, warm_active) = fold
        .execute_timed()
        .expect("warm RAM output-check fold should execute");
    let warm_wall = warm_start.elapsed();
    assert_eq!(cold, expected);
    assert_eq!(warm, expected);

    eprintln!(
        "ram-output-check addresses={} threads={} upload={upload_wall:?} setup={setup_wall:?} cpu-resident={cpu_wall:?} cpu-complete-deferred={complete_cpu_wall:?} cold-wall={cold_wall:?} cold-active={cold_active:?} warm-wall={warm_wall:?} warm-active={warm_active:?} borrowed-bytes={} private-bytes={} partial-tew={} partial-max-threads={} partial-static-tgmem={} frozen-cpu-us={:.3} five-x-cap-us={:.3} deferred-speedup={:.3}",
        plan.addresses(),
        plan.threads_per_threadgroup(),
        resident.resident_bytes(),
        fold.storage().private_bytes,
        fold.partial_pipeline_limits().thread_execution_width,
        fold.partial_pipeline_limits().max_total_threads_per_threadgroup,
        fold.partial_pipeline_limits().static_threadgroup_memory_length,
        RAM_OUTPUT_CHECK_TARGET_CPU_NS as f64 / 1e3,
        RAM_OUTPUT_CHECK_FIVE_X_CAP_NS as f64 / 1e3,
        RAM_OUTPUT_CHECK_TARGET_CPU_NS as f64 / complete_cpu_wall.as_nanos() as f64,
    );

    if observe_only {
        return;
    }

    let mut group = c.benchmark_group("metal_sumcheck/ram_output_check_prefix");
    let _ = group
        .sample_size(50)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64))
        .throughput(Throughput::Elements(plan.addresses() as u64));
    let suffix = format!("k{}_t{threads}", plan.addresses());
    let _ = group.bench_function(BenchmarkId::new("cpu_resident", &suffix), |bench| {
        bench.iter(|| {
            black_box(
                ram_output_check_fold_u64_low_prefix::<AkitaField>(resident.as_slice(), &weights)
                    .expect("resident CPU prefix fold should execute"),
            )
        });
    });
    let _ = group.bench_function(
        BenchmarkId::new("cpu_complete_deferred", &suffix),
        |bench| {
            bench.iter(|| {
                black_box(
                    evaluate_deferred_output_check_cpu(
                        &output_address,
                        plan.mask_start(),
                        plan.mask_end(),
                        &val_io,
                        &val_final,
                        &tail_challenges,
                    )
                    .expect("complete deferred CPU member should execute"),
                )
            });
        },
    );
    let _ = group.bench_function(BenchmarkId::new("metal_wall_resident", &suffix), |bench| {
        bench.iter(|| {
            black_box(
                fold.execute()
                    .expect("resident Metal prefix fold should execute"),
            )
        });
    });
    let _ = group.bench_function(
        BenchmarkId::new("metal_active_resident", &suffix),
        |bench| {
            bench.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    elapsed += fold
                        .execute_timed()
                        .expect("timed resident Metal prefix fold should execute")
                        .1;
                }
                elapsed
            });
        },
    );
    group.finish();
}

fn native_values(count: usize) -> Vec<u64> {
    (0..count)
        .map(|index| match index % 17 {
            0 => u64::MAX,
            1 => 0,
            _ => splitmix(index as u64 ^ 0xa409_3822_299f_31d0),
        })
        .collect()
}

fn challenge_values(count: usize) -> Vec<AkitaField> {
    (0..count)
        .map(|index| {
            AkitaField::from_u64(splitmix(index as u64 ^ 0x243f_6a88_85a3_08d3) & ((1 << 56) - 1))
        })
        .collect()
}

fn output_address_values(count: usize) -> Vec<AkitaField> {
    (0..count)
        .map(|index| {
            AkitaField::from_u64(splitmix(index as u64 ^ 0x1319_8a2e_0370_7344) & ((1 << 56) - 1))
        })
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}
