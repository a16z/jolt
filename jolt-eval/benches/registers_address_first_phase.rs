use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{Criterion, SamplingMode, Throughput};
use jolt_eval::objective::performance::registers_address_phase::{
    assert_small_scale_parity, RegistersAddressPhaseFixture,
};

fn criterion_benchmark(criterion: &mut Criterion) {
    assert_small_scale_parity();
    for log_cycles in [22, 24] {
        let fixture = Arc::new(RegistersAddressPhaseFixture::synthetic(log_cycles));
        let mut group =
            criterion.benchmark_group(format!("registers_address_phase/2^{log_cycles}"));
        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(1));
        group.throughput(Throughput::Elements(fixture.cycles() as u64));
        group.bench_function("binary", |bencher| {
            bencher.iter(|| black_box(fixture.run_binary()));
        });
        group.bench_function("radix4", |bencher| {
            bencher.iter(|| black_box(fixture.run_radix4()));
        });
        group.finish();
    }
}

fn gate_run() {
    assert_small_scale_parity();
    for log_cycles in [22, 24] {
        let fixture = RegistersAddressPhaseFixture::synthetic(log_cycles);

        let started = Instant::now();
        black_box(fixture.run_binary());
        let binary = started.elapsed();

        let started = Instant::now();
        black_box(fixture.run_radix4());
        let radix4 = started.elapsed();

        println!(
            "GATE_RESULT log_cycles={log_cycles} binary_s={:.9} radix4_s={:.9} ratio={:.6}",
            binary.as_secs_f64(),
            radix4.as_secs_f64(),
            radix4.as_secs_f64() / binary.as_secs_f64(),
        );
    }
}

fn main() {
    if std::env::args().any(|argument| argument == "--gate-run") {
        gate_run();
        return;
    }

    let mut criterion = Criterion::default().configure_from_args();
    criterion_benchmark(&mut criterion);
    criterion.final_summary();
}
