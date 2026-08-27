//! Metal single-kernel benchmark scaffold.
//!
//! Setup and buffer wrapping stay outside the timed loop; every sample is a
//! synchronous command-buffer dispatch, so Criterion observes completed GPU
//! work rather than host enqueue latency.

#![expect(
    clippy::expect_used,
    reason = "benchmark harness must fail loudly on device errors"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{Criterion, Throughput};
    use jolt_field::{Fr, Ring};
    use jolt_kernels::metal::testing::{gpu_lock, seeded_frs};
    use jolt_kernels::metal::{fr_to_u32_limbs, KernelId, MetalContext, PageAlignedVec};

    pub fn benchmark(c: &mut Criterion) {
        const LOG_INPUTS: usize = 20;
        let input_len = 1usize << LOG_INPUTS;
        let output_len = input_len / 2;
        let _gpu_lock = gpu_lock();
        let context = MetalContext::global().expect("Metal context");
        let input = PageAlignedVec::from_fn(input_len, |index| Fr::from_u64(index as u64 + 1));
        let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), output_len);
        let input_buffer = input.device_buffer(context).expect("input buffer");
        let output_buffer = output.device_buffer_mut(context).expect("output buffer");
        let mut params = vec![output_len as u32];
        params.extend_from_slice(&fr_to_u32_limbs(seeded_frs(0x4d45_5441, 1)[0]));

        context
            .run_once(
                KernelId::FrBind,
                &params,
                &[&input_buffer, &output_buffer],
                output_len,
            )
            .expect("warm dispatch");

        let mut group = c.benchmark_group("metal_kernel");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(input_len as u64));
        group.bench_function(format!("fr_bind/2^{LOG_INPUTS}"), |bencher| {
            bencher.iter(|| {
                context
                    .run_once(
                        KernelId::FrBind,
                        black_box(&params),
                        &[&input_buffer, &output_buffer],
                        output_len,
                    )
                    .expect("FrBind dispatch");
            });
        });
        group.finish();
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_fr_bind requires macOS");
}
