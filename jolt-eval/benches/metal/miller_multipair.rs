//! W4-fly isolated objective: the stage-8 reduce-round multi-pairing hook
//! (`multi_pair_device`) at production dispatch sizes, fused fly kernel
//! (`JOLT_MILLER_FLY_SPLIT=0`, the default) vs the split ladder (`=1`).
//! The pipeline register cap toggles at context build, so capped arms need
//! their own invocation: `JOLT_METAL_PAIRING_TG_CAP=64 cargo bench …` vs
//! the uncapped default (W4-fly verdict: cap 64 is −8.6% at 2^13 but only
//! −2.8% at 2^17 — below the retention bar, ships uncapped).
//!
//! Geometry (2^27, `.journals/lane-reports/metal-w3-st8.md`): the resident
//! Dory loop serves 6 multi-pairs of n/2 pairs per round, n/2 from 2^17
//! down to the 2^11 gate. `2^17` is the round-0 shape carrying most of the
//! mass; `2^13` is a mid-ladder round. Arms toggle `JOLT_MILLER_FLY_SPLIT`
//! (read per call by design); the GT of every arm is asserted identical
//! before timing — the restructure may not change a single output bit.
//! (The W3 CPU-share arms were retired with that door: `JOLT_MILLER_CPU_PCT`
//! stays at its measured-NO-GO default 0 here.)
//!
//! Each Criterion sample is one synchronous hook call under `gpu_lock()`,
//! GPU otherwise idle.

#![expect(
    clippy::expect_used,
    reason = "benchmark harness must fail loudly on device errors"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{Criterion, Throughput};
    use jolt_kernels::metal::miller::{
        multi_pair_device, seeded_pairing_inputs, ENV_MILLER_FLY_SPLIT,
    };
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::MetalContext;

    /// (env value, arm label): fused kernel first so the split arm never
    /// rides warmer silicon than its baseline.
    const ARMS: [(&str, &str); 2] = [("0", "fused"), ("1", "split")];

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        MetalContext::global().expect("Metal context");

        for log_n in [13usize, 17] {
            let n = 1usize << log_n;
            let (ps, qs) = seeded_pairing_inputs(0x57_38 ^ log_n as u64, n);

            // Parity gate before any timing: both arms serve the call and
            // produce the identical GT.
            let reference = {
                std::env::set_var(ENV_MILLER_FLY_SPLIT, ARMS[0].0);
                multi_pair_device(&ps, &qs).expect("fused arm serves the call")
            };
            for (value, label) in &ARMS[1..] {
                std::env::set_var(ENV_MILLER_FLY_SPLIT, value);
                let arm = multi_pair_device(&ps, &qs).expect("split arm serves the call");
                assert_eq!(arm.0, reference.0, "GT drift in the {label} arm");
            }

            let mut group = c.benchmark_group("miller_multipair");
            group.sample_size(10);
            group.warm_up_time(Duration::from_secs(2));
            group.measurement_time(Duration::from_secs(8));
            group.throughput(Throughput::Elements(n as u64));
            for (value, label) in ARMS {
                std::env::set_var(ENV_MILLER_FLY_SPLIT, value);
                group.bench_function(format!("2^{log_n}/{label}"), |bencher| {
                    bencher.iter(|| {
                        multi_pair_device(black_box(&ps), black_box(&qs))
                            .expect("hook serves the call")
                    });
                });
            }
            group.finish();
        }
        std::env::remove_var(ENV_MILLER_FLY_SPLIT);
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("miller_multipair requires macOS");
}
