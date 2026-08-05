//! W2-st6b lane microbench: the stage-6b lazy-RA device round pipelines in
//! isolation (see `metal::st6b_bench` for the protocol and arm
//! definitions) — the instruction-RA-virtualization slot and the
//! booleanity-cycle slot, sync vs deferred adoption.
//!
//! ```text
//! cargo run --release -p jolt-kernels --example st6b_rav_microbench \
//!     --features metal -- [--slot rav|bool|both] [--log-t 22,24] \
//!     [--passes 3] [--skip-oracle]
//! ```
//!
//! Arms run in balanced AB/BA order across passes (same-window interleave;
//! min plus all samples reported), GPU otherwise idle, every device number
//! bracketed by the pipeline's own waits. `JOLT_METAL_CB_TRACE=1` adds
//! per-command-buffer gpu/blocked audit lines.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "benchmark harness: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("st6b_rav_microbench requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use jolt_kernels::metal::st6b_bench::{
        assert_arm_parity, assert_bool_arm_parity, bench_rows, bool_bench_rows,
        run_bool_device_pass, run_device_pass, BenchRows, BoolBenchConfig, PassTiming,
        RavBenchConfig,
    };
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::MetalContext;

    struct Args {
        log_ts: Vec<usize>,
        passes: usize,
        oracle: bool,
        rav: bool,
        bool_slot: bool,
    }

    fn parse_args() -> Args {
        let mut args = Args {
            log_ts: vec![22, 24],
            passes: 3,
            oracle: true,
            rav: true,
            bool_slot: true,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--log-t" => {
                    let value = iter.next().expect("--log-t needs a value");
                    args.log_ts = value
                        .split(',')
                        .map(|s| s.trim().parse().expect("log-t"))
                        .collect();
                }
                "--passes" => {
                    args.passes = iter
                        .next()
                        .expect("--passes needs a value")
                        .parse()
                        .unwrap();
                }
                "--skip-oracle" => args.oracle = false,
                "--slot" => match iter.next().expect("--slot needs a value").as_str() {
                    "rav" => args.bool_slot = false,
                    "bool" => args.rav = false,
                    "both" => {}
                    other => panic!("unknown slot {other}"),
                },
                other => panic!("unknown argument {other}"),
            }
        }
        args
    }

    fn report(arm: &str, log_t: usize, best: &PassTiming) {
        println!(
            "  {arm:>13} 2^{log_t}: total {:>8.2} ms | lazy {:>8.2} ms | adopt {:>8.2} ms | dense {:>7.2} ms | host-tail {:>7.2} ms | GPU waits {:>8.2} ms",
            best.total_s * 1e3,
            best.lazy_span_s() * 1e3,
            best.adopt_span_s() * 1e3,
            best.dense_device_span_s() * 1e3,
            best.host_tail_span_s() * 1e3,
            best.launched_wait_s() * 1e3,
        );
        let hot: Vec<String> = best
            .rounds
            .iter()
            .enumerate()
            .filter(|(_, (b, c))| (b + c) * 1e3 >= 1.0)
            .map(|(r, (b, c))| {
                let phase = match r {
                    0 => "w1",
                    1 => "w2",
                    2 => "w4",
                    _ if r == best.adopt_round => "adopt",
                    3 => "w8",
                    _ if best.launched[r] => "dense",
                    _ => "host",
                };
                format!("r{r}/{phase} {:.1}+{:.1}", b * 1e3, c * 1e3)
            })
            .collect();
        println!("           rounds ≥1ms (begin+collect): {}", hot.join("  "));
    }

    fn measure(
        label: &str,
        arms: [&str; 2],
        log_t: usize,
        passes: usize,
        mut pass: impl FnMut(bool) -> PassTiming,
    ) {
        let mut samples: [Vec<PassTiming>; 2] = [Vec::new(), Vec::new()];
        for pass_index in 0..passes {
            let order = if pass_index % 2 == 0 {
                [(0usize, false), (1usize, true)]
            } else {
                [(1usize, true), (0usize, false)]
            };
            for (slot, selected) in order {
                samples[slot].push(pass(selected));
            }
        }
        for slot in 0..2 {
            let best = samples[slot]
                .iter()
                .min_by(|left, right| left.total_s.total_cmp(&right.total_s))
                .unwrap();
            report(&format!("{label} {}", arms[slot]), log_t, best);
            let totals = samples[slot]
                .iter()
                .map(|timing| format!("{:.2}", timing.total_s * 1e3))
                .collect::<Vec<_>>()
                .join(", ");
            let lazy = samples[slot]
                .iter()
                .map(|timing| format!("{:.2}", timing.lazy_span_s() * 1e3))
                .collect::<Vec<_>>()
                .join(", ");
            println!("           samples total [{totals}] ms | lazy [{lazy}] ms");
        }
    }

    pub fn run() {
        let _lock = gpu_lock();
        let _context = MetalContext::global().expect("Metal context");
        let args = parse_args();
        // Full-device pipeline regardless of size (the bench sizes are all
        // production-shaped; gates stay honest in production).
        std::env::remove_var("JOLT_METAL_DISABLE");
        for &log_t in &args.log_ts {
            if args.rav {
                let config = RavBenchConfig::production(log_t);
                println!(
                    "st6b RAV pipeline 2^{log_t} (16 committed, batch 4, 8-bit chunks), min over {} passes:",
                    args.passes
                );
                let rows = bench_rows(&config);
                if args.oracle {
                    assert_arm_parity(&config, &rows, false);
                    assert_arm_parity(&config, &rows, true);
                    println!("  oracle: both arms byte-equal to the CPU twin ✓");
                }
                measure("rav", ["sync", "defer"], log_t, args.passes, |deferred| {
                    run_device_pass(&config, &rows, deferred)
                });
            }
            if args.bool_slot {
                let config = BoolBenchConfig::production(log_t);
                println!(
                    "st6b Bool pipeline 2^{log_t} (20 polys = 16 instr + 2 bytecode + 2 ram, 8-bit chunks), min over {} passes:",
                    args.passes
                );
                let rows: BenchRows = bool_bench_rows(&config);
                if args.oracle {
                    assert_bool_arm_parity(&config, &rows, false);
                    assert_bool_arm_parity(&config, &rows, true);
                    println!("  oracle: both arms byte-equal to the CPU twin ✓");
                }
                measure("bool", ["sync", "defer"], log_t, args.passes, |deferred| {
                    run_bool_device_pass(&config, &rows, deferred)
                });
            }
        }
    }
}
