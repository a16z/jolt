//! W2-st6b lane microbench: the stage-6b instruction-RA-virtualization
//! device round pipeline in isolation (see `metal::st6b_bench` for the
//! protocol and arm definitions).
//!
//! ```text
//! cargo run --release -p jolt-kernels --example st6b_rav_microbench \
//!     --features metal -- [--log-t 22,24] [--passes 3] [--skip-oracle]
//! ```
//!
//! Arms alternate sync/deferred within each pass (ABBA-free interleave, min
//! over passes reported), GPU otherwise idle, every device number bracketed
//! by the pipeline's own waits. `JOLT_METAL_CB_TRACE=1` adds per-command-
//! buffer gpu/blocked audit lines.

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
        assert_arm_parity, bench_rows, run_device_pass, RavBenchConfig, RavPassTiming,
    };
    use jolt_kernels::metal::testing::gpu_lock;

    struct Args {
        log_ts: Vec<usize>,
        passes: usize,
        oracle: bool,
    }

    fn parse_args() -> Args {
        let mut args = Args {
            log_ts: vec![22, 24],
            passes: 3,
            oracle: true,
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
                other => panic!("unknown argument {other}"),
            }
        }
        args
    }

    fn report(arm: &str, log_t: usize, best: &RavPassTiming) {
        println!(
            "  {arm:>8} 2^{log_t}: total {:>8.2} ms | adopt r{} begin {:>8.2} ms span {:>8.2} ms | Σbegin {:>8.2} ms",
            best.total_s * 1e3,
            best.adopt_round,
            best.adopt_begin_s() * 1e3,
            best.adopt_span_s() * 1e3,
            best.begin_serial_s() * 1e3,
        );
        let hot: Vec<String> = best
            .rounds
            .iter()
            .enumerate()
            .filter(|(_, (b, c))| (b + c) * 1e3 >= 1.0)
            .map(|(r, (b, c))| format!("r{r} {:.1}+{:.1}", b * 1e3, c * 1e3))
            .collect();
        println!("           rounds ≥1ms (begin+collect): {}", hot.join("  "));
    }

    pub fn run() {
        let _lock = gpu_lock();
        let args = parse_args();
        // Full-device pipeline regardless of size (the bench sizes are all
        // production-shaped; gates stay honest in production).
        std::env::remove_var("JOLT_METAL_DISABLE");
        for &log_t in &args.log_ts {
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
            let mut best: [Option<RavPassTiming>; 2] = [None, None];
            for _ in 0..args.passes {
                for (slot, deferred) in [(0usize, false), (1usize, true)] {
                    let pass = run_device_pass(&config, &rows, deferred);
                    let better = best[slot]
                        .as_ref()
                        .is_none_or(|old| pass.total_s < old.total_s);
                    if better {
                        best[slot] = Some(pass);
                    }
                }
            }
            report("sync", log_t, best[0].as_ref().unwrap());
            report("deferred", log_t, best[1].as_ref().unwrap());
        }
    }
}
