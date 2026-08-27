//! Metal microbenchmark harness (campaign ceiling-analysis input).
//!
//! ```text
//! cargo run --release -p jolt-kernels --example metal_microbench --features metal
//! ```
//!
//! Measures, on the current machine (G-rule: min over ≥3 warm passes, GPU
//! otherwise idle, every device number bracketed by `waitUntilCompleted`):
//!
//! - **D1** empty-kernel dispatch latency: full commit+wait round trip for a
//!   1-dispatch command buffer, and per-dispatch cost inside a 100-dispatch
//!   command buffer (separates commit overhead from encode/dispatch cost).
//! - **D2** streaming bind throughput (GB/s) at 2^20/2^22/2^24 input
//!   elements, device vs CPU rayon, both zero-copy on the device side.
//! - **D2r4** one radix-4 Lagrange bind vs two binary binds in one command
//!   buffer, isolating the intermediate-table traffic removed by fusion.
//! - **D2b** device-vs-CPU bind cutover sweep — the evidence behind
//!   `JOLT_METAL_MIN_TERMS`.
//! - **D3** compute-bound Montgomery throughput: chained squarings
//!   (`x^(2^64)`), device vs 1-thread and all-core CPU.
//! - **D4** bus contention: device streaming bind co-running with a CPU
//!   rayon field-mul loop on separate data; both sides' degradation vs solo.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "benchmark harness: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("metal_microbench requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;
    use std::time::Instant;

    use jolt_field::{Fr, Ring};
    use jolt_kernels::metal::testing::{gpu_lock, host_bind, seeded_frs};
    use jolt_kernels::metal::{
        fr_to_u32_limbs, DeviceBuffer, KernelId, MetalContext, PageAlignedVec, FR_U32_LIMBS,
    };
    use jolt_poly::UnivariatePoly;
    use rayon::prelude::*;

    const FR_BYTES: usize = FR_U32_LIMBS * 4;

    /// Minimum of `passes` timed runs of `f`, after one warm pass.
    fn min_secs(passes: usize, mut f: impl FnMut()) -> f64 {
        f();
        (0..passes)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Cheap deterministic fill: throughput here is data-oblivious, so a
    /// splitmix64 stream through `from_u64` is enough variety.
    fn fill_frs(n: usize) -> Vec<Fr> {
        (0..n as u64)
            .map(|i| {
                let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                Fr::from_u64(z)
            })
            .collect()
    }

    fn bind_params(n_out: usize, r: Fr) -> Vec<u32> {
        let mut params = vec![n_out as u32];
        params.extend_from_slice(&fr_to_u32_limbs(r));
        params
    }

    fn bind4_params(n_out: usize, z: Fr) -> Vec<u32> {
        let mut params = vec![n_out as u32];
        for index in 1..4 {
            params.extend_from_slice(&fr_to_u32_limbs(UnivariatePoly::evaluate_basis(
                4, index, z,
            )));
        }
        params
    }

    fn cpu_bind(a: &[Fr], out: &mut [Fr], r: Fr) {
        out.par_iter_mut()
            .zip(a.par_chunks_exact(2))
            .for_each(|(o, p)| *o = p[0] + r * (p[1] - p[0]));
    }

    /// One synchronous device bind pass over pre-wrapped buffers.
    fn device_bind_pass(
        ctx: &MetalContext,
        a: &DeviceBuffer<'_>,
        out: &DeviceBuffer<'_>,
        params: &[u32],
        n_out: usize,
    ) {
        ctx.run_once(KernelId::FrBind, params, &[a, out], n_out)
            .expect("bind dispatch");
    }

    pub fn run() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        println!("device: {}", ctx.device_name());
        println!(
            "cpu threads: {} | protocol: min over ≥3 warm passes, sync-bracketed\n",
            rayon::current_num_threads()
        );

        d1_dispatch_latency(ctx);
        let bind_rows = d2_streaming_bind(ctx);
        d2r4_radix4_bind(ctx);
        let cutover = d2b_cutover_sweep(ctx);
        d3_compute_bound(ctx);
        d4_bus_contention(ctx);

        println!("\n== summary ==");
        for row in bind_rows {
            println!("{row}");
        }
        if cutover == usize::MAX {
            println!("bind cutover: device never beat all-core CPU in the sweep");
        } else {
            println!("bind cutover (device beats all-core CPU): ~2^{cutover}");
        }
    }

    fn d1_dispatch_latency(ctx: &MetalContext) {
        println!("== D1: dispatch latency (empty kernel) ==");

        // Single dispatch per command buffer: the full synchronous round trip.
        for _ in 0..10 {
            ctx.run_once(KernelId::Noop, &[], &[], 1).unwrap();
        }
        let mut best = f64::INFINITY;
        for _ in 0..1000 {
            let t = Instant::now();
            ctx.run_once(KernelId::Noop, &[], &[], 1).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "1 dispatch / 1 CB round trip: {:.1} µs (min of 1000)",
            best * 1e6
        );

        // 100 dispatches in one command buffer: amortized encode+dispatch.
        let hundred = min_secs(30, || {
            let mut pass = ctx.begin_pass().unwrap();
            for _ in 0..100 {
                pass.dispatch(KernelId::Noop, &[], &[], 1);
            }
            pass.run().unwrap();
        });
        println!(
            "100 dispatches / 1 CB: {:.1} µs total → {:.2} µs/dispatch (CB commit+wait overhead ≈ {:.1} µs)\n",
            hundred * 1e6,
            hundred * 1e6 / 100.0,
            (best - hundred / 100.0).max(0.0) * 1e6,
        );
    }

    fn d2_streaming_bind(ctx: &MetalContext) -> Vec<String> {
        println!("== D2: streaming bind throughput (out[i] = a[2i] + r·(a[2i+1]-a[2i])) ==");
        let r = seeded_frs(42, 1)[0];
        let mut rows = Vec::new();

        for log_n in [20usize, 22, 24] {
            let n_in = 1usize << log_n;
            let n_out = n_in / 2;
            let bytes = (n_in + n_out) * FR_BYTES;

            let input = PageAlignedVec::from_slice(&fill_frs(n_in));
            let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_out);
            let params = bind_params(n_out, r);

            let device = {
                let a_buf = input.device_buffer(ctx).unwrap();
                let out_buf = output.device_buffer_mut(ctx).unwrap();
                assert!(!a_buf.was_copied() && !out_buf.was_copied());
                min_secs(5, || {
                    device_bind_pass(ctx, &a_buf, &out_buf, &params, n_out);
                })
            };

            // Spot-check correctness once per size.
            assert_eq!(output[0], host_bind(&input[..2], r)[0]);

            let mut cpu_out = vec![Fr::from_u64(0); n_out];
            let cpu = min_secs(5, || cpu_bind(&input, &mut cpu_out, r));

            let dev_gbs = bytes as f64 / device / 1e9;
            let cpu_gbs = bytes as f64 / cpu / 1e9;
            let row = format!(
                "bind 2^{log_n}: device {:.2} ms ({dev_gbs:.1} GB/s) | cpu {:.2} ms ({cpu_gbs:.1} GB/s) | device/cpu {:.2}×",
                device * 1e3,
                cpu * 1e3,
                cpu / device,
            );
            println!("{row}");
            rows.push(row);
        }
        println!();
        rows
    }

    fn d2r4_radix4_bind(ctx: &MetalContext) {
        println!("== D2r4: radix-4 direct bind vs two binary binds / one CB ==");
        let [r0, r1, z] = seeded_frs(0x4ad1, 3).try_into().unwrap();

        for log_n in [20usize, 22, 24] {
            let n_in = 1usize << log_n;
            let n_half = n_in / 2;
            let n_out = n_in / 4;
            let input = PageAlignedVec::from_slice(&fill_frs(n_in));
            let input_buf = input.device_buffer(ctx).unwrap();
            let intermediate = ctx.alloc_u32s(n_half * FR_U32_LIMBS).unwrap();
            let binary_out = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
            let radix4_out = ctx.alloc_u32s(n_out * FR_U32_LIMBS).unwrap();
            let first_params = bind_params(n_half, r0);
            let second_params = bind_params(n_out, r1);
            let radix4_params = bind4_params(n_out, z);

            let binary = min_secs(5, || {
                let mut pass = ctx.begin_pass().unwrap();
                pass.dispatch(
                    KernelId::FrBind,
                    &first_params,
                    &[&input_buf, &intermediate],
                    n_half,
                );
                pass.dispatch(
                    KernelId::FrBind,
                    &second_params,
                    &[&intermediate, &binary_out],
                    n_out,
                );
                pass.run().unwrap();
            });
            let radix4 = min_secs(5, || {
                ctx.run_once(
                    KernelId::FrBind4,
                    &radix4_params,
                    &[&input_buf, &radix4_out],
                    n_out,
                )
                .unwrap();
            });
            let binary_bytes = (n_in + 2 * n_half + n_out) * FR_BYTES;
            let radix4_bytes = (n_in + n_out) * FR_BYTES;

            println!(
                "2^{log_n}: binary×2 {:.2} ms ({:.1} GB/s) | radix-4 {:.2} ms ({:.1} GB/s) | speedup {:.2}×",
                binary * 1e3,
                binary_bytes as f64 / binary / 1e9,
                radix4 * 1e3,
                radix4_bytes as f64 / radix4 / 1e9,
                binary / radix4,
            );
        }
        println!();
    }

    /// Sweep small sizes for the device-vs-CPU cutover that motivates the
    /// `JOLT_METAL_MIN_TERMS` default.
    fn d2b_cutover_sweep(ctx: &MetalContext) -> usize {
        println!("== D2b: bind cutover sweep (device incl. dispatch vs all-core CPU) ==");
        let r = seeded_frs(43, 1)[0];
        let mut cutover = usize::MAX;

        for log_n in 12..=21 {
            let n_in = 1usize << log_n;
            let n_out = n_in / 2;
            let input = PageAlignedVec::from_slice(&fill_frs(n_in));
            let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_out);
            let params = bind_params(n_out, r);

            let device = {
                let a_buf = input.device_buffer(ctx).unwrap();
                let out_buf = output.device_buffer_mut(ctx).unwrap();
                min_secs(5, || {
                    device_bind_pass(ctx, &a_buf, &out_buf, &params, n_out);
                })
            };

            let mut cpu_out = vec![Fr::from_u64(0); n_out];
            let cpu = min_secs(5, || cpu_bind(&input, &mut cpu_out, r));

            let winner = if device < cpu { "device" } else { "cpu" };
            println!(
                "2^{log_n:2}: device {:8.1} µs | cpu {:8.1} µs | {winner}",
                device * 1e6,
                cpu * 1e6
            );
            if device < cpu && cutover == usize::MAX {
                cutover = log_n;
            }
        }
        println!();
        cutover
    }

    fn d3_compute_bound(ctx: &MetalContext) {
        println!("== D3: compute-bound mont_mul (x ← x², chained k=64) ==");
        let k = 64u32;

        // Device: 2^20 elements × 64 dependent squarings.
        let n_dev = 1usize << 20;
        let muls_dev = (n_dev as f64) * f64::from(k);
        let input = PageAlignedVec::from_slice(&fill_frs(n_dev));
        let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_dev);
        let device = {
            let a_buf = input.device_buffer(ctx).unwrap();
            let out_buf = output.device_buffer_mut(ctx).unwrap();
            min_secs(5, || {
                ctx.run_once(
                    KernelId::FrPow2k,
                    &[n_dev as u32, k],
                    &[&a_buf, &out_buf],
                    n_dev,
                )
                .unwrap();
            })
        };
        println!(
            "device (2^20 elems): {:.2} ms → {:.2} Gmul/s",
            device * 1e3,
            muls_dev / device / 1e9
        );

        // CPU single-thread.
        let n_1t = 1usize << 16;
        let cpu_in = fill_frs(n_1t);
        let cpu_1t = min_secs(3, || {
            for &x in &cpu_in {
                let mut acc = x;
                for _ in 0..k {
                    acc = acc * acc;
                }
                let _ = std::hint::black_box(acc);
            }
        });
        println!(
            "cpu 1 thread (2^16 elems): {:.2} ms → {:.2} Gmul/s",
            cpu_1t * 1e3,
            (n_1t as f64) * f64::from(k) / cpu_1t / 1e9
        );

        // CPU all cores.
        let n_mt = 1usize << 20;
        let cpu_mt_in = fill_frs(n_mt);
        let cpu_mt = min_secs(3, || {
            cpu_mt_in.par_iter().for_each(|&x| {
                let mut acc = x;
                for _ in 0..k {
                    acc = acc * acc;
                }
                let _ = std::hint::black_box(acc);
            });
        });
        println!(
            "cpu all cores (2^20 elems): {:.2} ms → {:.2} Gmul/s\n",
            cpu_mt * 1e3,
            (n_mt as f64) * f64::from(k) / cpu_mt / 1e9
        );
    }

    /// Windowed passes-per-second rate; returns bytes/s given per-pass bytes.
    fn windowed_rate(window_secs: f64, bytes_per_pass: usize, mut f: impl FnMut()) -> f64 {
        f();
        let start = Instant::now();
        let mut passes = 0usize;
        while start.elapsed().as_secs_f64() < window_secs {
            f();
            passes += 1;
        }
        (passes * bytes_per_pass) as f64 / start.elapsed().as_secs_f64()
    }

    fn d4_bus_contention(ctx: &'static MetalContext) {
        println!("== D4: bus contention (device bind ∥ CPU rayon field-mul, separate data) ==");

        // Device workload: bind over 2^22 input elements.
        let n_in = 1usize << 22;
        let n_out = n_in / 2;
        let dev_bytes = (n_in + n_out) * FR_BYTES;
        let r = seeded_frs(44, 1)[0];

        // CPU workload: elementwise c[i] = a[i]·b[i] over 2^22 elements.
        let n_cpu = 1usize << 22;
        let cpu_bytes = 3 * n_cpu * FR_BYTES;
        let cpu_a = fill_frs(n_cpu);
        let cpu_b = fill_frs(n_cpu);
        let mut cpu_c = vec![Fr::from_u64(0); n_cpu];

        // Solo baselines: best of 3 one-second windows each.
        let input = PageAlignedVec::from_slice(&fill_frs(n_in));
        let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_out);
        let params = bind_params(n_out, r);
        let dev_solo = {
            let a_buf = input.device_buffer(ctx).unwrap();
            let out_buf = output.device_buffer_mut(ctx).unwrap();
            (0..3)
                .map(|_| {
                    windowed_rate(1.0, dev_bytes, || {
                        device_bind_pass(ctx, &a_buf, &out_buf, &params, n_out);
                    })
                })
                .fold(0.0f64, f64::max)
        };
        let cpu_solo = (0..3)
            .map(|_| {
                windowed_rate(1.0, cpu_bytes, || {
                    cpu_c
                        .par_iter_mut()
                        .zip(cpu_a.par_iter().zip(cpu_b.par_iter()))
                        .for_each(|(c, (a, b))| *c = *a * *b);
                })
            })
            .fold(0.0f64, f64::max);
        println!(
            "solo: device {:.1} GB/s | cpu {:.1} GB/s ({:.2} Gmul/s)",
            dev_solo / 1e9,
            cpu_solo / 1e9,
            cpu_solo / (3.0 * FR_BYTES as f64) / 1e9,
        );

        // Co-run: device loop on its own thread, CPU rayon loop here, both
        // between a start barrier and a shared stop flag.
        let barrier = Barrier::new(2);
        let stop = AtomicBool::new(false);
        let window = 3.0f64;

        let (dev_co, cpu_co) = std::thread::scope(|scope| {
            let device_thread = scope.spawn(|| {
                let input = PageAlignedVec::from_slice(&fill_frs(n_in));
                let mut output = PageAlignedVec::from_elem(Fr::from_u64(0), n_out);
                let a_buf = input.device_buffer(ctx).unwrap();
                let out_buf = output.device_buffer_mut(ctx).unwrap();
                let params = bind_params(n_out, r);
                // Warm before synchronizing.
                device_bind_pass(ctx, &a_buf, &out_buf, &params, n_out);

                let _ = barrier.wait();
                let start = Instant::now();
                let mut passes = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    device_bind_pass(ctx, &a_buf, &out_buf, &params, n_out);
                    passes += 1;
                }
                (passes * dev_bytes) as f64 / start.elapsed().as_secs_f64()
            });

            // Warm the CPU side too, then synchronize.
            cpu_c
                .par_iter_mut()
                .zip(cpu_a.par_iter().zip(cpu_b.par_iter()))
                .for_each(|(c, (a, b))| *c = *a * *b);
            let _ = barrier.wait();
            let start = Instant::now();
            let mut passes = 0usize;
            while start.elapsed().as_secs_f64() < window {
                cpu_c
                    .par_iter_mut()
                    .zip(cpu_a.par_iter().zip(cpu_b.par_iter()))
                    .for_each(|(c, (a, b))| *c = *a * *b);
                passes += 1;
            }
            let cpu_rate = (passes * cpu_bytes) as f64 / start.elapsed().as_secs_f64();
            stop.store(true, Ordering::Relaxed);
            let dev_rate = device_thread.join().expect("device thread");
            (dev_rate, cpu_rate)
        });

        println!(
            "co-run: device {:.1} GB/s ({:+.0}%) | cpu {:.1} GB/s ({:+.0}%)",
            dev_co / 1e9,
            (dev_co / dev_solo - 1.0) * 100.0,
            cpu_co / 1e9,
            (cpu_co / cpu_solo - 1.0) * 100.0,
        );
        println!(
            "combined bus draw: solo-sum {:.1} GB/s → co-run {:.1} GB/s",
            (dev_solo + cpu_solo) / 1e9,
            (dev_co + cpu_co) / 1e9,
        );
    }
}
