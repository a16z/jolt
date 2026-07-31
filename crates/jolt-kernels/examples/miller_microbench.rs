//! W6 Miller-loop microbenchmark — the phase-1 go/no-go evidence.
//!
//! ```text
//! cargo run --release -p jolt-kernels --example miller_microbench --features metal
//! ```
//!
//! Measures (G-rule: min over ≥3 warm passes, sync-bracketed, GPU otherwise
//! idle):
//!
//! - **T1** tower op rates: dense Fq12 mul / squaring / sparse 034 chains —
//!   the register-pressure probe (a cratered chain rate means spills).
//! - **T2** `jk_miller_table` (stage-0 shape) at one tier-2 column's pair
//!   count, pairs-per-thread swept, plus a thread-scaling probe as the
//!   occupancy proxy.
//! - **T3** `jk_miller_fly` (stage-8 shape) at the same pair count.
//! - **C1/C2** CPU references on the same pairs: arkworks
//!   `multi_miller_loop` all-core (prepared in advance, 512-pair chunks =
//!   `jolt_dory::tier2`'s shape) and the absorb-real variant including the
//!   per-pair prepared-G2 clone; plus a single-thread run for the
//!   thread-normalized rate.
//! - **P1** the one-time coefficient flatten cost (per commit pass).
//! - **X1** device Miller co-running with an all-core CPU field-mul soak —
//!   the first-order contention picture for the st0 pipeline.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "benchmark harness: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("miller_microbench requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;
    use std::time::Instant;

    use ark_bn254::{Bn254, Fq12, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::pairing::{MillerLoopOutput, Pairing};
    use ark_ec::CurveGroup;
    use ark_ff::{One, UniformRand};
    use jolt_kernels::metal::miller::{
        ell_coeffs_per_pair, flatten_prepared_coeffs, fq12_to_device_limbs,
        miller_fly_indexed_partials, miller_fly_partials, miller_table_partials,
        product_of_partials, uniform_seg_starts, ArkG2Prepared, ELL_COEFF_U32S, FQ12_U32S,
    };
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::{KernelId, MetalContext};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use rayon::prelude::*;

    /// One tier-2 column's rows at 2^22 (and, flat across scales, 2^23).
    const N_PAIRS: usize = 8192;

    /// Fq-mul equivalents per op (Karatsuba tower: Fq12 mul = 18 Fq2 = 54,
    /// sqr = 12 Fq2 = 36, 034 = 13 Fq2 + 2 fp-scalings = 43).
    const MUL_EQ_FQ12_MUL: f64 = 54.0;
    const MUL_EQ_FQ12_SQR: f64 = 36.0;
    const MUL_EQ_034: f64 = 43.0;

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

    pub fn run() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        println!("device: {}", ctx.device_name());
        println!(
            "cpu threads: {} | pairs: {N_PAIRS} | coeffs/pair: {} | protocol: min over ≥3 warm passes\n",
            rayon::current_num_threads(),
            ell_coeffs_per_pair(),
        );

        let mut rng = ChaCha20Rng::seed_from_u64(0x6d69);
        // Distinct-enough fixtures without minutes of point sampling: small
        // random seed sets, permuted (Miller cost is data-oblivious).
        let g1_seeds: Vec<G1Affine> = (0..128)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let g2_seeds: Vec<G2Affine> = (0..128)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect();
        let ps: Vec<G1Affine> = (0..N_PAIRS).map(|i| g1_seeds[(i * 31 + 7) % 128]).collect();
        let qs: Vec<G2Affine> = (0..N_PAIRS).map(|i| g2_seeds[(i * 17 + 3) % 128]).collect();

        t1_tower_rates(ctx, &mut rng);
        let flatten = p1_flatten(&qs);
        let dev_table = t2_miller_table(ctx, &ps, &flatten.coeffs);
        let dev_fly = t3_miller_fly(ctx, &ps, &qs);
        let (cpu_all, cpu_absorb, cpu_1t) = c_cpu_references(&ps, &qs, &flatten.preps);
        x1_contention(ctx, &ps, &flatten.coeffs);

        println!("\n== summary (per pair-eval) ==");
        let per = |w: f64| w / N_PAIRS as f64 * 1e6;
        println!("device table (best ppt): {:.2} µs", per(dev_table));
        println!("device fly:              {:.2} µs", per(dev_fly));
        println!(
            "cpu all-core prepared:   {:.2} µs  | absorb-real: {:.2} µs | 1-thread: {:.1} µs",
            per(cpu_all),
            per(cpu_absorb),
            cpu_1t / 512.0 * 1e6,
        );
        println!(
            "flatten (once per pass): {:.1} ms for {N_PAIRS} rows ({:.0} MB table)",
            flatten.secs * 1e3,
            (flatten.coeffs.len() * 4) as f64 / 1e6,
        );
        let speedup_prepared = cpu_all / dev_table;
        let speedup_absorb = cpu_absorb / dev_table;
        println!(
            "\ndevice-table vs cpu all-core: {speedup_prepared:.2}× (prepared) / {speedup_absorb:.2}× (absorb-real)"
        );
        println!(
            "projected @2^22 (328k pair-evals): device {:.2} s vs cpu-lane {:.2} s (absorb-real)",
            dev_table / N_PAIRS as f64 * 328_000.0,
            cpu_absorb / N_PAIRS as f64 * 328_000.0,
        );
    }

    fn t1_tower_rates(ctx: &MetalContext, rng: &mut ChaCha20Rng) {
        println!("== T1: tower op chains (n = 2^14 threads × k = 256 chained ops) ==");
        let n = 1usize << 14;
        let k = 256u32;
        let xs: Vec<Fq12> = (0..4).map(|_| Fq12::rand(rng)).collect();
        let x_limbs: Vec<u32> = (0..n)
            .flat_map(|i| fq12_to_device_limbs(&xs[i % 4]))
            .collect();
        let ops = n as f64 * f64::from(k);

        for (name, kernel, bufs, mul_eq) in [
            ("fq12 mul", KernelId::Fq12Mul, 3usize, MUL_EQ_FQ12_MUL),
            ("fq12 sqr", KernelId::Fq12Sqr, 2, MUL_EQ_FQ12_SQR),
        ] {
            let a_buf = ctx.wrap_slice(&x_limbs).unwrap();
            let b_buf = ctx.wrap_slice(&x_limbs).unwrap();
            let out_buf = ctx.alloc_u32s(n * FQ12_U32S).unwrap();
            let buffers: Vec<&jolt_kernels::metal::DeviceBuffer<'_>> = if bufs == 3 {
                vec![&a_buf, &b_buf, &out_buf]
            } else {
                vec![&a_buf, &out_buf]
            };
            let secs = min_secs(5, || {
                ctx.run_once(kernel, &[n as u32, k], &buffers, n).unwrap();
            });
            println!(
                "{name}: {:.2} ms → {:.2} Mop/s ≈ {:.2} Gmul-eq/s",
                secs * 1e3,
                ops / secs / 1e6,
                ops * mul_eq / secs / 1e9,
            );
        }

        // 034 chain: same coeff re-applied (register-resident) — pure ALU.
        let coeff_limbs: Vec<u32> = (0..n * ELL_COEFF_U32S)
            .map(|i| x_limbs[i % (2 * FQ12_U32S)])
            .collect();
        let mut g1_rng = ChaCha20Rng::seed_from_u64(9);
        let pt = G1Projective::rand(&mut g1_rng).into_affine();
        let pts: Vec<G1Affine> = vec![pt; n];
        let f_buf = ctx.wrap_slice(&x_limbs).unwrap();
        let c_buf = ctx.wrap_slice(&coeff_limbs).unwrap();
        let p_buf = ctx
            .wrap_slice(jolt_kernels::metal::bases_as_u32s(&pts))
            .unwrap();
        let out_buf = ctx.alloc_u32s(n * FQ12_U32S).unwrap();
        let secs = min_secs(5, || {
            ctx.run_once(
                KernelId::Fq12Mul034,
                &[n as u32, k],
                &[&f_buf, &c_buf, &p_buf, &out_buf],
                n,
            )
            .unwrap();
        });
        println!(
            "fq12 034: {:.2} ms → {:.2} Mop/s ≈ {:.2} Gmul-eq/s\n",
            secs * 1e3,
            ops / secs / 1e6,
            ops * MUL_EQ_034 / secs / 1e9,
        );
    }

    struct Flatten {
        preps: Vec<ArkG2Prepared>,
        coeffs: Vec<u32>,
        secs: f64,
    }

    fn p1_flatten(qs: &[G2Affine]) -> Flatten {
        let t = Instant::now();
        let preps: Vec<ArkG2Prepared> = qs.par_iter().map(|q| (*q).into()).collect();
        let prep_secs = t.elapsed().as_secs_f64();
        let refs: Vec<&ArkG2Prepared> = preps.iter().collect();
        let mut coeffs = Vec::new();
        let secs = min_secs(3, || flatten_prepared_coeffs(&refs, &mut coeffs));
        println!(
            "== P1: G2 preparation {:.0} ms (all-core) + flatten {:.0} ms, once per pass ==\n",
            prep_secs * 1e3,
            secs * 1e3
        );
        Flatten {
            preps,
            coeffs,
            secs,
        }
    }

    fn t2_miller_table(ctx: &MetalContext, ps: &[G1Affine], coeffs: &[u32]) -> f64 {
        println!("== T2: jk_miller_table ({N_PAIRS} pairs) ==");
        let indices: Vec<u32> = (0..N_PAIRS as u32).collect();
        let mut best = f64::INFINITY;
        for ppt in [2usize, 4, 8, 16, 32] {
            let segs = uniform_seg_starts(N_PAIRS, ppt);
            let secs = min_secs(3, || {
                let partials =
                    miller_table_partials(ctx, ps, &indices, &segs, coeffs, N_PAIRS).unwrap();
                let _ = std::hint::black_box(&partials);
            });
            // Host fold of the partials, timed apart (it stays on the CPU
            // in production, overlapped with the next dispatch).
            let partials =
                miller_table_partials(ctx, ps, &indices, &segs, coeffs, N_PAIRS).unwrap();
            let t = Instant::now();
            let product = product_of_partials(&partials);
            let fold = t.elapsed().as_secs_f64();
            let _ = std::hint::black_box(product);
            println!(
                "ppt {ppt:2}: {:.1} ms ({:.2} µs/pair) | host partial-fold {:.2} ms ({} partials)",
                secs * 1e3,
                secs / N_PAIRS as f64 * 1e6,
                fold * 1e3,
                partials.len() / FQ12_U32S,
            );
            best = best.min(secs);
        }

        // Production table shape (2 pairs/thread), directly comparable to
        // the indexed-fly scale sweep below.
        for n in [512usize, 1024, 2048, 4096, 8192] {
            let segs = uniform_seg_starts(n, 2);
            let secs = min_secs(3, || {
                let partials =
                    miller_table_partials(ctx, &ps[..n], &indices[..n], &segs, coeffs, N_PAIRS)
                        .unwrap();
                let _ = std::hint::black_box(&partials);
            });
            println!(
                "  scale ppt=2 n={n:5}: {:.1} ms ({:.2} µs/pair, {} threads)",
                secs * 1e3,
                secs / n as f64 * 1e6,
                n / 2,
            );
        }
        println!();
        best
    }

    fn t3_miller_fly(ctx: &MetalContext, ps: &[G1Affine], qs: &[G2Affine]) -> f64 {
        println!("== T3: jk_miller_fly ({N_PAIRS} pairs, 1/thread) ==");
        let direct_secs = min_secs(3, || {
            let partials = miller_fly_partials(ctx, ps, qs).unwrap();
            let _ = std::hint::black_box(&partials);
        });
        let indices: Vec<u32> = (0..N_PAIRS as u32)
            .map(|i| (i * 17 + 3) % N_PAIRS as u32)
            .collect();
        let secs = min_secs(3, || {
            let partials = miller_fly_indexed_partials(ctx, ps, &indices, qs).unwrap();
            let _ = std::hint::black_box(&partials);
        });
        println!(
            "fly direct:  {:.1} ms ({:.2} µs/pair)",
            direct_secs * 1e3,
            direct_secs / N_PAIRS as f64 * 1e6
        );
        println!(
            "fly indexed: {:.1} ms ({:.2} µs/pair)",
            secs * 1e3,
            secs / N_PAIRS as f64 * 1e6
        );
        for n in [512usize, 1024, 2048, 4096, 8192] {
            let scale_secs = min_secs(3, || {
                let partials =
                    miller_fly_indexed_partials(ctx, &ps[..n], &indices[..n], qs).unwrap();
                let _ = std::hint::black_box(&partials);
            });
            println!(
                "  scale n={n:5}: {:.1} ms ({:.2} µs/pair, {n} threads)",
                scale_secs * 1e3,
                scale_secs / n as f64 * 1e6,
            );
        }
        println!();
        secs
    }

    fn c_cpu_references(
        ps: &[G1Affine],
        qs: &[G2Affine],
        preps: &[ArkG2Prepared],
    ) -> (f64, f64, f64) {
        println!("== C1/C2: CPU references (same pairs) ==");

        // All-core, G2 prepared in advance, 512-pair chunks — the
        // best-case CPU lane (jolt_dory::tier2::MILLER_CHUNK shape).
        let cpu_all = min_secs(3, || {
            let f = ps
                .par_chunks(512)
                .zip(preps.par_chunks(512))
                .map(|(pc, qc)| Bn254::multi_miller_loop(pc.iter().copied(), qc.iter().cloned()))
                .reduce(
                    || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
                    |a, b| MillerLoopOutput(a.0 * b.0),
                );
            let _ = std::hint::black_box(f.0);
        });
        println!(
            "all-core prepared: {:.1} ms ({:.2} µs/pair, {:.1} thread-µs/pair)",
            cpu_all * 1e3,
            cpu_all / N_PAIRS as f64 * 1e6,
            cpu_all / N_PAIRS as f64 * 1e6 * rayon::current_num_threads() as f64,
        );

        // Absorb-real: Tier2Accumulator::absorb's exact shape — per-call
        // prepared-G2 clone included (row_indices gather).
        let indices: Vec<u32> = (0..N_PAIRS as u32).collect();
        let cpu_absorb = min_secs(3, || {
            let f = ps
                .par_chunks(512)
                .zip(indices.par_chunks(512))
                .map(|(pc, rows)| {
                    let qc: Vec<ArkG2Prepared> = rows
                        .iter()
                        .map(|&row| preps[row as usize].clone())
                        .collect();
                    Bn254::multi_miller_loop(pc.iter().copied(), qc)
                })
                .reduce(
                    || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
                    |a, b| MillerLoopOutput(a.0 * b.0),
                );
            let _ = std::hint::black_box(f.0);
        });
        println!(
            "absorb-real (clones): {:.1} ms ({:.2} µs/pair)",
            cpu_absorb * 1e3,
            cpu_absorb / N_PAIRS as f64 * 1e6,
        );

        // Single thread over 512 pairs: the thread-normalized rate.
        let cpu_1t = min_secs(3, || {
            let f =
                Bn254::multi_miller_loop(ps[..512].iter().copied(), preps[..512].iter().cloned());
            let _ = std::hint::black_box(f.0);
        });
        println!(
            "1-thread 512 pairs: {:.1} ms ({:.1} µs/pair)",
            cpu_1t * 1e3,
            cpu_1t / 512.0 * 1e6
        );

        // On-the-fly CPU twin for T3: unprepared multi_pair (prep included).
        let cpu_unprepared = min_secs(3, || {
            let f = ps
                .par_chunks(512)
                .zip(qs.par_chunks(512))
                .map(|(pc, qc)| Bn254::multi_miller_loop(pc.iter().copied(), qc.iter().copied()))
                .reduce(
                    || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
                    |a, b| MillerLoopOutput(a.0 * b.0),
                );
            let _ = std::hint::black_box(f.0);
        });
        println!(
            "all-core unprepared (fly twin): {:.1} ms ({:.2} µs/pair)\n",
            cpu_unprepared * 1e3,
            cpu_unprepared / N_PAIRS as f64 * 1e6,
        );

        (cpu_all, cpu_absorb, cpu_1t)
    }

    fn x1_contention(ctx: &'static MetalContext, ps: &[G1Affine], coeffs: &[u32]) {
        println!("== X1: device miller ∥ all-core CPU field-mul soak ==");
        let indices: Vec<u32> = (0..N_PAIRS as u32).collect();
        let segs = uniform_seg_starts(N_PAIRS, 8);
        let solo = min_secs(3, || {
            let partials =
                miller_table_partials(ctx, ps, &indices, &segs, coeffs, N_PAIRS).unwrap();
            let _ = std::hint::black_box(&partials);
        });

        let n_cpu = 1usize << 20;
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let a: Vec<ark_bn254::Fr> = (0..n_cpu).map(|_| ark_bn254::Fr::rand(&mut rng)).collect();
        let cpu_pass = |xs: &[ark_bn254::Fr]| {
            let s: ark_bn254::Fr = xs.par_iter().map(|x| *x * *x).sum();
            let _ = std::hint::black_box(s);
        };
        let cpu_solo = min_secs(3, || cpu_pass(&a));

        let barrier = Barrier::new(2);
        let stop = AtomicBool::new(false);
        let (dev_co, cpu_co) = std::thread::scope(|scope| {
            let dev = scope.spawn(|| {
                let _ = barrier.wait();
                let mut best = f64::INFINITY;
                let mut passes = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let t = Instant::now();
                    let partials =
                        miller_table_partials(ctx, ps, &indices, &segs, coeffs, N_PAIRS).unwrap();
                    let _ = std::hint::black_box(&partials);
                    best = best.min(t.elapsed().as_secs_f64());
                    passes += 1;
                }
                (best, passes)
            });
            let _ = barrier.wait();
            let start = Instant::now();
            let mut best = f64::INFINITY;
            while start.elapsed().as_secs_f64() < 3.0 {
                let t = Instant::now();
                cpu_pass(&a);
                best = best.min(t.elapsed().as_secs_f64());
            }
            stop.store(true, Ordering::Relaxed);
            let (dev_best, dev_passes) = dev.join().expect("device thread");
            assert!(dev_passes > 0, "device never completed a co-run pass");
            (dev_best, best)
        });
        println!(
            "device: solo {:.1} ms → co-run {:.1} ms ({:+.0}%) | cpu pass: solo {:.1} ms → co-run {:.1} ms ({:+.0}%)",
            solo * 1e3,
            dev_co * 1e3,
            (dev_co / solo - 1.0) * 100.0,
            cpu_solo * 1e3,
            cpu_co * 1e3,
            (cpu_co / cpu_solo - 1.0) * 100.0,
        );
    }
}
