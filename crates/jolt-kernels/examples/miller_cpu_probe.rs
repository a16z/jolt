//! W3-st8 decomposition probe for the multi-pair CPU co-execution arm:
//! bisects the 5× CPU-rate anomaly between the wave-1 microbench rates and
//! the first co-execution measurements — point provenance, batch size, and
//! preparation-vs-Miller phase — then times the production join shape.
//!
//! ```text
//! cargo run --release -p jolt-kernels --example miller_cpu_probe --features metal
//! ```

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "diagnostic probe: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    probe::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("miller_cpu_probe requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod probe {
    use std::time::Instant;

    use ark_bn254::{Bn254, G1Projective, G2Affine, G2Projective};
    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::{One, UniformRand};
    use jolt_kernels::metal::miller::{
        miller_fly_partials, multi_pair_device, seeded_pairing_inputs, ArkG2Prepared,
        ENV_MILLER_CPU_PCT,
    };
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::MetalContext;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use rayon::prelude::*;

    fn min_secs(passes: usize, f: impl Fn()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..passes {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    }

    fn prep_rate(label: &str, qs: &[G2Affine]) {
        let t = min_secs(3, || {
            let preps: Vec<ArkG2Prepared> =
                qs.par_iter().map(|q| ArkG2Prepared::from(*q)).collect();
            let _ = std::hint::black_box(preps);
        });
        println!(
            "prep  {label:<28} {:>6} pts: {:8.1} ms ({:5.2} us/pt)",
            qs.len(),
            t * 1e3,
            t / qs.len() as f64 * 1e6
        );
    }

    pub fn run() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("Metal context");
        println!("rayon threads: {}", rayon::current_num_threads());

        let rand_qs = |n: usize, seed: u64| -> Vec<G2Affine> {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            (0..n)
                .map(|_| G2Projective::rand(&mut rng).into_affine())
                .collect()
        };
        let rand_ps = |n: usize, seed: u64| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            (0..n)
                .map(|_| G1Projective::rand(&mut rng).into_affine())
                .collect::<Vec<_>>()
        };

        let n = 1usize << 17;
        let (sps, sqs) = seeded_pairing_inputs(0x50_0b, n);
        let sps_affine =
            G1Projective::normalize_batch(&sps.iter().map(|p| p.0).collect::<Vec<_>>());
        let sqs_affine =
            G2Projective::normalize_batch(&sqs.iter().map(|q| q.0).collect::<Vec<_>>());
        let tail = n / 5;

        // --- 1. G2 preparation: provenance × size ---------------------------
        prep_rate("random 8192", &rand_qs(8192, 1));
        prep_rate("seeded 8192", &sqs_affine[..8192]);
        prep_rate("random 26214", &rand_qs(26214, 2));
        prep_rate("seeded 26214 (tail slice)", &sqs_affine[n - tail..]);

        // --- 2. Miller phase alone (pre-prepared, borrowed) ------------------
        let ps8k = rand_ps(8192, 3);
        let qs8k = rand_qs(8192, 4);
        let preps8k: Vec<ArkG2Prepared> =
            qs8k.par_iter().map(|q| ArkG2Prepared::from(*q)).collect();
        let t = min_secs(3, || {
            let f = ps8k
                .par_chunks(512)
                .zip(preps8k.par_chunks(512))
                .map(|(pc, qc)| Bn254::multi_miller_loop(pc.iter().copied(), qc.iter().cloned()).0)
                .reduce(ark_bn254::Fq12::one, |a, b| a * b);
            let _ = std::hint::black_box(f);
        });
        println!(
            "miller prepared ark c=512 (C1 replica)  8192 pairs: {:8.1} ms ({:5.2} us/pair)",
            t * 1e3,
            t / 8192_f64 * 1e6
        );

        // --- 2b. jolt prepared ladder, borrowed prebuilt preps ---------------
        for chunk in [128usize, 512] {
            let t = min_secs(3, || {
                let f = ps8k
                    .par_chunks(chunk)
                    .zip(preps8k.par_chunks(chunk))
                    .map(|(pc, qc)| {
                        let pairs: Vec<_> = pc
                            .iter()
                            .zip(qc)
                            .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
                            .collect();
                        jolt_dory::multi_miller_prepared_pairs(&pairs)
                    })
                    .reduce(ark_bn254::Fq12::one, |a, b| a * b);
                let _ = std::hint::black_box(f);
            });
            println!(
                "jolt ladder borrowed c={chunk:<4} 8192 pairs: {:8.1} ms ({:5.2} us/pair)",
                t * 1e3,
                t / 8192_f64 * 1e6
            );
        }

        // --- 2c. two-phase inside one timed region: par prep, then ladder ----
        let t = min_secs(3, || {
            let preps: Vec<ArkG2Prepared> =
                qs8k.par_iter().map(|q| ArkG2Prepared::from(*q)).collect();
            let f = ps8k
                .par_chunks(512)
                .zip(preps.par_chunks(512))
                .map(|(pc, qc)| {
                    let pairs: Vec<_> = pc
                        .iter()
                        .zip(qc)
                        .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
                        .collect();
                    jolt_dory::multi_miller_prepared_pairs(&pairs)
                })
                .reduce(ark_bn254::Fq12::one, |a, b| a * b);
            let _ = std::hint::black_box(f);
        });
        println!(
            "two-phase prep+ladder c=512  8192 pairs: {:8.1} ms ({:5.2} us/pair)",
            t * 1e3,
            t / 8192_f64 * 1e6
        );

        // --- 2d. single-thread jolt ladder, 512 borrowed pairs ---------------
        let pairs512: Vec<_> = ps8k[..512]
            .iter()
            .zip(&preps8k[..512])
            .map(|(p, q)| (*p, q.ell_coeffs.as_slice()))
            .collect();
        let t = min_secs(3, || {
            let _ = std::hint::black_box(jolt_dory::multi_miller_prepared_pairs(&pairs512));
        });
        println!(
            "jolt ladder 1-thread 512 pairs: {:8.1} ms ({:5.2} us/pair)",
            t * 1e3,
            t / 512_f64 * 1e6
        );

        // --- 3. The deterministic path: provenance × size --------------------
        for (label, ps, qs) in [
            ("random 8192", &ps8k[..], &qs8k[..]),
            ("seeded 8192", &sps_affine[..8192], &sqs_affine[..8192]),
            (
                "seeded 26214 (tail)",
                &sps_affine[n - tail..],
                &sqs_affine[n - tail..],
            ),
        ] {
            let t = min_secs(3, || {
                let _ = std::hint::black_box(jolt_dory::multi_miller_affine(ps, qs));
            });
            println!(
                "multi_miller_affine {label:<22} {:>6} pairs: {:8.1} ms ({:5.2} us/pair)",
                ps.len(),
                t * 1e3,
                t / ps.len() as f64 * 1e6
            );
        }

        // --- 4. Production join shape + full hook ----------------------------
        let head = n - tail;
        let t = min_secs(3, || {
            let (partials, cpu) = rayon::join(
                || miller_fly_partials(ctx, &sps_affine[..head], &sqs_affine[..head]).unwrap(),
                || jolt_dory::multi_miller_affine(&sps_affine[head..], &sqs_affine[head..]),
            );
            let _ = std::hint::black_box((partials, cpu));
        });
        println!(
            "join(device head, prepared-ladder tail): {:8.1} ms",
            t * 1e3
        );

        for pct in ["0", "20"] {
            std::env::set_var(ENV_MILLER_CPU_PCT, pct);
            let t = min_secs(3, || {
                let _ = std::hint::black_box(multi_pair_device(&sps, &sqs).unwrap());
            });
            println!(
                "multi_pair_device pct={pct:<3} {n} pairs: {:8.1} ms",
                t * 1e3
            );
        }
        std::env::remove_var(ENV_MILLER_CPU_PCT);

        // --- 5. Partial-product fold: sequential vs parallel ------------------
        let partials = miller_fly_partials(ctx, &sps_affine, &sqs_affine).unwrap();
        let t = min_secs(3, || {
            let _ =
                std::hint::black_box(jolt_kernels::metal::miller::product_of_partials(&partials));
        });
        println!("fold sequential {n} partials: {:8.1} ms", t * 1e3);
        let t = min_secs(3, || {
            let _ = std::hint::black_box(jolt_kernels::metal::miller::product_of_partials_par(
                &partials,
            ));
        });
        println!("fold parallel   {n} partials: {:8.1} ms", t * 1e3);
    }
}
