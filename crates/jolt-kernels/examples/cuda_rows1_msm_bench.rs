#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use dory::backends::arkworks::{ArkFr, ArkG1};
use dory::primitives::arithmetic::DoryRoutines;
use jolt_dory::JoltG1Routines;
use jolt_field::Fr;
use jolt_kernels::cuda::{shared_context, AffineLimbs, JacobianLimbs};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const LENGTHS: [usize; 10] = [1, 8, 64, 256, 512, 1024, 2048, 4096, 8192, 16384];
const SAMPLES: usize = 5;

fn limbs(point: &ark_bn254::G1Projective) -> JacobianLimbs {
    JacobianLimbs {
        x: point.x.0 .0,
        y: point.y.0 .0,
        z: point.z.0 .0,
    }
}

fn affine_limbs(point: &ark_bn254::G1Projective) -> AffineLimbs {
    let affine = point.into_affine();
    if affine.infinity {
        return AffineLimbs::IDENTITY;
    }
    AffineLimbs {
        x: affine.x.0 .0,
        y: affine.y.0 .0,
        infinity: false,
    }
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device");
        return;
    };
    let mut rng = ChaCha20Rng::seed_from_u64(20_260_820);
    let bases: Vec<ark_bn254::G1Projective> = (0..*LENGTHS.last().unwrap_or(&1))
        .map(|_| ark_bn254::G1Projective::rand(&mut rng))
        .collect();
    let scalars: Vec<ark_bn254::Fr> = (0..bases.len())
        .map(|_| ark_bn254::Fr::rand(&mut rng))
        .collect();

    println!(
        "{:>7}  {:>11}  {:>11}  {:>11}  {:>8}",
        "len", "host ms", "chain ms", "bucket ms", "bucket/host"
    );
    for len in LENGTHS {
        let ark_bases: Vec<ArkG1> = bases[..len].iter().copied().map(ArkG1).collect();
        let ark_scalars: Vec<ArkFr> = scalars[..len].iter().copied().map(ArkFr).collect();
        let device_bases: Vec<JacobianLimbs> = bases[..len].iter().map(limbs).collect();
        let device_scalars: Vec<Fr> = scalars[..len].iter().map(|s| Fr::from(*s)).collect();

        let expected = JoltG1Routines::msm(&ark_bases, &ark_scalars);
        let mut host = f64::MAX;
        for _ in 0..SAMPLES {
            let now = Instant::now();
            let _ = JoltG1Routines::msm(&ark_bases, &ark_scalars);
            host = host.min(now.elapsed().as_secs_f64() * 1e3);
        }

        let warm = context.msm_rows_shared_scalars(&device_bases, &device_scalars, 1);
        let mut device = f64::MAX;
        for _ in 0..SAMPLES {
            let now = Instant::now();
            let _ = context.msm_rows_shared_scalars(&device_bases, &device_scalars, 1);
            device = device.min(now.elapsed().as_secs_f64() * 1e3);
        }

        match warm {
            Ok(rows) => {
                let got = rows.first().map(|row| {
                    ark_bn254::G1Projective::new_unchecked(
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.x)),
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.y)),
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.z)),
                    )
                });
                assert_eq!(
                    got.map(|point| point.into_affine()),
                    Some(expected.0.into_affine()),
                    "the device MSM diverged at len {len}"
                );
            }
            Err(error) => {
                println!("{len:>7}  device declined: {error:?}");
                continue;
            }
        }
        let affine_bases: Vec<AffineLimbs> = bases[..len].iter().map(affine_limbs).collect();
        let uploaded = context
            .upload_g1_bases(&affine_bases)
            .expect("upload affine bases");
        let uploaded_scalars = context.upload(&device_scalars).expect("upload scalars");
        let bucket_warm = context.msm_rows_fr(&uploaded, &uploaded_scalars, len);
        let mut bucket = f64::MAX;
        for _ in 0..SAMPLES {
            let now = Instant::now();
            let _ = context.msm_rows_fr(&uploaded, &uploaded_scalars, len);
            bucket = bucket.min(now.elapsed().as_secs_f64() * 1e3);
        }
        match bucket_warm {
            Ok(rows) => {
                let got = rows.first().map(|row| {
                    ark_bn254::G1Projective::new_unchecked(
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.x)),
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.y)),
                        ark_bn254::Fq::new_unchecked(ark_ff::BigInt(row.z)),
                    )
                });
                assert_eq!(
                    got.map(|point| point.into_affine()),
                    Some(expected.0.into_affine()),
                    "the bucket MSM diverged at len {len}"
                );
            }
            Err(error) => {
                println!("{len:>7}  bucket declined: {error:?}");
                continue;
            }
        }

        println!(
            "{len:>7}  {host:>11.2}  {device:>11.2}  {bucket:>11.2}  {:>7.2}x",
            host / bucket
        );
    }

    println!("\nbucket cost vs rows (row_len fixed; per-MSM = total/rows)");
    println!(
        "{:>8}  {:>5}  {:>10}  {:>11}  {:>10}",
        "row_len", "rows", "total ms", "per-MSM ms", "host ms"
    );
    for row_len in [512usize, 2048, 4096, 8192] {
        let affine_bases: Vec<AffineLimbs> = bases[..row_len].iter().map(affine_limbs).collect();
        let uploaded = context
            .upload_g1_bases(&affine_bases)
            .expect("upload affine bases");
        let ark_bases: Vec<ArkG1> = bases[..row_len].iter().copied().map(ArkG1).collect();
        let ark_scalars: Vec<ArkFr> = scalars[..row_len].iter().copied().map(ArkFr).collect();
        let mut host = f64::MAX;
        for _ in 0..SAMPLES {
            let now = Instant::now();
            let _ = JoltG1Routines::msm(&ark_bases, &ark_scalars);
            host = host.min(now.elapsed().as_secs_f64() * 1e3);
        }
        for rows in [1usize, 2, 4, 8, 16, 32] {
            let wide: Vec<Fr> = (0..rows * row_len)
                .map(|index| Fr::from(scalars[index % scalars.len()]))
                .collect();
            let uploaded_scalars = context.upload(&wide).expect("upload scalars");
            let warm = context.msm_rows_fr(&uploaded, &uploaded_scalars, row_len);
            if let Err(error) = warm {
                println!("{row_len:>8}  {rows:>5}  declined: {error:?}");
                continue;
            }
            let mut total = f64::MAX;
            for _ in 0..SAMPLES {
                let now = Instant::now();
                let _ = context.msm_rows_fr(&uploaded, &uploaded_scalars, row_len);
                total = total.min(now.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "{row_len:>8}  {rows:>5}  {total:>10.2}  {:>11.2}  {host:>10.2}",
                total / rows as f64
            );
        }
    }
}
