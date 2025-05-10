use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ff::PrimeField;
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_core::msm::{arkmsm_init, use_arkmsm, icicle_init, disable_arkmsm_fallback, arkmsm_msm, msm_bigint};
use std::time::Duration;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

// Benchmark sizes - adjusted to show scaling behavior
const BENCH_SIZES: [usize; 3] = [128, 1024, 1 << 16];
const MAX_NUM_BITS: usize = 254;

fn bench_msm_standard_vs_arkmsm(c: &mut Criterion) {
    // Initialize feature flags
    let _has_icicle = icicle_init(); // Initialize icicle first to avoid conflicts
    
    // Always initialize arkmsm
    disable_arkmsm_fallback(); // Ensure that fallback is disabled
    let has_arkmsm = arkmsm_init(); // Initialize arkmsm feature
    println!("arkmsm optimizations {}", if use_arkmsm() { "enabled" } else { "not enabled" });
    
    // Create a benchmark group
    let mut group = c.benchmark_group("MSM Implementations");
    group.measurement_time(Duration::from_secs(10));
    
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    
    // Benchmark each size
    for size in BENCH_SIZES.iter() {
        // Skip the largest size for validation to avoid stack overflow
        let should_validate = *size < 1 << 16;
        
        // Generate random input data
        let bases: Vec<G1Affine> = (0..*size).map(|_| G1Affine::rand(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..*size).map(|_| Fr::rand(&mut rng)).collect();
        let scalar_bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
        
        // Standard MSM implementation benchmark
        let mut standard_result = None;
        group.bench_with_input(BenchmarkId::new("Standard", size), size, |b, _| {
            b.iter(|| {
                let result = msm_bigint::<Fr, G1Projective>(
                    black_box(&bases),
                    black_box(&scalar_bigints),
                    black_box(MAX_NUM_BITS),
                );
                if should_validate && standard_result.is_none() {
                    standard_result = Some(result);
                }
                black_box(result)
            });
        });
        
        // Only benchmark arkmsm if the feature is enabled
        if has_arkmsm {
            // Force re-initialization
            println!("Running arkmsm benchmarks for size {}", size);
            disable_arkmsm_fallback();
            assert!(use_arkmsm(), "ArkMSM should be enabled for benchmarks");
            
            // ArkMSM optimized implementation benchmark
            let mut arkmsm_result = None;
            group.bench_with_input(BenchmarkId::new("ArkMSM", size), size, |b, _| {
                b.iter(|| {
                    let result = arkmsm_msm::<Fr, G1Projective>(
                        black_box(&bases),
                        black_box(&scalar_bigints),
                        black_box(MAX_NUM_BITS),
                    );
                    if should_validate && arkmsm_result.is_none() {
                        arkmsm_result = Some(result);
                    }
                    black_box(result)
                });
            });
            
            // Validate results if we're not dealing with the largest size
            if should_validate {
                match (standard_result, arkmsm_result) {
                    (Some(std_res), Some(ark_res)) => {
                        assert_eq!(std_res, ark_res, 
                            "Results from standard and arkmsm implementations differ for size {}", size);
                        println!("✓ Validation passed for size {}: Results match", size);
                    },
                    _ => {
                        println!("⚠ Could not validate results for size {}", size);
                    }
                }
            }
        }
    }
    
    group.finish();
}

criterion_group!(benches, bench_msm_standard_vs_arkmsm);
criterion_main!(benches); 
