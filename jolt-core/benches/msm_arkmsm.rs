use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jolt_core::field::JoltField;
use jolt_core::msm::{arkmsm_init, use_arkmsm, icicle_init, use_icicle, GpuBaseType, VariableBaseMSM};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use std::time::Duration;

// Benchmark sizes
const BENCH_SIZES: [usize; 5] = [1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20];
const MAX_NUM_BITS: usize = 256;

fn setup_bench(size: usize) -> (Vec<G1Affine>, MultilinearPolynomial<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    
    // Generate random bases
    let bases: Vec<G1Affine> = std::iter::repeat_with(|| G1Affine::rand(&mut rng))
        .take(size)
        .collect();
    
    // Generate random scalars
    let scalars: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(size)
        .collect();
    
    let poly = MultilinearPolynomial::from(scalars);
    
    (bases, poly)
}

fn bench_msm_implementations(c: &mut Criterion) {
    // Initialize both backends
    icicle_init();
    arkmsm_init();
    
    // Enable JoltField lookup tables for optimized operations
    let small_value_lookup_tables = <Fr as JoltField>::compute_lookup_tables();
    <Fr as JoltField>::initialize_lookup_tables(small_value_lookup_tables);
    
    let mut group = c.benchmark_group("MSM Implementations");
    group.measurement_time(Duration::from_secs(10));
    
    for size in BENCH_SIZES.iter() {
        let (bases, poly) = setup_bench(*size);
        
        // Benchmark standard implementation
        group.bench_with_input(BenchmarkId::new("Standard", size), size, |b, _| {
            b.iter(|| {
                let msm = <G1Projective as VariableBaseMSM>::msm(
                    black_box(&bases), 
                    black_box(None), 
                    black_box(&poly), 
                    black_box(Some(MAX_NUM_BITS))
                );
                let _ = msm.expect("MSM failed");
            });
        });
        
        // Benchmark arkmsm implementation
        group.bench_with_input(BenchmarkId::new("ArkMSM", size), size, |b, _| {
            b.iter(|| {
                // Force arkmsm to be used
                assert!(use_arkmsm());
                
                let msm = <G1Projective as VariableBaseMSM>::msm(
                    black_box(&bases), 
                    black_box(None), 
                    black_box(&poly), 
                    black_box(Some(MAX_NUM_BITS))
                );
                let _ = msm.expect("MSM failed");
            });
        });
        
        // Benchmark Icicle implementation if available
        if use_icicle() {
            #[cfg(feature = "icicle")]
            {
                // Convert bases to GPU format
                let gpu_bases: Vec<GpuBaseType<G1Projective>> = bases
                    .iter()
                    .map(|base| G1Projective::from_ark_affine(base))
                    .collect();
                
                group.bench_with_input(BenchmarkId::new("Icicle", size), size, |b, _| {
                    b.iter(|| {
                        let msm = <G1Projective as VariableBaseMSM>::msm(
                            black_box(&bases), 
                            black_box(Some(&gpu_bases)), 
                            black_box(&poly), 
                            black_box(Some(MAX_NUM_BITS))
                        );
                        let _ = msm.expect("MSM failed");
                    });
                });
            }
        }
    }
    
    group.finish();
}

criterion_group!(benches, bench_msm_implementations);
criterion_main!(benches); 
