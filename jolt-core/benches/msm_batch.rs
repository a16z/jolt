use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ff::BigInteger;
use ark_std::rand::seq::SliceRandom;
use ark_std::rand::Rng;
use ark_std::UniformRand;
use ark_std::{One, Zero};
use criterion::Criterion;
use jolt_core::field::JoltField;
#[cfg(feature = "icicle")]
use jolt_core::msm::Icicle;
use jolt_core::msm::{icicle_init, GpuBaseType, MsmType, VariableBaseMSM};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "icicle")]
use rayon::prelude::*;

const SRS_SIZE: usize = 1 << 14;

// Sets up the benchmark
fn setup_bench<PCS, F, ProofTranscript>(
    batch_config: BatchConfig,
) -> (
    Vec<G1Affine>,
    Option<Vec<GpuBaseType<G1Projective>>>,
    Vec<Vec<Fr>>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut rng = ChaCha20Rng::seed_from_u64(SRS_SIZE as u64);
    // For each type in the batch config create a vector of scalars
    let mut scalar_batches: Vec<Vec<Fr>> = vec![];

    (0..batch_config.small)
        .into_iter()
        .for_each(|_| scalar_batches.push(get_scalars(MsmType::Small(0 /* unused */), SRS_SIZE)));
    (0..batch_config.medium)
        .into_iter()
        .for_each(|_| scalar_batches.push(get_scalars(MsmType::Medium(0 /* unused */), SRS_SIZE)));
    (0..batch_config.large)
        .into_iter()
        .for_each(|_| scalar_batches.push(get_scalars(MsmType::Large(0 /* unused */), SRS_SIZE)));
    scalar_batches.shuffle(&mut rng);

    let bases: Vec<G1Affine> = std::iter::repeat_with(|| G1Affine::rand(&mut rng))
        .take(SRS_SIZE)
        .collect();
    #[cfg(feature = "icicle")]
    let gpu_bases = Some(
        bases
            .par_iter()
            .map(|base| G1Projective::from_ark_affine(base))
            .collect(),
    );
    #[cfg(not(feature = "icicle"))]
    let gpu_bases = None;
    (bases, gpu_bases, scalar_batches)
}

fn get_scalars(msm_type: MsmType, size: usize) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(size as u64);
    match msm_type {
        MsmType::Zero => {
            vec![Fr::zero(); size]
        }
        MsmType::One => {
            vec![Fr::one(); size]
        }
        MsmType::Small(_) => (0..size)
            .into_iter()
            .map(|_| {
                let i = rng.gen_range(0..(1 << 10));
                <Fr as JoltField>::from_u64(i).unwrap()
            })
            .collect(),
        MsmType::Medium(_) => (0..size)
            .into_iter()
            .map(|_| {
                let i = rng.next_u64();
                <Fr as JoltField>::from_u64(i).unwrap()
            })
            .collect(),
        MsmType::Large(_) => (0..size)
            .into_iter()
            .map(|_| Fr::random(&mut rng))
            .collect(),
    }
}

fn benchmark_msm_batch<PCS, F, ProofTranscript>(
    c: &mut Criterion,
    name: &str,
    batch_config: BatchConfig,
) where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let (bases, gpu_bases, scalar_batches) = setup_bench::<PCS, F, ProofTranscript>(batch_config);
    let scalar_batches_ref: Vec<_> = scalar_batches
        .iter()
        .map(|inner_vec| inner_vec.as_slice())
        .collect();
    icicle_init();
    println!("Running benchmark for {:?}", batch_config);
    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{} [mode:JOLT CPU]", name);
    c.bench_function(&id, |b| {
        b.iter(|| {
            let msm = <G1Projective as VariableBaseMSM>::batch_msm(
                &bases,
                gpu_bases.as_deref(),
                &scalar_batches_ref,
            );
            assert_eq!(msm.len(), scalar_batches.len());
        });
    });
}

#[derive(Debug, Clone, Copy)]
struct BatchConfig {
    small: usize,
    medium: usize,
    large: usize,
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(10));
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Large)",
        BatchConfig {
            small: 100,
            medium: 100,
            large: 300,
        },
    );
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Medium)",
        BatchConfig {
            small: 100,
            medium: 300,
            large: 100,
        },
    );
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Small)",
        BatchConfig {
            small: 300,
            medium: 100,
            large: 100,
        },
    );
    criterion.final_summary();
}
