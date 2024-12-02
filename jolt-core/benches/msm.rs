use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_ff::{BigInteger, PrimeField};
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
use rayon::prelude::*;

const SRS_SIZE: usize = 1 << 20;

// Sets up the benchmark
fn setup_bench<PCS, F, ProofTranscript>(
    msm_type: MsmType,
) -> (
    Vec<G1Affine>,
    Option<Vec<GpuBaseType<G1Projective>>>,
    Vec<Fr>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut rng = ChaCha20Rng::seed_from_u64(SRS_SIZE as u64);

    let scalars = match msm_type {
        MsmType::Zero => {
            vec![Fr::zero(); SRS_SIZE]
        }
        MsmType::One => {
            vec![Fr::one(); SRS_SIZE]
        }
        MsmType::Small(_) => (0..SRS_SIZE)
            .into_iter()
            .map(|_| {
                let i = rng.gen_range(0..(1 << 10));
                <Fr as JoltField>::from_u64(i).unwrap()
            })
            .collect(),
        MsmType::Medium(_) => (0..SRS_SIZE)
            .into_iter()
            .map(|_| {
                let i = rng.next_u64();
                <Fr as JoltField>::from_u64(i).unwrap()
            })
            .collect(),
        MsmType::Large(_) => (0..SRS_SIZE)
            .into_iter()
            .map(|_| {
                let values: [u64; 4] = [
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ];
                let bigint = ark_ff::BigInteger256::new(values);
                <Fr as JoltField>::from_bytes(&bigint.to_bytes_le())
            })
            .collect(),
    };

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

    let max_num_bits = scalars
        .par_iter()
        .map(|s| s.clone().into_bigint().num_bits())
        .max()
        .unwrap();

    println!("Using max num bits: {}", max_num_bits);
    #[cfg(not(feature = "icicle"))]
    let gpu_bases = None;
    (bases, gpu_bases, scalars)
}

fn benchmark_msm<PCS, F, ProofTranscript>(c: &mut Criterion, name: &str, msm_type: MsmType)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let (bases, gpu_bases, scalars) = setup_bench::<PCS, F, ProofTranscript>(msm_type);
    icicle_init();
    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{} [mode:JOLT CPU]", name);
    c.bench_function(&id, |b| {
        b.iter(|| {
            let msm =
                <G1Projective as VariableBaseMSM>::msm(&bases, gpu_bases.as_deref(), &scalars);
            let _ = msm.expect("MSM failed");
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(5));
    benchmark_msm::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm(Large)",
        MsmType::Large(0 /* unused */),
    );
    benchmark_msm::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm(Medium)",
        MsmType::Medium(0 /* unused */),
    );
    benchmark_msm::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm(Small)",
        MsmType::Small(0 /* unused */),
    );
    criterion.final_summary();
}
