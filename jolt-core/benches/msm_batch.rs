use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_std::rand::seq::SliceRandom;
use ark_std::UniformRand;
use criterion::Criterion;
use jolt_core::field::JoltField;
#[cfg(feature = "icicle")]
use jolt_core::msm::Icicle;
use jolt_core::msm::{icicle_init, GpuBaseType, VariableBaseMSM};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "icicle")]
use rayon::prelude::*;

const SRS_SIZE: usize = 1 << 14;

// Sets up the benchmark
fn setup_bench<PCS, F, ProofTranscript>(
    max_num_bits: Vec<usize>,
) -> (
    Vec<G1Affine>,
    Option<Vec<GpuBaseType<G1Projective>>>,
    Vec<MultilinearPolynomial<Fr>>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut rng = ChaCha20Rng::seed_from_u64(SRS_SIZE as u64);
    // For each `max_num_bits` value, create a polynomial
    let mut polys: Vec<_> = max_num_bits
        .into_iter()
        .map(|num_bits| random_poly(num_bits, SRS_SIZE))
        .collect();

    polys.shuffle(&mut rng);

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
    (bases, gpu_bases, polys)
}

fn random_poly(max_num_bits: usize, len: usize) -> MultilinearPolynomial<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(len as u64);
    match max_num_bits {
        0 => MultilinearPolynomial::from(vec![0u8; len]),
        1..=8 => MultilinearPolynomial::from(
            (0..len)
                .into_iter()
                .map(|_| (rng.next_u32() & ((1 << max_num_bits) - 1)) as u8)
                .collect::<Vec<_>>(),
        ),
        9..=16 => MultilinearPolynomial::from(
            (0..len)
                .into_iter()
                .map(|_| (rng.next_u32() & ((1 << max_num_bits) - 1)) as u16)
                .collect::<Vec<_>>(),
        ),
        17..=32 => MultilinearPolynomial::from(
            (0..len)
                .into_iter()
                .map(|_| (rng.next_u64() & ((1 << max_num_bits) - 1)) as u32)
                .collect::<Vec<_>>(),
        ),
        33..=64 => MultilinearPolynomial::from(
            (0..len)
                .into_iter()
                .map(|_| rng.next_u64() & ((1 << max_num_bits) - 1))
                .collect::<Vec<_>>(),
        ),
        _ => MultilinearPolynomial::from(
            (0..len)
                .into_iter()
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>(),
        ),
    }
}

fn benchmark_msm_batch<PCS, F, ProofTranscript>(
    c: &mut Criterion,
    name: &str,
    max_num_bits: Vec<usize>,
) where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let (bases, gpu_bases, polys) = setup_bench::<PCS, F, ProofTranscript>(max_num_bits);
    let polys_ref: Vec<_> = polys.iter().collect();
    icicle_init();
    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{} [mode:JOLT CPU]", name);
    c.bench_function(&id, |b| {
        b.iter(|| {
            let msm = <G1Projective as VariableBaseMSM>::batch_msm(
                &bases,
                gpu_bases.as_deref(),
                &polys_ref,
            );
            assert_eq!(msm.len(), polys.len());
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(10));

    let max_num_bits = [vec![8; 100], vec![32; 100], vec![256; 300]].concat();
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Large)",
        max_num_bits,
    );

    let max_num_bits = [vec![8; 100], vec![32; 300], vec![256; 100]].concat();
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Medium)",
        max_num_bits,
    );

    let max_num_bits = [vec![8; 300], vec![32; 100], vec![256; 100]].concat();
    benchmark_msm_batch::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_batch(bias: Small)",
        max_num_bits,
    );
    criterion.final_summary();
}
