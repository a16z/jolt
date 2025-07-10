use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_std::UniformRand;
use jolt_core::fast_msm::VariableBaseMSM as FastVariableBaseMsm;
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
use std::time::Instant;
const SRS_SIZE: usize = 1 << 17;

// Sets up the benchmark
fn setup_bench<PCS, F, ProofTranscript>(
    max_num_bits: usize,
) -> (
    Vec<G1Affine>,
    Option<Vec<GpuBaseType<G1Projective>>>,
    MultilinearPolynomial<Fr>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    let mut rng = ChaCha20Rng::seed_from_u64(SRS_SIZE as u64);
    let poly = match max_num_bits {
        0 => MultilinearPolynomial::from(vec![0u8; SRS_SIZE]),
        1..=8 => MultilinearPolynomial::from(
            (0..SRS_SIZE)
                .map(|_| (rng.next_u32() & ((1 << max_num_bits) - 1)) as u8)
                .collect::<Vec<_>>(),
        ),
        9..=16 => MultilinearPolynomial::from(
            (0..SRS_SIZE)
                .map(|_| (rng.next_u32() & ((1 << max_num_bits) - 1)) as u16)
                .collect::<Vec<_>>(),
        ),
        17..=32 => MultilinearPolynomial::from(
            (0..SRS_SIZE)
                .map(|_| (rng.next_u64() & ((1 << max_num_bits) - 1)) as u32)
                .collect::<Vec<_>>(),
        ),
        33..=64 => MultilinearPolynomial::from(
            (0..SRS_SIZE)
                .map(|_| rng.next_u64() & ((1 << max_num_bits) - 1))
                .collect::<Vec<_>>(),
        ),
        _ => MultilinearPolynomial::from(
            (0..SRS_SIZE)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>(),
        ),
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
    #[cfg(not(feature = "icicle"))]
    let gpu_bases = None;

    println!("Using max num bits: {max_num_bits}");
    (bases, gpu_bases, poly)
}

fn benchmark_msm<PCS, F, ProofTranscript>(name: &str, max_num_bits: usize)
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    let (bases, gpu_bases, poly) = setup_bench::<PCS, F, ProofTranscript>(max_num_bits);
    icicle_init();
    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{name} [mode:JOLT CPU]");
    println!("{id}");

    let start = Instant::now();
    let msm = <G1Projective as VariableBaseMSM>::msm(&bases, gpu_bases.as_deref(), &poly, None);
    //let _ = msm.expect("MSM failed");
    let duration = start.elapsed().as_micros();
    println!("Jolt (OLD): took {duration} micro s");
    let start = Instant::now();
    let ark_msm = <G1Projective as FastVariableBaseMsm>::msm(&bases, &poly, None);
    let duration = start.elapsed().as_micros();
    println!("Jolt (New) took {duration} micro s\n");
    assert_eq!(ark_msm.unwrap(), msm.unwrap());
}

fn main() {
    let small_value_lookup_tables = <Fr as JoltField>::compute_lookup_tables();
    <Fr as JoltField>::initialize_lookup_tables(small_value_lookup_tables);

    benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
        "VariableBaseMSM::msm(256 bit scalars)",
        256,
    );
    //benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
    //    "VariableBaseMSM::msm(64 bit scalars)",
    //    64,
    //);
    benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
        "VariableBaseMSM::msm(32 bit scalars)",
        32,
    );
    benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
        "VariableBaseMSM::msm(16 bit scalars)",
        16,
    );
    benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
        "VariableBaseMSM::msm(8 bit scalars)",
        8,
    );
    benchmark_msm::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
        "VariableBaseMSM::msm(1 bit scalars)",
        1,
    );
}
