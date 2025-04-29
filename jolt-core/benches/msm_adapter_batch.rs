use ark_bn254::{Bn254, Fr, G1Affine, G1Projective};
use ark_std::rand::seq::SliceRandom;
use ark_std::UniformRand;
use criterion::Criterion;
use jolt_core::field::JoltField;
#[cfg(not(feature = "icicle"))]
use jolt_core::msm::VariableBaseMSM;
#[cfg(feature = "icicle")]
use jolt_core::msm::{icicle_batch_msm, Icicle};
use jolt_core::msm::{icicle_init, GpuBaseType};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

// This bench uses icicle directly and bypasses the JOLT msm wrapper
// useful to test the icicle rust api almost directly, still goes through our adapter

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
                .map(|_| {
                    let mask = if max_num_bits == 8 {
                        u8::MAX
                    } else {
                        (1u8 << max_num_bits) - 1
                    };
                    (rng.next_u32() & (mask as u32)) as u8
                })
                .collect::<Vec<_>>(),
        ),
        9..=16 => MultilinearPolynomial::from(
            (0..len)
                .map(|_| {
                    let mask = if max_num_bits == 16 {
                        u16::MAX
                    } else {
                        (1u16 << max_num_bits) - 1
                    };
                    (rng.next_u32() & (mask as u32)) as u16
                })
                .collect::<Vec<_>>(),
        ),
        17..=32 => MultilinearPolynomial::from(
            (0..len)
                .map(|_| {
                    let mask = if max_num_bits == 32 {
                        u32::MAX
                    } else {
                        (1u32 << max_num_bits) - 1
                    };
                    (rng.next_u64() & (mask as u64)) as u32
                })
                .collect::<Vec<_>>(),
        ),
        33..=64 => MultilinearPolynomial::from(
            (0..len)
                .map(|_| {
                    let mask = if max_num_bits == 64 {
                        u64::MAX
                    } else {
                        (1u64 << max_num_bits) - 1
                    };
                    rng.next_u64() & mask
                })
                .collect::<Vec<_>>(),
        ),
        _ => {
            MultilinearPolynomial::from((0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
        }
    }
}

#[allow(unused_variables)]
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
    let max_bit_size = polys
        .iter()
        .map(MultilinearPolynomial::max_num_bits)
        .max()
        .unwrap();

    let polys: Vec<_> = polys
        .iter()
        .map(|poly| match poly {
            MultilinearPolynomial::LargeScalars(poly) => poly.evals(),
            MultilinearPolynomial::U16Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U32Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U64Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::I64Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U8Scalars(poly) => poly.coeffs_as_field_elements(),
        })
        .collect();
    let polys_ref: Vec<_> = polys.iter().map(|p| p.as_slice()).collect();

    icicle_init();
    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{name} [mode:JOLT CPU]");
    c.bench_function(&id, |b| {
        b.iter(|| {
            #[cfg(feature = "icicle")]
            icicle_batch_msm::<G1Projective>(gpu_bases.as_ref().unwrap(), &polys_ref, max_bit_size);
            #[cfg(not(feature = "icicle"))]
            {
                let _res: Vec<_> = polys_ref
                    .par_iter()
                    .map(|poly| {
                        let bases_slice = &bases[..poly.len()];
                        <G1Projective as VariableBaseMSM>::msm_field_elements(
                            bases_slice,
                            None,
                            poly,
                            Some(max_bit_size),
                            false,
                        )
                        .unwrap()
                    })
                    .collect();
            }
        });
    });
}

fn main() {
    let small_value_lookup_tables = <Fr as JoltField>::compute_lookup_tables();
    <Fr as JoltField>::initialize_lookup_tables(small_value_lookup_tables);

    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(10));

    let max_num_bits = vec![256; 1000];
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(1000 256bit scalars)",
        max_num_bits,
    );

    let max_num_bits = vec![64; 1000];
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(1000 64bit scalars)",
        max_num_bits,
    );

    let max_num_bits = vec![32; 1000];
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(1000 32bit scalars)",
        max_num_bits,
    );

    let max_num_bits = vec![16; 1000];
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(1000 16bit scalars)",
        max_num_bits,
    );

    let max_num_bits = vec![8; 1000];
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(1000 8bit scalars)",
        max_num_bits,
    );

    let max_num_bits = [vec![8; 450], vec![32; 450], vec![256; 100]].concat();
    benchmark_msm_batch::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "VariableBaseMSM::msm_adapter_batch(Mixed scalars, max 256 bits)",
        max_num_bits,
    );
    criterion.final_summary();
}
