use std::cmp::min;
use std::collections::HashMap;
use std::time::Instant;
use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::Criterion;
#[cfg(feature = "icicle")]
use icicle_bn254::curve::ScalarField;
#[cfg(feature = "icicle")]
use icicle_bn254::program::bn254::FieldReturningValueProgram;
#[cfg(feature = "icicle")]
use icicle_bn254::sumcheck::SumcheckWrapper;
#[cfg(feature = "icicle")]
use icicle_core::curve::Curve;
#[cfg(feature = "icicle")]
use icicle_core::program::ReturningValueProgram;
use jolt_core::field::JoltField;
#[cfg(feature = "icicle")]
use jolt_core::msm::{icicle_init, icicle_to_jolt};
use jolt_core::utils::transcript::{AppendToTranscript, KeccakTranscript, Transcript};
use rand_core::RngCore;
#[cfg(not(feature = "icicle"))]
use jolt_core::subprotocols::shout::prove_core_shout_piop;
#[cfg(feature = "icicle")]
use icicle_core::sumcheck::{Sumcheck, SumcheckConfig, SumcheckTranscriptConfig};
#[cfg(feature = "icicle")]
use icicle_core::sumcheck::SumcheckProofOps;
#[cfg(feature = "icicle")]
use icicle_core::traits::FieldImpl;
#[cfg(feature = "icicle")]
use icicle_core::traits::GenerateRandom;
#[cfg(feature = "icicle")]
use icicle_runtime::memory::{DeviceVec, HostSlice};
#[cfg(feature = "icicle")]
use icicle_runtime::stream::IcicleStream;
use rayon::prelude::*;
use jolt_core::msm::Icicle;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation};
use jolt_core::poly::unipoly::{CompressedUniPoly, UniPoly};
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;

const TABLE_SIZE: usize = 1 << 26;
const NUM_LOOKUPS: usize = 1 << 22;

#[inline]
unsafe fn reinterpret_field_slice<F, T>(input: &[F]) -> &[T] {
    assert_eq!(size_of::<F>(), size_of::<T>());
    std::slice::from_raw_parts(input.as_ptr() as *const T, input.len())
}

#[inline]
unsafe fn reinterpret_field<F, T>(input: &F) -> T
where
    T: Copy,
{
    assert_eq!(size_of::<F>(), size_of::<T>());
    *(input as *const F as *const T)
}

pub fn setup_F<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: &[F],
    read_addresses: &[usize],
    transcript: &mut ProofTranscript,
) -> Vec<F> {
    let K = lookup_table.len();
    let T = read_addresses.len();
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);

    // if T <= 1 << 20 {
        let mut F: Vec<F> = unsafe_allocate_zero_vec(K);

        for (cycle, &address) in read_addresses.iter().enumerate() {
            if address < K {
                F[address] += E[cycle];
            }
        }
        F
    // } else {
    //     let num_chunks = rayon::current_num_threads()
    //     .next_power_of_two()
    //     .min(read_addresses.len());
    //     let chunk_size = (read_addresses.len() / num_chunks).max(1);
    //     let F: Vec<_> = read_addresses
    //         .par_chunks(chunk_size)
    //         .enumerate()
    //         .map(|(chunk_index, addresses)| {
    //             let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
    //             let mut cycle = chunk_index * chunk_size;
    //             for address in addresses {
    //                 result[*address] += E[cycle];
    //                 cycle += 1;
    //             }
    //             result
    //         })
    //         .reduce(
    //             || unsafe_allocate_zero_vec(K),
    //             |mut running, new| {
    //                 running
    //                     .iter_mut()
    //                     .zip(new.into_iter())
    //                     .for_each(|(x, y)| *x += y);
    //                 running
    //             },
    //         );
    //     F
    // }
}

// pub fn setup_sumcheck_claim2<F: JoltField, ProofTranscript: Transcript>(
//     lookup_table: &[F],
//     F: &[F]) -> F {
//     let mut stream = IcicleStream::create().unwrap();
//     let mut vals = DeviceVec::<<SumcheckWrapper as Sumcheck>::Field>::device_malloc_async(lookup_table.len(), &stream).unwrap();
//     let mut ra = DeviceVec::<<SumcheckWrapper as Sumcheck>::Field>::device_malloc_async(F.len(), &stream).unwrap();
//     let mut result = DeviceVec::<<SumcheckWrapper as Sumcheck>::Field>::device_malloc_async(F.len(), &stream).unwrap();
//     unsafe {
//         vals.copy_from_host_async(HostSlice::from_slice(reinterpret_field_slice(lookup_table)), &stream).unwrap();
//         ra.copy_from_host_async(HostSlice::from_slice(reinterpret_field_slice(F)), &stream).unwrap();
//     }

//     // Element-wise multiply on GPU
//     let cfg = icicle_core::vec_ops::VecOpsConfig::default();
//     icicle_core::vec_ops::mul_scalars(&vals, &ra, &mut result, &cfg).unwrap();

//     // Copy result back to host
//     let mut res = vec![<SumcheckWrapper as Sumcheck>::Field::zero(); F.len()];
//     let mut host_result = HostSlice::from_mut_slice(&mut res);
//     result.copy_to_host_async(&mut host_result, &stream).unwrap();

//     // Wait for stream to finish before summing on CPU
//     stream.synchronize().unwrap();
//     stream.destroy().unwrap();

//     // Sum on CPU
//     host_result.as_slice().par_iter().map(|scalar| icicle_to_jolt::<F, ScalarField>(scalar)).sum()
// }

pub fn setup_sumcheck_claim<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: &[F],
    F: &[F]) -> F {
    F.par_iter().zip(lookup_table.par_iter()).map(|(&ra, &val)| ra * val).sum()
}

pub fn setup_sumcheck<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: &[usize],
    transcript: &mut ProofTranscript,
) -> (
    usize,
    Vec<Vec<F>>,
    F,
) {
    // K becomes the number of variables in the poly
    let K = lookup_table.len();
    let F = setup_F(&lookup_table, read_addresses, transcript);

    let sumcheck_claim = setup_sumcheck_claim::<F, ProofTranscript>(&lookup_table, &F);

    let mle_polys = vec![F, lookup_table];
    const DEGREE: usize = 2;
    (
        K,
        mle_polys,
        sumcheck_claim,
    )
}

#[cfg(feature = "icicle")]
fn icicle_sumcheck<F>(prover_transcript: &mut KeccakTranscript, K: usize, mle_polys: &[Vec<F>], sumcheck_claim: F)
where
    F: JoltField
{
    let sumcheck_claim_icicle  = unsafe { reinterpret_field(&sumcheck_claim) };
    let sumcheck_config = SumcheckConfig::default();
    let mle_poly_hosts = mle_polys
        .iter()
        .map(|coeffs| {
            let coeffs = unsafe {
                reinterpret_field_slice(coeffs)
                // std::slice::from_raw_parts(coeffs.as_ptr() as *const <SW as Sumcheck>::Field, coeffs.len())
            };

            HostSlice::from_slice(coeffs)
        })
        .collect::<Vec<_>>();

    let hasher = icicle_hash::keccak::Keccak256::new(32u64).unwrap();
    let seed_rng = <<SumcheckWrapper as Sumcheck>::FieldConfig>::generate_random(1)[0];

    let combine_func = |vars: &mut Vec<<FieldReturningValueProgram as ReturningValueProgram>::ProgSymbol>| -> <FieldReturningValueProgram as ReturningValueProgram>::ProgSymbol {
        let ra = vars[0]; // Shallow copies pointing to the same memory in the backend
        let val = vars[1];
        return ra * val;
    };
    let combine_func = FieldReturningValueProgram::new(combine_func, 2).unwrap();

    let config = SumcheckTranscriptConfig::new(
        &hasher,
        b"Domain".to_vec(),
        b"Round".to_vec(),
        b"Challenge".to_vec(),
        true, // little endian
        seed_rng,
    );

    let sumcheck = SumcheckWrapper::new().unwrap();
    let proof = sumcheck.prove(
        &mle_poly_hosts,
        K as u64,
        sumcheck_claim_icicle,
        combine_func,
        &config,
        &sumcheck_config,
    );

    let proof_round_polys =
        <<SumcheckWrapper as Sumcheck>::Proof>::get_round_polys(&proof).unwrap();
    // Convert this into compressed Polys
    let compressed_polys = proof_round_polys.par_iter().map(|coeffs| {
        let coeffs = coeffs.iter().map(|c| icicle_to_jolt(c)).collect::<Vec<_>>();
        UniPoly::from_evals(&coeffs).compress()
    }).collect::<Vec<CompressedUniPoly<F>>>();

    // Add these to our transcript to remain consistent with the rest of the proof
    compressed_polys.iter().for_each(|p| {
        p.append_to_transcript(prover_transcript);
        prover_transcript.challenge_scalar::<F>();
    });

    // #[cfg(feature = "icicle")]
    // let (_, _, _, _) =
    //         prove_core_shout_piop_icicle::<_, _, SumcheckWrapper, icicle_bn254::program::bn254::FieldReturningValueProgram>(lookup_table.clone(), read_addresses.clone(), &mut prover_transcript);
}

#[cfg(feature = "icicle")]
fn icicle_sumcheck2<F>(prover_transcript: &mut KeccakTranscript, K: usize, mle_poly_device: &[&DeviceVec<ScalarField>], stream: &mut IcicleStream, sumcheck_claim: F)
where
    F: JoltField
{
    // let mut stream = IcicleStream::create().unwrap();
    let sumcheck_claim_icicle  = unsafe { reinterpret_field(&sumcheck_claim) };
    let mut sumcheck_config = SumcheckConfig::default();
    sumcheck_config.are_inputs_on_device = true;
    sumcheck_config.is_async = true;
    sumcheck_config.stream = stream.clone().into();
    // print the available memory on the device now
    println!("Available Memory Start: {:?}", icicle_runtime::get_available_memory());

    // let mle_poly_hosts: Vec<DeviceVec<ScalarField>> = mle_polys
    //     .iter()
    //     .map(|coeffs| {
    //         let coeffs = unsafe {
    //             reinterpret_field_slice(coeffs)
    //         };

    //         let host_slice = HostSlice::from_slice(coeffs);
    //         let mut device_vec = DeviceVec::<<SumcheckWrapper as Sumcheck>::Field>::device_malloc_async(coeffs.len(), &stream).unwrap();
    //         device_vec.copy_from_host_async(host_slice, &stream).unwrap();
    //         device_vec
    //     })
    //     .collect::<Vec<_>>();
    //     let mle_poly_hosts = mle_poly_hosts.iter().collect::<Vec<_>>();

    let hasher = icicle_hash::keccak::Keccak256::new(32u64).unwrap();
    let seed_rng = <<SumcheckWrapper as Sumcheck>::FieldConfig>::generate_random(1)[0];

    let combine_func = |vars: &mut Vec<<FieldReturningValueProgram as ReturningValueProgram>::ProgSymbol>| -> <FieldReturningValueProgram as ReturningValueProgram>::ProgSymbol {
        let ra = vars[0]; // Shallow copies pointing to the same memory in the backend
        let val = vars[1];
        return ra * val;
    };
    let combine_func = FieldReturningValueProgram::new(combine_func, 2).unwrap();

    let config = SumcheckTranscriptConfig::new(
        &hasher,
        b"Domain".to_vec(),
        b"Round".to_vec(),
        b"Challenge".to_vec(),
        true, // little endian
        seed_rng,
    );


    println!("Available Memory Before Prove: {:?}", icicle_runtime::get_available_memory());
    let sumcheck = SumcheckWrapper::new().unwrap();
    let proof = sumcheck.prove(
        mle_poly_device,
        K as u64,
        sumcheck_claim_icicle,
        combine_func,
        &config,
        &sumcheck_config,
    );
    println!("Available Memory After Prove: {:?}", icicle_runtime::get_available_memory());

    stream.synchronize().expect("Failed to synchronize the stream");
    // stream.destroy().unwrap();

    let proof_round_polys =
        <<SumcheckWrapper as Sumcheck>::Proof>::get_round_polys(&proof).unwrap();
    // Convert this into compressed Polys
    let compressed_polys = proof_round_polys.par_iter().map(|coeffs| {
        let coeffs = coeffs.iter().map(|c| icicle_to_jolt(c)).collect::<Vec<_>>();
        UniPoly::from_evals(&coeffs).compress()
    }).collect::<Vec<CompressedUniPoly<F>>>();
    drop(proof_round_polys);

    // Add these to our transcript to remain consistent with the rest of the proof
    compressed_polys.iter().for_each(|p| {
        p.append_to_transcript(prover_transcript);
        prover_transcript.challenge_scalar::<F>();
    });
}

fn sumcheck<F: JoltField>(transcript: &mut KeccakTranscript, K: usize, mle_polys: &[Vec<F>], sumcheck_claim: F) {
    let mut previous_claim = sumcheck_claim;
    let num_rounds = K.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    let mut ra = MultilinearPolynomial::from(mle_polys[0].clone());
    let mut val = MultilinearPolynomial::from(mle_polys[1].clone());

    const DEGREE: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let _ra_claim = ra.final_sumcheck_claim();
}

fn benchmark_sumcheck<F, ProofTranscript>(c: &mut Criterion, name: &str, benchmark_mode: BenchmarkMode)
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let mut rng = test_rng();

    let lookup_table: Vec<F> = (0..TABLE_SIZE).map(|_| F::random(&mut rng)).collect();
    println!("lookup_table.len() = {}", lookup_table.len());
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        .collect();

    #[cfg(feature = "icicle")]
    icicle_init();

    // setup the benchmark
    let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
    // #[cfg(feature = "icicle")]
    let (K, mle_polys, sumcheck_claim) =
        setup_sumcheck::<F, KeccakTranscript>(lookup_table.clone(), &read_addresses.clone(), &mut prover_transcript);
    println!("mle_polys.len() = {}", mle_polys[0].len());

    #[cfg(feature = "icicle")]
    let mut stream = IcicleStream::create().unwrap();
    #[cfg(feature = "icicle")]
    let mut mle_polys_device: Vec<DeviceVec<ScalarField>> = vec![];
    #[cfg(feature = "icicle")]
    if benchmark_mode == BenchmarkMode::Full2 {
        let now = Instant::now();
        mle_polys_device = mle_polys
            .iter()
            .map(|coeffs| {
                let coeffs = unsafe {
                    reinterpret_field_slice(coeffs)
                };
    
                let host_slice = HostSlice::from_slice(coeffs);
                let mut device_vec = DeviceVec::<<SumcheckWrapper as Sumcheck>::Field>::device_malloc_async(coeffs.len(), &stream).unwrap();
                println!("Copying to device item of len {}", coeffs.len());
                device_vec.copy_from_host_async(host_slice, &stream).unwrap();
                device_vec
            })
            .collect::<Vec<_>>();
    
        #[cfg(feature = "icicle")]
        stream.synchronize().unwrap();
        println!("Copy to Device took {:.2?}", now.elapsed());
    }
    #[cfg(feature = "icicle")]
    let mle_polys_device = mle_polys_device.iter().collect::<Vec<_>>();

    // #[cfg(not(feature = "icicle"))]
    // let (K, mle_polys, sumcheck_claim) =
    //     crate::setup_sumcheck::<F, KeccakTranscript>(lookup_table, read_addresses.clone(), &mut prover_transcript);
    let F = setup_F(&lookup_table, &read_addresses, &mut prover_transcript);

    #[cfg(feature = "icicle")]
    let id = format!("{} [mode:Icicle]", name);
    #[cfg(not(feature = "icicle"))]
    let id = format!("{} [mode:JOLT CPU]", name);
    c.bench_function(&id, |b| {
        b.iter(|| {
            match benchmark_mode {
                BenchmarkMode::F => { setup_F(&lookup_table, &read_addresses, &mut prover_transcript); },
                BenchmarkMode::Claim => { setup_sumcheck_claim::<F, ProofTranscript>(&lookup_table, &F); },
                // BenchmarkMode::Claim2 => { setup_sumcheck_claim2::<F, ProofTranscript>(&lookup_table, &F); },
                BenchmarkMode::Full => {
                    // let (K, mle_polys, sumcheck_claim ) = setup_sumcheck(lookup_table.clone(), &read_addresses, &mut prover_transcript);
                    {
                        #[cfg(feature = "icicle")]
                        icicle_sumcheck(&mut prover_transcript, K, &mle_polys, sumcheck_claim);
                        #[cfg(not(feature = "icicle"))]
                        sumcheck(
                            &mut prover_transcript,
                            K,
                            &mle_polys,
                            sumcheck_claim,
                        );
                    }
                },
                BenchmarkMode::Full2 => {
                    // let (K, mle_polys, sumcheck_claim ) = setup_sumcheck(lookup_table.clone(), &read_addresses, &mut prover_transcript);
                    #[cfg(feature = "icicle")]
                    icicle_sumcheck2(&mut prover_transcript, K, &mle_polys_device, &mut stream, sumcheck_claim);
                    #[cfg(not(feature = "icicle"))]
                    sumcheck(
                        &mut prover_transcript,
                        K,
                        &mle_polys,
                        sumcheck_claim,
                    );

                },
            }
        });
    });
    #[cfg(feature = "icicle")]
    stream.destroy().unwrap();
    #[cfg(feature = "icicle")]
    drop(mle_polys_device);
}

#[derive(PartialEq)]
enum BenchmarkMode {
    F,
    Claim,
    // Claim2,
    Full,
    Full2,
}

fn main() {
    let small_value_lookup_tables = <Fr as JoltField>::compute_lookup_tables();
    <Fr as JoltField>::initialize_lookup_tables(small_value_lookup_tables);
    jolt_core::msm::icicle_init();

    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(5));
    // benchmark_sumcheck::<Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "Sumcheck::shout_core_piop",
    // );
    // benchmark_sumcheck::<Fr, KeccakTranscript>(
    //   &mut criterion,
    //   "Sumcheck::shout_setup_F",
    //   BenchmarkMode::F
    // );
    // benchmark_sumcheck::<Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "Sumcheck::shout_setup_sumcheck_claim",
    //     BenchmarkMode::Claim
    // );
    // benchmark_sumcheck::<Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "Sumcheck::shout_setup_sumcheck_claim_gpu",
    //     BenchmarkMode::Claim2
    // );

    // benchmark_sumcheck::<Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "Sumcheck::shout_setup_sumcheck_prove",
    //     BenchmarkMode::Full
    // );
    benchmark_sumcheck::<Fr, KeccakTranscript>(
        &mut criterion,
        "Sumcheck::shout_setup_sumcheck_prove_gpu2",
        BenchmarkMode::Full2
    );
    criterion.final_summary();
}
