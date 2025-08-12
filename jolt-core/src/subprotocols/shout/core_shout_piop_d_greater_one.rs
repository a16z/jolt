#![allow(unused_imports)]
use super::helpers::*;
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof};
use crate::utils::counters::{get_mult_count, reset_mult_count};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::{errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

/// Implements the sumcheck prover for the generic core Shout PIOP for d>1.
/// See Figure 7 of https://eprint.iacr.org/2025/105
/// Reference implementation without Gruen Or Split Eq Optimisation
pub fn prove_generic_core_shout_pip_d_greater_than_one<
    F: JoltField,
    ProofTranscript: Transcript,
>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    d: usize,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    F,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();
    let N = (K as f64).powf(1.0 / d as f64).round() as usize;
    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k
    let C = construct_vector_c_in_shout(K, &read_addresses, &E_star);
    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    // The sum check answer (for d=1, it's the same as normal one)
    let sumcheck_claim: F = C
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;

    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C);
    let mut val = MultilinearPolynomial::from(lookup_table);

    // Binding the first log_2 K variables
    // How many evaluations we need
    const DEGREE_ADDR: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _addr_idx in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE_ADDR] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE_ADDR=2
            })
            .reduce(
                || [F::zero(); DEGREE_ADDR],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        // Construct coefficients of univariate polynomial from evaluations
        // No Gruen optimisation here for now
        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();

        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate(&r_j);

        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();

    // At this point we should have bound the first log K variables
    // Making E_star into a ML poly
    let mut eq_r_cycle = MultilinearPolynomial::from(E_star);
    // As d > 1, we will have d arrays each of length T
    let eq_taus: Vec<Vec<F>> = compute_eq_taus_parallel(&r_address, d, N.log_2());

    // This is the same E as the one referenced on Page 51 of Shetty/Thaler
    let mut E: Vec<Vec<F>> = vec![vec![F::zero(); T]; d];

    E.par_iter_mut().enumerate().for_each(|(j, e_j)| {
        // for a fixed table e_j iterate through all of time stamps
        e_j.par_iter_mut().enumerate().for_each(|(y, e)| {
            // take the memory cell to be read at time y and extrac j'th digit: addr_j
            // (which is also the index in array e_j[y: addr_j])
            // Here when j=0 we get the MSB and when j=d-1 we get the LSB
            let addr_j = digit_j_of(read_addresses[y], j, d, N);
            // Since eq_taus[0] contains the first log N bits of of read_address
            // we adjust the indexing
            *e = eq_taus[d - j - 1][addr_j];
        });
    });
    E.reverse();
    // The E tables will be dropped once we make ra_taus
    let mut ra_taus: Vec<MultilinearPolynomial<F>> =
        E.into_par_iter().map(MultilinearPolynomial::from).collect();

    let DEGREE_TME: usize = d + 1;
    for _time_round_idx in 0..T.log_2() {
        let univariate_poly_evals: Vec<F> = (0..ra_taus[0].len() / 2)
            .into_par_iter()
            .map(|index| {
                let eq_r_cycle_evals =
                    eq_r_cycle.sumcheck_evals(index, DEGREE_TME, BindingOrder::LowToHigh);
                // For each of the d ra_taus we should get d sumcheck evals as the evaluation at 1
                // is constructed from the previous claim
                // This happens only d times so there's no need to parallelise
                let ra_evals_per_tau: Vec<Vec<F>> = ra_taus
                    .iter()
                    .map(|ra_tau| ra_tau.sumcheck_evals(index, DEGREE_TME, BindingOrder::LowToHigh))
                    .collect();

                // The parallelisation should be over ra_evals_per_tau which can be as
                // large as ra_taus[0].len()/2 = T/2 initially.
                // It shrinks by half at each round
                // TODO: once the size of ra_taus[0] is small enough
                // we should swtich from par_iter to iter.
                let result: Vec<F> = (0..DEGREE_TME)
                    .map(|i| {
                        let col_product = ra_evals_per_tau
                            .par_iter()
                            .map(|row| row[i])
                            .reduce(|| F::one(), |a, b| a * b);

                        col_product * eq_r_cycle_evals[i]
                    })
                    .collect();
                result
            })
            .reduce(
                || vec![F::zero(); DEGREE_TME],
                |running, new| {
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(a, b)| *a + *b)
                        .collect()
                },
            );

        let d_plus_two_evaluations = construct_final_sumcheck_evals(
            &univariate_poly_evals,
            val_claim,
            previous_claim,
            DEGREE_TME,
        );
        let univariate_poly = UniPoly::from_evals(&d_plus_two_evaluations);

        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate(&r_j);
        rayon::join(
            || {
                ra_taus.par_iter_mut().for_each(|ra_tau| {
                    ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            },
            || eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ras_raddress_rtime_product: F = ra_taus
        .par_iter()
        .map(|ra| ra.final_sumcheck_claim())
        .reduce(|| F::one(), |acc, val| acc * val);

    let eq_r_cycle_at_r_time = eq_r_cycle.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ras_raddress_rtime_product,
        val_claim,
        eq_r_cycle_at_r_time,
        previous_claim,
    )
}

/// Implements the sumcheck prover for the generic core Shout PIOP for d>1.
/// Include the split-eq + gruen optimisations.
/// See Figure 7 of https://eprint.iacr.org/2025/105
/// The latest Gruen + Split Poly optimisation  can be found in Figure 5 of
/// https://eprint.iacr.org/2025/1117.pdf
pub fn prove_generic_core_shout_pip_d_greater_than_one_with_gruen<
    F: JoltField,
    ProofTranscript: Transcript,
>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    d: usize,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    F,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();
    let N = (K as f64).powf(1.0 / d as f64).round() as usize;
    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k

    let C = construct_vector_c_in_shout(K, &read_addresses, &E_star);
    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    let sumcheck_claim: F = C
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;

    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C);
    let mut val = MultilinearPolynomial::from(lookup_table);

    // Binding the first log_2 K variables
    const DEGREE_ADDR: usize = 2; // independent of d
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _addr_idx in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE_ADDR] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE_ADDR, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE_ADDR=2
            })
            .reduce(
                || [F::zero(); DEGREE_ADDR],
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

        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();
    // At this point we should have bound the first log K variables
    // As d > 1, we will have d arrays each of length T
    let eq_taus: Vec<Vec<F>> = compute_eq_taus_serial(&r_address, d, N.log_2());
    // This is the same E as the one referenced on Page 51 of Shetty/Thaler
    let mut Es: Vec<Vec<F>> = vec![vec![F::zero(); T]; d];
    // Filling out E involve any multiplications (Just lookups)

    // d is never very large : no need to parallelise over d
    // ENDIAN STUFF -- this because extact last digits first so E[j] updates is being mapped to eq_taus[d-j-1]
    for (j, e_j) in Es.iter_mut().rev().enumerate() {
        // each e_j has T slots and T is large so this should be parallel
        e_j.par_iter_mut().enumerate().for_each(|(y, e)| {
            let addr_j = digit_j_of(read_addresses[y], j, d, N);
            *e = eq_taus[d - j - 1][addr_j];
        });
    }

    // The E tables will be dropped once we make ra_taus
    // d is small : no need to parallelise
    let mut ra_taus: Vec<MultilinearPolynomial<F>> =
        Es.into_iter().map(MultilinearPolynomial::from).collect();

    // Making E_star into a SplitEqPoly
    // The 2 EqPolynomials here should not be parellelised
    // as they start with size \sqrt{T} which is below 16
    // which is the parallel threshold as max T = 2**32

    let mut greq_r_cycle = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);
    // This how many evals we need to evaluate t(x)
    // The degree of t is d
    let degree = d;
    for _time_round_idx in 0..T.log_2() {
        let E_2 = greq_r_cycle.E_in_current();
        let E_1 = greq_r_cycle.E_out_current();

        let mut evals_of_t: Vec<F> = (0..E_1.len())
            .into_par_iter()
            .map(|x1| {
                // Parallel over x2, compute scaled eval vectors by E_2[x2]
                let inner_sum: Vec<F> = (0..E_2.len())
                    .into_par_iter()
                    .map(|x2| {
                        let idx = x1 * E_2.len() + x2;

                        // d x degree matrix
                        // columns are evals at [0, 2,..,degree]
                        // normally with d+1 uni poly we'd need d+2 locations
                        // but with gruen and split eq we only need
                        let ra_evals_per_tau: Vec<Vec<F>> = ra_taus
                            .iter()
                            .map(|ra_tau| {
                                ra_tau.sumcheck_evals(idx, degree, BindingOrder::LowToHigh)
                            })
                            .collect();

                        let mut prod = vec![F::one(); degree];
                        // reduce column wise to get product
                        for row in &ra_evals_per_tau {
                            for i in 0..degree {
                                prod[i] *= row[i];
                            }
                        }

                        // Multiply by E_2[x2] only here
                        for p in prod.iter_mut() {
                            *p *= E_2[x2];
                        }

                        prod
                    })
                    .reduce(
                        || vec![F::zero(); degree],
                        |mut acc, v| {
                            for (a, b) in acc.iter_mut().zip(v) {
                                *a += b;
                            }
                            acc
                        },
                    );

                // Multiply the entire combined vector by E_1[x1]
                inner_sum
                    .into_iter()
                    .map(|v| v * E_1[x1])
                    .collect::<Vec<F>>()
            })
            .reduce(
                || vec![F::zero(); degree],
                |mut acc, v| {
                    for (a, b) in acc.iter_mut().zip(v) {
                        *a += b;
                    }
                    acc
                },
            );

        let ell_at_0 = val_claim
            * greq_r_cycle.get_current_scalar()
            * (F::one() - greq_r_cycle.get_current_w());
        let ell_at_1 = val_claim * greq_r_cycle.get_current_scalar() * greq_r_cycle.get_current_w();
        // ell is linear: so do interpolation, don't use multiplication
        let mut d_plus_two_evaluations_of_ell: Vec<F> = vec![F::zero(); d + 2];
        d_plus_two_evaluations_of_ell[0] = ell_at_0;
        let m = ell_at_1 - ell_at_0;
        let mut eval = ell_at_1;
        for i in 1..(d + 2) {
            d_plus_two_evaluations_of_ell[i] = eval;
            eval += m;
        }

        // Procedure 8 from Dao/Thaler/Domb/Baggad
        let ell_one_inverse = ell_at_1
            .inverse()
            .expect("Tried to invert zero (ell_at_1 has no inverse)");

        let t_at_zero = evals_of_t[0];
        let t_at_one = ell_one_inverse * (previous_claim - t_at_zero * ell_at_0);
        evals_of_t.insert(1, t_at_one);

        // d + 2 evaluations of t are sufficient to construct the polynomial
        let t_x = UniPoly::from_evals(&evals_of_t);
        //We need this to evaluate t @ d+2 which we don't hae right now
        // but this will only take d mults instead of 2^{\log T - round_i}

        let d_plus_two_evaluations: Vec<F> = (0..(d + 2))
            .map(|i| {
                if i == d + 1 {
                    t_x.evaluate(&F::from_u16(i as u16)) * d_plus_two_evaluations_of_ell[i]
                } else {
                    evals_of_t[i] * d_plus_two_evaluations_of_ell[i]
                }
            })
            .collect();

        // Construct coefficients of univariate polynomial from evaluations
        let univariate_poly = UniPoly::from_evals(&d_plus_two_evaluations);

        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate(&r_j);

        rayon::join(
            || {
                ra_taus.iter_mut().for_each(|ra_tau| {
                    ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            },
            || greq_r_cycle.bind(r_j),
        );
    }

    let ras_raddress_rtime_product: F = ra_taus
        .par_iter()
        .map(|ra| ra.final_sumcheck_claim())
        .reduce(|| F::one(), |acc, val| acc * val);

    let eq_r_cycle_at_r_time = greq_r_cycle.get_current_scalar();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ras_raddress_rtime_product,
        val_claim,
        eq_r_cycle_at_r_time,
        previous_claim,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    //use crate::field::tracked_ark::TrackedFr as Fr; // Use to track mults
    use ark_ff::UniformRand;
    use ark_ff::{One, Zero}; // often from ark_ff, depending on your setup
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::test_rng;
    use rand_core::RngCore;

    fn decompose_one_hot_matrix<F: JoltField>(
        read_addresses: &[usize],
        K: usize,
        d: usize,
    ) -> Vec<Vec<Vec<F>>> {
        let T = read_addresses.len();
        let N = (K as f64).powf(1.0 / d as f64).round() as usize;
        assert_eq!(N.pow(d as u32), K, "K must be a perfect power of N");

        // Step 1: compute base-N digits for each address
        let digits_per_addr: Vec<Vec<usize>> = read_addresses
            .par_iter()
            .map(|&addr| {
                let mut digits = vec![0; d];
                let mut rem = addr;
                for j in (0..d).rev() {
                    digits[j] = rem % N;
                    rem /= N;
                }
                digits
            })
            .collect();

        // Step 2: build d matrices of shape T x N
        let mut result = vec![vec![vec![F::zero(); N]; T]; d];

        for (i, digits) in digits_per_addr.iter().enumerate() {
            for (j, &digit) in digits.iter().enumerate() {
                result[d - j - 1][i][digit] = F::one();
            }
        }

        result
    }

    #[test]
    fn test_core_shout_generic_d_greater_than_one_shout_sumcheck() {
        //------- PROBLEM SETUP----------------------
        const POWER_OF_2: usize = 22;
        const K: usize = 64; // 2**6
        const T: usize = 1 << POWER_OF_2; // 2**power
        const D: usize = 3;
        let N = (K as f64).powf(1.0 / D as f64).round() as usize;
        assert_eq!(N.pow(D as u32), K, "K must be a perfect power of N");

        let seed1: u64 = 42;
        let mut rng1 = StdRng::seed_from_u64(seed1);
        let lookup_table: Vec<Fr> = (0..K).map(|_| Fr::rand(&mut rng1)).collect();
        let read_addresses: Vec<usize> = (0..T).map(|_| (rng1.next_u32() as usize) % K).collect();
        assert_eq!(T, read_addresses.len());
        assert_eq!(K, lookup_table.len());
        //-------------------------------------------------

        let ras: Vec<Vec<Vec<Fr>>> = decompose_one_hot_matrix(&read_addresses, K, D);
        let flattened_ras: Vec<Vec<Fr>> = (0..D)
            .into_par_iter()
            .map(|d| {
                ras[d]
                    .iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect::<Vec<Fr>>()
            })
            .collect();

        let ra_polys: Vec<MultilinearPolynomial<Fr>> = flattened_ras
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect();
        let val = MultilinearPolynomial::from(lookup_table.clone());
        //-------------------------------------------------------------------------------
        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");

        let start = Instant::now();
        let (
            _sumcheck_proof_wo,
            _verifier_challenges_wo,
            _sumcheck_claim_wo,
            ra_address_time_claim_wo,
            val_tau_claim_wo,
            eq_rcycle_rtime_claim_wo,
            final_claim_wo,
        ) = prove_generic_core_shout_pip_d_greater_than_one(
            lookup_table.clone(),
            read_addresses.clone(),
            D,
            &mut prover_transcript,
        );
        let end = start.elapsed().as_millis();
        println!("Took {end} ms\n");

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        reset_mult_count();
        let start = Instant::now();
        let (
            sumcheck_proof,
            verifier_challenges,
            sumcheck_claim,
            ra_address_time_claim,
            val_tau_claim,
            eq_rcycle_rtime_claim,
            final_claim,
        ) = prove_generic_core_shout_pip_d_greater_than_one_with_gruen(
            lookup_table.clone(),
            read_addresses.clone(),
            D,
            &mut prover_transcript,
        );
        let end = start.elapsed().as_millis();
        let gruen_opt_prover = get_mult_count();

        println!(
            "Took {} ms\nMultiplications: Optimised: {}: Assymptotics {}",
            end,
            gruen_opt_prover,
            (D * D + D + 1) * T + 5 * K + 4 * (1 << (POWER_OF_2 / 2))
        );

        // Thesea are sanity checks to see that the openings
        // of the sum_check_with_split_eq and gruen opts are
        // the same as the linear time prover.
        // The only difference should be one is a lot faster than the other.
        assert_eq!(val_tau_claim, val_tau_claim_wo);
        assert_eq!(eq_rcycle_rtime_claim, eq_rcycle_rtime_claim_wo);
        assert_eq!(ra_address_time_claim, ra_address_time_claim_wo);
        assert_eq!(final_claim, final_claim_wo);
        let _product = ra_address_time_claim * val_tau_claim * eq_rcycle_rtime_claim;

        // Now we check if the verification aligns with thee opening
        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        // Already in Big ENDIAN
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            K.log_2() + T.log_2(),
            D + 2,
            &mut verifier_transcript,
        );
        let (final_claim, _vfr_challenges) = verification_result.unwrap();
        //-------------------------------------------------------------------------

        // Simulating the Polynomial commitment opening
        let (r_address, r_time) = verifier_challenges.split_at(K.log_2());
        let mut r_address_rev = r_address.to_vec();
        r_address_rev.reverse();
        let val_at_r_address = val.evaluate(&r_address_rev);

        // Now i need to take r_address and split it into D chunks
        let chunk_size = N.log_2();
        let r_address_chunked: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let start = i * chunk_size;
                let end = (i + 1) * chunk_size;
                r_address[start..end].to_vec()
            })
            .collect();

        // this is r_address_chunkded || r_time
        let full_random_locations: Vec<Vec<Fr>> = (0..D)
            .map(|i| {
                let mut combined = r_address_chunked[i].clone(); // clone the chunk
                assert_eq!(combined.len(), N.log_2());
                combined.extend_from_slice(r_time); // append r_time
                combined
            })
            .collect();

        let evaluations: Vec<Fr> = (0..D)
            .map(|i| {
                let mut random_location_rev = full_random_locations[i].clone();
                random_location_rev.reverse();
                assert_eq!(random_location_rev.len(), T.log_2() + N.log_2());
                ra_polys[i].evaluate(&random_location_rev) // no semicolon, return this value
            })
            .collect();

        let evaluations_product: Fr = evaluations.iter().fold(Fr::one(), |acc, val| acc * val);
        assert_eq!(evaluations_product, ra_address_time_claim);
        assert_eq!(val_at_r_address, val_tau_claim);
        let mut r_time_rev: Vec<Fr> = r_time.to_vec();
        r_time_rev.reverse();
        let eq_r_cycle_at_r_time =
            MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle)).evaluate(&r_time_rev);
        assert_eq!(eq_r_cycle_at_r_time, eq_rcycle_rtime_claim);

        let final_oracle_answer = val_at_r_address * evaluations_product * eq_r_cycle_at_r_time;
        assert_eq!(final_oracle_answer, final_claim);
    }
}
