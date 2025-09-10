#![allow(unused_imports)]
use super::helpers::*;
use crate::field::MontU128;
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
use std::time::Instant;
use std::{cell::RefCell, rc::Rc};
/// Implements the sumcheck prover for the generic core Shout PIOP for d=1.
/// See Figure 7 of https://eprint.iacr.org/2025/105
/// This is a reference implementation without Gruen/Split Poly
pub fn prove_generic_core_shout_pip<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<MontU128>,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();

    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<MontU128> = transcript.challenge_vector_u128(T.log_2());

    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k
    let C: Vec<_> = (0..K) // This is C[x] = ra(r_cycle, x)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| {
                    if *address == k {
                        // this check will be more complex for d > 1 but let's keep
                        // this for now
                        Some(E_star[cycle])
                    } else {
                        None
                    }
                })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<MontU128> = Vec::with_capacity(num_rounds);

    // The sum check answer (for d=1, it's the same as normal one)
    let sumcheck_claim: F = C
        .iter()
        .zip(lookup_table.iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;
    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C);
    let mut val = MultilinearPolynomial::from(lookup_table);

    // Binding the first log_2 K variables
    const DEGREE: usize = 2;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _addr_idx in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE=2
            })
            .reduce(
                || [F::zero(); DEGREE],
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

        // Get challenge that binds the variable
        let r_j = transcript.challenge_u128();
        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate_u128(&r_j);
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();

    // At this point we should have bound the first log K variables
    // Binding the second log T variables
    let mut eq_r_cycle = MultilinearPolynomial::from(E_star);
    // Endian issue
    let mut r_address_reversed = r_address.clone();
    r_address_reversed.reverse();
    let eq_tau: Vec<F> = EqPolynomial::evals(&r_address_reversed);
    let mut E = vec![F::zero(); T];
    E.par_iter_mut().enumerate().for_each(|(y, e)| {
        *e = eq_tau[read_addresses[y]];
    });
    let mut ra_tau = MultilinearPolynomial::from(E);

    for _round_time in 0..T.log_2() {
        //let start = Instant::now();
        let univariate_poly_evals: [F; 2] = (0..ra_tau.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra_tau.sumcheck_evals(index, 2, BindingOrder::LowToHigh);
                let val_evals = eq_r_cycle.sumcheck_evals(index, 2, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            val_claim * univariate_poly_evals[0],
            previous_claim - val_claim * univariate_poly_evals[0],
            val_claim * univariate_poly_evals[1],
        ]);
        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);
        // Get challenge that binds the variable
        let r_j = transcript.challenge_u128();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate_u128(&r_j);

        rayon::join(
            || ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh),
            || eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_tau_claim = ra_tau.final_sumcheck_claim();
    let eq_r_cycle_at_r_time = eq_r_cycle.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ra_tau_claim,
        val_claim,
        eq_r_cycle_at_r_time,
    )
}

/// Implements the sumcheck prover for the generic core Shout PIOP for d=1.
/// With Gruen And Split Poly Opts Included
/// See Figure 7 of https://eprint.iacr.org/2025/105
pub fn prove_generic_core_shout_piop_d_is_one_w_gruen<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<MontU128>,
    F,
    F,
    F,
    F,
) {
    // This assumes that K and T are powers of 2
    let K = lookup_table.len();
    let T = read_addresses.len();

    // A random field element F^{\log_2 T} for Schwartz-Zippll
    // This is stored in Big Endian
    let r_cycle: Vec<MontU128> = transcript.challenge_vector_u128(T.log_2());
    // Page 50: eq(44)
    let E_star: Vec<F> = EqPolynomial::evals(&r_cycle);
    // Page 50: eq(47) : what the paper calls v_k

    // This is sub-optimal -> TODO
    // read_addresses is much bigger
    // so this a thing to check
    let C = construct_vector_c_in_shout(K, &read_addresses, &E_star);

    // The sum check answer.
    // Note that it does not matter whether d=1 or d > 1
    // The final sum-check answer is the same.
    let sumcheck_claim: F = C
        .iter()
        .zip(lookup_table.iter())
        .map(|(&ra, &val)| ra * val)
        .sum();

    let mut previous_claim = sumcheck_claim;

    // These are the polynomials the prover commits to
    let mut ra = MultilinearPolynomial::from(C); // \widetilde{ra}(r_cycle, X_1, ..., X_logK)
    let mut val = MultilinearPolynomial::from(lookup_table); // \widetilde{Val}(X_1, ..., X_logK)

    // Binding the first log_2 K variables
    const DEGREE: usize = 2;
    let num_rounds = K.log_2() + T.log_2();
    // The vector storing the verifiers sum-check challenges
    let mut r_address: Vec<MontU128> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let start = Instant::now();
    for _ in 0..K.log_2() {
        // Page 51: (eq 51)
        let univariate_poly_evals: [F; DEGREE] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]] // since DEGREE=2
            })
            .reduce(
                || [F::zero(); DEGREE],
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

        // Get challenge that binds the variable
        let r_j = transcript.challenge_u128();
        r_address.push(r_j);

        previous_claim = univariate_poly.evaluate_u128(&r_j);

        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }
    let duration = start.elapsed();
    println!(
        "\n Large (d is one, Gruen Opts)- Execution time: {}",
        duration.as_nanos()
    );

    // tau = r_address (the verifiers challenges which bind all log K variables of memory)
    // This is \widetilde{Val}(\tau) from the paper (eq 52)
    let val_claim = val.final_sumcheck_claim();

    // At this point we should have bound the first log K variables
    // Binding the second log T variables
    //let mut eq_r_cycle = MultilinearPolynomial::from(E_star);

    // Endian issue: Polynomial Evaluation is currently BigEndian
    // But Sum-check evaluation is LittleEndian -- Lo To Hi
    // Thus, reversing is needed
    let mut r_address_reversed = r_address.clone();
    r_address_reversed.reverse();

    let eq_tau: Vec<F> = EqPolynomial::evals(&r_address_reversed); // This takes K mults.
    let mut E = vec![F::zero(); T];
    E.par_iter_mut().enumerate().for_each(|(y, e)| {
        *e = eq_tau[read_addresses[y]];
    });
    // widetilde{ra}(r_1, ..., r_address, Y_1, ..., Y_logT)
    let mut ra_tau = MultilinearPolynomial::<F>::from(E);
    let mut greq_r_cycle = GruenSplitEqPolynomial::<F>::new(&r_cycle, BindingOrder::LowToHigh);
    for _round_i in 0..T.log_2() {
        let E_2 = greq_r_cycle.E_in_current();
        let E_1 = greq_r_cycle.E_out_current();

        let degree = 1;
        //let start = Instant::now();
        let mut evals_of_t: Vec<F> = (0..E_1.len())
            .into_par_iter()
            .map(|x1| {
                // Partial sum over x2
                let mut partial_sum = vec![F::zero(); degree];
                for x2 in 0..E_2.len() {
                    let idx = x1 * E_2.len() + x2;
                    let evals = ra_tau.sumcheck_evals(idx, degree, BindingOrder::LowToHigh);

                    // Multiply evals by E_2[x2] once here
                    for (p, e) in partial_sum.iter_mut().zip(evals) {
                        *p += e * E_2[x2];
                    }
                }

                // Now multiply the whole partial sum by E_1[x1] once
                for p in partial_sum.iter_mut() {
                    *p *= E_1[x1];
                }

                partial_sum
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

        // Procedure 8
        let ell_at_0 = val_claim
            * greq_r_cycle.get_current_scalar()
            * (F::one() - F::from_u128_mont(greq_r_cycle.get_current_w()));
        let ell_at_1 = (val_claim*greq_r_cycle.get_current_scalar()).mul_u128_mont_form(greq_r_cycle.get_current_w());

        // One inverse
        let ell_one_inverse = ell_at_1
            .inverse()
            .expect("Tried to invert zero (ell_at_1 has no inverse)");
        let t_at_zero = evals_of_t[0];

        // 2 Mults per round
        let t_at_one = ell_one_inverse * (previous_claim - t_at_zero * ell_at_0);

        evals_of_t.insert(1, t_at_one);

        // when d = 1; t is also a linear polynomial
        //let t_x = UniPoly::from_evals(&evals_of_t);
        //let t_at_two = t_x.evaluate(&F::from_u8(2));
        let t_at_two = t_at_one + t_at_one - t_at_zero;

        let ell_at_two = ell_at_1 + ell_at_1 - ell_at_0;
        let s_at_two = ell_at_two * t_at_two;

        // Construct coefficients of univariate polynomial from evaluations
        // This still needs 3 points
        // 3 mults per rounds
        let univariate_poly =
            UniPoly::from_evals(&[ell_at_0 * t_at_zero, ell_at_1 * t_at_one, s_at_two]);

        // Skip the linear term when storing coeffs as we can always re-construct it
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        // Get challenge that binds the variable
        let r_j = transcript.challenge_u128();
        r_address.push(r_j);
        previous_claim = univariate_poly.evaluate_u128(&r_j);
        ra_tau.bind_parallel(r_j, BindingOrder::LowToHigh);
        greq_r_cycle.bind(r_j);
    }

    let ra_tau_claim = ra_tau.final_sumcheck_claim();
    let eq_r_cycle_at_r_time = greq_r_cycle.get_current_scalar();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ra_tau_claim,
        val_claim,
        eq_r_cycle_at_r_time,
    )
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    // use crate::field::tracked_ark::TrackedFr as Fr;
    use ark_ff::UniformRand;
    use ark_ff::{One, Zero}; // often from ark_ff, depending on your setup
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn test_core_shout_generic_d_is_one() {
        //------- PROBLEM SETUP----------------------
        const TABLE_SIZE: usize = 64; // 2**6
        const NUM_LOOKUPS: usize = 1 << 16; // 2**10

        let seed1: u64 = 42;
        let mut rng1 = StdRng::seed_from_u64(seed1);
        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::rand(&mut rng1)).collect();

        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| (rng1.next_u32() as usize) % TABLE_SIZE)
            .collect();

        let table_size = lookup_table.len();
        let num_lookups = read_addresses.len();
        let ra: Vec<Vec<Fr>> = read_addresses
            .par_iter()
            .map(|&addr| {
                (0..table_size)
                    .into_par_iter()
                    .map(|j| if j == addr { Fr::one() } else { Fr::zero() })
                    .collect()
            })
            .collect();

        let flattened: Vec<Fr> = ra.iter().flat_map(|row| row.iter().cloned()).collect();

        // What the prover commits to
        let ra_poly = MultilinearPolynomial::from(flattened);
        let val = MultilinearPolynomial::from(lookup_table.clone());

        //-------------------------------------------------------------------------------

        reset_mult_count();
        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let (
            _sumcheck_proof,
            _verifier_challenges_wo_gruen,
            _sumcheck_claim,
            _ra_address_time_claim,
            _val_tau_claim,
            _eq_rcycle_rtime_claim,
        ) = prove_generic_core_shout_pip(
            lookup_table.clone(),
            read_addresses.clone(),
            &mut prover_transcript,
        );

        let start = Instant::now();
        let linear_prover = get_mult_count();
        reset_mult_count();
        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let (
            sumcheck_proof,
            _verifier_challenges,
            sumcheck_claim,
            ra_address_time_claim,
            val_tau_claim,
            eq_rcycle_rtime_claim,
        ) = prove_generic_core_shout_piop_d_is_one_w_gruen(
            lookup_table,
            read_addresses,
            &mut prover_transcript,
        );
        let gruen_opt_prover = get_mult_count();

        const T: usize = NUM_LOOKUPS;
        println!(
            "Lin: {}\tOptimised: {}: savings {} ~~ {}",
            linear_prover,
            gruen_opt_prover,
            linear_prover - gruen_opt_prover,
            2 * T + 4 * (1 << 5)
        );
        // See page 51 of Twist and Shout paper for a derivation of the above asymptotics
        let end = start.elapsed();
        println!("Elapsed time: {:.5?}", end.as_micros());

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let r_cycle: Vec<MontU128> = verifier_transcript.challenge_vector_u128(num_lookups.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            table_size.log_2() + num_lookups.log_2(),
            2,
            &mut verifier_transcript,
        );
        let (final_claim, verifier_challenges) = verification_result.unwrap();
        //-----------------------------------------------------------------------

        let (r_address, r_time) = verifier_challenges.split_at(table_size.log_2());
        let mut r_address = r_address.to_vec();
        r_address.reverse();
        let val_at_r_address = val.evaluate(&r_address);

        let mut full_random_location = verifier_challenges.clone();
        full_random_location.reverse();
        let ra_evaluated_r_address_r_time = ra_poly.evaluate(&full_random_location);
        let eq_r_cycle = MultilinearPolynomial::from(EqPolynomial::<Fr>::evals(&r_cycle));
        let mut r_time = r_time.to_vec();
        r_time.reverse();
        let eq_r_cycle_r_time = eq_r_cycle.evaluate(&r_time);

        // These are the 3 product terms evaluated at the final veerifiers
        // challenges
        assert_eq!(ra_evaluated_r_address_r_time, ra_address_time_claim);
        assert_eq!(val_at_r_address, val_tau_claim);
        assert_eq!(eq_r_cycle_r_time, eq_rcycle_rtime_claim);
        assert_eq!(
            final_claim,
            ra_evaluated_r_address_r_time * eq_r_cycle_r_time * val_at_r_address,
            "GRUEN FAILS,"
        );
    }
}
