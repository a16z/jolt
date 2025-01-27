use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};
use rayon::prelude::*;

pub struct ShoutProof<F: JoltField, ProofTranscript: Transcript> {
    core_piop_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    booleanity_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_prime: F,
    rv_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
    pub fn prove(
        lookup_table: Vec<F>,
        read_addresses: Vec<usize>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = lookup_table.len();
        let T = read_addresses.len();
        debug_assert_eq!(r_cycle.len(), T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let num_rounds = K.log_2();
        let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

        let E: Vec<F> = EqPolynomial::evals(&r_cycle);
        let F: Vec<_> = (0..K)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum::<F>()
            })
            .collect();

        let rv_claim: F = F
            .par_iter()
            .zip(lookup_table.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();
        // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
        let mut previous_claim = rv_claim + z;

        let mut ra = MultilinearPolynomial::from(F.clone());
        let mut val = MultilinearPolynomial::from(lookup_table);

        const DEGREE: usize = 2;

        // Prove the core PIOP and Hamming weight sumchecks in parallel
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    [
                        ra_evals[0] * (z + val_evals[0]),
                        ra_evals[1] * (z + val_evals[1]),
                    ]
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
                || ra.bind(r_j, BindingOrder::LowToHigh),
                || val.bind(r_j, BindingOrder::LowToHigh),
            );
        }

        let ra_claim = ra.final_sumcheck_claim();

        let core_piop_sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

        let (booleanity_sumcheck_proof, r_address_prime, r_cycle_prime, ra_claim_prime) =
            prove_booleanity(read_addresses, &r_address, E, F, transcript);

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Self {
            core_piop_sumcheck: core_piop_sumcheck_proof,
            booleanity_sumcheck: booleanity_sumcheck_proof,
            ra_claim,
            ra_claim_prime,
            rv_claim,
        }
    }

    pub fn verify(
        &self,
        K: usize,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let _z: F = transcript.challenge_scalar();

        self.core_piop_sumcheck
            .verify(self.rv_claim, K.log_2(), 2, transcript)?;

        self.booleanity_sumcheck
            .verify(F::zero(), K.log_2() + T.log_2(), 3, transcript)?;

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Ok(())
    }
}

/// Implements the sumcheck prover for the core Shout PIOP when d = 1. See
/// Figure 5 from the Twist+Shout paper.
pub fn prove_core_shout_piop<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());

    // Sumcheck for the core Shout PIOP (Figure 5)
    let num_rounds = K.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let sumcheck_claim: F = F
        .par_iter()
        .zip(lookup_table.par_iter())
        .map(|(&ra, &val)| ra * val)
        .sum();
    let mut previous_claim = sumcheck_claim;

    let mut ra = MultilinearPolynomial::from(F);
    let mut val = MultilinearPolynomial::from(lookup_table);

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
            || ra.bind(r_j, BindingOrder::LowToHigh),
            || val.bind(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address,
        sumcheck_claim,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the Booleanity check in step 3 of
/// Figure 6 in the Twist+Shout paper. The efficient implementation of this
/// sumcheck is described in Section 6.3.
pub fn prove_booleanity<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F) {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = read_addresses.len();
    debug_assert_eq!(D.len(), T);
    debug_assert_eq!(G.len(), K);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(r)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
    F[0] = F::one();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(K.log_2());
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    let eq_km_c: [[F; DEGREE]; 2] = [
        [
            F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
            F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
            F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
        ],
        [
            F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
            F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
            F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
        ],
    ];
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    for round in 0..K.log_2() {
        let m = round + 1;
        let F_squared: Vec<_> = F[..(1 << round)]
            .par_iter()
            .map(|F_k| F_k.square())
            .collect();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        [
                            G_k * (eq_km_c_squared[k % 2][0] * F_squared[k >> 1]
                                - eq_km_c[k % 2][0] * F[k >> 1]),
                            G_k * (eq_km_c_squared[k % 2][1] * F_squared[k >> 1]
                                - eq_km_c[k % 2][1] * F[k >> 1]),
                            G_k * (eq_km_c_squared[k % 2][2] * F_squared[k >> 1]
                                - eq_km_c[k % 2][2] * F[k >> 1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind(r_j, BindingOrder::LowToHigh);

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << m);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x + r_j;
                *x -= *y;
            });
    }

    let eq_r_r = B.final_sumcheck_claim();
    let H: Vec<F> = read_addresses.par_iter().map(|&k| F[k]).collect();
    let mut H = MultilinearPolynomial::from(H);
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // Last log(T) rounds of sumcheck
    for _ in 0..T.log_2() {
        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                    D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                    D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals = [
            eq_r_r * univariate_poly_evals[0],
            eq_r_r * univariate_poly_evals[1],
            eq_r_r * univariate_poly_evals[2],
        ];

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind(r_j, BindingOrder::LowToHigh),
            || H.bind(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = H.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_prime,
        r_cycle_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the Hamming weight 1 check in step 5 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle_prime: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle_prime.len());

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle_prime);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let mut ra = MultilinearPolynomial::from(F);
    let mut previous_claim = F::one();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_eval: F = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| ra.get_bound_coeff(2 * i))
            .sum();

        let univariate_poly =
            UniPoly::from_evals(&[univariate_poly_eval, previous_claim - univariate_poly_eval]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.bind(r_j, BindingOrder::LowToHigh);
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_double_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the raf-evaluation sumcheck in step 6 of
/// Figure 6 in the Twist+Shout paper.
pub fn prove_raf_evaluation<F: JoltField, ProofTranscript: Transcript>(
    lookup_table: Vec<F>,
    read_addresses: Vec<usize>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, F) {
    let K = lookup_table.len();
    let T = read_addresses.len();
    debug_assert_eq!(T.log_2(), r_cycle.len());

    let E: Vec<F> = EqPolynomial::evals(&r_cycle);
    let F: Vec<_> = (0..K)
        .into_par_iter()
        .map(|k| {
            read_addresses
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| if *address == k { Some(E[cycle]) } else { None })
                .sum::<F>()
        })
        .collect();

    let num_rounds = K.log_2();
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let mut ra = MultilinearPolynomial::from(F);
    let mut int = IdentityPolynomial::new(num_rounds);

    let mut previous_claim = claimed_evaluation;

    const DEGREE: usize = 2;

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let int_evals: Vec<F> = int.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [ra_evals[0] * int_evals[0], ra_evals[1] * int_evals[1]]
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
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind(r_j, BindingOrder::LowToHigh),
            || int.bind(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = ra.final_sumcheck_claim();
    (SumcheckInstanceProof::new(compressed_polys), ra_claim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn shout_e2e() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let proof = ShoutProof::prove(
            lookup_table,
            read_addresses,
            &r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let verification_result = proof.verify(TABLE_SIZE, NUM_LOOKUPS, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn core_shout_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let (sumcheck_proof, _, sumcheck_claim, _) =
            prove_core_shout_piop(lookup_table, read_addresses, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let verification_result = sumcheck_proof.verify(
            sumcheck_claim,
            TABLE_SIZE.log_2(),
            2,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn booleanity_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(TABLE_SIZE.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let E: Vec<Fr> = EqPolynomial::evals(&r_prime);
        let F: Vec<_> = (0..TABLE_SIZE)
            .into_par_iter()
            .map(|k| {
                read_addresses
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(cycle, address)| if *address == k { Some(E[cycle]) } else { None },
                    )
                    .sum::<Fr>()
            })
            .collect();

        let (sumcheck_proof, _, _, _) =
            prove_booleanity(read_addresses, &r, E, F, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(TABLE_SIZE.log_2());
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result = sumcheck_proof.verify(
            Fr::zero(),
            TABLE_SIZE.log_2() + NUM_LOOKUPS.log_2(),
            3,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn hamming_weight_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle_prime: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let (sumcheck_proof, _, _) = prove_hamming_weight(
            lookup_table,
            read_addresses,
            r_cycle_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(Fr::one(), TABLE_SIZE.log_2(), 1, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn raf_evaluation_sumcheck() {
        const TABLE_SIZE: usize = 64;
        const NUM_LOOKUPS: usize = 1 << 10;

        let mut rng = test_rng();

        let lookup_table: Vec<Fr> = (0..TABLE_SIZE).map(|_| Fr::random(&mut rng)).collect();
        let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
            .map(|_| rng.next_u32() as usize % TABLE_SIZE)
            .collect();
        let raf = MultilinearPolynomial::from(
            read_addresses.iter().map(|a| *a as u32).collect::<Vec<_>>(),
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let raf_eval = raf.evaluate(&r_cycle);
        let (sumcheck_proof, _) = prove_raf_evaluation(
            lookup_table,
            read_addresses,
            r_cycle,
            raf_eval,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());

        let verification_result =
            sumcheck_proof.verify(raf_eval, TABLE_SIZE.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
