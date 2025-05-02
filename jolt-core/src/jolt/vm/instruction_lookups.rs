use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::{instruction::LookupQuery, lookup_table::LookupTables},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{
        sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{unsafe_allocate_zero_array, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: SumcheckInstanceProof<F, ProofTranscript>,
    rv_claim: F,
    ra_claims: [F; 4],
    flag_claims: Vec<F>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(preprocessing: (), lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        generators: &PCS::Setup,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let log_T = trace.len().log_2();
        let r_cycle: Vec<F> = transcript.challenge_vector(log_T);
        let (read_checking_proof, rv_claim, ra_claims, flag_claims) =
            prove_sparse_dense_shout::<WORD_SIZE, _, _>(&trace, r_cycle, transcript);

        prove_ra_booleanity::<F, ProofTranscript>(trace, transcript);
        prove_ra_hamming_weight::<F, ProofTranscript>(trace, transcript);

        // TODO(moodlezoup): Interleaved raf evaluation

        Self {
            read_checking_proof,
            rv_claim,
            ra_claims,
            flag_claims,
            log_T,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &self.read_checking_proof,
            self.log_T,
            r_cycle,
            self.rv_claim,
            self.ra_claims,
            &self.flag_claims,
            transcript,
        )
    }
}

#[tracing::instrument(skip_all)]
fn prove_ra_booleanity<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    transcript: &mut ProofTranscript,
) {
    const DEGREE: usize = 3;
    const LOG_K: usize = 16;
    const K: usize = 1 << LOG_K;
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let r: Vec<F> = transcript.challenge_vector(LOG_K);
    let r_prime: Vec<F> = transcript.challenge_vector(T.log_2());
    let z: F = transcript.challenge_scalar();
    let z_squared = z.square();
    let z_cubed = z_squared * z;

    // Can reuse from read-checking sumcheck
    let D: Vec<F> = EqPolynomial::evals(&r_prime);

    let span = tracing::span!(tracing::Level::INFO, "compute G");
    let _guard = span.enter();

    let G: [Vec<F>; 4] = trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result = [
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
            ];
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                let k = lookup_index % K as u64;
                result[3][k as usize] += D[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[2][k as usize] += D[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[1][k as usize] += D[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[0][k as usize] += D[j];
                j += 1;
            }
            result
        })
        .reduce(
            || {
                [
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                ]
            },
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        );

    drop(_guard);
    drop(span);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(&r)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: [F; K] = unsafe_allocate_zero_array();
    F[0] = F::one();

    let num_rounds = LOG_K + T.log_2();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(LOG_K);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
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
    // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    let span = tracing::span!(
        tracing::Level::INFO,
        "First log(K) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    for round in 0..LOG_K {
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F = (G[0][k_G]
                            + z * G[1][k_G]
                            + z_squared * G[2][k_G]
                            + z_cubed * G[3][k_G])
                            * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                            G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                            G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
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

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update F");
        let _inner_guard = inner_span.enter();

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << round);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(
        tracing::Level::INFO,
        "Last log(T) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    let eq_r_r = B.final_sumcheck_claim();
    let mut H: [MultilinearPolynomial<F>; 4] = std::array::from_fn(|i| {
        let coeffs: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                let k = (lookup_index >> (LOG_K * (3 - i))) % K as u64;
                F[k as usize]
            })
            .collect();
        MultilinearPolynomial::from(coeffs)
    });
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
        #[cfg(test)]
        {
            let expected: F = eq_r_r
                * (0..H.len())
                    .map(|j| {
                        let D_j = D.get_bound_coeff(j);
                        let H_j = [
                            H[0].get_bound_coeff(j),
                            H[1].get_bound_coeff(j),
                            H[2].get_bound_coeff(j),
                            H[3].get_bound_coeff(j),
                        ];
                        D_j * ((H_j[0].square() - H_j[0])
                            + z * (H_j[1].square() - H_j[1])
                            + z_squared * (H_j[2].square() - H_j[2])
                            + z_cubed * (H_j[3].square() - H_j[3]))
                    })
                    .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = [
                    H[0].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[1].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[2].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                    H[3].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh),
                ];

                let mut evals = [
                    H_evals[0][0].square() - H_evals[0][0],
                    H_evals[0][1].square() - H_evals[0][1],
                    H_evals[0][2].square() - H_evals[0][2],
                ];

                evals[0] += z * (H_evals[1][0].square() - H_evals[1][0]);
                evals[1] += z * (H_evals[1][1].square() - H_evals[1][1]);
                evals[2] += z * (H_evals[1][2].square() - H_evals[1][2]);

                evals[0] += z_squared * (H_evals[2][0].square() - H_evals[2][0]);
                evals[1] += z_squared * (H_evals[2][1].square() - H_evals[2][1]);
                evals[2] += z_squared * (H_evals[2][2].square() - H_evals[2][2]);

                evals[0] += z_cubed * (H_evals[3][0].square() - H_evals[3][0]);
                evals[1] += z_cubed * (H_evals[3][1].square() - H_evals[3][1]);
                evals[2] += z_cubed * (H_evals[3][2].square() - H_evals[3][2]);

                [
                    D_evals[0] * evals[0],
                    D_evals[1] * evals[1],
                    D_evals[2] * evals[2],
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

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        H.par_iter_mut()
            .chain([&mut D].into_par_iter())
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
    }
}

#[tracing::instrument(skip_all)]
fn prove_ra_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    transcript: &mut ProofTranscript,
) {
    const LOG_K: usize = 16;
    const K: usize = 1 << LOG_K;
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let r_cycle_prime: Vec<F> = transcript.challenge_vector(T.log_2());
    let z: F = transcript.challenge_scalar();
    let z_squared = z.square();
    let z_cubed = z_squared * z;

    let num_rounds = LOG_K;
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let E: Vec<F> = EqPolynomial::evals(&r_cycle_prime);
    let mut F: [Vec<F>; 4] = trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result = [
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
                unsafe_allocate_zero_vec(K),
            ];
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                let k = lookup_index % K as u64;
                result[3][k as usize] += E[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[2][k as usize] += E[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[1][k as usize] += E[j];

                lookup_index = lookup_index >> LOG_K;
                let k = lookup_index % K as u64;
                result[0][k as usize] += E[j];
                j += 1;
            }
            result
        })
        .reduce(
            || {
                [
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                ]
            },
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        );

    let mut ra = [
        MultilinearPolynomial::from(std::mem::take(&mut F[0])),
        MultilinearPolynomial::from(std::mem::take(&mut F[1])),
        MultilinearPolynomial::from(std::mem::take(&mut F[2])),
        MultilinearPolynomial::from(std::mem::take(&mut F[3])),
    ];
    let mut previous_claim = F::one();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let mut univariate_poly_eval: F = (0..ra[0].len() / 2)
            .into_par_iter()
            .map(|i| ra[0].get_bound_coeff(2 * i))
            .sum();
        univariate_poly_eval += (0..ra[1].len() / 2)
            .into_par_iter()
            .map(|i| ra[1].get_bound_coeff(2 * i))
            .sum::<F>()
            * z;
        univariate_poly_eval += (0..ra[2].len() / 2)
            .into_par_iter()
            .map(|i| ra[2].get_bound_coeff(2 * i))
            .sum::<F>()
            * z_squared;
        univariate_poly_eval += (0..ra[3].len() / 2)
            .into_par_iter()
            .map(|i| ra[3].get_bound_coeff(2 * i))
            .sum::<F>()
            * z_cubed;

        let univariate_poly =
            UniPoly::from_evals(&[univariate_poly_eval, previous_claim - univariate_poly_eval]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
    }
}
