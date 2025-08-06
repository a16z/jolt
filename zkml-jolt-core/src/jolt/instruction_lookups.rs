use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use rayon::prelude::*;
use std::marker::PhantomData;

use jolt_core::{
    field::JoltField,
    jolt::lookup_table::LookupTables,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};

use crate::{
    jolt::{
        JoltProverPreprocessing,
        execution_trace::{JoltONNXCycle, ONNXLookupQuery},
    },
    subprotocols::sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: ReadCheckingProof<F, ProofTranscript>,
    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadCheckingProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    rv_claim: F,
    ra_claims: [F; 4],
    add_sub_mul_flag_claim: F,
    flag_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; 4],
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(_preprocessing: (), _lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        _preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        execution_trace: &[JoltONNXCycle],
        _opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = execution_trace.len() * MAX_TENSOR_SIZE;
        let log_T = T.log_2();
        let r_cycle: Vec<F> = transcript.challenge_vector(log_T);
        let (
            read_checking_sumcheck,
            rv_claim,
            ra_claims,
            add_sub_mul_flag_claim,
            flag_claims,
            eq_r_cycle,
        ) = prove_sparse_dense_shout::<_, _>(execution_trace, &r_cycle, transcript);
        let read_checking_proof = ReadCheckingProof {
            sumcheck_proof: read_checking_sumcheck,
            rv_claim,
            ra_claims,
            add_sub_mul_flag_claim,
            flag_claims,
        };

        // TODO(moodlezoup): Openings
        let (booleanity_sumcheck, _, _, ra_claims) = prove_ra_booleanity::<F, ProofTranscript>(
            execution_trace,
            eq_r_cycle.clone(),
            transcript,
        );
        let booleanity_proof = BooleanityProof {
            sumcheck_proof: booleanity_sumcheck,
            ra_claims,
        };

        // TODO(moodlezoup): Openings
        let (hamming_weight_sumcheck, _r_address, ra_claims) =
            prove_ra_hamming_weight::<F, ProofTranscript>(execution_trace, eq_r_cycle, transcript);
        let hamming_weight_proof = HammingWeightProof {
            sumcheck_proof: hamming_weight_sumcheck,
            ra_claims,
        };

        // TODO: Openings: https://github.com/ICME-Lab/zkml-jolt/issues/66
        // let unbound_ra_polys = vec![
        //     CommittedPolynomials::InstructionRa(0).generate_witness(preprocessing, trace),
        //     CommittedPolynomials::InstructionRa(1).generate_witness(preprocessing, trace),
        //     CommittedPolynomials::InstructionRa(2).generate_witness(preprocessing, trace),
        //     CommittedPolynomials::InstructionRa(3).generate_witness(preprocessing, trace),
        // ];

        // let r_address_rev = r_address.iter().copied().rev().collect::<Vec<_>>();

        // opening_accumulator.append_sparse(
        //     unbound_ra_polys,
        //     r_address_rev,
        //     r_cycle,
        //     ra_claims.to_vec(),
        // );

        // TODO(moodlezoup): Interleaved raf evaluation

        Self {
            read_checking_proof,
            booleanity_proof,
            hamming_weight_proof,
            log_T,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        // commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            &self.read_checking_proof.sumcheck_proof,
            self.log_T,
            r_cycle.clone(),
            self.read_checking_proof.rv_claim,
            self.read_checking_proof.ra_claims,
            self.read_checking_proof.add_sub_mul_flag_claim,
            &self.read_checking_proof.flag_claims,
            transcript,
        )?;

        let mut r_address: Vec<F> = transcript.challenge_vector(16);
        let z_booleanity: F = transcript.challenge_scalar();
        let z_booleanity_squared: F = z_booleanity.square();
        let z_booleanity_cubed: F = z_booleanity_squared * z_booleanity;
        let (sumcheck_claim, r_booleanity) = self.booleanity_proof.sumcheck_proof.verify(
            F::zero(),
            16 + self.log_T,
            3,
            transcript,
        )?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(16);

        r_address = r_address.into_iter().rev().collect();
        let eq_eval_address = EqPolynomial::mle(&r_address, r_address_prime);
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&r_cycle_rev, r_cycle_prime);

        assert_eq!(
            eq_eval_address
                * eq_eval_cycle
                * ((self.booleanity_proof.ra_claims[0].square()
                    - self.booleanity_proof.ra_claims[0])
                    + z_booleanity
                        * (self.booleanity_proof.ra_claims[1].square()
                            - self.booleanity_proof.ra_claims[1])
                    + z_booleanity_squared
                        * (self.booleanity_proof.ra_claims[2].square()
                            - self.booleanity_proof.ra_claims[2])
                    + z_booleanity_cubed
                        * (self.booleanity_proof.ra_claims[3].square()
                            - self.booleanity_proof.ra_claims[3])),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        let z_hamming_weight: F = transcript.challenge_scalar();
        let z_hamming_weight_squared: F = z_hamming_weight.square();
        let z_hamming_weight_cubed: F = z_hamming_weight_squared * z_hamming_weight;
        let (sumcheck_claim, _r_hamming_weight) = self.hamming_weight_proof.sumcheck_proof.verify(
            F::one() + z_hamming_weight + z_hamming_weight_squared + z_hamming_weight_cubed,
            16,
            1,
            transcript,
        )?;

        assert_eq!(
            self.hamming_weight_proof.ra_claims[0]
                + z_hamming_weight * self.hamming_weight_proof.ra_claims[1]
                + z_hamming_weight_squared * self.hamming_weight_proof.ra_claims[2]
                + z_hamming_weight_cubed * self.hamming_weight_proof.ra_claims[3],
            sumcheck_claim,
            "Hamming weight sumcheck failed"
        );

        // let r_hamming_weight: Vec<_> = r_hamming_weight.iter().copied().rev().collect();
        // for i in 0..4 {
        //     opening_accumulator.append(
        //         &[&commitments.commitments[CommittedPolynomials::InstructionRa(i).to_index()]],
        //         [r_hamming_weight.as_slice(), r_cycle.as_slice()].concat(),
        //         &[self.hamming_weight_proof.ra_claims[i]],
        //         transcript,
        //     );
        // }

        Ok(())
    }
}

#[tracing::instrument(skip_all)]
fn prove_ra_booleanity<F: JoltField, ProofTranscript: Transcript>(
    trace: &[JoltONNXCycle],
    eq_r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    Vec<F>,
    [F; 4],
) {
    const DEGREE: usize = 3;
    const LOG_K: usize = 16;
    const K: usize = 1 << LOG_K;
    let T = trace.len() * MAX_TENSOR_SIZE;
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let r_address: Vec<F> = transcript.challenge_vector(LOG_K);
    let z: F = transcript.challenge_scalar();
    let z_squared = z.square();
    let z_cubed = z_squared * z;

    let span = tracing::span!(tracing::Level::INFO, "compute G");
    let _guard = span.enter();

    let lookup_indices: Vec<u64> = trace
        .par_iter()
        .flat_map(ONNXLookupQuery::<32>::to_lookup_index)
        .collect();
    assert_eq!(lookup_indices.len(), T);
    let G: [Vec<F>; 4] = lookup_indices
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
            for lookup_index in trace_chunk {
                let mut lookup_index = *lookup_index;
                let k = lookup_index % K as u64;
                result[3][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[2][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[1][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[0][k as usize] += eq_r_cycle[j];
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

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(&r_address)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
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
        let coeffs: Vec<F> = lookup_indices
            .par_iter()
            .map(|lookup_index| {
                let k = (lookup_index >> (LOG_K * (3 - i))) % K as u64;
                F[k as usize]
            })
            .collect();
        MultilinearPolynomial::from(coeffs)
    });
    let mut D = MultilinearPolynomial::from(eq_r_cycle);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // TODO(moodlezoup): Implement optimization from Section 6.2.2 "An optimization leveraging small memory size"
    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
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

    let ra_claims = [
        H[0].final_sumcheck_claim(),
        H[1].final_sumcheck_claim(),
        H[2].final_sumcheck_claim(),
        H[3].final_sumcheck_claim(),
    ];

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_prime,
        r_cycle_prime,
        ra_claims,
    )
}

#[tracing::instrument(skip_all)]
fn prove_ra_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    trace: &[JoltONNXCycle],
    eq_r_cycle: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 4]) {
    const LOG_K: usize = 16;
    const K: usize = 1 << LOG_K;
    let T = trace.len() * MAX_TENSOR_SIZE;
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let z: F = transcript.challenge_scalar();
    let z_squared = z.square();
    let z_cubed = z_squared * z;

    let num_rounds = LOG_K;
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let lookup_indices: Vec<u64> = trace
        .par_iter()
        .flat_map(ONNXLookupQuery::<32>::to_lookup_index)
        .collect();
    assert_eq!(lookup_indices.len(), T);
    let mut F: [Vec<F>; 4] = lookup_indices
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
            for lookup_index in trace_chunk {
                let mut lookup_index = *lookup_index;
                let k = lookup_index % K as u64;
                result[3][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[2][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[1][k as usize] += eq_r_cycle[j];

                lookup_index >>= LOG_K;
                let k = lookup_index % K as u64;
                result[0][k as usize] += eq_r_cycle[j];
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
    let mut previous_claim = F::one() + z + z_squared + z_cubed;

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

    let ra_claims = [
        ra[0].final_sumcheck_claim(),
        ra[1].final_sumcheck_claim(),
        ra[2].final_sumcheck_claim(),
        ra[3].final_sumcheck_claim(),
    ];

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_double_prime,
        ra_claims,
    )
}
