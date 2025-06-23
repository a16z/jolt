use super::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof};
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
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    core_piop_claims: ShoutSumcheckClaims<F>,
    ra_claim_prime: F,
}

struct ShoutProverState<F: JoltField> {
    K: usize,
    rv_claim: F,
    z: F,
    ra: MultilinearPolynomial<F>,
    val: MultilinearPolynomial<F>,
}

impl<F: JoltField> ShoutProverState<F> {
    #[tracing::instrument(skip_all)]
    fn initialize<ProofTranscript: Transcript>(
        lookup_table: Vec<F>,
        read_addresses: &[usize],
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let K = lookup_table.len();
        let T = read_addresses.len();
        debug_assert_eq!(r_cycle.len(), T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let E: Vec<F> = EqPolynomial::evals(r_cycle);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);
        let F: Vec<_> = read_addresses
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, addresses)| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut cycle = chunk_index * chunk_size;
                for address in addresses {
                    result[*address] += E[cycle];
                    cycle += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );
        drop(_guard);
        drop(span);

        let rv_claim: F = F
            .par_iter()
            .zip(lookup_table.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();

        let ra = MultilinearPolynomial::from(F.clone());
        let val = MultilinearPolynomial::from(lookup_table);

        let prover_state = Self {
            K,
            z,
            rv_claim,
            ra,
            val,
        };
        (prover_state, E, F)
    }
}

#[derive(Clone)]
struct ShoutSumcheckClaims<F: JoltField> {
    ra_claim: F,
    rv_claim: F,
}

struct ShoutVerifierState<F: JoltField> {
    K: usize,
    z: F,
    val: MultilinearPolynomial<F>,
}

impl<F: JoltField> ShoutVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        lookup_table: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = lookup_table.len();
        let z: F = transcript.challenge_scalar();
        let val = MultilinearPolynomial::from(lookup_table);
        Self { K, z, val }
    }
}

struct ShoutSumcheck<F: JoltField> {
    verifier_state: Option<ShoutVerifierState<F>>,
    prover_state: Option<ShoutProverState<F>>,
    claims: Option<ShoutSumcheckClaims<F>>,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ShoutSumcheck<F>
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().K.log_2()
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().K.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            let ShoutProverState { rv_claim, z, .. } = self.prover_state.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else if self.verifier_state.is_some() {
            let ShoutVerifierState { z, .. } = self.verifier_state.as_ref().unwrap();
            let ShoutSumcheckClaims { rv_claim, .. } = self.claims.as_ref().unwrap();
            // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
            *rv_claim + z
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let ShoutProverState { ra, val, z, .. } = self.prover_state.as_ref().unwrap();

        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (*z + val_evals[0]),
                    ra_evals[1] * (*z + val_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let ShoutProverState { ra, val, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let ShoutProverState { rv_claim, ra, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(ShoutSumcheckClaims {
            ra_claim: ra.final_sumcheck_claim(),
            rv_claim: *rv_claim,
        });
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ShoutVerifierState { z, val, .. } = self.verifier_state.as_ref().unwrap();
        let ShoutSumcheckClaims { ra_claim, .. } = self.claims.as_ref().unwrap();

        let r_address: Vec<F> = r.iter().rev().copied().collect();
        *ra_claim * (*z + val.evaluate(&r_address))
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ShoutProof::prove")]
    pub fn prove(
        lookup_table: Vec<F>,
        read_addresses: Vec<usize>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let (core_piop_prover_state, E, F) =
            ShoutProverState::initialize(lookup_table, &read_addresses, r_cycle, transcript);
        let booleanity_prover_state =
            BooleanityProverState::initialize(read_addresses, E, F, transcript);

        let mut core_piop_sumcheck = ShoutSumcheck {
            prover_state: Some(core_piop_prover_state),
            verifier_state: None,
            claims: None,
        };

        let mut booleanity_sumcheck = BooleanitySumcheck {
            prover_state: Some(booleanity_prover_state),
            verifier_state: None,
            ra_claim: None,
        };

        let (sumcheck_proof, _r_sumcheck) = BatchedSumcheck::prove(
            vec![&mut core_piop_sumcheck, &mut booleanity_sumcheck],
            transcript,
        );

        let core_piop_claims = core_piop_sumcheck.claims.unwrap();

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Self {
            sumcheck_proof,
            core_piop_claims,
            ra_claim_prime: booleanity_sumcheck.ra_claim.unwrap(),
        }
    }

    pub fn verify(
        &self,
        lookup_table: Vec<F>,
        r_cycle: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = lookup_table.len();

        let core_piop_verifier_state = ShoutVerifierState::initialize(lookup_table, transcript);
        let booleanity_verifier_state = BooleanityVerifierState::initialize(r_cycle, K, transcript);

        let core_piop_sumcheck = ShoutSumcheck {
            prover_state: None,
            verifier_state: Some(core_piop_verifier_state),
            claims: Some(self.core_piop_claims.clone()),
        };

        let booleanity_sumcheck = BooleanitySumcheck {
            prover_state: None,
            verifier_state: Some(booleanity_verifier_state),
            ra_claim: Some(self.ra_claim_prime),
        };

        let _r_sumcheck = BatchedSumcheck::verify(
            &self.sumcheck_proof,
            vec![&core_piop_sumcheck, &booleanity_sumcheck],
            transcript,
        )?;

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
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
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

struct BooleanityProverState<F: JoltField> {
    read_addresses: Vec<usize>,
    K: usize,
    T: usize,
    B: MultilinearPolynomial<F>,
    F: Vec<F>,
    G: Vec<F>,
    D: MultilinearPolynomial<F>,
    /// Initialized after first log(K) rounds of sumcheck
    H: Option<MultilinearPolynomial<F>>,
    /// EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
    eq_km_c: [[F; 3]; 2],
    /// EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    eq_km_c_squared: [[F; 3]; 2],
}

impl<F: JoltField> BooleanityProverState<F> {
    #[tracing::instrument(skip_all)]
    fn initialize<ProofTranscript: Transcript>(
        read_addresses: Vec<usize>,
        D: Vec<F>,
        G: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = G.len();
        let r: Vec<F> = transcript.challenge_vector(K.log_2());
        const DEGREE: usize = 3;
        let T = read_addresses.len();

        let D = MultilinearPolynomial::from(D);
        let B = MultilinearPolynomial::from(EqPolynomial::evals(&r)); // (53)
        let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
        F[0] = F::one();

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

        Self {
            read_addresses,
            K,
            T,
            B,
            F,
            G,
            D,
            H: None,
            eq_km_c,
            eq_km_c_squared,
        }
    }
}

struct BooleanityVerifierState<F: JoltField> {
    eq_r_address: EqPolynomial<F>,
    eq_r_cycle: EqPolynomial<F>,
}

impl<F: JoltField> BooleanityVerifierState<F> {
    fn initialize<ProofTranscript: Transcript>(
        r_cycle: &[F],
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let r_address: Vec<F> = transcript
            .challenge_vector(K.log_2())
            .into_iter()
            .rev()
            .collect();

        Self {
            eq_r_cycle: EqPolynomial::new(r_cycle),
            eq_r_address: EqPolynomial::new(r_address),
        }
    }
}

struct BooleanitySumcheck<F: JoltField> {
    verifier_state: Option<BooleanityVerifierState<F>>,
    prover_state: Option<BooleanityProverState<F>>,
    ra_claim: Option<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            let BooleanityProverState { K, T, .. } = self.prover_state.as_ref().unwrap();
            K.log_2() + T.log_2()
        } else if self.verifier_state.is_some() {
            let BooleanityVerifierState {
                eq_r_cycle,
                eq_r_address,
            } = self.verifier_state.as_ref().unwrap();
            let K = eq_r_address.len();
            let T = eq_r_cycle.len();
            K.log_2() + T.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&self, round: usize) -> Vec<F> {
        const DEGREE: usize = 3;
        let BooleanityProverState {
            K,
            B,
            F,
            G,
            D,
            H,
            eq_km_c,
            eq_km_c_squared,
            ..
        } = self.prover_state.as_ref().unwrap();

        if round < K.log_2() {
            // First log(K) rounds of sumcheck
            let m = round + 1;

            let univariate_poly_evals: [F; DEGREE] = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                    let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                        .par_iter()
                        .enumerate()
                        .map(|(k, &G_k)| {
                            // Since we're binding variables from low to high, k_m is the high bit
                            let k_m = k >> (m - 1);
                            // We then index into F using (k_{m-1}, ..., k_1)
                            let F_k = F[k % (1 << (m - 1))];
                            // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                            let G_times_F = G_k * F_k;
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
                            || [F::zero(); DEGREE],
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
                    || [F::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            univariate_poly_evals.to_vec()
        } else {
            // Last log(T) rounds of sumcheck

            let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
                .into_par_iter()
                .map(|i| {
                    let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                    let H_evals =
                        H.as_ref()
                            .unwrap()
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

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

            let eq_r_r = B.final_sumcheck_claim();
            univariate_poly_evals = [
                eq_r_r * univariate_poly_evals[0],
                eq_r_r * univariate_poly_evals[1],
                eq_r_r * univariate_poly_evals[2],
            ];

            univariate_poly_evals.to_vec()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let BooleanityProverState {
            K,
            B,
            F,
            D,
            H,
            read_addresses,
            ..
        } = self.prover_state.as_mut().unwrap();
        if round < K.log_2() {
            // First log(K) rounds of sumcheck
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

            if round == K.log_2() - 1 {
                // Transition point; initialize H
                *H = Some(MultilinearPolynomial::from(
                    read_addresses.par_iter().map(|&k| F[k]).collect::<Vec<_>>(),
                ));
            }
        } else {
            // Last log(T) rounds of sumcheck
            rayon::join(
                || D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || {
                    H.as_mut()
                        .unwrap()
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claim.is_none());
        let BooleanityProverState { H, .. } = self.prover_state.as_ref().unwrap();
        let ra_claim = H.as_ref().unwrap().final_sumcheck_claim();
        self.ra_claim = Some(ra_claim);
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let BooleanityVerifierState {
            eq_r_address,
            eq_r_cycle,
        } = self.verifier_state.as_ref().unwrap();
        let K = eq_r_address.len();
        let (r_address_prime, r_cycle_prime) = r.split_at(K.log_2());
        let ra_claim = self.ra_claim.unwrap();

        eq_r_address.evaluate(r_address_prime)
            * eq_r_cycle.evaluate(r_cycle_prime)
            * (ra_claim.square() - ra_claim)
    }
}

/// Implements the sumcheck prover for the Booleanity check in step 3 of
/// Figure 6 in the Twist+Shout paper. The efficient implementation of this
/// sumcheck is described in Section 6.3.
#[tracing::instrument(skip_all, name = "Shout booleanity sumcheck")]
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

    for round in 0..K.log_2() {
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;
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
    let H: Vec<F> = read_addresses.par_iter().map(|&k| F[k]).collect();
    let mut H = MultilinearPolynomial::from(H);
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
                        let H_j = H.get_bound_coeff(j);
                        D_j * (H_j.square() - H_j)
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

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind_parallel(r_j, BindingOrder::LowToHigh),
            || H.bind_parallel(r_j, BindingOrder::LowToHigh),
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

        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || int.bind_parallel(r_j, BindingOrder::LowToHigh),
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
            lookup_table.clone(),
            read_addresses,
            &r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(NUM_LOOKUPS.log_2());
        let verification_result = proof.verify(lookup_table, &r_cycle, &mut verifier_transcript);
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
