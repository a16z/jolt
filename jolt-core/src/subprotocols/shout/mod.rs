#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "prover")]
pub mod sparse_dense;
mod lookup_bits;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
pub use lookup_bits::*;

use strum::IntoEnumIterator;
#[cfg(feature = "prover")]
pub use prover::*;

use crate::{field::JoltField, into_optimal_iter, optimal_iter, optimal_iter_mut, poly::{
    multilinear_polynomial::{
        MultilinearPolynomial,
    },
}, utils::{
    errors::ProofVerifyError,
    math::Math,
    transcript::{Transcript},
}};
use crate::jolt::lookup_table::LookupTables;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof};

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

struct ShoutSumcheck<F: JoltField> {
    verifier_state: Option<ShoutVerifierState<F>>,
    prover_state: Option<ShoutProverState<F>>,
    claims: Option<ShoutSumcheckClaims<F>>,
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

        let degree = <ShoutSumcheck<F> as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: [F; 2] = into_optimal_iter!((0..ra.len() / 2))
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

struct BooleanitySumcheck<F: JoltField> {
    verifier_state: Option<BooleanityVerifierState<F>>,
    prover_state: Option<BooleanityProverState<F>>,
    ra_claim: Option<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> ShoutProof<F, ProofTranscript> {
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

pub fn verify_sparse_dense_shout<
    const WORD_SIZE: usize,
    F: JoltField,
    ProofTranscript: Transcript,
>(
    proof: &SumcheckInstanceProof<F, ProofTranscript>,
    log_T: usize,
    r_cycle: Vec<F>,
    rv_claim: F,
    ra_claims: [F; 4],
    flag_claims: &[F],
    transcript: &mut ProofTranscript,
) -> Result<(), ProofVerifyError> {
    let log_K = 2 * WORD_SIZE;
    let first_log_K_rounds = SumcheckInstanceProof::new(proof.compressed_polys[..log_K].to_vec());
    let last_log_T_rounds = SumcheckInstanceProof::new(proof.compressed_polys[log_K..].to_vec());
    // The first log(K) rounds' univariate polynomials are degree 2
    let (sumcheck_claim, r_address) = first_log_K_rounds.verify(rv_claim, log_K, 2, transcript)?;
    // The last log(T) rounds' univariate polynomials are degree 6
    let (sumcheck_claim, r_cycle_prime) =
        last_log_T_rounds.verify(sumcheck_claim, log_T, 6, transcript)?;

    let val_evals: Vec<_> = LookupTables::<WORD_SIZE>::iter()
        .map(|table| table.evaluate_mle(&r_address))
        .collect();
    let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(&r_cycle_prime);

    assert_eq!(
        eq_eval_cycle
            * ra_claims.iter().product::<F>()
            * flag_claims
            .iter()
            .zip(val_evals.iter())
            .map(|(flag, val)| *flag * val)
            .sum::<F>(),
        sumcheck_claim,
        "Read-checking sumcheck failed"
    );

    Ok(())
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

            let univariate_poly_evals: [F; DEGREE] = into_optimal_iter!((0..B.len() / 2))
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                    let inner_sum = optimal_iter!(G[k_prime << m..(k_prime + 1) << m])
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

            let mut univariate_poly_evals: [F; 3] = into_optimal_iter!((0..D.len() / 2))
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
            optimal_iter_mut!(F_left)
                .zip(optimal_iter_mut!(F_right))
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            if round == K.log_2() - 1 {
                // Transition point; initialize H
                *H = Some(MultilinearPolynomial::from(
                    optimal_iter!(read_addresses).map(|&k| F[k]).collect::<Vec<_>>(),
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
