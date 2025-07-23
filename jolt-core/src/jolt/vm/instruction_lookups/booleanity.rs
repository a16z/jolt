use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};
use tracer::instruction::RV32IMCycle;

use super::{D, K_CHUNK, LOG_K_CHUNK};

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::{
        instruction::LookupQuery,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};

const DEGREE: usize = 3;

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: [Vec<F>; D],
    H_indices: [Vec<usize>; D],
    H: Option<[MultilinearPolynomial<F>; D]>,
    F: Vec<F>,
    eq_r_r: Option<F>,
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Precomputed powers of gamma - batching chgallenge
    gamma: [F; D],
    prover_state: Option<BooleanityProverState<F>>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    log_T: usize,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        eq_r_cycle: Vec<F>,
        G: [Vec<F>; D],
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(LOG_K_CHUNK);
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let trace = sm.get_prover_data().1;

        Self {
            gamma: gamma_powers,
            prover_state: Some(BooleanityProverState::new(trace, eq_r_cycle, G, &r_address)),
            r_address,
            r_cycle,
            log_T: trace.len().log_2(),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let log_T = r_cycle.len();
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(LOG_K_CHUNK);
        Self {
            gamma: gamma_powers,
            prover_state: None,
            r_address,
            r_cycle,
            log_T,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(trace: &[RV32IMCycle], eq_r_cycle: Vec<F>, G: [Vec<F>; D], r_address: &[F]) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_address));
        let mut F: Vec<F> = unsafe_allocate_zero_vec(K_CHUNK);
        F[0] = F::one();

        let H_indices: [Vec<usize>; D] = std::array::from_fn(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<32>::to_lookup_index(cycle);
                    ((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u64) as usize
                })
                .collect()
        });

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
        BooleanityProverState {
            B,
            D: MultilinearPolynomial::from(eq_r_cycle),
            G,
            H_indices,
            H: None,
            F,
            eq_r_r: None,
            eq_km_c,
            eq_km_c_squared,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(
        skip_all,
        name = "InstructionBooleanitySumcheck::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        if round < LOG_K_CHUNK {
            // Phase 1: First log(K_CHUNK) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < LOG_K_CHUNK {
            // Phase 1: Bind B and update F
            ps.B.bind_parallel(r_j, BindingOrder::LowToHigh);
            // Update F for this round (see Equation 55)
            let (F_left, F_right) = ps.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });
            if round == LOG_K_CHUNK - 1 {
                ps.H = Some(std::array::from_fn(|i| {
                    let coeffs: Vec<F> = std::mem::take(&mut ps.H_indices[i])
                        .into_par_iter()
                        .map(|j| ps.F[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                }));
                ps.eq_r_r = Some(ps.B.final_sumcheck_claim());
            }
        } else {
            // Phase 2: Bind D and H
            ps.H.as_mut()
                .unwrap()
                .into_par_iter()
                .chain(rayon::iter::once(&mut ps.D))
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_prime: &[F],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            let accumulator = accumulator.as_ref().unwrap();
            let accumulator = accumulator.borrow();
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionBooleanity,
                )
                .1
        });
        EqPolynomial::mle(
            r_prime,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.r_cycle.iter().cloned().rev())
                .collect::<Vec<F>>(),
        ) * self
            .gamma
            .iter()
            .zip(ra_claims)
            .fold(F::zero(), |acc, (gamma, ra)| {
                (ra.square() - ra) * gamma + acc
            })
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(LOG_K_CHUNK);
        let mut r_big_endian: Vec<F> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();
        let ra_claims =
            ps.H.as_ref()
                .unwrap()
                .iter()
                .map(|ra| ra.final_sumcheck_claim())
                .collect::<Vec<F>>();
        let polynomials = (0..D).map(CommittedPolynomial::InstructionRa).collect();

        accumulator.borrow_mut().append_sparse(
            polynomials,
            SumcheckId::InstructionBooleanity,
            opening_point.r[..LOG_K_CHUNK].to_vec(),
            opening_point.r[LOG_K_CHUNK..].to_vec(),
            ra_claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let polynomials = (0..D).map(CommittedPolynomial::InstructionRa).collect();

        accumulator.borrow_mut().append_sparse(
            polynomials,
            SumcheckId::InstructionBooleanity,
            r_sumcheck.r,
        );
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..p.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals =
                    p.B.sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = (0..1 << m)
                    .into_par_iter()
                    .map(|k| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = p.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let k_G = (k_prime << m) + k;
                        let G_times_F =
                            p.G.iter()
                                .zip(self.gamma.iter())
                                .map(|(g, gamma)| g[k_G] * gamma)
                                .sum::<F>()
                                * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (p.eq_km_c_squared[k_m][0] * F_k - p.eq_km_c[k_m][0]),
                            G_times_F * (p.eq_km_c_squared[k_m][1] * F_k - p.eq_km_c[k_m][1]),
                            G_times_F * (p.eq_km_c_squared[k_m][2] * F_k - p.eq_km_c[k_m][2]),
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

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let univariate_poly_evals: [F; 3] = (0..p.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals =
                    p.D.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H = p.H.as_ref().unwrap();
                let H_evals = H
                    .iter()
                    .map(|h| h.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh))
                    .collect::<Vec<_>>();

                let mut evals = [
                    H_evals[0][0].square() - H_evals[0][0],
                    H_evals[0][1].square() - H_evals[0][1],
                    H_evals[0][2].square() - H_evals[0][2],
                ];

                for i in 1..D {
                    evals[0] += self.gamma[i] * (H_evals[i][0].square() - H_evals[i][0]);
                    evals[1] += self.gamma[i] * (H_evals[i][1].square() - H_evals[i][1]);
                    evals[2] += self.gamma[i] * (H_evals[i][2].square() - H_evals[i][2]);
                }

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

        vec![
            p.eq_r_r.unwrap() * univariate_poly_evals[0],
            p.eq_r_r.unwrap() * univariate_poly_evals[1],
            p.eq_r_r.unwrap() * univariate_poly_evals[2],
        ]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: [F; D],
}
