use std::cell::RefCell;
use std::rc::Rc;

use crate::poly::opening_proof::{
    OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::SumcheckInstance,
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: Vec<Vec<F>>,
    pc_by_cycle: Vec<Vec<usize>>,
    H: Option<Vec<MultilinearPolynomial<F>>>,
    F: Vec<F>,
    eq_r_r: Option<F>,
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

pub struct BooleanitySumcheck<F: JoltField> {
    gamma: Vec<F>,
    d: usize,
    log_T: usize,
    log_K_chunk: usize,
    prover_state: Option<BooleanityProverState<F>>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        eq_r_cycle: Vec<F>,
        G: Vec<Vec<F>>,
    ) -> Self {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let d = preprocessing.shared.bytecode.d;
        let log_K = preprocessing.shared.bytecode.bytecode.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K_chunk);

        let (r_cycle, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        Self {
            gamma: gamma_powers,
            prover_state: Some(BooleanityProverState::new(
                trace,
                &preprocessing.shared.bytecode,
                eq_r_cycle,
                G,
                &r_address,
                d,
            )),
            d,
            log_T: trace.len().log_2(),
            log_K_chunk,
            r_address,
            r_cycle: r_cycle.into(),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K_chunk);
        let (r_cycle, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        Self {
            gamma: gamma_powers,
            prover_state: None,
            r_address,
            log_T: r_cycle.len(),
            r_cycle: r_cycle.into(),
            log_K_chunk,
            d,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        trace: &[RV32IMCycle],
        preprocessing: &BytecodePreprocessing,
        eq_r_cycle: Vec<F>,
        G: Vec<Vec<F>>,
        r_address: &[F],
        d: usize,
    ) -> Self {
        let log_K = preprocessing.code_size.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_address));

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F_vec[0] = F::one();

        let pc_by_cycle = (0..d)
            .into_par_iter()
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let k = preprocessing.get_pc(cycle);
                        (k >> (log_K_chunk * (d - i - 1))) % K_chunk
                    })
                    .collect()
            })
            .collect();
        let D = MultilinearPolynomial::from(eq_r_cycle);

        // Precompute EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c: [[F; 3]; 2] = [
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

        // Precompute EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c_squared: [[F; 3]; 2] = [
            [F::one(), F::one(), F::from_u8(4)],
            [F::zero(), F::from_u8(4), F::from_u8(9)],
        ];
        BooleanityProverState {
            B,
            D,
            H: None,
            G,
            F: F_vec,
            eq_r_r: None,
            eq_km_c,
            eq_km_c_squared,
            pc_by_cycle,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_K_chunk + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        if round < self.log_K_chunk {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < self.log_K_chunk {
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

            // If transitioning to phase 2, prepare H
            if round == self.log_K_chunk - 1 {
                ps.H = Some(
                    ps.pc_by_cycle
                        .par_iter_mut()
                        .map(|pc_by_cycle| {
                            let coeffs: Vec<F> = std::mem::take(pc_by_cycle)
                                .into_par_iter()
                                .map(|j| ps.F[j])
                                .collect();
                            MultilinearPolynomial::from(coeffs)
                        })
                        .collect(),
                );
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
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let ra_claims = (0..self.d)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeBooleanity,
                    )
                    .1
            })
            .collect::<Vec<F>>();

        EqPolynomial::mle(
            r,
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
            .map(|(gamma, ra)| (ra.square() - ra) * gamma)
            .sum::<F>()
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point[..self.log_K_chunk].reverse();
        opening_point[self.log_K_chunk..].reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();

        let claims: Vec<F> =
            ps.H.as_ref()
                .unwrap()
                .iter()
                .map(|H| H.final_sumcheck_claim())
                .collect();

        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            opening_point.r[..self.log_K_chunk].to_vec(),
            opening_point.r[self.log_K_chunk..].to_vec(),
            claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            opening_point.r,
        );
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..p.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
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
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..p.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
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

                for i in 1..self.d {
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
