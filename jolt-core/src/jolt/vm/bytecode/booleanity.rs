use std::cell::RefCell;
use std::rc::Rc;

use crate::dag::stage::StagedSumcheck;
use crate::dag::state_manager::StateManager;
use crate::jolt::vm::bytecode::BytecodePreprocessing;
use crate::poly::opening_proof::{
    OpeningPoint, OpeningsKeys, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::subprotocols::sumcheck::CacheSumcheckOpenings;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::BatchableSumcheckInstance,
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    G: Vec<F>,
    pc_by_cycle: Vec<usize>,
    H: Option<MultilinearPolynomial<F>>,
    F: Vec<F>,
    eq_r_r: Option<F>,
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
    unbound_ra_poly: Option<MultilinearPolynomial<F>>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    log_T: usize,
    log_K: usize,
    prover_state: Option<BooleanityProverState<F>>,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    ra_claim: Option<F>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        eq_r_cycle: Vec<F>,
        G: Vec<F>,
        unbound_ra_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let log_K = preprocessing.shared.bytecode.bytecode.len().log_2();

        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K);

        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();

        Self {
            prover_state: Some(BooleanityProverState::new(
                trace,
                &preprocessing.shared.bytecode,
                eq_r_cycle,
                G,
                unbound_ra_poly,
                &r_address,
            )),
            log_T: trace.len().log_2(),
            log_K,
            r_address,
            r_cycle,
            ra_claim: None,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let log_K = sm.get_bytecode().len().log_2();
        let r_address: Vec<F> = sm.transcript.borrow_mut().challenge_vector(log_K);
        let r_cycle = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput))
            .unwrap()
            .r
            .clone();
        let ra_claim = sm.get_opening(OpeningsKeys::BytecodeBooleanityRa);
        Self {
            prover_state: None,
            r_address,
            log_T: r_cycle.len(),
            r_cycle,
            ra_claim: Some(ra_claim),
            log_K,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        trace: &[RV32IMCycle],
        preprocessing: &BytecodePreprocessing,
        eq_r_cycle: Vec<F>,
        G: Vec<F>,
        unbound_ra_poly: MultilinearPolynomial<F>,
        r_address: &[F],
    ) -> Self {
        let log_K = r_address.len();
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r_address));

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F_vec[0] = F::one();

        let pc_by_cycle = trace
            .par_iter()
            .map(|cycle| preprocessing.get_pc(cycle))
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
            unbound_ra_poly: Some(unbound_ra_poly),
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        if round < self.log_K {
            // Phase 1: First log(K) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - self.log_K)
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_K {
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
            if round == self.log_K - 1 {
                ps.eq_r_r = Some(ps.B.final_sumcheck_claim());
                let H_vec: Vec<F> = std::mem::take(&mut ps.pc_by_cycle)
                    .into_par_iter()
                    .map(|pc| ps.F[pc])
                    .collect();
                ps.H = Some(MultilinearPolynomial::from(H_vec));
            }
        } else {
            // Phase 2: Bind D and H
            rayon::join(
                || ps.D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || {
                    ps.H.as_mut()
                        .unwrap()
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim = self.ra_claim.as_ref().expect("ra_claim not set");

        let eq_eval = EqPolynomial::mle(
            r,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(self.r_cycle.iter().cloned().rev())
                .collect::<Vec<F>>(),
        );

        eq_eval * (ra_claim.square() - ra_claim)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for BooleanitySumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        let ra_claim = ps.H.as_ref().unwrap().final_sumcheck_claim();

        let accumulator = accumulator.expect("accumulator is needed");
        accumulator.borrow_mut().append_sparse(
            vec![ps.unbound_ra_poly.take().unwrap()],
            opening_point.r[..self.log_K].to_vec(),
            opening_point.r[self.log_K..].to_vec(),
            vec![ra_claim],
            Some(vec![OpeningsKeys::BytecodeBooleanityRa]),
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::BytecodeBooleanityRa, opening_point);
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..prover_state.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals = prover_state
                    .B
                    .sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let inner_sum = prover_state.G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = prover_state.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;

                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][0] * F_k
                                    - prover_state.eq_km_c[k_m][0]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][1] * F_k
                                    - prover_state.eq_km_c[k_m][1]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][2] * F_k
                                    - prover_state.eq_km_c[k_m][2]),
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
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self, _round: usize) -> Vec<F> {
        let ps = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;

        let mut univariate_poly_evals: [F; 3] = (0..ps.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals =
                    ps.D.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H_evals =
                    ps.H.as_ref()
                        .unwrap()
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0].square() - H_evals[0]),
                    D_evals[1] * (H_evals[1].square() - H_evals[1]),
                    D_evals[2] * (H_evals[2].square() - H_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        for eval in &mut univariate_poly_evals {
            *eval *= ps.eq_r_r.unwrap();
        }

        univariate_poly_evals.to_vec()
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for BooleanitySumcheck<F>
{
}
