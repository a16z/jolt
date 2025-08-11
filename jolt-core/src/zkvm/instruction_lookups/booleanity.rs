use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};
use tracer::instruction::RV32IMCycle;

use super::{D, K_CHUNK, LOG_K_CHUNK};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcript::Transcript,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::dag::state_manager::StateManager,
    zkvm::{
        instruction::LookupQuery,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

const DEGREE: usize = 3;

struct BooleanityProverState<F: JoltField> {
    B: GruenSplitEqPolynomial<F>,
    D: GruenSplitEqPolynomial<F>,
    G: [Vec<F>; D],
    H_indices: [Vec<usize>; D],
    H: [MultilinearPolynomial<F>; D],
    F: Vec<F>,
    eq_r_r: F,
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
            prover_state: Some(BooleanityProverState::new(trace, G, &r_address, &r_cycle)),
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
    fn new(trace: &[RV32IMCycle], G: [Vec<F>; D], r_address: &[F], r_cycle: &[F]) -> Self {
        let B = GruenSplitEqPolynomial::new(r_address, BindingOrder::LowToHigh);

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

        BooleanityProverState {
            B,
            D: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            G,
            H_indices,
            H: std::array::from_fn(|_| MultilinearPolynomial::from(vec![F::zero()])),
            F,
            eq_r_r: F::zero(),
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
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < LOG_K_CHUNK {
            // Phase 1: First log(K_CHUNK) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - LOG_K_CHUNK, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < LOG_K_CHUNK {
            // Phase 1: Bind B and update F
            ps.B.bind(r_j);
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
                let mut h_indices = std::mem::take(&mut ps.H_indices);
                let f_ref = &ps.F;
                ps.H = std::array::from_fn(|i| {
                    let coeffs: Vec<F> = std::mem::take(&mut h_indices[i])
                        .into_par_iter()
                        .map(|j| f_ref[j])
                        .collect();
                    MultilinearPolynomial::from(coeffs)
                });
                ps.eq_r_r = ps.B.current_scalar;

                // Drop G arrays, F array, and remaining H_indices as they're no longer needed in phase 2
                // Replace G with empty vectors
                let g: [Vec<F>; D] = std::array::from_fn(|i| std::mem::take(&mut ps.G[i]));
                drop_in_background_thread(g);

                let f = std::mem::take(&mut ps.F);
                drop_in_background_thread(f);

                drop_in_background_thread(h_indices);
            }
        } else {
            // Phase 2: Bind D and H
            ps.D.bind(r_j);
            ps.H.par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r_prime: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let ra_claims = (0..D).map(|i| {
            accumulator
                .borrow()
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
            ps.H.iter()
                .map(|ra| ra.final_sumcheck_claim())
                .collect::<Vec<F>>();

        accumulator.borrow_mut().append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
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
        accumulator.borrow_mut().append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionBooleanity,
            r_sumcheck.r,
        );
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        let B = &p.B;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        // Compute quadratic coefficients to interpolate for Gruen
        let quadratic_coeffs: [F; DEGREE - 1] = if B.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_eval = B.E_out_current()[k_prime];

                    let inner_sum = (0..1 << m)
                        .into_par_iter()
                        .map(|k| {
                            let k_m = k >> (m - 1);
                            let F_k = p.F[k % (1 << (m - 1))];
                            let k_G = (k_prime << m) + k;
                            let G_ref = &p.G;
                            let G_times_F = G_ref
                                .iter()
                                .zip(self.gamma.iter())
                                .map(|(g, gamma)| g[k_G] * gamma)
                                .sum::<F>()
                                * F_k;

                            // For c \in {0, infty} compute:
                            // G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                            let eval_infty = G_times_F * F_k;
                            let eval_0 = if k_m == 0 {
                                eval_infty - G_times_F
                            } else {
                                F::zero()
                            };
                            [eval_0, eval_infty]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [B_eval * inner_sum[0], B_eval * inner_sum[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound
            let num_x_in_bits = B.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..B.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let B_E_out_eval = B.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|k_prime| {
                            let x_in = k_prime & x_bitmask;
                            let B_E_in_eval = B.E_in_current()[x_in];

                            let inner_sum = (0..1 << m)
                                .into_par_iter()
                                .map(|k| {
                                    let k_m = k >> (m - 1);
                                    let F_k = p.F[k % (1 << (m - 1))];
                                    let k_G = (k_prime << m) + k;
                                    let G_ref = &p.G;
                                    let G_times_F = G_ref
                                        .iter()
                                        .zip(self.gamma.iter())
                                        .map(|(g, gamma)| g[k_G] * gamma)
                                        .sum::<F>()
                                        * F_k;

                                    let eval_infty = G_times_F * F_k;
                                    let eval_0 = if k_m == 0 {
                                        eval_infty - G_times_F
                                    } else {
                                        F::zero()
                                    };
                                    [eval_0, eval_infty]
                                })
                                .reduce(
                                    || [F::zero(); DEGREE - 1],
                                    |running, new| [running[0] + new[0], running[1] + new[1]],
                                );

                            [B_E_in_eval * inner_sum[0], B_E_in_eval * inner_sum[1]]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [B_E_out_eval * chunk_evals[0], B_E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Use Gruen optimization to get cubic evaluations from quadratic coefficients
        B.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let D_poly = &p.D;
        let H = &p.H;

        let quadratic_coeffs: [F; DEGREE - 1] = if D_poly.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..D_poly.len() / 2)
                .into_par_iter()
                .map(|j_prime| {
                    let D_eval = D_poly.E_out_current()[j_prime];
                    let mut coeffs = [F::zero(); DEGREE - 1];

                    for i in 0..D {
                        let h_poly = &H[i];

                        let h_0 = h_poly.get_bound_coeff(2 * j_prime); // h(0)
                        let h_1 = h_poly.get_bound_coeff(2 * j_prime + 1); // h(1)

                        // For c = 0: h(0)^2 - h(0)
                        coeffs[0] += self.gamma[i] * (h_0.square() - h_0);

                        // For quadratic coefficient: b^2 where b = h(1) - h(0) is the linear coefficient
                        let b = h_1 - h_0; // Linear coefficient of h
                        coeffs[1] += self.gamma[i] * b.square(); // Quadratic coefficient of h^2 - h
                    }

                    [D_eval * coeffs[0], D_eval * coeffs[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound - use nested structure
            let num_x_in_bits = D_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..D_poly.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let D_E_out_eval = D_poly.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|j_prime| {
                            let x_in = j_prime & x_bitmask;
                            let D_E_in_eval = D_poly.E_in_current()[x_in];
                            let mut coeffs = [F::zero(); DEGREE - 1];

                            for i in 0..D {
                                let h_poly = &H[i];

                                let h_0 = h_poly.get_bound_coeff(2 * j_prime); // h(0)
                                let h_1 = h_poly.get_bound_coeff(2 * j_prime + 1); // h(1)

                                // For c = 0: h(0)^2 - h(0)
                                coeffs[0] += self.gamma[i] * (h_0.square() - h_0);

                                // For quadratic coefficient: b^2 where b = h(1) - h(0) is the linear coefficient
                                let b = h_1 - h_0; // Linear coefficient of h
                                coeffs[1] += self.gamma[i] * b.square(); // Quadratic coefficient of h^2 - h
                            }

                            [D_E_in_eval * coeffs[0], D_E_in_eval * coeffs[1]]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [D_E_out_eval * chunk_evals[0], D_E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Adjust the previous claim by dividing out eq_r_r
        let adjusted_claim = previous_claim / p.eq_r_r;

        let gruen_evals =
            D_poly.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);
        vec![
            p.eq_r_r * gruen_evals[0],
            p.eq_r_r * gruen_evals[1],
            p.eq_r_r * gruen_evals[2],
        ]
    }
}
