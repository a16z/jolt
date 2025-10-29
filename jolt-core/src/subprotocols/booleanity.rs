use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
    },
    transcripts::Transcript,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

use crate::subprotocols::sumcheck::SumcheckInstance;

/// Common prover state for all booleanity sumchecks
#[derive(Allocative)]
pub struct BooleanityProverState<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    pub B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    pub D: GruenSplitEqPolynomial<F>,
    /// G as in the Twist and Shout paper
    pub G: Vec<Vec<F>>,
    /// H as in the Twist and Shout paper
    pub H: Vec<RaPolynomial<u8, F>>,
    /// F: Expanding table
    pub F: Vec<F>,
    /// eq_r_r
    pub eq_r_r: F,
    /// Indices for H polynomials
    pub H_indices: Vec<Vec<Option<u8>>>,
}

/// Unified Booleanity Sumcheck implementation for RAM, Bytecode, and Instruction lookups
#[derive(Allocative)]
pub struct BooleanitySumcheck<F: JoltField> {
    /// Number of address chunks
    d: usize,
    /// Log of chunk size
    log_k_chunk: usize,
    /// Log of trace length
    log_t: usize,
    /// Batching challenges
    gamma: Vec<F::Challenge>,
    /// Address binding point
    r_address: Vec<F::Challenge>,
    /// Cycle binding point
    r_cycle: Vec<F::Challenge>,
    /// Polynomial types for opening accumulator
    polynomial_types: Vec<CommittedPolynomial>,
    /// Sumcheck ID for opening accumulator
    sumcheck_id: SumcheckId,
    /// Optional virtual polynomial for r_cycle
    virtual_poly: Option<VirtualPolynomial>,
    /// Prover state (only for prover)
    prover_state: Option<BooleanityProverState<F>>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_prover(
        d: usize,
        log_k_chunk: usize,
        log_t: usize,
        r_cycle: Vec<F::Challenge>,
        r_address: Vec<F::Challenge>,
        gamma: Vec<F::Challenge>,
        G: Vec<Vec<F>>,
        H_indices: Vec<Vec<Option<u8>>>,
        polynomial_types: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        virtual_poly: Option<VirtualPolynomial>,
    ) -> Self {
        let B = GruenSplitEqPolynomial::new(&r_address, BindingOrder::LowToHigh);
        let D_poly = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);

        let k_chunk = 1 << log_k_chunk;
        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(k_chunk);
        F_vec[0] = F::one();

        let prover_state = BooleanityProverState {
            B,
            D: D_poly,
            G,
            H_indices,
            H: vec![],
            F: F_vec,
            eq_r_r: F::zero(),
        };

        Self {
            d,
            log_k_chunk,
            log_t,
            gamma,
            r_address,
            r_cycle,
            polynomial_types,
            sumcheck_id,
            virtual_poly,
            prover_state: Some(prover_state),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_verifier(
        d: usize,
        log_k_chunk: usize,
        log_t: usize,
        r_cycle: Vec<F::Challenge>,
        r_address: Vec<F::Challenge>,
        gamma: Vec<F::Challenge>,
        polynomial_types: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        virtual_poly: Option<VirtualPolynomial>,
    ) -> Self {
        Self {
            d,
            log_k_chunk,
            log_t,
            gamma,
            r_address,
            r_cycle,
            polynomial_types,
            sumcheck_id,
            virtual_poly,
            prover_state: None,
        }
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        if !self.r_cycle.is_empty() {
            self.r_cycle.clone()
        } else {
            let virtual_poly = self
                .virtual_poly
                .expect("virtual_poly must be set when r_cycle is empty");
            accumulator
                .get_virtual_polynomial_opening(virtual_poly, SumcheckId::SpartanOuter)
                .0
                .r
                .clone()
        }
    }

    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> Vec<F> {
        let p = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let m = round + 1;
        const DEGREE: usize = 3;

        let B = &p.B;

        // Compute quadratic coefficients to interpolate for Gruen
        let quadratic_coeffs: [F; DEGREE - 1] = if B.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_eval = B.E_out_current()[k_prime];

                    let coeffs = (0..self.d)
                        .into_par_iter()
                        .map(|i| {
                            let G_i = &p.G[i];
                            let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                                .par_iter()
                                .enumerate()
                                .map(|(k, &G_k)| {
                                    let k_m = k >> (m - 1);
                                    let F_k = p.F[k % (1 << (m - 1))];
                                    let G_times_F = G_k * F_k;

                                    // For c in {0, infty}:
                                    // G[k] * (F(..., c)^2 - F(..., c))
                                    let eval_infty = G_times_F * F_k;
                                    let eval_0 = if k_m == 0 {
                                        eval_infty - G_times_F
                                    } else {
                                        F::zero()
                                    };
                                    [eval_0, eval_infty]
                                })
                                .fold_with(
                                    [F::Unreduced::<5>::zero(); DEGREE - 1],
                                    |running, new| {
                                        [
                                            running[0] + new[0].as_unreduced_ref(),
                                            running[1] + new[1].as_unreduced_ref(),
                                        ]
                                    },
                                )
                                .reduce(
                                    || [F::Unreduced::zero(); DEGREE - 1],
                                    |running, new| [running[0] + new[0], running[1] + new[1]],
                                );

                            [
                                self.gamma[i] * F::from_barrett_reduce(inner_sum[0]),
                                self.gamma[i] * F::from_barrett_reduce(inner_sum[1]),
                            ]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        B_eval.mul_unreduced::<9>(coeffs[0]),
                        B_eval.mul_unreduced::<9>(coeffs[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
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

                            let coeffs = (0..self.d)
                                .into_par_iter()
                                .map(|i| {
                                    let G_i = &p.G[i];
                                    let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                                        .par_iter()
                                        .enumerate()
                                        .map(|(k, &G_k)| {
                                            let k_m = k >> (m - 1);
                                            let F_k = p.F[k % (1 << (m - 1))];
                                            let G_times_F = G_k * F_k;

                                            let eval_infty = G_times_F * F_k;
                                            let eval_0 = if k_m == 0 {
                                                eval_infty - G_times_F
                                            } else {
                                                F::zero()
                                            };
                                            [eval_0, eval_infty]
                                        })
                                        .fold_with(
                                            [F::Unreduced::<5>::zero(); DEGREE - 1],
                                            |running, new| {
                                                [
                                                    running[0] + new[0].as_unreduced_ref(),
                                                    running[1] + new[1].as_unreduced_ref(),
                                                ]
                                            },
                                        )
                                        .reduce(
                                            || [F::Unreduced::zero(); DEGREE - 1],
                                            |running, new| {
                                                [running[0] + new[0], running[1] + new[1]]
                                            },
                                        );

                                    [
                                        self.gamma[i] * F::from_barrett_reduce(inner_sum[0]),
                                        self.gamma[i] * F::from_barrett_reduce(inner_sum[1]),
                                    ]
                                })
                                .reduce(
                                    || [F::zero(); DEGREE - 1],
                                    |running, new| [running[0] + new[0], running[1] + new[1]],
                                );

                            [
                                B_E_in_eval.mul_unreduced::<9>(coeffs[0]),
                                B_E_in_eval.mul_unreduced::<9>(coeffs[1]),
                            ]
                        })
                        .reduce(
                            || [F::Unreduced::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        B_E_out_eval.mul_unreduced::<9>(F::from_montgomery_reduce(chunk_evals[0])),
                        B_E_out_eval.mul_unreduced::<9>(F::from_montgomery_reduce(chunk_evals[1])),
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };

        // Use Gruen optimization to get cubic evaluations from quadratic coefficients
        B.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> Vec<F> {
        let p = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 3;

        let D_poly = &p.D;

        let quadratic_coeffs = if D_poly.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..D_poly.len() / 2)
                .into_par_iter()
                .map(|j_prime| {
                    let D_eval = D_poly.E_out_current()[j_prime];
                    let coeffs =
                        p.H.iter()
                            .zip(self.gamma.iter())
                            .map(|(h, gamma)| {
                                let h_0 = h.get_bound_coeff(2 * j_prime);
                                let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                                let b = h_1 - h_0;
                                [(h_0.square() - h_0) * *gamma, b.square() * *gamma]
                            })
                            .fold([F::zero(); 2], |running, new| {
                                [running[0] + new[0], running[1] + new[1]]
                            });

                    [
                        D_eval.mul_unreduced::<9>(coeffs[0]),
                        D_eval.mul_unreduced::<9>(coeffs[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound
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
                            let coeffs =
                                p.H.iter()
                                    .zip(self.gamma.iter())
                                    .map(|(h, gamma)| {
                                        let h_0 = h.get_bound_coeff(2 * j_prime);
                                        let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                                        let b = h_1 - h_0;
                                        [(h_0.square() - h_0) * *gamma, b.square() * *gamma]
                                    })
                                    .fold([F::zero(); 2], |running, new| {
                                        [running[0] + new[0], running[1] + new[1]]
                                    });

                            [
                                D_E_in_eval.mul_unreduced::<9>(coeffs[0]),
                                D_E_in_eval.mul_unreduced::<9>(coeffs[1]),
                            ]
                        })
                        .reduce(
                            || [F::Unreduced::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        D_E_out_eval.mul_unreduced::<9>(F::from_montgomery_reduce(chunk_evals[0])),
                        D_E_out_eval.mul_unreduced::<9>(F::from_montgomery_reduce(chunk_evals[1])),
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Convert to field elements
        let quadratic_coeffs_f: [F; DEGREE - 1] = [
            F::from_montgomery_reduce(quadratic_coeffs[0]),
            F::from_montgomery_reduce(quadratic_coeffs[1]),
        ];

        // previous_claim is s(0)+s(1) of the scaled polynomial; divide out eq_r_r to get inner claim
        let adjusted_claim = previous_claim * p.eq_r_r.inverse().unwrap();
        let gruen_evals =
            D_poly.gruen_evals_deg_3(quadratic_coeffs_f[0], quadratic_coeffs_f[1], adjusted_claim);
        vec![
            p.eq_r_r * gruen_evals[0],
            p.eq_r_r * gruen_evals[1],
            p.eq_r_r * gruen_evals[2],
        ]
    }
}

impl<F, T> SumcheckInstance<F, T> for BooleanitySumcheck<F>
where
    F: JoltField,
    T: Transcript,
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, _acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < self.log_k_chunk {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_k_chunk {
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

            // If transitioning to phase 2, prepare H polynomials
            if round == self.log_k_chunk - 1 {
                ps.eq_r_r = ps.B.get_current_scalar();

                // Initialize H polynomials using RaPolynomial
                let F = std::mem::take(&mut ps.F);
                let H_indices = std::mem::take(&mut ps.H_indices);
                ps.H = H_indices
                    .into_iter()
                    .map(|indices| RaPolynomial::new(Arc::new(indices), F.clone()))
                    .collect();

                // Drop G arrays as they're no longer needed
                let g = std::mem::take(&mut ps.G);
                drop_in_background_thread(g);
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
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let ra_claims = self
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(*poly_type, self.sumcheck_id)
                    .1
            })
            .collect::<Vec<F>>();

        let r_cycle = self.get_r_cycle(&*accumulator.borrow());

        let combined_r: Vec<F::Challenge> = self
            .r_address
            .iter()
            .cloned()
            .rev()
            .chain(r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(r, &combined_r)
            * self
                .gamma
                .iter()
                .zip(ra_claims)
                .map(|(gamma, ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claims: Vec<F> = ps.H.iter().map(|H| H.final_sumcheck_claim()).collect();

        accumulator.borrow_mut().append_sparse(
            transcript,
            self.polynomial_types.clone(),
            self.sumcheck_id,
            opening_point.r[..self.log_k_chunk].to_vec(),
            opening_point.r[self.log_k_chunk..].to_vec(),
            claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_sparse(
            transcript,
            self.polynomial_types.clone(),
            self.sumcheck_id,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
