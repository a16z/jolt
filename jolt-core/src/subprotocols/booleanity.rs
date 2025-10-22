use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::{JoltField, MulTrunc},
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
    zkvm::witness::CommittedPolynomial,
};

use crate::subprotocols::sumcheck::SumcheckInstance;

/// Common prover state for all booleanity sumchecks
#[derive(Allocative)]
pub struct BooleanityProverState<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    pub B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    pub D: GruenSplitEqPolynomial<F>,
    /// G[i]: pre-aggregated routing mass per address chunk i.
    pub G: Vec<Vec<F>>,
    /// H[i]: RaPolynomial
    pub H: Vec<RaPolynomial<u8, F>>,
    /// F: Expanding table
    pub F: Vec<F>,
    /// eq_r_r
    pub eq_r_r: F,
    /// Indices for H polynomials
    pub H_indices: Vec<Vec<Option<u8>>>,
}

pub trait BooleanityConfig {
    fn d(&self) -> usize;

    fn log_k_chunk(&self) -> usize;

    fn k_chunk(&self) -> usize {
        1 << self.log_k_chunk()
    }

    fn log_t(&self) -> usize;

    fn polynomial_type(i: usize) -> CommittedPolynomial;

    fn sumcheck_id() -> SumcheckId;
}

/// Extension trait for booleanity sumchecks that provides default implementations
pub trait BooleanitySumcheck<F: JoltField, T: Transcript>:
    SumcheckInstance<F, T> + BooleanityConfig + Send + Sync
{
    fn gamma(&self) -> &[F::Challenge];

    fn r_address(&self) -> &[F::Challenge];

    fn prover_state(&self) -> Option<&BooleanityProverState<F>>;

    fn prover_state_mut(&mut self) -> Option<&mut BooleanityProverState<F>>;

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge>;

    fn booleanity_degree(&self) -> usize {
        3
    }

    fn booleanity_num_rounds(&self) -> usize {
        self.log_k_chunk() + self.log_t()
    }

    fn booleanity_input_claim(&self, _acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        F::zero()
    }

    fn booleanity_compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < self.log_k_chunk() {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round, previous_claim)
        }
    }

    fn booleanity_bind(&mut self, r_j: F::Challenge, round: usize) {
        let log_k_chunk = self.log_k_chunk();
        let ps = self
            .prover_state_mut()
            .expect("Prover state not initialized");

        if round < log_k_chunk {
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
            if round == log_k_chunk - 1 {
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

    fn booleanity_expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let ra_claims = (0..self.d())
            .map(|i| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(Self::polynomial_type(i), Self::sumcheck_id())
                    .1
            })
            .collect::<Vec<F>>();

        let r_cycle = self.get_r_cycle(&*accumulator.borrow());

        let combined_r: Vec<F::Challenge> = self
            .r_address()
            .iter()
            .cloned()
            .rev()
            .chain(r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(r, &combined_r)
            * self
                .gamma()
                .iter()
                .zip(ra_claims)
                .map(|(gamma, ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
    }

    fn booleanity_normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point[..self.log_k_chunk()].reverse();
        opening_point[self.log_k_chunk()..].reverse();
        opening_point.into()
    }

    fn booleanity_cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state().expect("Prover state not initialized");

        let claims: Vec<F> = ps.H.iter().map(|H| H.final_sumcheck_claim()).collect();

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d()).map(Self::polynomial_type).collect(),
            Self::sumcheck_id(),
            opening_point.r[..self.log_k_chunk()].to_vec(),
            opening_point.r[self.log_k_chunk()..].to_vec(),
            claims,
        );
    }

    fn booleanity_cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d()).map(Self::polynomial_type).collect(),
            Self::sumcheck_id(),
            opening_point.r,
        );
    }

    // Phase 1 message computation
    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state().expect("Prover state not initialized");
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

                    let coeffs = (0..self.d())
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
                                self.gamma()[i] * F::from_barrett_reduce(inner_sum[0]),
                                self.gamma()[i] * F::from_barrett_reduce(inner_sum[1]),
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

                            let coeffs = (0..self.d())
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
                                        self.gamma()[i] * F::from_barrett_reduce(inner_sum[0]),
                                        self.gamma()[i] * F::from_barrett_reduce(inner_sum[1]),
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

    // Phase 2 message computation
    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state().expect("Prover state not initialized");
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
                            .zip(self.gamma().iter())
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
                                    .zip(self.gamma().iter())
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

    #[cfg(feature = "allocative")]
    fn booleanity_update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// Blanket implementation of SumcheckInstance for all types that implement BooleanitySumcheck
impl<F, T, B> SumcheckInstance<F, T> for B
where
    F: JoltField,
    T: Transcript,
    B: BooleanitySumcheck<F, T>,
{
    fn degree(&self) -> usize {
        self.booleanity_degree()
    }

    fn num_rounds(&self) -> usize {
        self.booleanity_num_rounds()
    }

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        self.booleanity_input_claim(acc)
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        self.booleanity_compute_prover_message(round, previous_claim)
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        self.booleanity_bind(r_j, round)
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        self.booleanity_expected_output_claim(accumulator, r)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.booleanity_normalize_opening_point(opening_point)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        self.booleanity_cache_openings_prover(accumulator, transcript, opening_point)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        self.booleanity_cache_openings_verifier(accumulator, transcript, opening_point)
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        self.booleanity_update_flamegraph(flamegraph)
    }
}
