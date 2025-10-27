use std::{cell::RefCell, rc::Rc, sync::Arc};

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use num_traits::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator,
            OpeningPoint,
            ProverOpeningAccumulator,
            SumcheckId,
            VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        bytecode::BytecodePreprocessing,
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

// Bytecode booleanity sumcheck
//
// Proves a zero-check of the form
//   0 = Σ_k Σ_j eq(r_address, k) · eq(r_cycle, j) · (Σ_{i=0}^{d-1} γ_i · (H_i(k, j)^2 − H_i(k, j)))
// where:
// - r_address are the address-chunk variables bound in phase 1
// - r_cycle are the time/cycle variables bound in phase 2
// - H_i is the routing/selection indicator for the i-th address chunk (boolean per point)

#[derive(Allocative)]
struct BooleanityProverState<F: JoltField> {
    /// B(k) := eq(r_address, k). Split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// D(j) := eq(r_cycle, j). Split-eq over time/cycle variables (phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// F_m[u] := eq(r_address[0..m-1], u) for u∈{0,1}^m; stored in first 2^m entries after m rounds.
    /// Eq-prefix weights reused to build H.
    F: Vec<F>,
    /// G_i[k] := Σ_j D(j) · 1[chunk_i(PC(j)) = k]. Pre-aggregated routing mass per address chunk i.
    G: Vec<Vec<F>>,
    /// pc_by_cycle[i][j] := chunk_i(PC(j)). Address-chunk index for chunk i at cycle j.
    pc_by_cycle: Vec<Vec<Option<u8>>>,
    /// H_i(k,j) := 1[chunk_i(PC(j)) = k] ∈ {0,1}. RaPolynomial routing indicator over chunk i.
    H: Vec<RaPolynomial<u8, F>>,
    /// eq(r_address, r_address′). Scalar after phase 1 collapse.
    eq_r_r: F,
}

#[derive(Allocative)]
pub struct BooleanitySumcheck<F: JoltField> {
    /// gamma: optimized batching challenges γ_i (length d).
    /// TODO: special casing for the first challenge to be F::one()
    gamma: Vec<F::Challenge>,
    /// d: number of address chunks in the decomposition.
    d: usize,
    /// log_T: number of time/cycle variables.
    log_T: usize,
    /// log_K_chunk: number of address-chunk variables per chunk.
    log_K_chunk: usize,
    /// prover_state: prover-side working state for both phases.
    prover_state: Option<BooleanityProverState<F>>,
    /// r_address: address binding point (for endianness and output claim).
    r_address: Vec<F::Challenge>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        r_cycle: Vec<F::Challenge>,
        G: Vec<Vec<F>>,
    ) -> Self {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let d = preprocessing.shared.bytecode.d;
        let log_K = preprocessing.shared.bytecode.bytecode.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);

        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(log_K_chunk);

        Self {
            gamma,
            prover_state: Some(BooleanityProverState::new(
                trace,
                &preprocessing.shared.bytecode,
                r_cycle,
                G,
                &r_address,
                d,
            )),
            d,
            log_T: trace.len().log_2(),
            log_K_chunk,
            r_address,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, _, T) = sm.get_verifier_data();
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_K = sm.get_bytecode().len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);
        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(log_K_chunk);
        Self {
            gamma,
            prover_state: None,
            log_T: T.log_2(),
            r_address,
            log_K_chunk,
            d,
        }
    }
}

impl<F: JoltField> BooleanityProverState<F> {
    fn new(
        trace: &[Cycle],
        preprocessing: &BytecodePreprocessing,
        r_cycle: Vec<F::Challenge>,
        G: Vec<Vec<F>>,
        r_address: &[F::Challenge],
        d: usize,
    ) -> Self {
        let log_K = preprocessing.code_size.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let B = GruenSplitEqPolynomial::new(r_address, BindingOrder::LowToHigh);

        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(log_K.pow2());
        F_vec[0] = F::one();

        let pc_by_cycle = (0..d)
            .into_par_iter()
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let k = preprocessing.get_pc(cycle);
                        Some(((k >> (log_K_chunk * (d - i - 1))) % K_chunk) as u8)
                    })
                    .collect()
            })
            .collect();
        let D = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);

        BooleanityProverState {
            B,
            D,
            H: vec![RaPolynomial::None; d],
            G,
            F: F_vec,
            eq_r_r: F::zero(),
            pc_by_cycle,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_K_chunk + self.log_T
    }

    fn input_claim(&self, _acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < self.log_K_chunk {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self.prover_state.as_mut().unwrap();

        if round < self.log_K_chunk {
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

            // Store eq_r_r at the end of phase 1
            if round == self.log_K_chunk - 1 {
                ps.eq_r_r = ps.B.get_current_scalar();
                // Initialize H polynomials using RaPolynomial
                let F = std::mem::take(&mut ps.F);
                let pc_by_cycle = std::mem::take(&mut ps.pc_by_cycle);
                ps.H = pc_by_cycle
                    .into_iter()
                    .map(|pc_indices| RaPolynomial::new(Arc::new(pc_indices), F.clone()))
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
        let (r_cycle, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        EqPolynomial::<F>::mle(
            r,
            &self
                .r_address
                .iter()
                .cloned()
                .rev()
                .chain(r_cycle.r.iter().cloned().rev())
                .collect::<Vec<F::Challenge>>(),
        ) * self
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
        opening_point[..self.log_K_chunk].reverse();
        opening_point[self.log_K_chunk..].reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();

        let claims: Vec<F> = ps.H.iter().map(|H| H.final_sumcheck_claim()).collect();

        accumulator.borrow_mut().append_sparse(
            transcript,
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
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
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
            // E_in has not been fully bound - use nested structure
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
        let p = self.prover_state.as_ref().unwrap();
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
