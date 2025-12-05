use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{iter::zip, sync::Arc};

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
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, thread::drop_in_background_thread},
    zkvm::witness::CommittedPolynomial,
};

/// Degree bound of the sumcheck round polynomials in [`BooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Number of address chunks
    pub d: usize,
    /// Log of chunk size
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Batching challenges
    pub gammas: Vec<F::Challenge>,
    /// Address binding point
    pub r_address: Vec<F::Challenge>,
    /// Cycle binding point
    pub r_cycle: Vec<F::Challenge>,
    /// Polynomial types for opening accumulator
    pub polynomial_types: Vec<CommittedPolynomial>,
    /// Sumcheck ID for opening accumulator
    pub sumcheck_id: SumcheckId,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = sumcheck_challenges.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }
}

/// Unified Booleanity Sumcheck implementation for RAM, Bytecode, and Instruction lookups
#[derive(Allocative)]
pub struct BooleanitySumcheckProver<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// G as in the Twist and Shout paper
    G: Vec<Vec<F>>,
    /// H as in the Twist and Shout paper
    H: Vec<RaPolynomial<u16, F>>,
    /// F: Expanding table
    F: ExpandingTable<F>,
    /// eq_r_r
    eq_r_r: F,
    /// Indices for H polynomials
    H_indices: Vec<Vec<Option<u16>>>,
    #[allocative(skip)]
    params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckProver<F> {
    pub fn gen(
        params: BooleanitySumcheckParams<F>,
        G: Vec<Vec<F>>,
        H_indices: Vec<Vec<Option<u16>>>,
    ) -> Self {
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let D_poly = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);

        let k_chunk = 1 << params.log_k_chunk;
        let mut F = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F.reset(F::one());

        Self {
            B,
            D: D_poly,
            G,
            H_indices,
            H: vec![],
            F,
            eq_r_r: F::zero(),
            params,
        }
    }

    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let B = &self.B;

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|k_prime| {
                let coeffs = (0..self.params.d)
                    .into_par_iter()
                    .map(|i| {
                        let G_i = &self.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = self.F[k % (1 << (m - 1))];
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
                                [F::Unreduced::<5>::zero(); DEGREE_BOUND - 1],
                                |running, new| {
                                    [
                                        running[0] + new[0].as_unreduced_ref(),
                                        running[1] + new[1].as_unreduced_ref(),
                                    ]
                                },
                            )
                            .reduce(
                                || [F::Unreduced::zero(); DEGREE_BOUND - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        [
                            self.params.gammas[i] * F::from_barrett_reduce(inner_sum[0]),
                            self.params.gammas[i] * F::from_barrett_reduce(inner_sum[1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                coeffs
            });

        // Use Gruen optimization to get cubic evaluations from quadratic coefficients
        B.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let D_poly = &self.D;

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        let quadratic_coeffs_f: [F; DEGREE_BOUND - 1] = D_poly
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                // Accumulate in unreduced form to minimize per-term reductions
                let mut acc_c = F::Unreduced::<9>::zero();
                let mut acc_e = F::Unreduced::<9>::zero();
                for (h, gamma) in zip(&self.H, &self.params.gammas) {
                    let h_0 = h.get_bound_coeff(2 * j_prime);
                    let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                    let b = h_1 - h_0;

                    // Compute gamma * h0, then a single unreduced multiply by (h0 - 1)
                    let g_h0 = *gamma * h_0;
                    let h0_minus_one = h_0 - F::one();
                    let c_unr = g_h0.mul_unreduced::<9>(h0_minus_one);
                    acc_c += c_unr;

                    // Compute gamma * b, then a single unreduced multiply by b
                    let g_b = *gamma * b;
                    let e_unr = g_b.mul_unreduced::<9>(b);
                    acc_e += e_unr;
                }
                [
                    F::from_montgomery_reduce::<9>(acc_c),
                    F::from_montgomery_reduce::<9>(acc_e),
                ]
            });

        // previous_claim is s(0)+s(1) of the scaled polynomial; divide out eq_r_r to get inner claim
        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            D_poly.gruen_poly_deg_3(quadratic_coeffs_f[0], quadratic_coeffs_f[1], adjusted_claim);

        gruen_poly * self.eq_r_r
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BooleanitySumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::compute_message", fields(variant = ?self.params.sumcheck_id))]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_k_chunk {
            // Phase 1: First log(K_chunk) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::ingest_challenge", fields(variant = ?self.params.sumcheck_id))]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.log_k_chunk {
            // Phase 1: Bind B and update F
            self.B.bind(r_j);

            // Update F for this round
            self.F.update(r_j);

            // If transitioning to phase 2, prepare H polynomials
            if round == self.params.log_k_chunk - 1 {
                self.eq_r_r = self.B.get_current_scalar();

                // Initialize H polynomials using RaPolynomial
                let F = std::mem::take(&mut self.F);
                let H_indices = std::mem::take(&mut self.H_indices);
                self.H = H_indices
                    .into_iter()
                    .map(|indices| RaPolynomial::new(Arc::new(indices), F.clone_values()))
                    .collect();

                // Drop G arrays as they're no longer needed
                let g = std::mem::take(&mut self.G);
                drop_in_background_thread(g);
            }
        } else {
            // Phase 2: Bind D and H
            self.D.bind(r_j);
            self.H
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let claims: Vec<F> = self.H.iter().map(|H| H.final_sumcheck_claim()).collect();
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            opening_point.r[..self.params.log_k_chunk].to_vec(),
            opening_point.r[self.params.log_k_chunk..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct BooleanitySumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BooleanitySumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims = self
            .params
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .get_committed_polynomial_opening(*poly_type, self.params.sumcheck_id)
                    .1
            })
            .collect::<Vec<F>>();

        let combined_r: Vec<F::Challenge> = self
            .params
            .r_address
            .iter()
            .cloned()
            .rev()
            .chain(self.params.r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.gammas, ra_claims)
                .map(|(gamma, ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            self.params.sumcheck_id,
            opening_point.r,
        );
    }
}
