use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
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
    zkvm::{
        dag::state_manager::StateManager,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
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

/// Type of booleanity sumcheck with associated parameters
#[derive(Clone, Debug, Allocative)]
pub enum BooleanityType {
    Ram { K: usize },
    Bytecode { d: usize, log_K: usize },
    Instruction,
}

/// Unified Booleanity Sumcheck implementation for RAM, Bytecode, and Instruction lookups
#[derive(Allocative)]
pub struct BooleanitySumcheck<F: JoltField> {
    /// Type of booleanity sumcheck
    booleanity_type: BooleanityType,
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
    /// Prover state (only for prover)
    prover_state: Option<BooleanityProverState<F>>,
    /// Optional virtual polynomial for r_cycle (for Bytecode)
    virtual_poly: Option<VirtualPolynomial>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    /// Create a new prover instance
    pub fn new_prover<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        booleanity_type: BooleanityType,
        state_manager: &mut StateManager<'_, F, T, PCS>,
        G: Option<Vec<Vec<F>>>,
        H_indices: Option<Vec<Vec<Option<u8>>>>,
    ) -> Self {
        let (d, log_k_chunk, log_t, r_cycle, r_address, gamma, virtual_poly, G, H_indices) =
            match &booleanity_type {
                BooleanityType::Ram { K } => {
                    use crate::poly::eq_poly::EqPolynomial;
                    use crate::zkvm::witness::{compute_d_parameter, DTH_ROOT_OF_K};

                    let (_, trace, program_io, _) = state_manager.get_prover_data();
                    let memory_layout = &program_io.memory_layout;
                    let d = compute_d_parameter(*K);
                    let log_k_chunk = DTH_ROOT_OF_K.log_2();
                    let log_t = trace.len().log_2();

                    let r_cycle: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_t);

                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_k_chunk);

                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(d);

                    // Compute G and H if not provided (for RAM, they depend on r_cycle)
                    let G = if let Some(g) = G {
                        g
                    } else {
                        let eq_r_cycle = EqPolynomial::<F>::evals(&r_cycle);
                        compute_ram_g_arrays_internal(trace, memory_layout, &eq_r_cycle, d)
                    };

                    let H_indices = if let Some(h) = H_indices {
                        h
                    } else {
                        compute_ram_h_indices_internal(trace, memory_layout, d)
                    };

                    (
                        d,
                        log_k_chunk,
                        log_t,
                        r_cycle,
                        r_address,
                        gamma,
                        None,
                        G,
                        H_indices,
                    )
                }
                BooleanityType::Bytecode { d, log_K } => {
                    let (_, trace, _, _) = state_manager.get_prover_data();
                    let log_k_chunk = log_K.div_ceil(*d);
                    let log_t = trace.len().log_2();

                    let r_cycle = state_manager
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::UnexpandedPC,
                            SumcheckId::SpartanOuter,
                        )
                        .0
                        .r
                        .clone();
                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(*d);

                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_k_chunk);

                    let virtual_poly = Some(VirtualPolynomial::UnexpandedPC);

                    (
                        *d,
                        log_k_chunk,
                        log_t,
                        r_cycle,
                        r_address,
                        gamma,
                        virtual_poly,
                        G.expect("G arrays must be provided for Bytecode"),
                        H_indices.expect("H_indices must be provided for Bytecode"),
                    )
                }
                BooleanityType::Instruction => {
                    let (_, trace, _, _) = state_manager.get_prover_data();
                    const D: usize = 16;
                    const LOG_K_CHUNK: usize = 8;
                    let log_t = trace.len().log_2();

                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(D);

                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(LOG_K_CHUNK);

                    let r_cycle = state_manager
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::LookupOutput,
                            SumcheckId::SpartanOuter,
                        )
                        .0
                        .r
                        .clone();

                    let virtual_poly = Some(VirtualPolynomial::LookupOutput);

                    (
                        D,
                        LOG_K_CHUNK,
                        log_t,
                        r_cycle,
                        r_address,
                        gamma,
                        virtual_poly,
                        G.expect("G arrays must be provided for Instruction"),
                        H_indices.expect("H_indices must be provided for Instruction"),
                    )
                }
            };

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
            booleanity_type,
            d,
            log_k_chunk,
            log_t,
            gamma,
            r_address,
            r_cycle,
            prover_state: Some(prover_state),
            virtual_poly,
        }
    }

    pub fn new_verifier<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        booleanity_type: BooleanityType,
        state_manager: &mut StateManager<'_, F, T, PCS>,
    ) -> Self {
        let (d, log_k_chunk, log_t, r_cycle, r_address, gamma, virtual_poly) =
            match &booleanity_type {
                BooleanityType::Ram { K } => {
                    use crate::zkvm::witness::{compute_d_parameter, DTH_ROOT_OF_K};
                    let (_, _, T) = state_manager.get_verifier_data();
                    let d = compute_d_parameter(*K);
                    let log_k_chunk = DTH_ROOT_OF_K.log_2(); // 8
                    let log_t = T.log_2();

                    // CRITICAL: Order must match prover for Fiat-Shamir consistency
                    // 1. First get r_cycle challenges
                    let r_cycle: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_t);

                    // 2. Then get r_address challenges
                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_k_chunk);

                    // 3. Finally get gamma challenges
                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(d);

                    (d, log_k_chunk, log_t, r_cycle, r_address, gamma, None)
                }
                BooleanityType::Bytecode { d, log_K } => {
                    let (_, _, T) = state_manager.get_verifier_data();
                    let log_k_chunk = log_K.div_ceil(*d);
                    let log_t = T.log_2();

                    // For bytecode verifier, r_cycle comes from virtual polynomial
                    let r_cycle = Vec::new(); // Will be populated from virtual polynomial

                    // Get gamma challenges then r_address for bytecode (original order)
                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(*d);

                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(log_k_chunk);

                    let virtual_poly = Some(VirtualPolynomial::UnexpandedPC);

                    (
                        *d,
                        log_k_chunk,
                        log_t,
                        r_cycle,
                        r_address,
                        gamma,
                        virtual_poly,
                    )
                }
                BooleanityType::Instruction => {
                    let (_, _, T) = state_manager.get_verifier_data();
                    const D: usize = 16;
                    const LOG_K_CHUNK: usize = 8;
                    let log_t = T.log_2();

                    // Get gamma challenges first for instruction (original order)
                    let gamma: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(D);

                    let r_address: Vec<F::Challenge> = state_manager
                        .transcript
                        .borrow_mut()
                        .challenge_vector_optimized::<F>(LOG_K_CHUNK);

                    // For instruction verifier, r_cycle comes from virtual polynomial
                    let r_cycle = Vec::new(); // Will be populated from virtual polynomial

                    (D, LOG_K_CHUNK, log_t, r_cycle, r_address, gamma, None)
                }
            };

        Self {
            booleanity_type,
            d,
            log_k_chunk,
            log_t,
            gamma,
            r_address,
            r_cycle,
            prover_state: None,
            virtual_poly,
        }
    }

    fn polynomial_type(&self, i: usize) -> CommittedPolynomial {
        match &self.booleanity_type {
            BooleanityType::Ram { .. } => CommittedPolynomial::RamRa(i),
            BooleanityType::Bytecode { .. } => CommittedPolynomial::BytecodeRa(i),
            BooleanityType::Instruction => CommittedPolynomial::InstructionRa(i),
        }
    }

    fn sumcheck_id(&self) -> SumcheckId {
        match &self.booleanity_type {
            BooleanityType::Ram { .. } => SumcheckId::RamBooleanity,
            BooleanityType::Bytecode { .. } => SumcheckId::BytecodeBooleanity,
            BooleanityType::Instruction => SumcheckId::InstructionBooleanity,
        }
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        if !self.r_cycle.is_empty() {
            self.r_cycle.clone()
        } else if let Some(virtual_poly) = &self.virtual_poly {
            // For bytecode, get from virtual polynomial
            accumulator
                .get_virtual_polynomial_opening(*virtual_poly, SumcheckId::SpartanOuter)
                .0
                .r
                .clone()
        } else {
            // For instruction verifier, get from virtual polynomial
            accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanOuter,
                )
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

// Implementation of SumcheckInstance
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
        let ra_claims = (0..self.d)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(self.polynomial_type(i), self.sumcheck_id())
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
            (0..self.d).map(|i| self.polynomial_type(i)).collect(),
            self.sumcheck_id(),
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
            (0..self.d).map(|i| self.polynomial_type(i)).collect(),
            self.sumcheck_id(),
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// Helper functions for computing RAM-specific data internally
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::Cycle;

fn compute_ram_g_arrays_internal<F: JoltField>(
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: &[F],
    d: usize,
) -> Vec<Vec<F>> {
    use crate::zkvm::witness::DTH_ROOT_OF_K;

    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let addresses: Vec<Option<u64>> = trace
        .par_iter()
        .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
        .collect();

    let mut G_arrays = Vec::with_capacity(d);
    for i in 0..d {
        let G: Vec<F> = addresses
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, address_chunk)| {
                let mut local_array = unsafe_allocate_zero_vec(DTH_ROOT_OF_K);
                let mut j = chunk_index * chunk_size;
                for address_opt in address_chunk {
                    if let Some(address) = address_opt {
                        let address_i = (address >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                            % DTH_ROOT_OF_K as u64;
                        local_array[address_i as usize] += eq_r_cycle[j];
                    }
                    j += 1;
                }
                local_array
            })
            .reduce(
                || unsafe_allocate_zero_vec(DTH_ROOT_OF_K),
                |mut running, new| {
                    for (r, n) in running.iter_mut().zip(new.iter()) {
                        *r += *n;
                    }
                    running
                },
            );
        G_arrays.push(G);
    }
    G_arrays
}

fn compute_ram_h_indices_internal(
    trace: &[Cycle],
    memory_layout: &MemoryLayout,
    d: usize,
) -> Vec<Vec<Option<u8>>> {
    use crate::zkvm::witness::DTH_ROOT_OF_K;

    let addresses: Vec<Option<u64>> = trace
        .par_iter()
        .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
        .collect();

    (0..d)
        .map(|i| {
            addresses
                .par_iter()
                .map(|address| {
                    address.map(|a| {
                        ((a >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i))) % DTH_ROOT_OF_K as u64) as u8
                    })
                })
                .collect()
        })
        .collect()
}
