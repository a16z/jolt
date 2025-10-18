/*
ShiftSumcheck: claim and proving strategy

Let T = 2^ell be the number of cycles (ell = log_2 T). We define two EqPlusOne kernels over the
cycle index y ∈ {0,1}^ell:
  EQP_c(y) = EqPlusOne(y; r_cycle)      // r_cycle = outer Spartan sumcheck opening bits (big-endian)
  EQP_p(y) = EqPlusOne(y; r_product)    // r_product = ShouldJump virtualization opening bits

We combine five per-cycle witness polynomials with batching scalars γ^i (i = 0..4):
  U(y)  = γ^0·UnexpandedPC(y) + γ^1·PC(y) + γ^2·IsVirtual(y) + γ^3·IsFirstInSequence(y)
  V(y)  = γ^4·(1 − IsNoop(y))

Target polynomial for this sumcheck over the cycle variables y is:
  F(y) = U(y)·EQP_c(y) + V(y)·EQP_p(y)

The sumcheck claim equals the value obtained when all y-variables are successively bound by the
verifier’s challenges r = (r_0, …, r_{ell−1}) (HighToLow). Algebraically, the initial claim equals
the “shifted” evaluations provided by the outer protocols via the EqPlusOne kernel identity:
  ∑_{y∈{0,1}^ell} U(y)·EqPlusOne(y; r_cycle) = U(r_cycle + 1)
  ∑_{y∈{0,1}^ell} V(y)·EqPlusOne(y; r_product) = V(r_product + 1)

Accordingly, the input_claim we commit to is:
  input_claim = (γ^0·NextUnexpandedPC + γ^1·NextPC + γ^2·NextIsVirtual + γ^3·NextIsFirstInSequence)
                + γ^4·(1 − NextIsNoop)
where the Next* evaluations are taken from the appropriate accumulators at the outer points.

Prover strategy
We build two dynamic, field-valued prefix–suffix decompositions (PS) of EqPlusOne(y; x) with x set
to r_cycle and r_product, respectively. Each PS splits the ell variables into two chunks:
  - Phase 0 (first_chunk_len = ceil(ell/2)): build P_0[x] over the first chunk and Q_0[x] by
    summing U (or V) against the suffix MLE over the remaining variables.
  - After binding the first chunk, rebuild P_1[x] over the second chunk and Q_1[x] with the bound
    witnesses (no suffix left in this phase).

At every round j we emit degree-2 univariate evaluations g_j(0), g_j(2) by aggregating, over terms
in the EqPlusOne expansion, the product of the current prefix univariate evaluations and the
appropriate left/right halves of Q. We do this independently for cycle and product PS, and sum the
results to obtain F’s prover message.

Verifier strategy
The verifier recomputes the expected folded claim at the end using:
  (γ^0·UnexpandedPC + γ^1·PC + γ^2·IsVirtual + γ^3·IsFirstInSequence)·EqPlusOne(r; r_cycle)
  + γ^4·(1 − IsNoop)·EqPlusOne(r; r_product)
with the same γ^i and r taken from this sumcheck’s transcript. Equality holds when all rounds are
consistent.
*/
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_plus_one_poly::EqPlusOnePS;
use crate::poly::eq_poly::EqPlusOnePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::prefix_suffix::{DynamicPrefixRegistry, PrefixSuffixDecompositionFieldDyn};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::lookup_bits::LookupBits;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::generate_shift_sumcheck_witnesses;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

#[derive(Allocative)]
struct ShiftSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_virtual_poly: MultilinearPolynomial<F>,
    is_first_in_sequence_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    // Dynamic field-valued prefix–suffix decompositions for EqPlusOne terms
    ps_cycle: PrefixSuffixDecompositionFieldDyn<F>,
    ps_product: PrefixSuffixDecompositionFieldDyn<F>,
    reg_cycle: DynamicPrefixRegistry<F>,
    reg_product: DynamicPrefixRegistry<F>,
    indices: Vec<usize>,
    lookup_bits: Vec<LookupBits>,
}

#[derive(Allocative)]
pub struct ShiftSumcheck<F: JoltField> {
    input_claim: F,
    gamma_powers: [F; 5],
    log_T: usize,
    prover_state: Option<ShiftSumcheckProverState<F>>,
}

impl<F: JoltField> ShiftSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let (preprocessing, trace, _program_io, _final_memory_state) =
            state_manager.get_prover_data();

        // Stream once to generate PC, UnexpandedPC and IsNoop witnesses
        let (unexpanded_pc_poly, pc_poly, is_noop_poly, is_virtual_poly, is_first_in_sequence_poly) =
            generate_shift_sumcheck_witnesses(&preprocessing.shared, trace);

        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get opening_point and claims from accumulator
        let accumulator = state_manager.get_prover_accumulator();
        let (outer_sumcheck_r, next_pc_eval) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_virtual_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_first_in_sequence_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );

        let (product_sumcheck_r, next_is_noop_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::ShouldJumpVirtualization,
            );

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        // Build dynamic field-valued prefix–suffix decompositions for EqPlusOne terms
        let ell = num_cycles_bits;
        let cutoff = ell.div_ceil(2);
        let eqps_cycle = EqPlusOnePS::<F>::new(r_cycle.r.clone(), ell, cutoff);
        let eqps_product = EqPlusOnePS::<F>::new(r_product.r.clone(), ell, cutoff);
        let mut reg_cycle = DynamicPrefixRegistry::new(eqps_cycle.order());
        let mut reg_product = DynamicPrefixRegistry::new(eqps_product.order());
        let mut ps_cycle =
            PrefixSuffixDecompositionFieldDyn::new(Box::new(eqps_cycle), cutoff, ell);
        let mut ps_product =
            PrefixSuffixDecompositionFieldDyn::new(Box::new(eqps_product), cutoff, ell);

        // Precompute index helpers once for full domain
        let t = 1usize << ell;
        let indices: Vec<usize> = (0..t).collect();
        let lookup_bits: Vec<LookupBits> =
            (0..t).map(|i| LookupBits::new(i as u128, ell)).collect();

        // Stage 0: build P and Q for both decompositions in one pass
        ps_cycle.init_P(&mut reg_cycle);
        ps_product.init_P(&mut reg_product);

        // Get batching challenge for combining claims (gamma powers) before using them
        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);

        let gamma = &gamma_powers; // [F; 5]
        let u_evals: Vec<F> = (0..t)
            .map(|i| {
                gamma[0] * unexpanded_pc_poly.get_coeff(i)
                    + gamma[1] * pc_poly.get_coeff(i)
                    + gamma[2] * is_virtual_poly.get_coeff(i)
                    + gamma[3] * is_first_in_sequence_poly.get_coeff(i)
            })
            .collect();
        let v_evals: Vec<F> = (0..t)
            .map(|i| gamma[4] * (F::one() - is_noop_poly.get_coeff(i)))
            .collect();

        PrefixSuffixDecompositionFieldDyn::init_Q_dual(
            &mut ps_cycle,
            &mut ps_product,
            &u_evals,
            &v_evals,
            &indices,
            &lookup_bits,
        );

        let input_claim = [
            next_unexpanded_pc_eval,
            next_pc_eval,
            next_is_virtual_eval,
            next_is_first_in_sequence_eval,
            F::one() - next_is_noop_eval,
        ]
        .iter()
        .zip(gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum();

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ShiftSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_virtual_poly,
                is_first_in_sequence_poly,
                is_noop_poly,
                ps_cycle,
                ps_product,
                reg_cycle,
                reg_product,
                indices,
                lookup_bits,
            }),
            gamma_powers: gamma_powers.try_into().unwrap(),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        // Get batching challenge for combining claims
        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);

        // Get the Next* evaluations from the accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let (_, next_pc_eval) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_virtual_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_first_in_sequence_eval) =
            accumulator.borrow().get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
        let (_, next_is_noop_eval) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ShouldJumpVirtualization,
        );

        let input_claim = [
            next_unexpanded_pc_eval,
            next_pc_eval,
            next_is_virtual_eval,
            next_is_first_in_sequence_eval,
            F::one() - next_is_noop_eval,
        ]
        .iter()
        .zip(gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum();
        let log_T = key.num_steps.log_2();

        Self {
            input_claim,
            prover_state: None,
            log_T,
            gamma_powers: gamma_powers.try_into().unwrap(),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ShiftSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        // Iterate over the current prefix-suffix domain size, not the raw witness size.
        debug_assert_eq!(prover_state.ps_cycle.Q_len(), prover_state.ps_product.Q_len());
        let q_half_len = core::cmp::min(
            prover_state.ps_cycle.Q_len(),
            prover_state.ps_product.Q_len(),
        ) / 2;
        let univariate_poly_evals: [F; DEGREE] = (0..q_half_len)
            .into_par_iter()
            .map(|i| {
                let (c0, c2) = prover_state.ps_cycle.sumcheck_evals(i);
                let (p0, p2) = prover_state.ps_product.sumcheck_evals(i);
                [c0 + p0, c2 + p2]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    running[0] += new[0];
                    running[1] += new[1];
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        let prev_len = prover_state.ps_cycle.Q_len();
        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .unexpanded_pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_virtual_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_first_in_sequence_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_noop_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| prover_state.ps_cycle.bind(r_j));
            s.spawn(|_| prover_state.ps_product.bind(r_j));
        });

        // Stage boundary: after finishing the first chunk, rebuild P and Q for next phase.
        // Detect when Q shrinks from 2 -> 1 and there is a remaining second chunk.
        let current_len = prover_state.ps_cycle.Q_len();
        let total_rounds = self.log_T;
        let cutoff = total_rounds.div_ceil(2);
        let rem = total_rounds - cutoff;
        if prev_len == 2 && current_len == 1 && rem > 0 {
            prover_state.reg_cycle.update_checkpoints();
            prover_state.reg_product.update_checkpoints();
            prover_state.ps_cycle.init_P(&mut prover_state.reg_cycle);
            prover_state
                .ps_product
                .init_P(&mut prover_state.reg_product);

            let t_next = prover_state.unexpanded_pc_poly.len();
            let u_evals: Vec<F> = (0..t_next)
                .map(|i| {
                    self.gamma_powers[0] * prover_state.unexpanded_pc_poly.get_bound_coeff(i)
                        + self.gamma_powers[1] * prover_state.pc_poly.get_bound_coeff(i)
                        + self.gamma_powers[2] * prover_state.is_virtual_poly.get_bound_coeff(i)
                        + self.gamma_powers[3]
                            * prover_state.is_first_in_sequence_poly.get_bound_coeff(i)
                })
                .collect();
            let v_evals: Vec<F> = (0..t_next)
                .map(|i| {
                    self.gamma_powers[4] * (F::one() - prover_state.is_noop_poly.get_bound_coeff(i))
                })
                .collect();

            PrefixSuffixDecompositionFieldDyn::init_Q_dual(
                &mut prover_state.ps_cycle,
                &mut prover_state.ps_product,
                &u_evals,
                &v_evals,
                &prover_state.indices,
                &prover_state.lookup_bits,
            );
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits = self.log_T;
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        let (product_sumcheck_opening, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ShouldJumpVirtualization,
        );
        let product_sumcheck_r = &product_sumcheck_opening.r;
        let (r_product, _) = product_sumcheck_r.split_at(num_cycles_bits);

        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_virtual_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let eq_plus_one_r_cycle_at_shift =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(r_product.to_vec()).evaluate(r);

        [
            unexpanded_pc_claim,
            pc_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .iter()
        .zip(self.gamma_powers.iter())
        .map(|(eval, gamma)| *gamma * eval)
        .sum::<F>()
            * eq_plus_one_r_cycle_at_shift
            + self.gamma_powers[4] * (F::one() - is_noop_claim) * eq_plus_one_r_product_at_shift
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let unexpanded_pc_eval = prover_state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover_state.pc_poly.final_sumcheck_claim();
        let is_virtual_eval = prover_state.is_virtual_poly.final_sumcheck_claim();
        let is_first_in_sequence_eval = prover_state
            .is_first_in_sequence_poly
            .final_sumcheck_claim();
        let is_noop_eval = prover_state.is_noop_poly.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_virtual_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_first_in_sequence_eval,
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
