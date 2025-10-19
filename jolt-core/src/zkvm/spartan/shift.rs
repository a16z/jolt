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
use crate::poly::eq_plus_one_poly::{EqPlusOnePolynomial, EqPlusOnePS};
use crate::poly::eq_poly::EqPolynomial;
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
use crate::zkvm::r1cs::inputs::{compute_shift_openings_at_point, generate_shift_sumcheck_witnesses};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;
use tracer::instruction::Cycle;

use rayon::prelude::*;

#[derive(Allocative)]
struct ShiftSumcheckProverState<F: JoltField> {
    #[allocative(skip)]
    pub preprocess: Arc<JoltSharedPreprocessing>,
    #[allocative(skip)]
    pub trace: Arc<Vec<Cycle>>,
    // Precomputed combined witness weights per cycle for phase 0 (length 2^ell)
    u_evals_full: Vec<F>,
    v_evals_full: Vec<F>,
    // Dynamic field-valued prefix–suffix decompositions for EqPlusOne terms
    ps_cycle: PrefixSuffixDecompositionFieldDyn<F>,
    ps_product: PrefixSuffixDecompositionFieldDyn<F>,
    reg_cycle: DynamicPrefixRegistry<F>,
    reg_product: DynamicPrefixRegistry<F>,
    // Full-domain helpers (phase 0)
    indices_full: Vec<usize>,
    lookup_bits_full: Vec<LookupBits>,
    // Phase-1 helpers over the remaining variables after cutoff
    indices_phase1: Vec<usize>,
    lookup_bits_phase1: Vec<LookupBits>,
    // Track the first-chunk challenges to condense U/V without binding base polynomials
    r_prefix: Vec<F::Challenge>,
    cutoff: usize,
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

        // Stream once to generate PC, UnexpandedPC and flags witnesses
        let (unexpanded_pc_poly, pc_poly, is_noop_poly, is_virtual_poly, is_first_in_sequence_poly) =
            generate_shift_sumcheck_witnesses::<F>(&preprocessing.shared, &trace);

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
        debug_assert_eq!(r_cycle.len(), num_cycles_bits);
        debug_assert_eq!(r_product.len(), num_cycles_bits);

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

        // Precompute index helpers once for full domain (phase 0)
        let t = 1usize << ell;
        let indices_full: Vec<usize> = (0..t).collect();
        let lookup_bits_full: Vec<LookupBits> =
            (0..t).map(|i| LookupBits::new(i as u128, ell)).collect();

        // Precompute helpers for phase 1 (remaining variables after cutoff)
        let rem = ell - cutoff;
        let (indices_phase1, lookup_bits_phase1) = if rem > 0 {
            let t1 = 1usize << rem;
            let idx: Vec<usize> = (0..t1).collect();
            let bits: Vec<LookupBits> = (0..t1).map(|i| LookupBits::new(i as u128, rem)).collect();
            (idx, bits)
        } else {
            (Vec::new(), Vec::new())
        };

        // Stage 0: build P and Q for both decompositions in one pass
        ps_cycle.init_P(&mut reg_cycle);
        ps_product.init_P(&mut reg_product);

        // Get batching challenge for combining claims (gamma powers) before using them
        let gamma_powers: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_scalar_powers(5);

        // Materialize base arrays (drop ML polys from state after use)
        let mut unexpanded_pc_vec = Vec::with_capacity(t);
        let mut pc_vec = Vec::with_capacity(t);
        let mut is_virtual_vec = Vec::with_capacity(t);
        let mut is_first_in_sequence_vec = Vec::with_capacity(t);
        let mut is_noop_vec = Vec::with_capacity(t);
        for i in 0..t {
            unexpanded_pc_vec.push(unexpanded_pc_poly.get_coeff(i));
            pc_vec.push(pc_poly.get_coeff(i));
            is_virtual_vec.push(is_virtual_poly.get_coeff(i));
            is_first_in_sequence_vec.push(is_first_in_sequence_poly.get_coeff(i));
            is_noop_vec.push(is_noop_poly.get_coeff(i));
        }

        let gamma = &gamma_powers; // [F; 5]
        let u_evals: Vec<F> = (0..t)
            .map(|i| {
                gamma[0] * unexpanded_pc_vec[i]
                    + gamma[1] * pc_vec[i]
                    + gamma[2] * is_virtual_vec[i]
                    + gamma[3] * is_first_in_sequence_vec[i]
            })
            .collect();
        let v_evals: Vec<F> = (0..t)
            .map(|i| gamma[4] * (F::one() - is_noop_vec[i]))
            .collect();

        PrefixSuffixDecompositionFieldDyn::init_Q_dual(
            &mut ps_cycle,
            &mut ps_product,
            &u_evals,
            &v_evals,
            &indices_full,
            &lookup_bits_full,
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

        // Debug: sanity-check non-wrap EqPlusOne identity against the assembled input_claim
        #[cfg(debug_assertions)]
        {
            let (_eq_tab_c, eqp_c) = EqPlusOnePolynomial::<F>::evals(&r_cycle.r, None);
            let (_eq_tab_p, eqp_p) = EqPlusOnePolynomial::<F>::evals(&r_product.r, None);
            debug_assert_eq!(eqp_c.len(), u_evals.len());
            debug_assert_eq!(eqp_p.len(), v_evals.len());
            let sum_u: F = (0..eqp_c.len()).map(|i| u_evals[i] * eqp_c[i]).sum();
            let sum_v: F = (0..eqp_p.len()).map(|i| v_evals[i] * eqp_p[i]).sum();
            debug_assert_eq!(
                sum_u + sum_v,
                input_claim,
                "EqPlusOne non-wrap identity failed"
            );
        }

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ShiftSumcheckProverState {
                preprocess: Arc::new(preprocessing.shared.clone()),
                trace: Arc::new(trace.to_vec()),
                u_evals_full: u_evals,
                v_evals_full: v_evals,
                ps_cycle,
                ps_product,
                reg_cycle,
                reg_product,
                indices_full,
                lookup_bits_full,
                indices_phase1,
                lookup_bits_phase1,
                r_prefix: Vec::with_capacity(cutoff),
                cutoff,
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
        debug_assert_eq!(
            prover_state.ps_cycle.Q_len(),
            prover_state.ps_product.Q_len()
        );
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
        // Track first-chunk challenges to condense U/V without binding the base polynomials
        if prover_state.r_prefix.len() < prover_state.cutoff {
            prover_state.r_prefix.push(r_j);
        }

        rayon::scope(|s| {
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

            // Condense U and V onto the remaining variables using the collected r_prefix
            let rem = total_rounds - cutoff;
            let suffix_len = rem; // low bits correspond to the remaining variables
            let poly_len_phase1 = 1usize << rem;

            let eq_prefix = EqPolynomial::<F>::evals(&prover_state.r_prefix);

            let mut u_evals: Vec<F> = vec![F::zero(); poly_len_phase1];
            let mut v_evals: Vec<F> = vec![F::zero(); poly_len_phase1];
            // Iterate full domain once; map to (prefix, suffix) and accumulate with eq(prefix)
            prover_state
                .indices_full
                .par_iter()
                .for_each(|&i| {
                    let k = prover_state.lookup_bits_full[i];
                    let _ = k.split(suffix_len);
                    // parallel stub suppressed; serial accumulation follows
                });

            // Serial accumulation to keep correctness and simplicity
            for i in 0..prover_state.indices_full.len() {
                let k = prover_state.lookup_bits_full[i];
                let (prefix_bits, suffix_bits) = k.split(suffix_len);
                let p: usize = prefix_bits.into();
                let sfx: usize = suffix_bits.into();
                let w = eq_prefix[p];
                u_evals[sfx] += w * prover_state.u_evals_full[i];
                v_evals[sfx] += w * prover_state.v_evals_full[i];
            }

            PrefixSuffixDecompositionFieldDyn::init_Q_dual(
                &mut prover_state.ps_cycle,
                &mut prover_state.ps_product,
                &u_evals,
                &v_evals,
                &prover_state.indices_phase1,
                &prover_state.lookup_bits_phase1,
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

        let (unexpanded_pc_eval, pc_eval, is_virtual_eval, is_first_in_sequence_eval, is_noop_eval) = compute_shift_openings_at_point(&prover_state.preprocess, &prover_state.trace, &opening_point.r);

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
