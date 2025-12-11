use std::sync::Arc;
use std::{array, mem};

use allocative::Allocative;
use ark_std::Zero;
use itertools::chain;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::eq_plus_one_poly::{EqPlusOnePolynomial, EqPlusOnePrefixSuffixPoly};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::ShiftSumcheckCycleState;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

// Spartan PC sumcheck
//
// Proves the batched identity over cycles j:
//   Σ_j EqPlusOne(r_outer, j) ⋅ (UnexpandedPC_shift(j) + γ·PC_shift(j) + γ²·IsNoop_shift(j))
//   = NextUnexpandedPC(r_outer) + γ·NextPC(r_outer) + γ²·NextIsNoop(r_outer),
//
// where:
// - EqPlusOne(r_outer, j): MLE of the function that,
//     on (i,j) returns 1 iff i = j + 1; no wrap-around at j = 2^{log T} − 1
// - UnexpandedPC_shift(j), PC_shift(j), IsNoop_shift(j):
//     SpartanShift MLEs encoding f(j+1) aligned at cycle j
// - NextUnexpandedPC(r_outer), NextPC(r_outer), NextIsNoop(r_outer)
//     are claims from Spartan outer sumcheck
// - γ: batching scalar drawn from the transcript

/// Degree bound of the sumcheck round polynomials in [`ShiftSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Default)]
pub struct ShiftSumcheckParams<F: JoltField> {
    pub gamma_powers: [F; 5],
    pub n_cycle_vars: usize, // = log(T)
    pub r_outer: OpeningPoint<BIG_ENDIAN, F>,
    pub r_product: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ShiftSumcheckParams<F> {
    pub fn new(
        n_cycle_vars: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma_powers = transcript.challenge_scalar_powers(5).try_into().unwrap();
        let (outer_sumcheck_r, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (r_outer, _rx_var) = outer_sumcheck_r.split_at(n_cycle_vars);
        let (product_sumcheck_r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );
        let (r_product, _) = product_sumcheck_r.split_at(n_cycle_vars);

        Self {
            gamma_powers,
            n_cycle_vars,
            r_outer,
            r_product,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ShiftSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, input_claim_next_pc) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, input_claim_next_unexpanded_pc) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_virtual) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_first_in_sequence) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
        let (_, input_claim_next_is_noop) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::ProductVirtualization,
        );

        input_claim_next_unexpanded_pc
            + input_claim_next_pc * self.gamma_powers[1]
            + input_claim_next_is_virtual * self.gamma_powers[2]
            + input_claim_next_is_first_in_sequence * self.gamma_powers[3]
            + (F::one() - input_claim_next_is_noop) * self.gamma_powers[4]
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        normalize_opening_point(challenges)
    }
}

fn normalize_opening_point<F: JoltField>(
    challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
}

/// Sumcheck prover for [`ShiftSumcheckVerifier`].
#[derive(Allocative)]
#[allow(clippy::large_enum_variant, private_interfaces)]
pub enum ShiftSumcheckProver<F: JoltField> {
    Phase1(Phase1Prover<F>), // 1st half (prefix-suffix sc)
    Phase2(Phase2Prover<F>), // 2st half (regular sc)
}

impl<F: JoltField> ShiftSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::initialize")]
    pub fn initialize(
        params: ShiftSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        Self::Phase1(Phase1Prover::gen(trace, bytecode_preprocessing, params))
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        match self {
            ShiftSumcheckProver::Phase1(phase1_prover) => &phase1_prover.params,
            ShiftSumcheckProver::Phase2(phase2_prover) => &phase2_prover.params,
        }
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match self {
            Self::Phase1(prover) => prover.compute_message(previous_claim),
            Self::Phase2(prover) => prover.compute_message(previous_claim),
        }
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match self {
            Self::Phase1(prover) => {
                if prover.should_transition_to_phase2() {
                    let params = mem::take(&mut prover.params);
                    let sumcheck_challenges = &mut prover.sumcheck_challenges;
                    sumcheck_challenges.push(r_j);
                    *self = Self::Phase2(Phase2Prover::gen(
                        &prover.trace,
                        &prover.bytecode_preprocessing,
                        sumcheck_challenges,
                        params,
                    ));
                    return;
                }

                prover.bind(r_j);
            }
            Self::Phase2(prover) => prover.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let Self::Phase2(prover) = &self else {
            panic!("Should finish sumcheck on phase 2");
        };

        let unexpanded_pc_eval = prover.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = prover.pc_poly.final_sumcheck_claim();
        let is_virtual_eval = prover.is_virtual_poly.final_sumcheck_claim();
        let is_first_in_sequence_eval = prover.is_first_in_sequence_poly.final_sumcheck_claim();
        let is_noop_eval = prover.is_noop_poly.final_sumcheck_claim();

        let opening_point = normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_virtual_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_first_in_sequence_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ShiftSumcheckVerifier<F: JoltField> {
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> ShiftSumcheckVerifier<F> {
    pub fn new(
        n_cycle_vars: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ShiftSumcheckParams::new(n_cycle_vars, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
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

        let r = normalize_opening_point::<F>(sumcheck_challenges);
        let eq_plus_one_r_outer_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_outer.r.to_vec()).evaluate(&r.r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_product.r.to_vec()).evaluate(&r.r);

        [
            unexpanded_pc_claim,
            pc_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .iter()
        .zip(&self.params.gamma_powers)
        .map(|(eval, gamma)| *gamma * eval)
        .sum::<F>()
            * eq_plus_one_r_outer_at_shift
            + self.params.gamma_powers[4]
                * (F::one() - is_noop_claim)
                * eq_plus_one_r_product_at_shift
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }
}

/// Prover for 1st half of the rounds.
///
/// Performs prefix-suffix sumcheck. See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
#[derive(Allocative)]
struct Phase1Prover<F: JoltField> {
    // All prefix-suffix (P, Q) buffers for this sumcheck.
    prefix_suffix_pairs: Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>)>,
    // Below all stored to gen phase 2 prover.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    sumcheck_challenges: Vec<F::Challenge>,
    #[allocative(skip)]
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> Phase1Prover<F> {
    fn gen(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        params: ShiftSumcheckParams<F>,
    ) -> Self {
        let EqPlusOnePrefixSuffixPoly {
            prefix_0: prefix_0_for_r_outer,
            suffix_0: suffix_0_for_r_outer,
            prefix_1: prefix_1_for_r_outer,
            suffix_1: suffix_1_for_r_outer,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_outer);
        let EqPlusOnePrefixSuffixPoly {
            prefix_0: prefix_0_for_r_prod,
            suffix_0: suffix_0_for_r_prod,
            prefix_1: prefix_1_for_r_prod,
            suffix_1: suffix_1_for_r_prod,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_product);

        let prefix_n_vars = prefix_0_for_r_outer.len().ilog2();
        let suffix_n_vars = suffix_0_for_r_outer.len().ilog2();

        // Prefix-suffix P and Q buffers.
        // See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
        let P_0_for_r_outer = prefix_0_for_r_outer;
        let P_1_for_r_outer = prefix_1_for_r_outer;
        let P_0_for_r_prod = prefix_0_for_r_prod;
        let P_1_for_r_prod = prefix_1_for_r_prod;
        let mut Q_0_for_r_outer = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_1_for_r_outer = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_0_for_r_prod = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_1_for_r_prod = vec![F::zero(); 1 << prefix_n_vars];

        const BLOCK_SIZE: usize = 32;
        (
            Q_0_for_r_outer.par_chunks_mut(BLOCK_SIZE),
            Q_1_for_r_outer.par_chunks_mut(BLOCK_SIZE),
            Q_0_for_r_prod.par_chunks_mut(BLOCK_SIZE),
            Q_1_for_r_prod.par_chunks_mut(BLOCK_SIZE),
        )
            .into_par_iter()
            .enumerate()
            .for_each(
                |(
                    chunk_i,
                    (
                        Q_0_for_r_outer_chunk,
                        Q_1_for_r_outer_chunk,
                        Q_0_for_r_prod_chunk,
                        Q_1_for_r_prod_chunk,
                    ),
                )| {
                    let chunk_len = Q_0_for_r_outer_chunk.len();
                    let mut Q_0_for_r_outer_unreduced = [F::Unreduced::<9>::zero(); BLOCK_SIZE];
                    let mut Q_1_for_r_outer_unreduced = [F::Unreduced::<9>::zero(); BLOCK_SIZE];
                    let mut Q_0_for_r_prod_unreduced = [F::Unreduced::<5>::zero(); BLOCK_SIZE];
                    let mut Q_1_for_r_prod_unreduced = [F::Unreduced::<5>::zero(); BLOCK_SIZE];

                    for x_hi in 0..1 << suffix_n_vars {
                        for i in 0..chunk_len {
                            let x_lo = chunk_i * BLOCK_SIZE + i;
                            let x = x_lo + (x_hi << prefix_n_vars);
                            let ShiftSumcheckCycleState {
                                unexpanded_pc,
                                pc,
                                is_virtual,
                                is_first_in_sequence,
                                is_noop,
                            } = ShiftSumcheckCycleState::new(&trace[x], bytecode_preprocessing);

                            let mut v =
                                F::from_u64(unexpanded_pc) + params.gamma_powers[1].mul_u64(pc);
                            if is_virtual {
                                v += params.gamma_powers[2];
                            }
                            if is_first_in_sequence {
                                v += params.gamma_powers[3];
                            }
                            Q_0_for_r_outer_unreduced[i] +=
                                v.mul_unreduced::<9>(suffix_0_for_r_outer[x_hi]);
                            Q_1_for_r_outer_unreduced[i] +=
                                v.mul_unreduced::<9>(suffix_1_for_r_outer[x_hi]);

                            // Q += suffix * (1 - is_noop)
                            if !is_noop {
                                Q_0_for_r_prod_unreduced[i] +=
                                    *suffix_0_for_r_prod[x_hi].as_unreduced_ref();
                                Q_1_for_r_prod_unreduced[i] +=
                                    *suffix_1_for_r_prod[x_hi].as_unreduced_ref();
                            }
                        }
                    }

                    for i in 0..chunk_len {
                        Q_0_for_r_outer_chunk[i] =
                            F::from_montgomery_reduce(Q_0_for_r_outer_unreduced[i]);
                        Q_1_for_r_outer_chunk[i] =
                            F::from_montgomery_reduce(Q_1_for_r_outer_unreduced[i]);
                        Q_0_for_r_prod_chunk[i] =
                            F::from_barrett_reduce(Q_0_for_r_prod_unreduced[i]);
                        Q_1_for_r_prod_chunk[i] =
                            F::from_barrett_reduce(Q_1_for_r_prod_unreduced[i]);
                    }
                },
            );

        chain!(&mut Q_0_for_r_prod, &mut Q_1_for_r_prod).for_each(|v| *v *= params.gamma_powers[4]);

        let prefix_suffix_pairs = vec![
            (P_0_for_r_outer.into(), Q_0_for_r_outer.into()),
            (P_1_for_r_outer.into(), Q_1_for_r_outer.into()),
            (P_0_for_r_prod.into(), Q_0_for_r_prod.into()),
            (P_1_for_r_prod.into(), Q_1_for_r_prod.into()),
        ];

        Self {
            prefix_suffix_pairs,
            trace,
            bytecode_preprocessing: bytecode_preprocessing.clone(),
            sumcheck_challenges: Vec::new(),
            params,
        }
    }

    fn compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let evals = self
            .prefix_suffix_pairs
            .par_iter()
            .map(|(p, q)| {
                let mut evals = [F::zero(); DEGREE_BOUND];
                for i in 0..p.len() / 2 {
                    let p_evals =
                        p.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                    let q_evals =
                        q.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                    evals = array::from_fn(|i| evals[i] + p_evals[i] * q_evals[i]);
                }
                evals
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        assert!(!self.should_transition_to_phase2());
        self.sumcheck_challenges.push(r_j);
        self.prefix_suffix_pairs.iter_mut().for_each(|(p, q)| {
            p.bind(r_j, BindingOrder::LowToHigh);
            q.bind(r_j, BindingOrder::LowToHigh);
        });
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.prefix_suffix_pairs[0].0.len().ilog2() == 1
    }
}

/// Prover for 2nd half of the rounds.
#[derive(Allocative)]
struct Phase2Prover<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_virtual_poly: MultilinearPolynomial<F>,
    is_first_in_sequence_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_r_outer: MultilinearPolynomial<F>,
    eq_plus_one_r_product: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> Phase2Prover<F> {
    fn gen(
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        sumcheck_challenges: &[F::Challenge],
        params: ShiftSumcheckParams<F>,
    ) -> Self {
        let n_remaining_rounds = params.r_outer.len() - sumcheck_challenges.len();
        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Gen eq+1(r_outer, (r_prefix, j)) for all j.
        let EqPlusOnePrefixSuffixPoly {
            prefix_0,
            suffix_0,
            prefix_1,
            suffix_1,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_outer);
        let prefix_0_eval = MultilinearPolynomial::from(prefix_0).evaluate(&r_prefix.r);
        let prefix_1_eval = MultilinearPolynomial::from(prefix_1).evaluate(&r_prefix.r);
        let eq_plus_one_r_outer: MultilinearPolynomial<F> = (0..suffix_0.len())
            .map(|i| prefix_0_eval * suffix_0[i] + prefix_1_eval * suffix_1[i])
            .collect::<Vec<F>>()
            .into();

        // Gen eq+1(r_product, (r_prefix, j)) for all j.
        let EqPlusOnePrefixSuffixPoly {
            prefix_0,
            suffix_0,
            prefix_1,
            suffix_1,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_product);
        let prefix_0_eval = MultilinearPolynomial::from(prefix_0).evaluate(&r_prefix.r);
        let prefix_1_eval = MultilinearPolynomial::from(prefix_1).evaluate(&r_prefix.r);
        let eq_plus_one_r_product: MultilinearPolynomial<F> = (0..suffix_0.len())
            .map(|i| prefix_0_eval * suffix_0[i] + prefix_1_eval * suffix_1[i])
            .collect::<Vec<F>>()
            .into();

        // Gen MLEs: UnexpandedPc(r_prefix, j), Pc(r_prefix, j), ...
        let eq_evals = EqPolynomial::evals(&r_prefix.r);
        let mut unexpanded_pc_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut pc_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_virtual_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_first_in_sequence_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_noop_poly = vec![F::zero(); 1 << n_remaining_rounds];
        (
            &mut unexpanded_pc_poly,
            &mut pc_poly,
            &mut is_virtual_poly,
            &mut is_first_in_sequence_poly,
            &mut is_noop_poly,
            trace.par_chunks(eq_evals.len()),
        )
            .into_par_iter()
            .for_each(
                |(
                    unexpanded_pc_eval,
                    pc_eval,
                    is_virtual_eval,
                    is_first_in_sequence_eval,
                    is_noop_eval,
                    trace_chunk,
                )| {
                    let mut unexpanded_pc_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut pc_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut is_virtual_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut is_first_in_sequence_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut is_noop_eval_unreduced = F::Unreduced::<5>::zero();

                    for (i, cycle) in trace_chunk.iter().enumerate() {
                        let ShiftSumcheckCycleState {
                            unexpanded_pc,
                            pc,
                            is_virtual,
                            is_first_in_sequence,
                            is_noop,
                        } = ShiftSumcheckCycleState::new(cycle, bytecode_preprocessing);
                        let eq_eval = eq_evals[i];
                        unexpanded_pc_eval_unreduced += eq_eval.mul_u64_unreduced(unexpanded_pc);
                        pc_eval_unreduced += eq_eval.mul_u64_unreduced(pc);
                        if is_virtual {
                            is_virtual_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                        if is_first_in_sequence {
                            is_first_in_sequence_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                        if is_noop {
                            is_noop_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                    }

                    *unexpanded_pc_eval = F::from_barrett_reduce(unexpanded_pc_eval_unreduced);
                    *pc_eval = F::from_barrett_reduce(pc_eval_unreduced);
                    *is_virtual_eval = F::from_barrett_reduce(is_virtual_eval_unreduced);
                    *is_first_in_sequence_eval =
                        F::from_barrett_reduce(is_first_in_sequence_eval_unreduced);
                    *is_noop_eval = F::from_barrett_reduce(is_noop_eval_unreduced);
                },
            );

        Self {
            unexpanded_pc_poly: unexpanded_pc_poly.into(),
            pc_poly: pc_poly.into(),
            is_virtual_poly: is_virtual_poly.into(),
            is_first_in_sequence_poly: is_first_in_sequence_poly.into(),
            is_noop_poly: is_noop_poly.into(),
            eq_plus_one_r_outer,
            eq_plus_one_r_product,
            params,
        }
    }

    fn compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let half_n = self.unexpanded_pc_poly.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let unexpanded_pc_evals = self
                .unexpanded_pc_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let pc_evals = self
                .pc_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_virtual_evals = self
                .is_virtual_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_first_in_sequence_evals = self
                .is_first_in_sequence_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_noop_evals = self
                .is_noop_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_plus_one_r_outer_evals = self
                .eq_plus_one_r_outer
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_plus_one_r_product_evals = self
                .eq_plus_one_r_product
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            evals = array::from_fn(|i| {
                evals[i]
                    + eq_plus_one_r_outer_evals[i]
                        * (unexpanded_pc_evals[i]
                            + self.params.gamma_powers[1] * pc_evals[i]
                            + self.params.gamma_powers[2] * is_virtual_evals[i]
                            + self.params.gamma_powers[3] * is_first_in_sequence_evals[i])
                    + self.params.gamma_powers[4]
                        * eq_plus_one_r_product_evals[i]
                        * (F::one() - is_noop_evals[i])
            });
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        let Self {
            unexpanded_pc_poly,
            pc_poly,
            is_virtual_poly,
            is_first_in_sequence_poly,
            is_noop_poly,
            eq_plus_one_r_outer,
            eq_plus_one_r_product,
            params: _,
        } = self;
        unexpanded_pc_poly.bind(r_j, BindingOrder::LowToHigh);
        pc_poly.bind(r_j, BindingOrder::LowToHigh);
        is_virtual_poly.bind(r_j, BindingOrder::LowToHigh);
        is_first_in_sequence_poly.bind(r_j, BindingOrder::LowToHigh);
        is_noop_poly.bind(r_j, BindingOrder::LowToHigh);
        eq_plus_one_r_outer.bind(r_j, BindingOrder::LowToHigh);
        eq_plus_one_r_product.bind(r_j, BindingOrder::LowToHigh);
    }
}
