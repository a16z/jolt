use std::{cell::RefCell, rc::Rc};

use crate::{
    field::{allocative_ark::MaybeAllocative, JoltField},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        program_io_polynomial::ProgramIOPolynomial,
        range_mask_polynomial::RangeMaskPolynomial,
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    utils::{expanding_table::ExpandingTable, math::Math, transcript::Transcript},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::RAM_START_ADDRESS;
use rayon::prelude::*;
use tracer::JoltDevice;

#[derive(Allocative)]
struct OutputSumcheckProverState<F: JoltField> {
    /// Val(k, 0)
    val_init: MultilinearPolynomial<F>,
    /// The MLE of the final RAM state
    val_final: MultilinearPolynomial<F>,
    /// Val_io(k) = Val_final(k) if k is in the "IO" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_io: MultilinearPolynomial<F>,
    /// EQ(k, r_address)
    eq_poly: MultilinearPolynomial<F>,
    /// io_mask(k) serves as a "mask" for the IO region of memory,
    /// i.e. io_mask(k) = 1 if k is in the "IO" region of memory,
    /// and 0 otherwise.
    io_mask: MultilinearPolynomial<F>,
    /// Updated to contain the table of evaluations
    /// EQ(x_1, ..., x_k, r_1, ..., r_k), where r_i is the
    /// random challenge for the i'th round of sumcheck.
    eq_table: ExpandingTable<F>,
}

impl<F: JoltField> OutputSumcheckProverState<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheckProverState::initialize")]
    fn new(
        initial_ram_state: Vec<u32>,
        final_ram_state: Vec<u32>,
        program_io: &JoltDevice,
        r_address: &[F],
    ) -> Self {
        let K = final_ram_state.len();
        debug_assert_eq!(initial_ram_state.len(), final_ram_state.len());
        debug_assert!(K.is_power_of_two());

        // Compute the witness indices corresponding to the start and end of the IO
        // region of memory
        let io_start = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        )
        .unwrap() as usize;
        let io_end = remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as usize;

        // Compute Val_io by copying the relevant slice of Val_final
        let mut val_io = vec![0; K];
        val_io[io_start..io_end]
            .par_iter_mut()
            .zip(final_ram_state[io_start..io_end].par_iter())
            .for_each(|(dest, src)| *dest = *src);

        // Compute io_mask by setting the relevant coefficients to 1
        let mut io_mask = vec![0u8; K];
        io_mask[io_start..io_end]
            .par_iter_mut()
            .for_each(|k| *k = 1);

        // Initialize the EQ table
        let mut eq_table = ExpandingTable::new(K);
        eq_table.reset(F::one());

        Self {
            val_init: initial_ram_state.into(),
            val_final: final_ram_state.into(),
            val_io: val_io.into(),
            eq_poly: EqPolynomial::evals(r_address).into(),
            io_mask: io_mask.into(),
            eq_table,
        }
    }
}

struct OutputSumcheckVerifierState<F: JoltField> {
    r_address: Vec<F>,
    program_io: JoltDevice,
}

/// Proves that the final RAM state is consistent with the claimed
/// program output.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct OutputProof<F: JoltField, ProofTranscript: Transcript> {
    output_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    val_final_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: F,
}

/// Sumcheck for the zero-check
///   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
/// In plain English: the final memory state (Val_final) should be consistent with
/// the expected program outputs (Val_io) at the indices where the program
/// inputs/outputs are stored (io_range).
#[derive(Allocative)]
pub struct OutputSumcheck<F: JoltField> {
    K: usize,
    #[allocative(skip)]
    verifier_state: Option<OutputSumcheckVerifierState<F>>,
    prover_state: Option<OutputSumcheckProverState<F>>,
}

impl<F: JoltField + MaybeAllocative> OutputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheck")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: Vec<u32>,
        final_ram_state: Vec<u32>,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, program_io, _) = state_manager.get_prover_data();
        let K = final_ram_state.len();

        let r_address = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        let output_sumcheck_prover_state = OutputSumcheckProverState::new(
            initial_ram_state,
            final_ram_state,
            program_io,
            &r_address,
        );

        OutputSumcheck {
            K,
            verifier_state: None,
            prover_state: Some(output_sumcheck_prover_state),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, program_io, _) = state_manager.get_verifier_data();

        let r_address = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        let output_sumcheck_verifier_state = OutputSumcheckVerifierState {
            program_io: program_io.clone(),
            r_address: r_address.to_vec(),
        };

        OutputSumcheck {
            K,
            verifier_state: Some(output_sumcheck_verifier_state),
            prover_state: None,
        }
    }
}

impl<F: JoltField + MaybeAllocative> SumcheckInstance<F> for OutputSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 3;
        let OutputSumcheckProverState {
            eq_poly,
            io_mask,
            val_final,
            val_io,
            ..
        } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|k| {
                let eq_evals = eq_poly.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let io_mask_evals =
                    io_mask.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_final_evals =
                    val_final.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_io_evals =
                    val_io.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                [
                    eq_evals[0] * io_mask_evals[0] * (val_final_evals[0] - val_io_evals[0]),
                    eq_evals[1] * io_mask_evals[1] * (val_final_evals[1] - val_io_evals[1]),
                    eq_evals[2] * io_mask_evals[2] * (val_final_evals[2] - val_io_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheck::bind")]
    fn bind(&mut self, r_j: F, _: usize) {
        // Bind address variable
        let OutputSumcheckProverState {
            val_init,
            val_final,
            val_io,
            eq_poly,
            io_mask,
            eq_table,
            ..
        } = self.prover_state.as_mut().unwrap();

        // We bind Val_init here despite the fact that it is not used in `compute_prover_message`
        // because we'll need Val_init(r) in `ValFinalSumcheck`
        [val_init, val_final, val_io, eq_poly, io_mask]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        eq_table.update(r_j);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let OutputSumcheckVerifierState {
            r_address,
            program_io,
        } = self.verifier_state.as_ref().unwrap();

        let val_final_claim = accumulator
            .as_ref()
            .unwrap()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let r_address_prime = &r[..r_address.len()];

        let io_mask = RangeMaskPolynomial::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            )
            .unwrap(),
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap(),
        );
        let val_io = ProgramIOPolynomial::new(program_io);

        let eq_eval = EqPolynomial::mle(r_address, r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
        let val_io_eval = val_io.evaluate(r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let OutputSumcheckProverState {
            val_final,
            val_init,
            ..
        } = self.prover_state.as_ref().unwrap();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final.final_sumcheck_claim(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
        );
    }
}

#[derive(Allocative)]
struct ValFinalSumcheckProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
}

/// This sumcheck virtualizes Val_final(k) as:
/// Val_final(k) = Val_init(k) + \sum_k Inc(j) * wa(k, j)
///   or equivalently:
/// Val_final(k) - Val_init(k) = \sum_k Inc(j) * wa(k, j)
/// We feed the output claim Val_final(r_address) from `OutputSumcheck`
/// into this sumcheck, which reduces it to claims about `Inc` and `wa`.
/// Note that the verifier is assumed to be able to evaluate Val_init
/// on its own.
#[derive(Allocative)]
pub struct ValFinalSumcheck<F: JoltField> {
    T: usize,
    prover_state: Option<ValFinalSumcheckProverState<F>>,
    val_init_eval: F,
    val_final_claim: F,
}

impl<F: JoltField + MaybeAllocative> ValFinalSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();

        let r_address = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        // TODO(moodlezoup): Can reuse from OutputSumcheck
        let eq_r_address = EqPolynomial::evals(&r_address);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // Compute the wa polynomial using the above table
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    .map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);

        // #[cfg(test)]
        // {
        //     let OutputSumcheckProverState {
        //         val_init,
        //         val_final,
        //         ..
        //     } = &output_sumcheck_prover_state;
        //     // Check that Val_init(r), wa(r, j), and Inc(j) are consistent with
        //     // the claim Val_final(r)
        //     let expected = val_final.final_sumcheck_claim();
        //     let actual = val_init.final_sumcheck_claim()
        //         + wa_r_address
        //             .par_iter()
        //             .enumerate()
        //             .map(|(j, wa)| inc.get_coeff(j) * wa)
        //             .sum::<F>();
        //     assert_eq!(
        //         expected, actual,
        //         "Val_final(r_address) ≠ Val_init(r_address) + \\sum_j wa(r_address, j) * Inc(j)"
        //     );
        // }

        let val_init_eval = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValInit,
                SumcheckId::RamOutputCheck,
            )
            .1;
        let val_final_claim = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        Self {
            T,
            prover_state: Some(ValFinalSumcheckProverState { wa, inc }),
            val_init_eval,
            val_final_claim,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u32],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        let r_address = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let val_init_eval = val_init.evaluate(&r_address);
        let val_final_claim = state_manager
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        Self {
            T,
            prover_state: None,
            val_init_eval,
            val_final_claim,
        }
    }
}

impl<F: JoltField + MaybeAllocative> SumcheckInstance<F> for ValFinalSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.val_final_claim - self.val_init_eval
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _: usize, _previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 2;

        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = inc.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let wa_evals = wa.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                [inc_evals[0] * wa_evals[0], inc_evals[1] * wa_evals[1]]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheck::bind")]
    fn bind(&mut self, r_j: F, _: usize) {
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::HighToLow),
            || wa.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();
        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;

        inc_claim * wa_claim
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let r_address = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
            &[inc.final_sumcheck_claim()],
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
            wa.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_address = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
        );
    }
}
