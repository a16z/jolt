use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::{
        vm::{ram::remap_address, JoltProverPreprocessing},
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        program_io_polynomial::ProgramIOPolynomial,
        range_mask_polynomial::RangeMaskPolynomial,
    },
    subprotocols::{
        sparse_dense_shout::ExpandingTable,
        sumcheck::{BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof},
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::RAM_START_ADDRESS;
use rayon::prelude::*;
use tracer::{instruction::RV32IMCycle, JoltDevice};

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
    fn initialize(
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
        ) as usize;
        let io_end = remap_address(RAM_START_ADDRESS, &program_io.memory_layout) as usize;

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

impl<F: JoltField> OutputSumcheckVerifierState<F> {
    fn initialize(r_address: &[F], program_io: &JoltDevice) -> Self {
        Self {
            r_address: r_address.to_vec(),
            program_io: program_io.clone(),
        }
    }
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
    /// Claimed evaluations Inc(r_cycle) and wa(r_cycle) output by `ValFinalSumcheck`
    output_claims: ValFinalSumcheckClaims<F>,
}

/// Sumcheck for the zero-check
///   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
/// In plain English: the final memory state (Val_final) should be consistent with
/// the expected program outputs (Val_io) at the indices where the program
/// inputs/outputs are stored (io_range).
pub struct OutputSumcheck<F: JoltField> {
    K: usize,
    T: usize,
    verifier_state: Option<OutputSumcheckVerifierState<F>>,
    prover_state: Option<OutputSumcheckProverState<F>>,
    /// Claimed evaluation Val_final(r_address) output by `OutputSumcheck`,
    /// proven using `ValFinalSumcheck`
    val_final_claim: Option<F>,
}

impl<F: JoltField> OutputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheck")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: Vec<u32>,
        final_ram_state: Vec<u32>,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let K = final_ram_state.len();
        let T = trace.len();

        let r_address = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        let output_sumcheck_prover_state = OutputSumcheckProverState::initialize(
            initial_ram_state,
            final_ram_state,
            program_io,
            &r_address,
        );

        OutputSumcheck {
            K,
            T,
            verifier_state: None,
            prover_state: Some(output_sumcheck_prover_state),
            val_final_claim: None,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, program_io, T) = state_manager.get_verifier_data();

        let r_address = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        let output_sumcheck_verifier_state = OutputSumcheckVerifierState {
            program_io: program_io.clone(),
            r_address: r_address.to_vec(),
        };

        let val_final_claim = state_manager
            .get_verifier_accumulator()
            .borrow()
            .get_opening(OpeningsKeys::RamValFinal);

        OutputSumcheck {
            K,
            T,
            verifier_state: Some(output_sumcheck_verifier_state),
            prover_state: None,
            val_final_claim: Some(val_final_claim),
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for OutputSumcheck<F> {
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
    fn compute_prover_message(&mut self, _: usize) -> Vec<F> {
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
                let eq_evals = eq_poly.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let io_mask_evals = io_mask.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let val_final_evals = val_final.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let val_io_evals = val_io.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
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

    fn expected_output_claim(&self, r: &[F]) -> F {
        let OutputSumcheckVerifierState {
            r_address,
            program_io,
        } = self.verifier_state.as_ref().unwrap();
        let val_final_claim = self.val_final_claim.as_ref().unwrap();

        let r_address_prime = &r[..r_address.len()];

        let io_mask = RangeMaskPolynomial::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            ),
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout),
        );
        let val_io = ProgramIOPolynomial::new(program_io);

        let eq_eval = EqPolynomial::mle(r_address, r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
        let val_io_eval = val_io.evaluate(r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (*val_final_claim - val_io_eval)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for OutputSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.val_final_claim.is_none());
        let OutputSumcheckProverState {
            val_final,
            val_init,
            ..
        } = self.prover_state.as_ref().unwrap();

        let accumulator = accumulator.expect("accumulator is needed");
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::RamValFinal,
            opening_point.clone(),
            val_final.final_sumcheck_claim(),
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::RamValInit,
            opening_point,
            val_init.final_sumcheck_claim(),
        );

        // TODO(moodlezoup): remove this
        self.val_final_claim = Some(val_final.final_sumcheck_claim());
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator
            .unwrap()
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::RamValFinal, opening_point);
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ValFinalSumcheckClaims<F: JoltField> {
    inc_claim: F,
    wa_claim: F,
}

struct ValFinalSumcheckProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
}

impl<F: JoltField> ValFinalSumcheckProverState<F> {
    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProverState::initialize")]
    fn initialize<PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        output_sumcheck_prover_state: &OutputSumcheckProverState<F>,
    ) -> Self {
        // For RAM, ra = wa but for the sake of intuition we'll call
        // this `write_addresses`
        let write_addresses: Vec<_> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                ) as usize
            })
            .collect();

        // wa(r_address, j)
        let eq_table = &output_sumcheck_prover_state.eq_table;
        let wa_r_address: Vec<_> = write_addresses.par_iter().map(|k| eq_table[*k]).collect();
        let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

        #[cfg(test)]
        {
            let OutputSumcheckProverState {
                val_init,
                val_final,
                ..
            } = &output_sumcheck_prover_state;
            // Check that Val_init(r), wa(r, j), and Inc(j) are consistent with
            // the claim Val_final(r)
            let expected = val_final.final_sumcheck_claim();
            let actual = val_init.final_sumcheck_claim()
                + wa_r_address
                    .par_iter()
                    .enumerate()
                    .map(|(j, wa)| inc.get_coeff(j) * wa)
                    .sum::<F>();
            assert_eq!(
                expected, actual,
                "Val_final(r_address) ≠ Val_init(r_address) + \\sum_j wa(r_address, j) * Inc(j)"
            );
        }

        Self {
            inc,
            wa: wa_r_address.into(),
        }
    }
}

/// This sumcheck virtualizes Val_final(k) as:
/// Val_final(k) = Val_init(k) + \sum_k Inc(j) * wa(k, j)
///   or equivalently:
/// Val_final(k) - Val_init(k) = \sum_k Inc(j) * wa(k, j)
/// We feed the output claim Val_final(r_address) from `OutputSumcheck`
/// into this sumcheck, which reduces it to claims about `Inc` and `wa`.
/// Note that the verifier is assumed to be able to evaluate Val_init
/// on its own.
pub struct ValFinalSumcheck<F: JoltField> {
    T: usize,
    prover_state: Option<ValFinalSumcheckProverState<F>>,
    val_init_eval: F,
    val_final_claim: F,
    output_claims: Option<ValFinalSumcheckClaims<F>>,
}

impl<F: JoltField> ValFinalSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();

        let r_address = state_manager
            .get_opening_point(OpeningsKeys::RamValFinal)
            .unwrap()
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
                let k = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                eq_r_address[k]
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

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

        Self {
            T,
            prover_state: Some(ValFinalSumcheckProverState { wa, inc }),
            val_init_eval: state_manager.get_opening(OpeningsKeys::RamValInit),
            val_final_claim: state_manager.get_opening(OpeningsKeys::RamValFinal),
            output_claims: None,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u32],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        let r_address = state_manager
            .get_opening_point(OpeningsKeys::RamValFinal)
            .unwrap()
            .r;

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address);

        let output_claims = ValFinalSumcheckClaims {
            inc_claim: state_manager.get_opening(OpeningsKeys::ValFinalInc),
            wa_claim: state_manager.get_opening(OpeningsKeys::ValFinalWa),
        };

        Self {
            T,
            prover_state: None,
            val_init_eval: init_eval,
            val_final_claim: state_manager.get_opening(OpeningsKeys::RamValFinal),
            output_claims: Some(output_claims),
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for ValFinalSumcheck<F> {
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
    fn compute_prover_message(&mut self, _: usize) -> Vec<F> {
        const DEGREE: usize = 2;

        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_ref().unwrap();

        let univariate_poly_evals: [F; DEGREE] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = inc.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                let wa_evals = wa.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
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

    fn expected_output_claim(&self, _: &[F]) -> F {
        let ValFinalSumcheckClaims {
            inc_claim,
            wa_claim,
        } = self.output_claims.as_ref().unwrap();
        *inc_claim * wa_claim
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for ValFinalSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.output_claims.is_none());
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_mut().unwrap();
        self.output_claims = Some(ValFinalSumcheckClaims {
            inc_claim: inc.final_sumcheck_claim(),
            wa_claim: wa.final_sumcheck_claim(),
        });

        let accumulator = accumulator.expect("accumulator is needed");
        let r_address = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::RamValFinal)
            .unwrap();
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::ValFinalInc,
            r_cycle_prime,
            inc.final_sumcheck_claim(),
        );
        accumulator.borrow_mut().append_virtual(
            OpeningsKeys::ValFinalWa,
            wa_opening_point,
            wa.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        let r_address = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::RamValFinal)
            .unwrap();
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::ValFinalInc, r_cycle_prime);
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::ValFinalWa, wa_opening_point);
    }
}
