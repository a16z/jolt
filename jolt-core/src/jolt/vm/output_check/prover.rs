use crate::jolt::vm::output_check::{
    OutputProof, OutputSumcheck, OutputSumcheckProverState, OutputSumcheckVerifierState,
    ValFinalSumcheck, ValFinalSumcheckClaims, ValFinalSumcheckProverState,
};
use crate::{
    field::JoltField,
    jolt::{
        vm::{ram::remap_address, JoltProverPreprocessing},
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation},
        program_io_polynomial::ProgramIOPolynomial,
        range_mask_polynomial::RangeMaskPolynomial,
    },
    subprotocols::{shout::sparse_dense::ExpandingTable, sumcheck::BatchableSumcheckInstance},
    utils::{math::Math, thread::drop_in_background_thread, transcript::Transcript},
};
use common::constants::RAM_START_ADDRESS;
use rayon::prelude::*;
use tracer::{instruction::RV32IMCycle, JoltDevice};

impl<F: JoltField> OutputSumcheckProverState<F> {
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

impl<F: JoltField> OutputSumcheck<F> {
    #[tracing::instrument(skip_all, name = "OutputSumcheck")]
    pub fn prove<ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        initial_ram_state: Vec<u32>,
        final_ram_state: Vec<u32>,
        program_io: &JoltDevice,
        r_address: &[F],
        transcript: &mut ProofTranscript,
    ) -> OutputProof<F, ProofTranscript> {
        let K = final_ram_state.len();
        let T = trace.len();

        let output_sumcheck_prover_state = OutputSumcheckProverState::initialize(
            initial_ram_state,
            final_ram_state,
            program_io,
            r_address,
        );
        let mut output_sumcheck = OutputSumcheck {
            K,
            T,
            verifier_state: None,
            prover_state: Some(output_sumcheck_prover_state),
            val_final_claim: None,
        };
        let (output_sumcheck_proof, _r_address) = output_sumcheck.prove_single(transcript);
        let output_sumcheck_prover_state = output_sumcheck.prover_state.as_ref().unwrap();

        let val_final_prover_state = ValFinalSumcheckProverState::initialize(
            preprocessing,
            trace,
            output_sumcheck_prover_state,
        );
        let mut val_final_sumcheck = ValFinalSumcheck {
            T,
            prover_state: Some(val_final_prover_state),
            val_init_eval: output_sumcheck_prover_state.val_init.final_sumcheck_claim(),
            val_final_claim: output_sumcheck.val_final_claim.unwrap(),
            output_claims: None,
        };
        let (val_final_sumcheck_proof, _r_cycle) = val_final_sumcheck.prove_single(transcript);
        let output_claims = std::mem::take(val_final_sumcheck.output_claims.as_mut().unwrap());
        let val_final_claim = val_final_sumcheck.val_final_claim;

        drop_in_background_thread((output_sumcheck, val_final_sumcheck));

        OutputProof {
            output_sumcheck_proof,
            val_final_sumcheck_proof,
            val_final_claim,
            output_claims,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for OutputSumcheck<F>
{
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

    fn cache_openings(&mut self) {
        debug_assert!(self.val_final_claim.is_none());
        let OutputSumcheckProverState { val_final, .. } = self.prover_state.as_ref().unwrap();
        self.val_final_claim = Some(val_final.final_sumcheck_claim());
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

impl<F: JoltField> ValFinalSumcheckProverState<F> {
    fn initialize<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    >(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
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

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ValFinalSumcheck<F>
{
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

    fn cache_openings(&mut self) {
        debug_assert!(self.output_claims.is_none());
        let ValFinalSumcheckProverState { inc, wa, .. } = self.prover_state.as_mut().unwrap();
        self.output_claims = Some(ValFinalSumcheckClaims {
            inc_claim: inc.final_sumcheck_claim(),
            wa_claim: wa.final_sumcheck_claim(),
        });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let ValFinalSumcheckClaims {
            inc_claim,
            wa_claim,
        } = self.output_claims.as_ref().unwrap();
        *inc_claim * wa_claim
    }
}
