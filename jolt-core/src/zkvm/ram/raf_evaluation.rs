use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::dag::state_manager::StateManager,
    zkvm::{ram::remap_address, witness::VirtualPolynomial},
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    raf_claim: F,
}

pub struct RafEvaluationProverState<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
}

pub struct RafEvaluationSumcheck<F: JoltField> {
    /// The initial claim (raf_claim)
    input_claim: F,
    /// log K (number of rounds)
    log_K: usize,
    /// Start address for unmap polynomial
    start_address: u64,
    prover_state: Option<RafEvaluationProverState<F>>,
    /// Cached ra_claim after sumcheck completion
    cached_claim: Option<F>,
}

impl<F: JoltField> RafEvaluationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        T: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let (r_cycle, raf_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );

        // TODO(moodlezoup): reuse
        let eq_r_cycle: Vec<F> = EqPolynomial::evals(&r_cycle.r);

        let ra_evals: Vec<F> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    if let Some(k) =
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                    {
                        result[k as usize] += eq_r_cycle[j];
                    }
                    j += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );
        let ra = MultilinearPolynomial::from(ra_evals);
        let unmap = UnmapRamAddressPolynomial::new(K.log_2(), memory_layout.input_start);

        Self {
            input_claim: raf_claim,
            log_K: K.log_2(),
            start_address: memory_layout.input_start,
            prover_state: Some(RafEvaluationProverState { ra, unmap }),
            cached_claim: None,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, program_io, _) = state_manager.get_verifier_data();
        let raf_claim = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .1;
        let ra_claim = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation)
            .1;

        Self {
            input_claim: raf_claim,
            log_K: K.log_2(),
            start_address: program_io.memory_layout.input_start,
            prover_state: None,
            cached_claim: Some(ra_claim),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RafEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = prover_state
                    .ra
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let unmap_evals =
                    prover_state
                        .unmap
                        .sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);

                // Compute the product evaluations
                [ra_evals[0] * unmap_evals[0], ra_evals[1] * unmap_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            rayon::join(
                || prover_state.ra.bind_parallel(r_j, BindingOrder::HighToLow),
                || {
                    prover_state
                        .unmap
                        .bind_parallel(r_j, BindingOrder::HighToLow)
                },
            );
        }
    }

    fn expected_output_claim(
        &self,
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // Compute unmap evaluation at r
        let unmap_eval = UnmapRamAddressPolynomial::new(self.log_K, self.start_address).evaluate(r);

        // Return unmap(r) * ra(r)
        let ra_claim = self.cached_claim.expect("ra_claim not cached");
        unmap_eval * ra_claim
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            prover_state.ra.final_sumcheck_claim(),
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
        );
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::transcripts::Blake2bTranscript;
//     use ark_bn254::Fr;

//     #[test]
//     fn test_raf_evaluation_no_ops() {
//         const K: usize = 1 << 16;
//         const T: usize = 1 << 8;

//         let memory_layout = MemoryLayout {
//             max_input_size: 256,
//             max_output_size: 256,
//             input_start: 0x80000000,
//             input_end: 0x80000100,
//             output_start: 0x80001000,
//             output_end: 0x80001100,
//             stack_size: 1024,
//             stack_end: 0x7FFFFF00,
//             memory_size: 0x10000,
//             memory_end: 0x80010000,
//             panic: 0x80002000,
//             termination: 0x80002001,
//             io_end: 0x80002002,
//         };

//         // Create trace with only no-ops (address = 0)
//         let mut trace = Vec::new();
//         for i in 0..T {
//             trace.push(RV32IMCycle::NoOp(i));
//         }

//         let mut prover_transcript = Blake2bTranscript::new(b"test_no_ops");
//         let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

//         // Prove
//         let proof =
//             RafEvaluationProof::prove(&trace, &memory_layout, r_cycle, K, &mut prover_transcript);

//         // Verify
//         let mut verifier_transcript = Blake2bTranscript::new(b"test_no_ops");
//         let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

//         let r_address_result = proof.verify(K, &mut verifier_transcript, &memory_layout);

//         assert!(
//             r_address_result.is_ok(),
//             "No-op RAF evaluation verification failed"
//         );
//     }
// }
