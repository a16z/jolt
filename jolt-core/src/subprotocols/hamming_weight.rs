use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K},
    },
};

use crate::subprotocols::sumcheck::SumcheckInstance;

#[derive(Allocative)]
pub struct HammingWeightProverState<F: JoltField> {
    /// ra polynomials
    pub ra: Vec<MultilinearPolynomial<F>>,
}

/// Type of hamming weight sumcheck
#[derive(Clone, Debug, Allocative)]
pub enum HammingWeightType {
    Ram,
    Bytecode { d: usize, log_K: usize },
    Instruction,
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    hamming_type: HammingWeightType,
    d: usize,
    num_rounds: usize,
    gamma_powers: Vec<F>,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    /// Create a new prover instance
    pub fn new_prover<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        hamming_type: HammingWeightType,
        state_manager: &mut StateManager<'_, F, T, PCS>,
        G: Option<Vec<Vec<F>>>, // Some for bytecode/instruction, None for RAM (computed internally)
    ) -> Self {
        let (d, num_rounds, log_chunk_size) = match &hamming_type {
            HammingWeightType::Ram => {
                let d = compute_d_parameter(state_manager.ram_K);
                (d, DTH_ROOT_OF_K.log_2(), DTH_ROOT_OF_K.log_2())
            }
            HammingWeightType::Bytecode { d, log_K } => {
                let log_K_chunk = log_K.div_ceil(*d);
                (*d, log_K_chunk, log_K_chunk)
            }
            HammingWeightType::Instruction => {
                // D = 16, LOG_K_CHUNK = LOG_K / D = 128 / 16 = 8
                // K_CHUNK = 2^8 = 256
                (16, 8, 8)
            }
        };

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // Compute ra polynomials
        let ra = match &hamming_type {
            HammingWeightType::Ram => {
                // Compute F arrays for RAM
                let (_, trace, program_io, _) = state_manager.get_prover_data();
                let memory_layout = &program_io.memory_layout;

                let T = trace.len();
                let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
                let chunk_size = (T / num_chunks).max(1);

                let (r_cycle, _) = state_manager.get_virtual_polynomial_opening(
                    VirtualPolynomial::RamHammingWeight,
                    SumcheckId::RamHammingBooleanity,
                );

                let eq_r_cycle = EqPolynomial::evals(&r_cycle.r);

                let mut F_arrays = Vec::with_capacity(d);
                for i in 0..d {
                    let F: Vec<F> = trace
                        .par_chunks(chunk_size)
                        .enumerate()
                        .map(|(chunk_index, trace_chunk)| {
                            let mut local_array = unsafe_allocate_zero_vec(DTH_ROOT_OF_K);
                            let mut j = chunk_index * chunk_size;
                            for cycle in trace_chunk {
                                if let Some(address) = remap_address(
                                    cycle.ram_access().address() as u64,
                                    memory_layout,
                                ) {
                                    let address_i = (address
                                        >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
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
                                running
                                    .par_iter_mut()
                                    .zip(new.into_par_iter())
                                    .for_each(|(x, y)| *x += y);
                                running
                            },
                        );
                    F_arrays.push(F);
                }

                // Verify each array has the right size
                for (i, arr) in F_arrays.iter().enumerate() {
                    assert_eq!(arr.len(), DTH_ROOT_OF_K, "F_array[{i}] has wrong size");
                    assert_eq!(
                        arr.len(),
                        1 << log_chunk_size,
                        "F_array[{i}] size doesn't match log_chunk_size"
                    );
                }

                F_arrays
                    .into_iter()
                    .map(MultilinearPolynomial::from)
                    .collect()
            }
            HammingWeightType::Bytecode { .. } | HammingWeightType::Instruction => {
                // Use provided G arrays
                let arrays = G.expect("G arrays must be provided for bytecode/instruction");

                // Verify each array has the right size for the number of rounds
                for (i, arr) in arrays.iter().enumerate() {
                    assert_eq!(
                        arr.len(),
                        1 << log_chunk_size,
                        "G array[{}] has size {} but expected {} (2^{})",
                        i,
                        arr.len(),
                        1 << log_chunk_size,
                        log_chunk_size
                    );
                }

                arrays
                    .into_iter()
                    .map(MultilinearPolynomial::from)
                    .collect()
            }
        };

        Self {
            hamming_type,
            d,
            num_rounds,
            gamma_powers,
            prover_state: Some(HammingWeightProverState { ra }),
        }
    }

    /// Create a new verifier instance
    pub fn new_verifier<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        hamming_type: HammingWeightType,
        state_manager: &mut StateManager<'_, F, T, PCS>,
    ) -> Self {
        let (d, num_rounds) = match &hamming_type {
            HammingWeightType::Ram => {
                let d = compute_d_parameter(state_manager.ram_K);
                (d, DTH_ROOT_OF_K.log_2())
            }
            HammingWeightType::Bytecode { d, log_K } => {
                let log_K_chunk = log_K.div_ceil(*d);
                (*d, log_K_chunk)
            }
            HammingWeightType::Instruction => {
                // D = 16, LOG_K_CHUNK = 8
                (16, 8)
            }
        };

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            hamming_type,
            d,
            num_rounds,
            gamma_powers,
            prover_state: None,
        }
    }

    fn polynomial_type(&self, i: usize) -> CommittedPolynomial {
        match self.hamming_type {
            HammingWeightType::Ram => CommittedPolynomial::RamRa(i),
            HammingWeightType::Bytecode { .. } => CommittedPolynomial::BytecodeRa(i),
            HammingWeightType::Instruction => CommittedPolynomial::InstructionRa(i),
        }
    }

    fn sumcheck_id(&self) -> SumcheckId {
        match self.hamming_type {
            HammingWeightType::Ram => SumcheckId::RamHammingWeight,
            HammingWeightType::Bytecode { .. } => SumcheckId::BytecodeHammingWeight,
            HammingWeightType::Instruction => SumcheckId::InstructionHammingWeight,
        }
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        match self.hamming_type {
            HammingWeightType::Ram => {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::RamHammingWeight,
                        SumcheckId::RamHammingBooleanity,
                    )
                    .0
                    .r
            }
            HammingWeightType::Bytecode { .. } | HammingWeightType::Instruction => {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupOutput,
                        SumcheckId::SpartanOuter,
                    )
                    .0
                    .r
            }
        }
    }
}

// Direct implementation of SumcheckInstance for HammingWeightSumcheck
impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1 // Hamming weight sumchecks always have degree 1
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        match self.hamming_type {
            HammingWeightType::Ram => {
                // RAM has special input claim handling
                let acc = acc.unwrap().borrow();
                let (_, hamming_booleanity_claim) = acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::RamHammingWeight,
                    SumcheckId::RamHammingBooleanity,
                );
                hamming_booleanity_claim * self.gamma_powers.iter().sum::<F>()
            }
            _ => {
                // Default: sum of gamma powers
                self.gamma_powers.iter().sum()
            }
        }
    }

    #[tracing::instrument(skip_all, name = "HammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let prover_msg = ps
            .ra
            .par_iter()
            .zip(self.gamma_powers.par_iter())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| {
                        running + new.as_unreduced_ref()
                    })
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
            })
            .reduce(F::Unreduced::zero, |running, new| running + new);

        vec![F::from_montgomery_reduce(prover_msg)]
    }

    #[tracing::instrument(skip_all, name = "HammingWeight::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(ps) = &mut self.prover_state {
            ps.ra
                .par_iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
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

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        ra_claims
            .iter()
            .zip(self.gamma_powers.iter())
            .map(|(ra_claim, gamma)| *ra_claim * gamma)
            .sum()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().cloned().rev().collect())
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

        let claims: Vec<F> = ps.ra.iter().map(|ra| ra.final_sumcheck_claim()).collect();

        let r_cycle = self.get_r_cycle(&*accumulator.borrow());

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d).map(|i| self.polynomial_type(i)).collect(),
            self.sumcheck_id(),
            opening_point.r,
            r_cycle,
            claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_cycle = self.get_r_cycle(&*accumulator.borrow());
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d).map(|i| self.polynomial_type(i)).collect(),
            self.sumcheck_id(),
            r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
