use crate::subprotocols::{
    booleanity::BooleanitySumcheck,
    hamming_weight::{HammingWeightSumcheck, HammingWeightType},
};
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::{stage::SumcheckStages, state_manager::StateManager},
        instruction::LookupQuery,
        instruction_lookups::{ra_virtual::RaSumcheck, read_raf_checking::ReadRafSumcheck},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;
pub mod ra_virtual;
pub mod read_raf_checking;

const LOG_K: usize = XLEN * 2;
const PHASES: usize = 8;
pub const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;
pub const D: usize = 16;
pub const LOG_K_CHUNK: usize = LOG_K / D;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

#[derive(Default)]
pub struct LookupsDag<F: JoltField> {
    G: Option<[Vec<F>; D]>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for LookupsDag<F>
{
    fn stage3_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        // Ensure G is available even if an earlier stage did not set it
        let G = if let Some(G) = self.G.take() {
            G
        } else {
            let (_, trace, _, _) = sm.get_prover_data();
            let r_cycle = sm
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanOuter,
                )
                .0
                .r
                .clone();
            let eq_r_cycle = EqPolynomial::evals(&r_cycle);
            compute_ra_evals(trace, &eq_r_cycle)
        };
        let hamming_weight = HammingWeightSumcheck::new_prover(
            HammingWeightType::Instruction,
            sm,
            G.into_iter().collect(),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "Instruction execution HammingWeightSumcheck",
                &hamming_weight,
            );
        }

        vec![Box::new(hamming_weight)]
    }

    fn stage3_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        let hamming_weight =
            HammingWeightSumcheck::new_verifier(HammingWeightType::Instruction, sm);

        vec![Box::new(hamming_weight)]
    }

    fn stage5_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        let read_raf = ReadRafSumcheck::new_prover(sm);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Instruction execution ReadRafSumcheck", &read_raf);
        }

        vec![Box::new(read_raf)]
    }

    fn stage5_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        let read_raf = ReadRafSumcheck::new_verifier(sm);

        vec![Box::new(read_raf)]
    }

    fn stage6_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        let ra_virtual = RaSumcheck::new_prover(sm);
        let (_, trace, _, _) = sm.get_prover_data();
        let r_cycle = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);
        let G = compute_ra_evals(trace, &eq_r_cycle);
        let H_indices = compute_instruction_h_indices(trace);

        const D_CONST: usize = D;
        const LOG_K_CHUNK_CONST: usize = LOG_K_CHUNK;
        let log_t = trace.len().log_2();

        let gamma: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(D_CONST);

        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(LOG_K_CHUNK_CONST);
        let polynomial_types: Vec<CommittedPolynomial> = (0..D_CONST)
            .map(CommittedPolynomial::InstructionRa)
            .collect();

        let booleanity = BooleanitySumcheck::new_prover(
            D_CONST,
            LOG_K_CHUNK_CONST,
            log_t,
            r_cycle,
            r_address,
            gamma,
            G.to_vec(),
            H_indices,
            polynomial_types,
            SumcheckId::InstructionBooleanity,
            Some(VirtualPolynomial::LookupOutput),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "Instruction execution RAVirtual sumcheck",
                &ra_virtual,
            );
            print_data_structure_heap_usage(
                "Instruction execution BooleanitySumcheck",
                &booleanity,
            );
        }

        vec![Box::new(ra_virtual), Box::new(booleanity)]
    }

    fn stage6_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, T>>> {
        let ra_virtual = RaSumcheck::new_verifier(sm);

        let (_, _, T_val) = sm.get_verifier_data();
        const D_CONST: usize = D;
        const LOG_K_CHUNK_CONST: usize = LOG_K_CHUNK;
        let log_t = T_val.log_2();

        let gamma: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(D_CONST);

        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(LOG_K_CHUNK_CONST);

        let r_cycle = Vec::new();

        let polynomial_types: Vec<CommittedPolynomial> = (0..D_CONST)
            .map(CommittedPolynomial::InstructionRa)
            .collect();

        let booleanity = BooleanitySumcheck::new_verifier(
            D_CONST,
            LOG_K_CHUNK_CONST,
            log_t,
            r_cycle,
            r_address,
            gamma,
            polynomial_types,
            SumcheckId::InstructionBooleanity,
            Some(VirtualPolynomial::LookupOutput),
        );

        vec![Box::new(ra_virtual), Box::new(booleanity)]
    }
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_instruction_h_indices")]
fn compute_instruction_h_indices(trace: &[Cycle]) -> Vec<Vec<Option<u8>>> {
    (0..D)
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u128) as u8)
                })
                .collect()
        })
        .collect()
}

#[inline(always)]
#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(trace: &[Cycle], eq_r_cycle: &[F]) -> [Vec<F>; D] {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: [Vec<F>; D] =
                std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK));
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                for i in (0..D).rev() {
                    let k = lookup_index % K_CHUNK as u128;
                    result[i][k as usize] += eq_r_cycle[j];
                    lookup_index >>= LOG_K_CHUNK;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK)),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}
