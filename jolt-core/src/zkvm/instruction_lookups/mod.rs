use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;

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
    utils::thread::unsafe_allocate_zero_vec,
    zkvm::{
        dag::{stage::SumcheckStages, state_manager::StateManager},
        instruction::LookupQuery,
        instruction_lookups::{
            booleanity::InstructionBooleanitySumcheck, hamming_weight::HammingWeightSumcheck,
            ra_virtual::RaSumcheck, read_raf_checking::ReadRafSumcheck,
        },
        witness::VirtualPolynomial,
    },
};

pub mod booleanity;
pub mod hamming_weight;
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
        let hamming_weight = HammingWeightSumcheck::new_prover(sm, G);

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
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

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
        let F = compute_ra_evals(trace, &eq_r_cycle);
        let booleanity = InstructionBooleanitySumcheck::new_prover(sm, F);

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
        let booleanity = InstructionBooleanitySumcheck::new_verifier(sm);
        vec![Box::new(ra_virtual), Box::new(booleanity)]
    }
}

#[inline(always)]
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
