use jolt_field::JoltField;
use jolt_sumcheck::{MemberFinish, MemberRound, RoundScheduler, SumcheckError};
use rayon::prelude::*;
use tracing::Span;

use crate::backend::{BuildRoundScheduler, ProofSession};

/// Runs at most two independent member calls together; transcript updates remain
/// in the batch engine. This caps member calls, not their internal allocations
/// or background drops.
/// An error still selects the first failing member in declaration order, but
/// other members may already have run before that error is returned.
pub(super) struct ParallelMembers;

impl<F: JoltField> BuildRoundScheduler<F> for ParallelMembers {
    fn build(&self, _session: &mut ProofSession) -> Box<dyn RoundScheduler<F>> {
        Box::new(Self)
    }
}

impl<F: JoltField> RoundScheduler<F> for ParallelMembers {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        // Large rounds already expose parallel work within each member. Running
        // their lazy materializations together also raises peak memory. Restrict
        // member concurrency to smaller rounds, using the actual runtime shape.
        // This scheduling threshold is not an allocation bound.
        const MAX_PARALLEL_MEMBER_ROUNDS: usize = 18;
        if work.len() <= 1
            || rayon::current_num_threads() == 1
            || work.iter().any(|item| {
                item.member.num_rounds().saturating_sub(item.local_round)
                    > MAX_PARALLEL_MEMBER_ROUNDS
            })
        {
            for item in work {
                item.run()?;
            }
            return Ok(());
        }
        let parallel: Option<Vec<_>> = work
            .iter_mut()
            .map(|item| {
                Some((
                    item.member.parallel_rounds()?,
                    item.bind,
                    item.local_round,
                    item.claim,
                ))
            })
            .collect();
        if let Some(mut parallel) = parallel {
            let span = Span::current();
            let mut results = Vec::with_capacity(parallel.len());
            for pair in parallel.chunks_mut(2) {
                let pair_results: Vec<_> = pair
                    .par_iter_mut()
                    .map(|(member, bind, round, claim)| {
                        span.in_scope(|| member.prove_round(*bind, *round, *claim))
                    })
                    .collect();
                results.extend(pair_results);
            }
            drop(parallel);
            for (item, result) in work.iter_mut().zip(results) {
                item.message = Some(result?);
            }
        } else {
            for item in work {
                item.run()?;
            }
        }
        Ok(())
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        if finishes.len() <= 1 || rayon::current_num_threads() == 1 {
            for item in finishes {
                item.run()?;
            }
            return Ok(());
        }
        let parallel: Option<Vec<_>> = finishes
            .iter_mut()
            .map(|item| Some((item.member.parallel_rounds()?, item.bind)))
            .collect();
        if let Some(mut parallel) = parallel {
            let span = Span::current();
            for pair in parallel.chunks_mut(2) {
                let results: Vec<_> = pair
                    .par_iter_mut()
                    .map(|(member, bind)| span.in_scope(|| member.finish_rounds(*bind)))
                    .collect();
                for result in results {
                    result?;
                }
            }
        } else {
            for item in finishes {
                item.run()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
