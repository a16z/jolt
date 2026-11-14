//! The register read/write check (stage 4) of the streaming register family.
//!
//! Cycle-first binding keeps no per-cycle state: the default kernel runs over
//! a streamed source ([`CycleSource::Streamed`]), whose cycle rounds read the
//! witness's resident rows in task-partitioned passes
//! ([`StreamedCycleState`]) until a level fits [`DENSE_TABLE_CAP`], and whose
//! two read-operand openings come from one more row walk
//! ([`operand_claims`]). Address-first binding keeps the default kernel's
//! collected state. Either way no `rd` indices carry to stage 5, whose value
//! evaluation
//! ([`StreamingRegistersValEvaluation`](crate::optimized::registers_val_evaluation::StreamingRegistersValEvaluation))
//! reads the rows.

use jolt_field::{Accumulator, JoltField};
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::address::OperandEq;
use super::rows::StreamedRows;
use super::sparse::StreamedCycleState;
use super::{
    read_write_order, CycleSource, OptimizedRegistersReadWrite, ReadWriteKernel, RegisterCycleRow,
};
use crate::optimized::read_write::ReadWriteOrder;
use crate::optimized::streaming::{task_ranges, DENSE_TABLE_CAP, VISIT_ROWS};
use crate::optimized::support::{try_par_sum_vecs, BundleStore};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

// The operand walk weights each visit callback by one `e_hi` entry, so a
// callback of at most one block of aligned rows must lie inside that block.
const _: () = assert!(VISIT_ROWS.is_power_of_two());

pub struct StreamingRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for StreamingRegistersReadWrite {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.register_dimensions().log_t();
        // Address-first binding keeps the default kernel, and so does a
        // two-cycle domain: its only cycle round is the counting pass, so no
        // round could fill a level.
        if read_write_order(&inputs)? == ReadWriteOrder::AddressFirst || log_t == 1 {
            return OptimizedRegistersReadWrite::prepare_kernel(None, witness, inputs);
        }
        let rows = BundleStore::resident(witness, log_t, "register read/write trace rows")?;
        Ok(Box::new(ReadWriteKernel::streamed(
            rows,
            &inputs,
            DENSE_TABLE_CAP,
        )?))
    }
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// The cycle-first kernel over `rows`, whose inputs passed
    /// [`read_write_order`]; `cap` bounds the bytes of the filled level's
    /// entries and increments, production passing [`DENSE_TABLE_CAP`].
    fn streamed(
        rows: BundleStore<RegisterCycleRow>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
        cap: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = u32::try_from(inputs.relation.register_dimensions().log_t())
            .ok()
            .filter(|&bits| bits <= u32::BITS)
            .and_then(|bits| 1usize.checked_shl(bits))
            .ok_or(KernelError::InvariantViolation {
                reason: "cycle count exceeds the seed entries' packed u32 row domain",
            })?;
        let rows = StreamedRows::new(rows, cycles)?;
        let state = StreamedCycleState::new(inputs.challenges.gamma, cap);
        Ok(Self::new(CycleSource::Streamed { state, rows }, inputs))
    }
}

/// The default kernel's two read-operand claims (`one_hot_operand_claims`,
/// [`OperandEq`]) from one walk over `rows` in place of its index columns:
/// the `e_hi` blocks that hold a prefix row. Tasks hold whole blocks, so each
/// callback lies in one block.
pub(super) fn operand_claims<F: JoltField>(
    rows: &StreamedRows,
    r_address: &[F],
    r_cycle: &[F],
) -> Result<(F, F), WitnessError> {
    let eq = OperandEq::new(r_address, r_cycle);
    let block = 1usize << eq.cycle_bits_in_lo;
    let ranges = task_ranges(rows.prefix_end(block, 1 << r_cycle.len()), block);
    let claims = try_par_sum_vecs(ranges.len(), 2, |task| -> Result<_, WitnessError> {
        let mut claims = vec![F::zero(); 2];
        rows.store.visit(
            ranges[task].clone(),
            VISIT_ROWS.min(block),
            |start, rows| {
                let mut sums = [F::Accumulator::default(); 2];
                for (j, row) in (start..).zip(rows) {
                    if let Some((rs1, _)) = row.rs1 {
                        sums[0].add(eq.lo[eq.low_index(j, rs1)]);
                    }
                    if let Some((rs2, _)) = row.rs2 {
                        sums[1].add(eq.lo[eq.low_index(j, rs2)]);
                    }
                }
                let e_hi = eq.hi[start >> eq.cycle_bits_in_lo];
                claims[0] += e_hi * sums[0].reduce();
                claims[1] += e_hi * sums[1].reduce();
                Ok(())
            },
        )?;
        Ok(claims)
    })?;
    Ok((claims[0], claims[1]))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;
