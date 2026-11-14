//! Streaming instruction read+RAF checking: the default kernel over the
//! witness's resident rows instead of the packed row cache, which this slot
//! neither collects nor parks. Every address phase scans the rows in one pass,
//! bucketing each block as it arrives, and condenses the default kernel's
//! cycle weights in place. The cycle rounds gather the virtual RA polynomials
//! from the rows ([`StreamedFoldedRa`]) until the bind at which the default
//! kernel's lazy tables materialize; from that bind on the default kernel's
//! rounds run over the family's dense state. The combined value keeps its
//! default tables over the one-byte claim column.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_field::{Accumulator, JoltField};
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::witnesses::LookupIndex;
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    claim_byte, CycleState, OptimizedInstructionReadRafKernel, PhaseScan, PhaseSums,
    RangePrefixSums, ScanBlock, ScanRow, ScanRows, TaskScan, SCAN_BLOCK, SCAN_TASKS_PER_THREAD,
    UNKNOWN_TABLE,
};
use crate::optimized::constant_suffix::ConstantSuffix;
use crate::optimized::streaming::{RowIndexSource, StreamedFoldedRa, VISIT_ROWS};
use crate::optimized::support::{
    accumulate_product_grid, eq_round_error, product_grid_scratch_len, scan_chunk_size,
    BundleStore, GruenRoundMessage, PrefixCycleWeights,
};
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const ROWS: &str = "instruction read-RAF trace rows";

/// The streaming stage-5 instruction read+RAF slot.
pub struct StreamingInstructionReadRaf;

impl<F: JoltField> PrepareKernel<F, InstructionReadRaf<F>> for StreamingInstructionReadRaf {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let dimensions = inputs.relation.dimensions();
        let log_t = dimensions.log_t();
        Ok(Box::new(OptimizedInstructionReadRafKernel::streamed(
            dimensions,
            &inputs.points.lookup_output,
            BundleStore::resident(witness, log_t, ROWS)?,
            BundleStore::resident(witness, log_t, ROWS)?,
            inputs.challenges.gamma,
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub(crate) struct LookupIndexRow {
    pub(crate) lookup_index: LookupIndex,
}

/// Column `i` is the `i`-th most significant `bits`-bit digit of a cycle's
/// lookup index, always hot.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct LookupIndexDigits {
    columns: usize,
    bits: usize,
}

impl LookupIndexDigits {
    /// `bits` is at most 32: stage 5 forms 8- and 16-bit columns, and the
    /// stage-6b kernel's constructor refuses wider committed chunks.
    pub(crate) fn new(columns: usize, bits: usize) -> Self {
        Self { columns, bits }
    }
}

impl RowIndexSource for LookupIndexDigits {
    type Row = LookupIndexRow;

    #[inline]
    fn for_each_hot(&self, row: &LookupIndexRow, mut visit: impl FnMut(usize, usize)) {
        let mask = (1u128 << self.bits) - 1;
        let mut index = row.lookup_index.0;
        for column in (0..self.columns).rev() {
            visit(column, (index & mask) as usize);
            index >>= self.bits;
        }
    }
}

impl ScanRow for InstructionReadRafWitness {
    #[inline(always)]
    fn lookup_index(&self) -> u128 {
        self.lookup_index.0
    }

    #[inline]
    fn table_index(&self) -> Option<usize> {
        self.table_index.0
    }

    #[inline]
    fn raf_flag(&self) -> bool {
        self.raf_flag.0
    }
}

pub(super) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS }
}

/// The claim bytes of rows `0..len` and the lookup tables those rows select
/// (bit `LookupTableKind::index()`), from one pass over the rows.
fn claim_bytes<F: JoltField>(
    rows: &BundleStore<InstructionReadRafWitness>,
    len: usize,
) -> Result<(Vec<u8>, u64), KernelError<F>> {
    let mut bytes = vec![0u8; len];
    let task_rows = scan_chunk_size(len);
    let task = |(task, bytes): (usize, &mut [u8])| -> Result<u64, KernelError<F>> {
        let start = task * task_rows;
        let mut present = 0u64;
        let mut unknown = false;
        rows.visit(start..start + bytes.len(), VISIT_ROWS, |first, rows| {
            for (byte, row) in bytes[first - start..].iter_mut().zip(rows) {
                if let Some(table) = row.table_index.0 {
                    if table >= LookupTableKind::<RISCV_XLEN>::COUNT {
                        unknown = true;
                        continue;
                    }
                    present |= 1 << table;
                }
                *byte = claim_byte(row.table_index.0, row.raf_flag.0);
            }
            Ok(())
        })?;
        if unknown {
            return Err(KernelError::InvariantViolation {
                reason: UNKNOWN_TABLE,
            });
        }
        Ok(present)
    };
    #[cfg(feature = "parallel")]
    let present = bytes
        .par_chunks_mut(task_rows)
        .enumerate()
        .map(task)
        .try_reduce(|| 0, |a, b| Ok(a | b))?;
    #[cfg(not(feature = "parallel"))]
    let present = bytes
        .chunks_mut(task_rows)
        .enumerate()
        .map(task)
        .try_fold(0, |present, part| part.map(|part| present | part))?;
    Ok((bytes, present))
}

impl<F: JoltField> PhaseScan<'_, F> {
    /// [`Self::run`] over the resident rows `0..u_evals.len()`, in tasks of
    /// the same consecutive blocks.
    pub(super) fn run_resident(
        &self,
        rows: &BundleStore<InstructionReadRafWitness>,
        range: &mut [Option<RangePrefixSums<F>>],
        u_evals: &mut [F],
    ) -> Result<PhaseSums<F>, WitnessError> {
        let task_blocks =
            scan_chunk_size(u_evals.len()).div_ceil(SCAN_TASKS_PER_THREAD * SCAN_BLOCK);
        let task_rows = task_blocks * SCAN_BLOCK;
        #[cfg(feature = "parallel")]
        {
            range
                .par_chunks_mut(task_blocks)
                .zip(u_evals.par_chunks_mut(task_rows))
                .enumerate()
                .map(|(task, (range, u_evals))| {
                    self.scan_resident(rows, task * task_rows, range, u_evals)
                })
                .try_reduce(PhaseSums::zero, |a, b| Ok(a.merge(b)))
        }
        #[cfg(not(feature = "parallel"))]
        {
            range
                .chunks_mut(task_blocks)
                .zip(u_evals.chunks_mut(task_rows))
                .enumerate()
                .map(|(task, (range, u_evals))| {
                    self.scan_resident(rows, task * task_rows, range, u_evals)
                })
                .try_fold(PhaseSums::zero(), |sums, part| {
                    part.map(|part| sums.merge(part))
                })
        }
    }

    /// One task's visit of the rows from `start` that `u_evals` weighs,
    /// bucketing each block as it arrives. No block coalesces repeated
    /// lookups, which takes a layout that outlives the scan.
    fn scan_resident(
        &self,
        rows: &BundleStore<InstructionReadRafWitness>,
        start: usize,
        range: &mut [Option<RangePrefixSums<F>>],
        u_evals: &mut [F],
    ) -> Result<PhaseSums<F>, WitnessError> {
        let mut task = TaskScan::new(self);
        rows.visit(start..start + u_evals.len(), SCAN_BLOCK, |first, rows| {
            let offset = first - start;
            task.block(
                &ScanBlock::new(rows),
                &mut range[offset / SCAN_BLOCK],
                rows,
                &mut u_evals[offset..offset + rows.len()],
            );
            Ok(())
        })?;
        Ok(task.finish())
    }
}

impl<F: JoltField> OptimizedInstructionReadRafKernel<F> {
    /// The kernel over a witness's resident rows: `rows` and `lookup_indices`
    /// are two typed views of the same `2^log_t` cycles. The claim column is
    /// packed here, in the pass that finds the present lookup tables, so no
    /// pass is left for the address/cycle handoff.
    pub(super) fn streamed(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: BundleStore<InstructionReadRafWitness>,
        lookup_indices: BundleStore<LookupIndexRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        Self::check_geometry(dimensions)?;
        let log_t = dimensions.log_t();
        if r_reduction.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction point".to_owned(),
                expected: log_t,
                got: r_reduction.len(),
            });
        }
        let cycles = u32::try_from(log_t)
            .ok()
            .and_then(|log_t| 1usize.checked_shl(log_t))
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: ROWS,
                reason: format!("a 2^{log_t}-cycle stage exceeds the address space"),
            })?;
        let prefix = rows.access().constant_suffix_start(cycles)?;
        let u_evals = PrefixCycleWeights::new(r_reduction, prefix).to_dense();
        let (claim_bytes, present_tables) = claim_bytes(&rows, u_evals.len())?;
        Self::start(
            dimensions,
            r_reduction,
            gamma,
            ScanRows::Resident {
                rows,
                lookup_indices,
            },
            u_evals,
            Arc::new(ConstantSuffix::new(claim_bytes, cycles)),
            present_tables,
        )
    }

    /// A cycle round before the virtual RA polynomials materialize: the
    /// default round's `q` samples, from the family's pair sums. A failed
    /// `q(0)` pass is held and returned, so its placeholder zero never
    /// reaches a round message.
    pub(super) fn streamed_cycle_message(
        &self,
        cycle: &CycleState<F>,
        ra: &StreamedFoldedRa<F, LookupIndexDigits>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let factors = 1 + self.dimensions.num_virtual_ra_polys();

        struct Scratch<F: JoltField> {
            val: [(F, F); 1],
            row_lanes: Vec<F::Accumulator>,
            evals: Vec<F>,
            steps: Vec<F>,
            grid: Vec<F>,
        }

        let q_evals = ra
            .sum_pair_terms(
                &cycle.gruen,
                factors,
                || Scratch::<F> {
                    val: [(F::zero(), F::zero())],
                    row_lanes: vec![F::Accumulator::default(); factors],
                    evals: vec![F::zero(); factors],
                    steps: vec![F::zero(); factors],
                    grid: vec![F::zero(); product_grid_scratch_len(factors)],
                },
                |scratch, row, lo, hi, terms| {
                    cycle.combined_val.lo_hi_all(row, &mut scratch.val);
                    let (val_lo, val_hi) = scratch.val[0];
                    scratch.evals[0] = val_hi;
                    scratch.steps[0] = val_hi - val_lo;
                    for (((eval, step), lo), hi) in scratch.evals[1..]
                        .iter_mut()
                        .zip(&mut scratch.steps[1..])
                        .zip(lo)
                        .zip(hi)
                    {
                        *eval = *hi;
                        *step = *hi - *lo;
                    }
                    scratch.row_lanes.fill(F::Accumulator::default());
                    accumulate_product_grid(
                        &scratch.evals,
                        &scratch.steps,
                        &mut scratch.row_lanes,
                        &mut scratch.grid,
                    );
                    for (term, lane) in terms.iter_mut().zip(&scratch.row_lanes) {
                        *term = lane.reduce();
                    }
                },
            )
            .map_err(rows_unavailable)?;
        let mut failure = None;
        let message = cycle
            .gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_toom(&q_evals, previous_claim, round, || {
                ra.sum_pair_terms(
                    &cycle.gruen,
                    1,
                    || [(F::zero(), F::zero())],
                    |val, row, lo, _, terms| {
                        cycle.combined_val.lo_hi_all(row, val);
                        terms[0] = lo.iter().fold(val[0].0, |product, value| product * *value);
                    },
                )
                .map_or_else(
                    |error| {
                        failure = Some(error);
                        F::zero()
                    },
                    |terms| terms[0],
                )
            });
        match failure {
            Some(error) => Err(rows_unavailable(error)),
            None => message,
        }
    }
}
