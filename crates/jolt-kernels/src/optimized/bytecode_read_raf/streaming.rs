//! Streaming bytecode read-RAF kernels: the default kernels over the
//! witness's resident rows instead of the packed row cache the default
//! kernels share. The address phase accumulates its stage pushforwards from
//! the rows in the default kernel's walk, which stops at the rows' constant
//! suffix and weighs that suffix once. The cycle phase gathers its committed
//! RA factors from the rows ([`StreamedFoldedRa`]): until the bind at which
//! the default kernel's factors turn dense, every round evaluates the default
//! kernel's summand on each pair row that holds a cycle before the rows'
//! constant suffix; the suffix term, the coefficient tables and every later
//! round are the default kernel's.

use jolt_field::JoltField;
use jolt_sumcheck::SumcheckError;
#[cfg(all(feature = "akita", feature = "parallel"))]
use jolt_utils::par_collect_windows;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::{BytecodePc, RaChunkSelector};
#[cfg(feature = "akita")]
use jolt_witness::witnesses::{FusedInc, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

#[cfg(feature = "akita")]
use super::LazyFusedInc;
use super::{AddressKernel, CycleKernel, CycleRa, CycleRows};
use crate::optimized::lazy_ra::DEFAULT_MAX_WIDTH;
use crate::optimized::streaming::{RowIndexSource, StreamedFoldedRa};
use crate::optimized::support::{merge_evals, BundleStore};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const ROWS: &str = "bytecode read-RAF trace rows";

/// What the bytecode read-RAF kernels read of a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub(super) struct BytecodeRow {
    bytecode_pc: BytecodePc,
    #[cfg(feature = "akita")]
    fused_inc: FusedInc,
}

/// The streaming stage-6a bytecode read-RAF address slot.
pub struct StreamingBytecodeReadRafAddress;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
    for StreamingBytecodeReadRafAddress
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.dimensions().log_t();
        let rows = BundleStore::<BytecodeRow>::resident(witness, log_t, ROWS)?;
        let access = rows.access();
        // The rows span `2^log_t` cycles, so the shift cannot overflow. The
        // walk weighs the suffix from `prefix` on by its row there, so
        // `prefix` stays inside the domain.
        let last = (1usize << log_t) - 1;
        let prefix = access.constant_suffix_start(last + 1)?.min(last);
        // `AddressKernel::new` reads a row only once this table is known to
        // hold the relation's address count, the length of its pushforwards.
        let table = witness.program_preprocessing().bytecode.bytecode.len();
        let row_at = |cycle: usize| {
            let row = access.row(cycle)?;
            if row.bytecode_pc.0 >= table {
                return Err(WitnessError::InvalidWitnessData {
                    label: ROWS,
                    reason: format!(
                        "cycle {cycle} has bytecode index {} in a table of {table} rows",
                        row.bytecode_pc.0
                    ),
                });
            }
            Ok(row)
        };
        #[cfg(not(feature = "akita"))]
        let row_weight = |_: &BytecodeRow| F::one();
        #[cfg(feature = "akita")]
        let row_weight = |row: &BytecodeRow| row.fused_inc.to_field::<F>();
        Ok(Box::new(AddressKernel::new(
            witness,
            &inputs,
            row_at,
            prefix,
            |row| row.bytecode_pc.0,
            row_weight,
        )?))
    }
}

/// The streaming stage-6b bytecode read-RAF cycle slot.
pub struct StreamingBytecodeReadRafCycle;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafCycle<F>> for StreamingBytecodeReadRafCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        Ok(Box::new(CycleKernel::streaming(witness, &inputs)?))
    }
}

pub(super) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS }
}

/// Streamed-RA index source: column `i` is committed chunk `i` of the
/// cycle's bytecode index, hot on every cycle. Like the default kernel's
/// source, it relies on the address phase, which runs first and refuses a
/// bytecode index outside the bytecode table, for every chunk to index
/// inside its table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct BytecodePcDigits {
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    selectors: Vec<RaChunkSelector>,
}

impl RowIndexSource for BytecodePcDigits {
    type Row = BytecodeRow;

    #[inline]
    fn for_each_hot(&self, row: &BytecodeRow, mut visit: impl FnMut(usize, usize)) {
        for (column, selector) in self.selectors.iter().enumerate() {
            visit(column, selector.chunk_usize(row.bytecode_pc.0));
        }
    }
}

impl<F: JoltField> CycleKernel<F> {
    /// The kernel over the witness's resident rows. Its RA factors follow
    /// the default kernel's schedule, and every table it holds by cycle stops
    /// where the rows' constant suffix starts, or at cycle 1 where that is
    /// later: the coefficient tables hold cycle 0's entry term explicitly.
    pub(super) fn streaming(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Self, KernelError<F>> {
        let log_t = inputs.relation.dimensions().log_t();
        Self::new(inputs, |chunk_eqs, selectors| {
            let ra = StreamedFoldedRa::factored(
                chunk_eqs,
                1,
                DEFAULT_MAX_WIDTH,
                BytecodePcDigits { selectors },
                BundleStore::resident(witness, log_t, ROWS)?,
                log_t,
                1,
            )?;
            let prefix_len = ra.prefix_len();
            Ok(CycleRows {
                #[cfg(feature = "akita")]
                fused_inc: LazyFusedInc::resident(witness, log_t, prefix_len)?,
                // A one-cycle domain has no bind to materialize at.
                ra: match ra.single_cycle()? {
                    Some(dense) => CycleRa::Lazy(dense),
                    None => CycleRa::Streamed(ra),
                },
                prefix_len,
            })
        })
    }

    /// The summand's samples summed over the pair rows that hold a cycle
    /// before the rows' constant suffix: the default kernel's group
    /// arithmetic on factors gathered from the rows, without its merging of
    /// repeated groups.
    pub(super) fn streamed_evals(
        &self,
        ra: &StreamedFoldedRa<F, BytecodePcDigits>,
    ) -> Result<Vec<F>, WitnessError> {
        ra.fold_pairs(
            || (vec![F::zero(); self.degree], Vec::new()),
            |(acc, ra_pairs), y, lo, hi| {
                ra_pairs.clear();
                ra_pairs.extend(lo.iter().copied().zip(hi.iter().copied()));
                Self::accumulate_coefficient(self.coefficient(y), acc, ra_pairs);
            },
            |(left, ra_pairs), (right, _)| (merge_evals(left, right), ra_pairs),
        )
        .map(|(evals, _)| evals)
    }
}

#[cfg(feature = "akita")]
impl<F: JoltField> LazyFusedInc<F> {
    /// The default kernel's dense state from the start, since its lazy one
    /// reads the packed row cache: the fused increment of every cycle before
    /// `prefix_len`, from which the rows of the `2^log_t`-cycle domain
    /// repeat.
    fn resident(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        prefix_len: usize,
    ) -> Result<Self, WitnessError> {
        let rows = BundleStore::<BytecodeRow>::resident(witness, log_t, ROWS)?;
        let access = rows.access();
        let value = |cycle: usize| access.row(cycle).map(|row| row.fused_inc.to_field::<F>());
        #[cfg(feature = "parallel")]
        let values = par_collect_windows(prefix_len, value)?;
        #[cfg(not(feature = "parallel"))]
        let values = (0..prefix_len).map(value).collect::<Result<_, _>>()?;
        Ok(Self::Prefix {
            values,
            // The rows span `2^log_t` cycles, so the shift cannot overflow.
            tail: value((1usize << log_t) - 1)?,
        })
    }
}
