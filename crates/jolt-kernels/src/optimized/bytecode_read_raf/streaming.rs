//! Streaming bytecode read-RAF kernels: the default kernels over the
//! witness's resident rows instead of the packed row cache the default
//! kernels share. The address phase accumulates its stage pushforwards from
//! the rows in the default kernel's walk, which stops at the rows' constant
//! suffix and weighs that suffix once. The cycle phase holds no table by
//! cycle until the default kernel's dense tables fit its [`DenseCap`]: every
//! round before that evaluates the default kernel's summand on each pair row
//! that holds a cycle before the rows' constant suffix, gathering the
//! committed RA factors and, in `akita` builds, the fused increment from the
//! rows ([`StreamedFoldedRa`]) and evaluating the coefficient tables from
//! their eq terms. The bind at which the tables fit fills them; the suffix
//! term and every later round are the default kernel's.

use jolt_field::JoltField;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::{BytecodePc, RaChunkSelector};
#[cfg(feature = "akita")]
use jolt_witness::witnesses::{FusedInc, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

#[cfg(feature = "akita")]
use super::LazyFusedInc;
use super::{AddressKernel, CycleKernel, CycleRows, LazyRows};
use crate::optimized::streaming::{DenseCap, RowIndexSource, StreamedFoldedRa, DENSE_CAP};
use crate::optimized::support::{merge_evals, BundleStore};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const ROWS: &str = "bytecode read-RAF trace rows";

/// The dense tables a cycle kernel holds by cycle beside its RA factors, one
/// field element per cycle each before any bind: the coefficient table and,
/// in `akita` builds, the fused increment and its coefficient table.
const NON_RA_TABLES: usize = if cfg!(feature = "akita") { 3 } else { 1 };

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
        Ok(Box::new(CycleKernel::streaming(
            witness, &inputs, DENSE_CAP,
        )?))
    }
}

pub(super) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS }
}

/// Streamed-RA index source: column `i` is committed chunk `i` of the
/// cycle's bytecode index, hot on every cycle. Any index gives chunks inside
/// their tables: a selector keeps `chunk_bits` bits and every chunk table has
/// `2^chunk_bits` entries. The chunks spell the index only if it lies inside
/// the bytecode table, which this source, like the default kernel's, leaves
/// to the address phase: that phase runs first and refuses any other index.
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

/// The fused increment at the bound index whose cycles are `rows`, the
/// cycles' branch weights being `weights`: a column of
/// [`StreamedFoldedRa::fold_pairs`].
#[cfg(feature = "akita")]
#[inline]
fn bound_fused_inc<F: JoltField>(weights: &[F], rows: &[BytecodeRow]) -> [F; 1] {
    [weights
        .iter()
        .zip(rows)
        .map(|(weight, row)| weight.mul_i128(row.fused_inc.0))
        .sum()]
}

/// The streaming family's state by cycle until the kernel's dense tables
/// fill: the RA factors' branch tables and, in `akita` builds, the fused
/// increment on the rows' constant suffix. Every round gathers the factors
/// and the increment before that suffix from the rows.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct StreamedRows<F: JoltField> {
    pub(super) ra: StreamedFoldedRa<F, BytecodePcDigits>,
    #[cfg(feature = "akita")]
    pub(super) fused_inc_tail: F,
}

impl<F: JoltField> StreamedRows<F> {
    /// Binds the next cycle variable. The family's materializing bind returns
    /// the default family's dense state, gathered in one pass over the rows,
    /// and this state is spent.
    pub(super) fn bind(&mut self, challenge: F) -> Result<Option<LazyRows<F>>, WitnessError> {
        #[cfg(not(feature = "akita"))]
        let dense = self.ra.bind(challenge)?.map(|ra| LazyRows { ra });
        #[cfg(feature = "akita")]
        let dense = self
            .ra
            .bind_with_columns(challenge, bound_fused_inc)?
            .map(|(ra, [values])| LazyRows {
                ra,
                fused_inc: LazyFusedInc::Prefix {
                    values,
                    tail: self.fused_inc_tail,
                },
            });
        Ok(dense)
    }
}

impl<F: JoltField> CycleKernel<F> {
    /// The kernel over the witness's resident rows. It fills the default
    /// kernel's dense tables, one per RA factor and [`NON_RA_TABLES`] more,
    /// at the first bind at which they fit `cap` together
    /// ([`StreamedFoldedRa::with_columns`]); production passes [`DENSE_CAP`].
    /// Until then it holds one coefficient, cycle 0's, for its entry term.
    pub(super) fn streaming(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
        cap: DenseCap,
    ) -> Result<Self, KernelError<F>> {
        let log_t = inputs.relation.dimensions().log_t();
        Self::new(inputs, |chunk_eqs, selectors| {
            let rows = BundleStore::<BytecodeRow>::resident(witness, log_t, ROWS)?;
            // The rows span `2^log_t` cycles, so the shift cannot overflow.
            #[cfg(feature = "akita")]
            let fused_inc_tail = rows
                .access()
                .row((1usize << log_t) - 1)?
                .fused_inc
                .to_field::<F>();
            // The suffix term sums eq terms only and cycle 0's coefficient
            // also holds the entry term, so the rounds visit its pair row
            // even where the rows' constant suffix starts at cycle 0.
            let ra = StreamedFoldedRa::with_columns(
                chunk_eqs,
                BytecodePcDigits { selectors },
                rows,
                log_t,
                cap,
                NON_RA_TABLES,
                1,
            )?;
            // A one-cycle domain has no bind to materialize at, and its
            // cycle is the last one, whose increment the tail holds.
            Ok(match ra.single_cycle()? {
                Some(ra) => CycleRows::Lazy(LazyRows {
                    ra,
                    #[cfg(feature = "akita")]
                    fused_inc: LazyFusedInc::Prefix {
                        values: Vec::new(),
                        tail: fused_inc_tail,
                    },
                }),
                None => CycleRows::Streamed(StreamedRows {
                    ra,
                    #[cfg(feature = "akita")]
                    fused_inc_tail,
                }),
            })
        })
    }

    /// The summand's samples summed over the pair rows that hold a cycle
    /// before the rows' constant suffix: the default kernel's arithmetic for
    /// one pair row, on RA factors and a fused increment gathered from the
    /// rows and on coefficients evaluated from their eq terms, without its
    /// merging of repeated groups.
    pub(super) fn streamed_evals(&self, rows: &StreamedRows<F>) -> Result<Vec<F>, WitnessError> {
        let ra = &rows.ra;
        let combined = self.combined.entries();
        #[cfg(not(feature = "akita"))]
        let columns = |_: &[F], _: &[BytecodeRow]| [];
        #[cfg(feature = "akita")]
        let (columns, fused_combined) = (bound_fused_inc::<F>, self.fused_combined.entries());
        ra.fold_pairs(
            || {
                (
                    vec![F::zero(); self.degree],
                    vec![(F::zero(), F::zero()); ra.num_polys()],
                )
            },
            columns,
            |(acc, ra_pairs), y, lo, hi, gathered| {
                let (c_lo, c_hi) = combined.pair(y);
                #[cfg(not(feature = "akita"))]
                let coefficient = {
                    let [[], []] = gathered;
                    [c_lo, c_hi - c_lo, F::zero()]
                };
                #[cfg(feature = "akita")]
                let coefficient = {
                    let [[fused_inc_lo], [fused_inc_hi]] = gathered;
                    Self::fused_coefficient(
                        (c_lo, c_hi),
                        fused_combined.pair(y),
                        (fused_inc_lo, fused_inc_hi),
                    )
                };
                Self::accumulate_samples(coefficient, acc, ra_pairs, |pairs| {
                    for ((pair, lo), hi) in pairs.iter_mut().zip(lo).zip(hi) {
                        *pair = (*lo, *hi);
                    }
                });
            },
            |(left, ra_pairs), (right, _)| (merge_evals(left, right), ra_pairs),
        )
        .map(|(evals, _)| evals)
    }
}
