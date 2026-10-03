//! The optimized field-registers read/write-checking (stage 4) kernel: the
//! integer-register sparse Twist ([`super::registers_read_write`]) at the
//! field-register geometry, byte-parity twin of
//! [`crate::reference::field_registers_read_write_checking`].
//!
//! The reference kernel binds six dense `2^(4 + log_T)` register-major grids per round.
//! This kernel computes the same round polynomials from the sparse structure of the
//! field-inline access pattern — the v2-port `SparseFieldRegState` design
//! (`specs/native-field-registers.md`, Stage 4) restated over today's relation shapes:
//!
//! - **Sparse cycle-major entries**: ≤ 3 entries per active field-inline cycle (rs2
//!   merges into rs1's cell, rd into either read's), built in one pass over
//!   the field-inline oracle's decoded rows. The witness boundary validates the
//!   K = 16 register history, so extraction can fill disjoint blocks in parallel.
//!   Between touches a field register is constant, so a missing merge partner
//!   is inferred from its neighbor's `prev_val`/`next_val` — field-valued
//!   here (the v2 delta vs the integer sibling's raw `u64`s). The integer
//!   access coefficients remain compact selector bits until the first bind.
//! - **γ-combined read coefficient**: one `ra = γ·rs1_ra + γ²·rs2_ra` column
//!   per entry (exact by distributivity).
//! - **Gruen split-eq factoring** for the cycle rounds, with the quadratic
//!   endpoints accumulated over the sparse rows only — a trace without field-inline activity
//!   has zero entries and the cycle rounds cost O(√T) eq-table work plus the
//!   increment column stays allocation-free when zero.
//! - **Rayon past a threshold**: the round accumulation and the bind shell
//!   out to pair-aligned parallel blocks once the entry count crosses
//!   [`PARALLEL_THRESHOLD`] (the v2 `par_chunk_by` convention); below it the
//!   sequential walks win.
//! - **Small fixed K**: after the cycle rounds the state collapses to three
//!   `K = 2^4` dense arrays plus two scalars; address rounds cost O(K).
//! - **Direct one-hot claims at extraction**: `rs1_ra(r)`/`rs2_ra(r)` come
//!   straight from the sparse per-cycle read indices with a 2-way split-eq
//!   walk (the sibling's `one_hot_operand_claims` — no γ⁻¹ recovery).
//!
//! Like the reference kernel, only the config-pinned field-inline phase split (phase 1
//! = all cycle rounds, phase 2 = the 4 address rounds) is supported.

use core::{mem::MaybeUninit, ops::Range};

use crate::field_inline::{FieldIncrementColumn, IncrementRounds};
use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldRegistersReadWriteChallenge,
    FieldRegistersReadWritePublic,
};
use jolt_claims::SumcheckChallenges as _;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_witness::field_inline::FieldInlineRegisterReadWriteRow;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::registers_read_write::address::{OperandEq, RegisterAddressState};
use super::registers_read_write::sparse::layout::{merge_bind, split_pair_group, Cell};
use super::registers_read_write::sparse::ops::{bind_sparse_entries_in_place, pair_aligned_bounds};
use super::support::{
    map_indices, map_reduce_chunks, pin_derived_term, GruenRoundMessage, RoundChallenges,
};
use std::sync::Arc;

use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_witness::field_inline::FieldInlineWitnessOracle;

/// Entry count above which the round accumulation and the bind run over
/// pair-aligned parallel blocks (the v2-port `DENSE_BIND_PAR_THRESHOLD`
/// convention — below it the sequential walk beats the fork/join overhead).
const PARALLEL_THRESHOLD: usize = 1 << 12;

/// One non-zero cell of the conceptual `K × T` field register matrices: the bound `Val`
/// coefficient plus the γ-combined read and write coefficients of one touched register
/// slice. All value fields are field elements — field registers hold full field values,
/// so there is no raw-scalar shortcut for the untouched-neighbor boundary values.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct FieldSparseEntry<F> {
    /// Bound `Val(col, row-slice)` coefficient (value *before* the access).
    val: F,
    /// Register value just before this entry's row slice.
    prev_val: F,
    /// Register value just after this entry's row slice.
    next_val: F,
    /// Bound `γ·rs1_ra + γ²·rs2_ra` coefficient.
    ra: F,
    wa: F,
    /// Cycle-domain row index (before binding: the cycle).
    row: usize,
    col: u8,
}

/// Before binding, Val equals its pre-value and the access coefficients are
/// selector bits. Keep only two field values and expand coefficients on demand.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct FieldSeed<F> {
    pre: F,
    post: F,
    row: usize,
    col: u8,
    reads: u8,
    write: bool,
}

impl<F: JoltField> Cell for FieldSeed<F> {
    fn row(&self) -> usize {
        self.row
    }
    fn col(&self) -> u8 {
        self.col
    }
}

impl<F: JoltField> FieldSeed<F> {
    fn row_entries(row: usize, access: &FieldInlineRegisterReadWriteRow<F>) -> [Option<Self>; 3] {
        let mut entries: [Option<Self>; 3] = [None; 3];
        let mut len = 0;
        let mut add = |col, pre, post, reads, write| {
            if let Some(entry) = entries[..len]
                .iter_mut()
                .flatten()
                .find(|entry| entry.col == col)
            {
                entry.reads |= reads;
                if write {
                    entry.write = true;
                    entry.post = post;
                }
            } else {
                entries[len] = Some(Self {
                    pre,
                    post,
                    row,
                    col,
                    reads,
                    write,
                });
                len += 1;
            }
        };
        if let Some(read) = access.rs1 {
            add(read.register, read.value, read.value, 1, false);
        }
        if let Some(read) = access.rs2 {
            add(read.register, read.value, read.value, 2, false);
        }
        if let Some(write) = access.rd {
            add(write.register, write.pre_value, write.post_value, 0, true);
        }
        entries.sort_unstable_by_key(|entry| entry.map(|entry| entry.col));
        entries
    }

    fn expand(self, reads: &[F; 4]) -> FieldSparseEntry<F> {
        FieldSparseEntry {
            val: self.pre,
            prev_val: self.pre,
            next_val: self.post,
            ra: reads[usize::from(self.reads)],
            wa: F::from_bool(self.write),
            row: self.row,
            col: self.col,
        }
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
enum FieldEntries<F: JoltField> {
    Seeds {
        entries: Vec<FieldSeed<F>>,
        reads: [F; 4],
    },
    Bound(Vec<FieldSparseEntry<F>>),
}

impl<F: JoltField> FieldEntries<F> {
    fn quadratic(&self, e_in: &[F], e_out: &[F], inc: &IncrementRounds<F>) -> [F; 2] {
        match self {
            Self::Seeds { entries, reads } => {
                sparse_quadratic(entries, e_in, e_out, inc, |entry| entry.expand(reads))
            }
            Self::Bound(entries) => sparse_quadratic(entries, e_in, e_out, inc, |entry| entry),
        }
    }

    fn q_at_one(&self, e_in: &[F], e_out: &[F], inc: &IncrementRounds<F>) -> F {
        match self {
            Self::Seeds { entries, reads } => sparse_at_one(
                entries.iter().map(|entry| entry.expand(reads)),
                e_in,
                e_out,
                inc,
            ),
            Self::Bound(entries) => sparse_at_one(entries.iter().copied(), e_in, e_out, inc),
        }
    }

    fn bind(&mut self, challenge: F) {
        if let Self::Seeds { entries, reads } = self {
            let expanded = map_indices(entries.len(), |index| entries[index].expand(reads));
            *self = Self::Bound(expanded);
        }
        if let Self::Bound(entries) = self {
            bind_sparse_entries_in_place(entries, |even, odd| {
                FieldSparseEntry::bind(even, odd, challenge)
            });
        }
    }

    fn take_bound(&mut self) -> Vec<FieldSparseEntry<F>> {
        match self {
            Self::Bound(entries) => std::mem::take(entries),
            Self::Seeds { .. } => unreachable!("at least one cycle binds before the address phase"),
        }
    }
}

impl<F: JoltField> FieldSparseEntry<F> {
    /// Bind two vertically adjacent cells (rows `2j`/`2j+1`, same column)
    /// with `r`. A missing side is an untouched slice: its `Val` is the
    /// neighbor's boundary value and its `ra`/`wa` are zero.
    fn bind(even: Option<&Self>, odd: Option<&Self>, r: F) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                Self {
                    val: even.val + r * (odd.val - even.val),
                    ra: even.ra + r * (odd.ra - even.ra),
                    wa: even.wa + r * (odd.wa - even.wa),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                    row: even.row / 2,
                    col: even.col,
                }
            }
            (Some(even), None) => Self {
                val: even.val + r * (even.next_val - even.val),
                ra: (F::one() - r) * even.ra,
                wa: (F::one() - r) * even.wa,
                prev_val: even.prev_val,
                next_val: even.next_val,
                row: even.row / 2,
                col: even.col,
            },
            (None, Some(odd)) => Self {
                val: odd.prev_val + r * (odd.val - odd.prev_val),
                ra: r * odd.ra,
                wa: r * odd.wa,
                prev_val: odd.prev_val,
                next_val: odd.next_val,
                row: odd.row / 2,
                col: odd.col,
            },
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    /// Accumulate this vertical pair's `[t = 0, t = ∞]` contributions to the
    /// quadratic inner factor `ra_t·val_t + wa_t·(val_t + inc_t)`, weighted
    /// by the pair's eq factor.
    fn accumulate_pair_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        weight: F,
        acc: &mut [F::Accumulator; 2],
    ) {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                acc[0].fmadd(
                    weight,
                    even.ra * even.val + even.wa * (even.val + inc_evals[0]),
                );
                let val_m = odd.val - even.val;
                acc[1].fmadd(
                    weight,
                    (odd.ra - even.ra) * val_m + (odd.wa - even.wa) * (val_m + inc_evals[1]),
                );
            }
            (Some(even), None) => {
                acc[0].fmadd(
                    weight,
                    even.ra * even.val + even.wa * (even.val + inc_evals[0]),
                );
                let val_m = even.next_val - even.val;
                acc[1].fmadd(
                    weight,
                    -(even.ra * val_m) - even.wa * (val_m + inc_evals[1]),
                );
            }
            (None, Some(odd)) => {
                // The even side has zero ra/wa, so the t = 0 term vanishes.
                let val_m = odd.val - odd.prev_val;
                acc[1].fmadd(weight, odd.ra * val_m + odd.wa * (val_m + inc_evals[1]));
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }
}

impl<F: JoltField> Cell for FieldSparseEntry<F> {
    fn row(&self) -> usize {
        self.row
    }
    fn col(&self) -> u8 {
        self.col
    }
}

fn sparse_at_one<F: JoltField>(
    entries: impl IntoIterator<Item = FieldSparseEntry<F>>,
    e_in: &[F],
    e_out: &[F],
    inc: &IncrementRounds<F>,
) -> F {
    let in_bits = e_in.len().trailing_zeros() as usize;
    let mask = e_in.len() - 1;
    let mut sum = F::Accumulator::default();
    for entry in entries {
        if entry.row.is_multiple_of(2) {
            continue;
        }
        let pair = entry.row / 2;
        let weight = e_out[pair >> in_bits] * e_in[pair & mask];
        let mut lanes = [F::Accumulator::default(), F::Accumulator::default()];
        FieldSparseEntry::accumulate_pair_evals(
            Some(&entry),
            None,
            [inc.value(entry.row), F::zero()],
            weight,
            &mut lanes,
        );
        let [constant, _] = lanes;
        sum.merge(constant);
    }
    sum.reduce()
}

/// The cycle-round quadratic inner factor `[q(0), leading coefficient]` over the sparse
/// entries: per row pair, the eq weight is `E_out[z >> in_bits] · E_in[z & mask]`
/// (recombined per pair — untouched pairs contribute nothing, so there is no
/// per-`x_out` factoring win at field-inline densities).
fn sparse_quadratic<F: JoltField, E: Cell>(
    entries: &[E],
    e_in: &[F],
    e_out: &[F],
    inc: &IncrementRounds<F>,
    expand: impl Fn(E) -> FieldSparseEntry<F> + Sync,
) -> [F; 2] {
    let in_bits = if e_in.len() <= 1 {
        0
    } else {
        e_in.len().trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let range_contribution = |range: Range<usize>| -> [F; 2] {
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for group in entries[range].chunk_by(|a, b| a.row() / 2 == b.row() / 2) {
            let z = group[0].row() / 2;
            let weight = if e_in.len() <= 1 {
                e_out[z]
            } else {
                e_out[z >> in_bits] * e_in[z & mask]
            };
            let j_prime = 2 * z;
            let inc_0 = inc.value(j_prime);
            let inc_evals = [inc_0, inc.value(j_prime + 1) - inc_0];
            let (evens, odds) = split_pair_group(group);
            merge_bind(
                evens,
                odds,
                &|even, odd| (even.copied().map(&expand), odd.copied().map(&expand)),
                |(even, odd)| {
                    FieldSparseEntry::accumulate_pair_evals(
                        even.as_ref(),
                        odd.as_ref(),
                        inc_evals,
                        weight,
                        &mut acc,
                    );
                },
            );
        }
        [acc[0].reduce(), acc[1].reduce()]
    };

    #[cfg(feature = "parallel")]
    if entries.len() >= PARALLEL_THRESHOLD {
        let bounds = pair_aligned_bounds(entries, 1);
        return (0..bounds.len() - 1)
            .into_par_iter()
            .map(|block| range_contribution(bounds[block]..bounds[block + 1]))
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    }
    range_contribution(0..entries.len())
}

/// The rd write slots of one proof's field-inline trace — `(cycle, register)` pairs of
/// every bytecode-active field-inline write — parked by the stage-4 kernel for the
/// stage-5 val-evaluation kernel (which folds the same one-hot `FieldRdWa` grid at its
/// address prefix).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedFieldRdWrites(pub(crate) Vec<(u32, u8)>);

type FieldRegisterRows<F> = Arc<Vec<(usize, FieldInlineRegisterReadWriteRow<F>)>>;

/// Sparse register rows shared by claim reduction and read/write checking.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedFieldRegisterRows<F: JoltField>(
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_shared_heap_free_elements))]
    pub(crate) FieldRegisterRows<F>,
);

pub(crate) fn field_register_rows<F: JoltField>(
    session: &mut ProofSession,
    field_inline: &dyn FieldInlineWitnessOracle<F>,
    cycles: usize,
) -> Result<FieldRegisterRows<F>, KernelError<F>> {
    if let Some(SharedFieldRegisterRows(rows)) = session.state::<SharedFieldRegisterRows<F>>() {
        return Ok(Arc::clone(rows));
    }
    let rows = field_inline.field_inline_register_read_write_rows()?;
    if rows.iter().any(|(cycle, _)| *cycle >= cycles)
        || rows.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(KernelError::InvariantViolation {
            reason: "field register rows must be ordered within the cycle domain",
        });
    }
    let rows = Arc::new(rows);
    session.park(SharedFieldRegisterRows(Arc::clone(&rows)));
    Ok(rows)
}

/// Sparse per-cycle field-inline access facts extracted from the oracle's decoded rows:
/// the ≤3-entries-per-active-cycle matrix cells plus the raw read/write index lists
/// (reads feed the final one-hot claims, writes feed stage 5).
pub(crate) struct FieldRegisterAccesses<F: JoltField> {
    entries: Vec<FieldSeed<F>>,
    rs1_reads: Vec<(u32, u8)>,
    rs2_reads: Vec<(u32, u8)>,
    pub(crate) rd_writes: Vec<(u32, u8)>,
}

impl<F: JoltField> FieldRegisterAccesses<F> {
    /// Count then fill disjoint spans over the already-validated sparse rows.
    /// Read/pre-write values are pinned to the register replay by the witness
    /// boundary, so entry construction needs no serial register-file replay.
    pub(crate) fn collect(
        rows: &[(usize, FieldInlineRegisterReadWriteRow<F>)],
        register_count: usize,
    ) -> Result<Self, KernelError<F>> {
        for (cycle, access) in rows {
            if u32::try_from(*cycle).is_err()
                || access
                    .rs1
                    .iter()
                    .map(|r| r.register)
                    .chain(access.rs2.iter().map(|r| r.register))
                    .chain(access.rd.iter().map(|r| r.register))
                    .any(|register| usize::from(register) >= register_count)
            {
                return Err(KernelError::InvariantViolation {
                    reason: "field register access exceeds its cycle or register domain",
                });
            }
        }
        const CHUNK: usize = 1 << 12;
        let counts = map_indices(rows.len().div_ceil(CHUNK), |chunk| {
            rows[chunk * CHUNK..((chunk + 1) * CHUNK).min(rows.len())]
                .iter()
                .map(|(cycle, access)| {
                    FieldSeed::row_entries(*cycle, access)
                        .into_iter()
                        .flatten()
                        .count()
                })
                .sum::<usize>()
        });
        let total = counts.iter().sum();
        let mut entries: Vec<FieldSeed<F>> = Vec::with_capacity(total);
        let mut spare = &mut entries.spare_capacity_mut()[..total];
        let spans: Vec<_> = counts
            .iter()
            .map(|&count| {
                let (head, tail) = std::mem::take(&mut spare).split_at_mut(count);
                spare = tail;
                head
            })
            .collect();
        let fill = |(chunk, span): (usize, &mut [MaybeUninit<FieldSeed<F>>])| {
            let mut output = span.iter_mut();
            for (cycle, access) in &rows[chunk * CHUNK..((chunk + 1) * CHUNK).min(rows.len())] {
                for entry in FieldSeed::row_entries(*cycle, access).into_iter().flatten() {
                    #[expect(
                        clippy::expect_used,
                        reason = "count and fill use the same row_entries constructor"
                    )]
                    let _ = output.next().expect("counted field entry").write(entry);
                }
            }
            assert!(
                output.next().is_none(),
                "field entry count and fill disagree"
            );
        };
        #[cfg(feature = "parallel")]
        spans.into_par_iter().enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        spans.into_iter().enumerate().for_each(fill);
        // SAFETY: each disjoint span was filled completely using the same constructor
        // that counted it; FieldSeed is Copy and no uninitialized slot is exposed.
        unsafe {
            entries.set_len(total);
        }
        Ok(Self {
            entries,
            rs1_reads: rows
                .iter()
                .filter_map(|(cycle, access)| access.rs1.map(|read| (*cycle as u32, read.register)))
                .collect(),
            rs2_reads: rows
                .iter()
                .filter_map(|(cycle, access)| access.rs2.map(|read| (*cycle as u32, read.register)))
                .collect(),
            rd_writes: rows
                .iter()
                .filter_map(|(cycle, access)| {
                    access.rd.map(|write| (*cycle as u32, write.register))
                })
                .collect(),
        })
    }
}

pub struct OptimizedFieldRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, FieldRegistersReadWriteChecking<F>>
    for OptimizedFieldRegistersReadWrite
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersReadWriteChecking<F>>>,
        KernelError<F>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        // The field-inline phase split is pinned by the compile-time protocol config
        // (phase 1 = log_t, phase 2 = log_k) — the same guard as the reference kernel:
        // a drifted config is a bug, not a capability gap.
        if dimensions.phase1_num_rounds() != dimensions.log_t()
            || dimensions.phase2_num_rounds() != dimensions.log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "field-register read-write dimensions drifted from the config-pinned phase split",
            });
        }
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason:
                    "optimized field-register read-write checking requires at least one cycle round",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason:
                    "field-register read-write upstream cycle point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers read-write checking field-inline oracle",
                }))?;
        let rows = field_register_rows(session, field_inline, cycles)?;
        let inc_column = FieldIncrementColumn::resolve(session, field_inline, cycles)?;
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "field-register read-write checking is missing its gamma challenge",
            })?;

        let FieldRegisterAccesses {
            entries,
            rs1_reads,
            rs2_reads,
            rd_writes,
        } = FieldRegisterAccesses::collect(&rows, 1usize << log_k)?;

        // Park the rd write slots for the stage-5 field-register value-evaluation
        // kernel.
        session.park(SharedFieldRdWrites(rd_writes));
        let _ = session.take::<SharedFieldRegisterRows<F>>();

        Ok(Box::new(FieldReadWriteKernel {
            log_t,
            log_k,
            entries: FieldEntries::Seeds {
                entries,
                reads: [F::zero(), gamma, gamma * gamma, gamma + gamma * gamma],
            },
            gruen: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            inc: IncrementRounds::new(inc_column),
            address: RegisterAddressState::default(),
            rs1_reads,
            rs2_reads,
            challenges: RoundChallenges::new(log_t + log_k),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldReadWriteKernel<F: JoltField> {
    log_t: usize,
    log_k: usize,
    /// Sparse cycle-major entries, sorted by `(row, col)`; drained at the
    /// cycle→address transition.
    entries: FieldEntries<F>,
    gruen: GruenSplitEqPolynomial<F>,
    inc: IncrementRounds<F>,
    address: RegisterAddressState<F>,
    rs1_reads: Vec<(u32, u8)>,
    rs2_reads: Vec<(u32, u8)>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> FieldReadWriteKernel<F> {
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining sparse rows, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let quadratic = self.entries.quadratic(
            self.gruen.e_in_current(),
            self.gruen.e_out_current(),
            &self.inc,
        );
        self.gruen
            .checked_cubic(quadratic[0], quadratic[1], previous_claim, round, || {
                self.entries.q_at_one(
                    self.gruen.e_in_current(),
                    self.gruen.e_out_current(),
                    &self.inc,
                )
            })
    }

    fn bind(&mut self, r: F) {
        if self.challenges.bound() < self.log_t {
            self.gruen.bind(r);
            self.inc.bind(r);
            self.entries.bind(r);
        } else {
            self.address.bind(r);
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.log_t {
            let register_count = 1usize << self.log_k;
            let mut ra = vec![F::zero(); register_count];
            let mut wa = vec![F::zero(); register_count];
            let mut val = vec![F::zero(); register_count];
            for entry in self.entries.take_bound() {
                debug_assert_eq!(entry.row, 0);
                ra[usize::from(entry.col)] = entry.ra;
                wa[usize::from(entry.col)] = entry.wa;
                val[usize::from(entry.col)] = entry.val;
            }
            self.address.ra = ra;
            self.address.wa = wa;
            self.address.val = val;
            self.address.eq_scalar = self.gruen.current_scalar();
            self.address.inc_scalar = self.inc.value(0);
        }
    }

    /// The bound opening point, split as `(r_address, r_cycle)` — the same
    /// reversal `FieldRegistersReadWriteDimensions::read_write_opening_point`
    /// applies under the config-pinned phase split.
    fn bound_point(&self) -> (Vec<F>, Vec<F>) {
        let r_cycle: Vec<F> = self.challenges.as_slice()[..self.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
        let r_address: Vec<F> = self.challenges.as_slice()[self.log_t..]
            .iter()
            .rev()
            .copied()
            .collect();
        (r_address, r_cycle)
    }

    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for both
    /// read operands — the direct MLE of a one-hot `(K × T)` grid at the
    /// bound point, walked over the sparse read lists (the sibling's
    /// `one_hot_operand_claims` with the dense scan replaced by the lists).
    /// Big-endian joint point `[r_cycle ‖ r_address]`, joint index
    /// `(j << addr_bits) | k`.
    fn one_hot_operand_claims(&self, r_address: &[F], r_cycle: &[F]) -> (F, F) {
        let eq = OperandEq::new(r_address, r_cycle);

        let claim = |reads: &[(u32, u8)]| -> F {
            map_reduce_chunks(
                reads.len(),
                1 << 12,
                |range| {
                    let mut sum = F::Accumulator::default();
                    for &(j, k) in &reads[range] {
                        let j = j as usize;
                        let lo_index = eq.low_index(j, k);
                        sum.fmadd(eq.hi[j >> eq.cycle_bits_in_lo], eq.lo[lo_index]);
                    }
                    sum.reduce()
                },
                |a, b| a + b,
                F::zero,
            )
        };
        (claim(&self.rs1_reads), claim(&self.rs2_reads))
    }
}

impl<F: JoltField> ProveRounds<F> for FieldReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        if self.challenges.bound() < self.log_t {
            self.cycle_round_message(round, previous_claim)
        } else {
            self.address.round_message(round, previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldReadWriteKernel<F> {
    type Relation = FieldRegistersReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersReadWriteOutputClaims;

        self.challenges.require_complete()?;
        let (r_address, r_cycle) = self.bound_point();
        let (rs1_ra, rs2_ra) = self.one_hot_operand_claims(&r_address, &r_cycle);
        Ok(FieldRegistersReadWriteOutputClaims {
            registers_val: self.address.val[0],
            rs1_ra,
            rs2_ra,
            rd_wa: self.address.wa[0],
            rd_inc: self.address.inc_scalar,
        })
    }

    /// The `EqCycle` cross-check: the fully bound Gruen scalar must equal the
    /// verifier's `derive_output_term` at the bound point (the reference
    /// kernel's tie-down on the table it materializes).
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        pin_derived_term(
            relation,
            FieldInlineDerivedId::from(FieldRegistersReadWritePublic::EqCycle),
            input_points,
            output_points,
            challenges,
            self.address.eq_scalar,
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::field_inline::relations::registers::{
        FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    };
    use jolt_field::{Fr, Ring};
    use jolt_riscv::FieldInlineOp;
    use jolt_verifier::config::JOLT_VERIFIER_CONFIG;
    use jolt_verifier::stages::relations::ConcreteSumcheck as _;

    use super::*;
    use crate::optimized::field_registers_testing::{
        inactive_field_register_fixture, structured_field_register_fixture,
        FieldRegisterTraceFixture,
    };
    use crate::optimized::parity::{probe_input_claim, synthetic_point, ExceptionalEq};
    use crate::optimized::registers_read_write::test_support::assert_kernel_parity_with_session;
    use crate::ReferenceBackend;

    fn run_parity(
        fixture: FieldRegisterTraceFixture,
        log_t: usize,
        seed: u64,
        expect_active: bool,
    ) {
        run_parity_case(fixture, log_t, seed, expect_active, None);
    }

    fn run_parity_case(
        fixture: FieldRegisterTraceFixture,
        log_t: usize,
        seed: u64,
        expect_active: bool,
        exceptional: Option<ExceptionalEq>,
    ) {
        fixture.with_plane(log_t, |backend| {
            let relation = FieldRegistersReadWriteChecking::<Fr>::new(
                JOLT_VERIFIER_CONFIG
                    .field_inline
                    .read_write_dimensions(log_t),
            );
            let round_challenges =
                synthetic_point(relation.rounds(), seed.wrapping_mul(0x9E37_79B9));
            let r_cycle = exceptional.map_or_else(
                || synthetic_point(log_t, seed),
                |case| case.point(log_t, round_challenges[0]),
            );
            let claims = FieldRegistersReadWriteInputClaims {
                rd_value: Fr::from_u64(0),
                rs1_value: Fr::from_u64(0),
                rs2_value: Fr::from_u64(0),
            };
            let points = FieldRegistersReadWriteInputClaims {
                rd_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let challenges = FieldRegistersReadWriteChallenges {
                gamma: Fr::from_u64(31 + seed),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut session = ProofSession::default();
            let mut reference = <ReferenceBackend as PrepareKernel<
                Fr,
                FieldRegistersReadWriteChecking<Fr>,
            >>::prepare(
                &ReferenceBackend, &mut session, backend, inputs()
            )
            .unwrap();
            let claim = probe_input_claim(reference.as_mut());

            if exceptional.is_none() && expect_active {
                assert!(
                    claim != Fr::from_u64(0),
                    "fixture with field-inline activity degenerated"
                );
            } else if exceptional.is_none() {
                assert_eq!(
                    claim,
                    Fr::from_u64(0),
                    "claim without field-inline activity must be zero"
                );
            }
            drop(reference);
            assert_kernel_parity_with_session(
                &mut session,
                &OptimizedFieldRegistersReadWrite,
                backend,
                &relation,
                &claims,
                &points,
                &challenges,
                claim,
                &round_challenges,
            );
            assert!(
                session.state::<SharedFieldRdWrites>().is_some(),
                "the optimized kernel must park the field-register write slots for stage 5",
            );
        });
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_field_register_fixture(16), 4, 101, true);
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_field_register_fixture(8), 3, 103, true);
    }

    #[test]
    fn parity_partially_padded_trace() {
        // Real rows in the front half only: the padding tail exercises the
        // constant-value slices the sparse boundary values reconstruct.
        run_parity(structured_field_register_fixture(9), 5, 107, true);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = FieldRegisterTraceFixture::new();
        fixture.load_imm(2, 99);
        fixture.arithmetic(FieldInlineOp::Mul, 2, 2, 2);
        run_parity(fixture, 1, 109, true);
    }

    #[test]
    fn parity_inactive_trace_is_degenerate_and_cheap() {
        run_parity(inactive_field_register_fixture(4), 3, 113, false);
    }
    #[test]
    fn parity_exceptional_equality_points_and_prefix() {
        for log_t in [3usize, 4] {
            for case in ExceptionalEq::ALL {
                run_parity_case(
                    structured_field_register_fixture(1 << log_t),
                    log_t,
                    89,
                    true,
                    Some(case),
                );
            }
        }
    }
}
