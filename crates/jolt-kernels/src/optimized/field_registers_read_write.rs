//! The optimized field-registers read/write-checking (stage 4) kernel: the
//! integer-register sparse Twist ([`super::registers_read_write`]) at the FR
//! geometry, byte-parity twin of
//! [`crate::reference::field_registers_read_write_checking`].
//!
//! The reference kernel binds six dense `2^(4 + log_T)` register-major grids
//! per round. This kernel computes the same round polynomials from the
//! sparse structure of the FR access pattern — the v2-port
//! `SparseFieldRegState` design (`specs/native-field-registers.md`, Stage 4)
//! restated over today's relation shapes:
//!
//! - **Sparse cycle-major entries**: ≤ 3 entries per FR-active cycle (rs2
//!   merges into rs1's cell, rd into either read's), built in one pass over
//!   the FR oracle's decoded per-cycle rows against a running K = 16 register
//!   file (the all-zero init the FR val-evaluation sumcheck enforces).
//!   Between touches an FR register is constant, so a missing merge partner
//!   is inferred from its neighbor's `prev_val`/`next_val` — field-valued
//!   here (the v2 delta vs the integer sibling's raw `u64`s). The integer
//!   sibling's u16 coefficient LUT is deliberately not ported: its win is
//!   peak memory at ≤ 3·T entries, and FR entries are ≤ 3·(FR-active cycles).
//! - **γ-combined read coefficient**: one `ra = γ·rs1_ra + γ²·rs2_ra` column
//!   per entry (exact by distributivity).
//! - **Gruen split-eq factoring** for the cycle rounds, with the quadratic
//!   endpoints accumulated over the sparse rows only — an FR-inactive trace
//!   has zero entries and the cycle rounds cost O(√T) eq-table work plus the
//!   dense `FieldRdInc` bind (an all-zero column).
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
//! Like the reference kernel, only the config-pinned FR phase split (phase 1
//! = all cycle rounds, phase 2 = the 4 address rounds) is supported.

use core::cmp::Ordering;
use core::ops::Range;

use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineDerivedId,
    FieldInlinePolynomialId, FieldRegistersReadWriteChallenge, FieldRegistersReadWritePublic,
};
use jolt_claims::SumcheckChallenges as _;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_verifier::VerifierError;
use jolt_witness::field_inline::FieldInlineRegisterReadWriteRow;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{bind_pairs, RoundChallenges};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Entry count above which the round accumulation and the bind run over
/// pair-aligned parallel blocks (the v2-port `DENSE_BIND_PAR_THRESHOLD`
/// convention — below it the sequential walk beats the fork/join overhead).
const PARALLEL_THRESHOLD: usize = 1 << 12;

/// Pair-aligned block target for the parallel walks.
const BLOCK_TARGET: usize = 1 << 12;

/// One non-zero cell of the conceptual `K × T` FR register matrices: the
/// bound `Val` coefficient plus the γ-combined read and write coefficients of
/// one touched register slice. All value fields are field elements — FR
/// registers hold full field values, so there is no raw-scalar shortcut for
/// the untouched-neighbor boundary values.
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
    /// Bound `rd_wa` coefficient.
    wa: F,
    /// Cycle-domain row index (before binding: the cycle).
    row: usize,
    /// FR register index.
    col: u8,
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

    /// Split a row-pair group (entries sharing `row / 2`) into its even and
    /// odd rows. Entries are sorted by `(row, col)`, so the evens form the
    /// prefix.
    fn split_pair_group(group: &[Self]) -> (&[Self], &[Self]) {
        let odd_start = group.partition_point(|entry| entry.row.is_multiple_of(2));
        group.split_at(odd_start)
    }

    /// Two-pointer merge walk over one row pair's (col-sorted) even and odd
    /// slices, calling `visit` per merged cell.
    fn merge_walk(
        evens: &[Self],
        odds: &[Self],
        mut visit: impl FnMut(Option<&Self>, Option<&Self>),
    ) {
        let mut i = 0;
        let mut j = 0;
        while i < evens.len() && j < odds.len() {
            match evens[i].col.cmp(&odds[j].col) {
                Ordering::Equal => {
                    visit(Some(&evens[i]), Some(&odds[j]));
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    visit(Some(&evens[i]), None);
                    i += 1;
                }
                Ordering::Greater => {
                    visit(None, Some(&odds[j]));
                    j += 1;
                }
            }
        }
        for even in &evens[i..] {
            visit(Some(even), None);
        }
        for odd in &odds[j..] {
            visit(None, Some(odd));
        }
    }
}

/// Pair-aligned block bounds over a sorted entry slice: fixed-size blocks
/// advanced to the next row-pair edge, so no merge group straddles a block.
fn pair_aligned_bounds<F: JoltField>(entries: &[FieldSparseEntry<F>]) -> Vec<usize> {
    let len = entries.len();
    let block_count = len.div_ceil(BLOCK_TARGET).max(1);
    let mut bounds: Vec<usize> = Vec::with_capacity(block_count + 1);
    bounds.push(0);
    for block in 1..block_count {
        let mut index = block * len / block_count;
        while index < len && index > 0 && entries[index].row / 2 == entries[index - 1].row / 2 {
            index += 1;
        }
        #[expect(clippy::unwrap_used, reason = "bounds starts non-empty")]
        if index > *bounds.last().unwrap() && index < len {
            bounds.push(index);
        }
    }
    bounds.push(len);
    bounds
}

/// The cycle-round quadratic inner factor `[q(0), leading coefficient]` over
/// the sparse entries: per row pair, the eq weight is
/// `E_out[z >> in_bits] · E_in[z & mask]` (recombined per pair — untouched
/// pairs contribute nothing, so there is no per-`x_out` factoring win at FR
/// densities).
fn sparse_quadratic<F: JoltField>(
    entries: &[FieldSparseEntry<F>],
    e_in: &[F],
    e_out: &[F],
    inc: &[F],
) -> [F; 2] {
    let in_bits = if e_in.len() <= 1 {
        0
    } else {
        e_in.len().trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let range_contribution = |range: Range<usize>| -> [F; 2] {
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for group in entries[range].chunk_by(|a, b| a.row / 2 == b.row / 2) {
            let z = group[0].row / 2;
            let weight = if e_in.len() <= 1 {
                e_out[z]
            } else {
                e_out[z >> in_bits] * e_in[z & mask]
            };
            let j_prime = 2 * z;
            let inc_0 = inc[j_prime];
            let inc_evals = [inc_0, inc[j_prime + 1] - inc_0];
            let (evens, odds) = FieldSparseEntry::split_pair_group(group);
            FieldSparseEntry::merge_walk(evens, odds, |even, odd| {
                FieldSparseEntry::accumulate_pair_evals(even, odd, inc_evals, weight, &mut acc);
            });
        }
        [acc[0].reduce(), acc[1].reduce()]
    };

    #[cfg(feature = "parallel")]
    if entries.len() >= PARALLEL_THRESHOLD {
        let bounds = pair_aligned_bounds(entries);
        return (0..bounds.len() - 1)
            .into_par_iter()
            .map(|block| range_contribution(bounds[block]..bounds[block + 1]))
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    }
    range_contribution(0..entries.len())
}

/// Bind one cycle variable of the sparse matrix: merge every adjacent row
/// pair into `output` (cleared and refilled — the caller swaps buffers).
fn bind_sparse_entries<F: JoltField>(
    entries: &[FieldSparseEntry<F>],
    r: F,
    output: &mut Vec<FieldSparseEntry<F>>,
) {
    output.clear();
    let merge_range = |range: Range<usize>, out: &mut Vec<FieldSparseEntry<F>>| {
        for group in entries[range].chunk_by(|a, b| a.row / 2 == b.row / 2) {
            let (evens, odds) = FieldSparseEntry::split_pair_group(group);
            FieldSparseEntry::merge_walk(evens, odds, |even, odd| {
                out.push(FieldSparseEntry::bind(even, odd, r));
            });
        }
    };

    #[cfg(feature = "parallel")]
    if entries.len() >= PARALLEL_THRESHOLD {
        let bounds = pair_aligned_bounds(entries);
        let blocks: Vec<Vec<FieldSparseEntry<F>>> = (0..bounds.len() - 1)
            .into_par_iter()
            .map(|block| {
                let range = bounds[block]..bounds[block + 1];
                let mut out = Vec::with_capacity(range.len());
                merge_range(range, &mut out);
                out
            })
            .collect();
        for block in blocks {
            output.extend_from_slice(&block);
        }
        return;
    }
    output.reserve(entries.len());
    merge_range(0..entries.len(), output);
}

/// The rd write slots of one proof's FR trace — `(cycle, register)` pairs of
/// every bytecode-active FR write — parked by the stage-4 kernel for the
/// stage-5 val-evaluation kernel (which folds the same one-hot `FieldRdWa`
/// grid at its address prefix).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedFieldRdWrites(pub(crate) Vec<(u32, u8)>);

/// Sparse per-cycle FR access facts extracted from the oracle's decoded rows:
/// the ≤3-entries-per-active-cycle matrix cells plus the raw read/write index
/// lists (reads feed the final one-hot claims, writes feed stage 5).
pub(crate) struct FieldRegisterAccesses<F: JoltField> {
    entries: Vec<FieldSparseEntry<F>>,
    rs1_reads: Vec<(u32, u8)>,
    rs2_reads: Vec<(u32, u8)>,
    pub(crate) rd_writes: Vec<(u32, u8)>,
}

impl<F: JoltField> FieldRegisterAccesses<F> {
    /// One pass over the decoded rows against a running register file (the
    /// all-zero initial state every FR execution shares — enforced by the
    /// stage-5 val-evaluation identity, not merely assumed). `Val` cells use
    /// the running value, exactly as the oracle's dense
    /// `FieldRegistersVal` materializer replays writes; the witness view's
    /// build-time validation pins the rows' claimed pre-values to the same
    /// replay.
    pub(crate) fn collect(
        rows: &[FieldInlineRegisterReadWriteRow<F>],
        register_count: usize,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let gamma_sq = gamma * gamma;
        let mut running: Vec<F> = vec![F::zero(); register_count];
        let mut entries: Vec<FieldSparseEntry<F>> = Vec::new();
        let mut rs1_reads: Vec<(u32, u8)> = Vec::new();
        let mut rs2_reads: Vec<(u32, u8)> = Vec::new();
        let mut rd_writes: Vec<(u32, u8)> = Vec::new();
        let in_domain = |register: u8| -> Result<usize, KernelError<F>> {
            let col = usize::from(register);
            if col >= register_count {
                return Err(KernelError::InvariantViolation {
                    reason: "FR register index outside the field-register domain",
                });
            }
            Ok(col)
        };
        for (row, access) in rows.iter().enumerate() {
            let start = entries.len();
            if let Some(read) = &access.rs1 {
                let col = in_domain(read.register)?;
                rs1_reads.push((row as u32, read.register));
                let val = running[col];
                entries.push(FieldSparseEntry {
                    val,
                    prev_val: val,
                    next_val: val,
                    ra: gamma,
                    wa: F::zero(),
                    row,
                    col: read.register,
                });
            }
            if let Some(read) = &access.rs2 {
                let col = in_domain(read.register)?;
                rs2_reads.push((row as u32, read.register));
                if let Some(entry) = entries[start..]
                    .iter_mut()
                    .find(|entry| usize::from(entry.col) == col)
                {
                    entry.ra += gamma_sq;
                } else {
                    let val = running[col];
                    entries.push(FieldSparseEntry {
                        val,
                        prev_val: val,
                        next_val: val,
                        ra: gamma_sq,
                        wa: F::zero(),
                        row,
                        col: read.register,
                    });
                }
            }
            if let Some(write) = &access.rd {
                let col = in_domain(write.register)?;
                rd_writes.push((row as u32, write.register));
                let post = write.post_value;
                if let Some(entry) = entries[start..]
                    .iter_mut()
                    .find(|entry| usize::from(entry.col) == col)
                {
                    entry.wa = F::one();
                    entry.next_val = post;
                } else {
                    let pre = running[col];
                    entries.push(FieldSparseEntry {
                        val: pre,
                        prev_val: pre,
                        next_val: post,
                        ra: F::zero(),
                        wa: F::one(),
                        row,
                        col: write.register,
                    });
                }
                running[col] = post;
            }
            entries[start..].sort_unstable_by_key(|entry| entry.col);
        }
        Ok(Self {
            entries,
            rs1_reads,
            rs2_reads,
            rd_writes,
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
        // The FR phase split is pinned by the compile-time protocol config
        // (phase 1 = log_t, phase 2 = log_k) — the same guard as the
        // reference kernel: a drifted config is a bug, not a capability gap.
        if dimensions.phase1_num_rounds() != dimensions.log_t()
            || dimensions.phase2_num_rounds() != dimensions.log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "FR read-write dimensions drifted from the config-pinned phase split",
            });
        }
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized FR read-write checking requires at least one cycle round",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "FR read-write upstream cycle point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers read-write checking field-inline oracle",
                }))?;
        let rows = field_inline.field_inline_register_read_write_rows()?;
        if rows.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "field-inline register read-write rows".to_owned(),
                expected: cycles,
                got: rows.len(),
            });
        }
        let inc_table = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        if inc_table.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc".to_owned(),
                expected: cycles,
                got: inc_table.len(),
            });
        }
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersReadWriteChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "FR read-write checking is missing its gamma challenge",
            })?;

        let FieldRegisterAccesses {
            entries,
            rs1_reads,
            rs2_reads,
            rd_writes,
        } = FieldRegisterAccesses::collect(&rows, 1usize << log_k, gamma)?;

        // Park the rd write slots for the stage-5 FR val-evaluation kernel.
        session.park(SharedFieldRdWrites(rd_writes));

        Ok(Box::new(FieldReadWriteKernel {
            log_t,
            log_k,
            entries,
            scratch: Vec::new(),
            gruen: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            inc: Polynomial::new(inc_table),
            ra: Vec::new(),
            wa: Vec::new(),
            val: Vec::new(),
            eq_scalar: F::zero(),
            inc_scalar: F::zero(),
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
    entries: Vec<FieldSparseEntry<F>>,
    scratch: Vec<FieldSparseEntry<F>>,
    gruen: GruenSplitEqPolynomial<F>,
    inc: Polynomial<F>,
    // Address-phase dense state (K = 16), materialized at the transition.
    ra: Vec<F>,
    wa: Vec<F>,
    val: Vec<F>,
    /// Fully bound `eq(r_cycle, ·)` — constant across the address rounds.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    eq_scalar: F,
    /// Fully bound `FieldRdInc` — constant across the address rounds.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    inc_scalar: F,
    rs1_reads: Vec<(u32, u8)>,
    rs2_reads: Vec<(u32, u8)>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> FieldReadWriteKernel<F> {
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining sparse rows, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let quadratic = sparse_quadratic(
            &self.entries,
            self.gruen.e_in_current(),
            self.gruen.e_out_current(),
            self.inc.evals(),
        );
        self.gruen
            .gruen_poly_deg_3(quadratic[0], quadratic[1], previous_claim)
    }

    /// Address-round message over the K-sized dense arrays. Cheap enough to
    /// sample all `degree + 1` points directly, so the naive tier's running
    /// claim self-check is kept.
    fn address_round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let half = self.ra.len() / 2;
        let mut evals = [F::zero(); 4];
        for y in 0..half {
            let pair = |table: &[F]| {
                let lo = table[2 * y];
                (lo, table[2 * y + 1] - lo)
            };
            let (ra_0, ra_m) = pair(&self.ra);
            let (wa_0, wa_m) = pair(&self.wa);
            let (val_0, val_m) = pair(&self.val);
            let (mut ra_t, mut wa_t, mut val_t) = (ra_0, wa_0, val_0);
            for eval in &mut evals {
                *eval += wa_t * (self.inc_scalar + val_t) + ra_t * val_t;
                ra_t += ra_m;
                wa_t += wa_m;
                val_t += val_m;
            }
        }
        let evals = evals.map(|eval| self.eq_scalar * eval);
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    /// Bind the pending challenge: cycle rounds bind eq/inc and merge the
    /// sparse rows; the final cycle bind collapses to the K-sized dense
    /// address state; address rounds bind the three dense arrays.
    fn bind(&mut self, r: F) {
        if self.challenges.bound() < self.log_t {
            self.gruen.bind(r);
            self.inc.bind_with_order(r, BindingOrder::LowToHigh);
            bind_sparse_entries(&self.entries, r, &mut self.scratch);
            core::mem::swap(&mut self.entries, &mut self.scratch);
        } else {
            for table in [&mut self.ra, &mut self.wa, &mut self.val] {
                bind_pairs(table, r);
            }
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.log_t {
            let register_count = 1usize << self.log_k;
            let mut ra = vec![F::zero(); register_count];
            let mut wa = vec![F::zero(); register_count];
            let mut val = vec![F::zero(); register_count];
            for entry in self.entries.drain(..) {
                debug_assert_eq!(entry.row, 0);
                ra[usize::from(entry.col)] = entry.ra;
                wa[usize::from(entry.col)] = entry.wa;
                val[usize::from(entry.col)] = entry.val;
            }
            // Free the scratch here rather than at kernel drop.
            self.scratch = Vec::new();
            self.ra = ra;
            self.wa = wa;
            self.val = val;
            self.eq_scalar = self.gruen.current_scalar();
            self.inc_scalar = self.inc.evals()[0];
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
        let log_t = r_cycle.len();
        let addr_bits = r_address.len();
        let n = log_t + addr_bits;
        let hi_bits = core::cmp::min(log_t, n.div_ceil(2));

        let r_joint: Vec<F> = r_cycle.iter().chain(r_address.iter()).copied().collect();
        let (r_hi, r_lo) = r_joint.split_at(hi_bits);
        let e_hi = EqPolynomial::<F>::evals(r_hi, None);
        let e_lo = EqPolynomial::<F>::evals(r_lo, None);

        let cycle_bits_in_lo = (n - hi_bits) - addr_bits;
        let cycle_lo_mask = (1usize << cycle_bits_in_lo) - 1;

        let claim = |reads: &[(u32, u8)]| -> F {
            let mut sum = F::Accumulator::default();
            for &(j, k) in reads {
                let j = j as usize;
                let lo_index = ((j & cycle_lo_mask) << addr_bits) | usize::from(k);
                sum.fmadd(e_hi[j >> cycle_bits_in_lo], e_lo[lo_index]);
            }
            sum.reduce()
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
            Ok(self.cycle_round_message(previous_claim))
        } else {
            self.address_round_message(round, previous_claim)
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
            registers_val: self.val[0],
            rs1_ra,
            rs2_ra,
            rd_wa: self.wa[0],
            rd_inc: self.inc_scalar,
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
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersReadWritePublic::EqCycle),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.eq_scalar;
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersReadWriteChecking".to_string(),
                    reason: format!(
                        "bound eq scalar {got:?}, but derive_output_term gives {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}

/// Byte parity against the reference kernel on register-consistent FR
/// traces: identical round polynomials at every round (cycle and address
/// phases), equal typed output claims, and both kernels' derived-table
/// validation — plus the FR-inactive degenerate case, where the sparse state
/// is empty and every round polynomial is honestly zero.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::field_inline::relations::registers::{
        FieldRegistersReadWriteChallenges, FieldRegistersReadWriteInputClaims,
    };
    use jolt_field::{Fr, Ring};
    use jolt_riscv::FieldInlineOp;
    use jolt_verifier::stages::stage4::field_inline::read_write_member;

    use super::*;
    use crate::optimized::field_registers_testing::{
        inactive_fr_fixture, structured_fr_fixture, FrTraceFixture,
    };
    use crate::optimized::parity::{
        probe_input_claim, run_lockstep, run_lockstep_degenerate, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn run_parity(fixture: FrTraceFixture, log_t: usize, seed: u64, expect_active: bool) {
        fixture.with_plane(log_t, |backend| {
            let relation = read_write_member::<Fr>(log_t);
            let r_cycle = synthetic_point(log_t, seed);
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
            let mut optimized = OptimizedFieldRegistersReadWrite
                .prepare(&mut session, backend, inputs())
                .unwrap();
            assert!(
                session.state::<SharedFieldRdWrites>().is_some(),
                "the optimized kernel must park the FR write slots for stage 5",
            );

            let claim = probe_input_claim(reference.as_mut());
            let round_challenges =
                synthetic_point(relation.rounds(), seed.wrapping_mul(0x9E37_79B9));
            if expect_active {
                assert!(claim != Fr::from_u64(0), "FR-active fixture degenerated");
                run_lockstep(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            } else {
                assert_eq!(claim, Fr::from_u64(0), "FR-inactive claim must be zero");
                run_lockstep_degenerate(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            }
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fr_fixture(16), 4, 101, true);
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fr_fixture(8), 3, 103, true);
    }

    #[test]
    fn parity_partially_padded_trace() {
        // Real rows in the front half only: the padding tail exercises the
        // constant-value slices the sparse boundary values reconstruct.
        run_parity(structured_fr_fixture(9), 5, 107, true);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = FrTraceFixture::new();
        fixture.load_imm(2, 99);
        fixture.arithmetic(FieldInlineOp::Mul, 2, 2, 2);
        run_parity(fixture, 1, 109, true);
    }

    #[test]
    fn parity_inactive_trace_is_degenerate_and_cheap() {
        run_parity(inactive_fr_fixture(4), 3, 113, false);
    }
}
