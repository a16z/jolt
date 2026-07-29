//! The optimized registers read/write-checking (stage 4) kernel: the legacy
//! prover's sparse-matrix algorithm behind the `PrepareKernel` seam.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_read_write.rs`), which sums the
//! summand over dense `2^(log_K + log_T)` register-major tables. This kernel
//! computes the same polynomials from the sparse structure of the one-hot
//! grids — field arithmetic is exact, so algebraic refactorings (eq
//! factoring, γ-combined ra, deferred-reduction accumulation) preserve every
//! wire byte.
//!
//! Techniques ported from
//! `jolt-prover-legacy/src/zkvm/registers/read_write_checking.rs` and
//! `subprotocols/read_write_matrix/{cycle_major,registers}.rs`:
//!
//! - **Sparse cycle-major matrix**: `rd_wa`/`rs1_ra`/`rs2_ra`/`Val` are
//!   represented by ≤ 3 entries per cycle (the touched registers) instead of
//!   three dense `K × T` grids. Between touches a register's value is
//!   constant, so a missing merge partner is inferred from its neighbor's
//!   raw `prev_val`/`next_val` (a constant slice binds to itself).
//! - **γ-combined read coefficient**: one `ra = γ·rs1_ra + γ²·rs2_ra` column
//!   per entry (exact by distributivity).
//! - **Gruen split-eq factoring** for the cycle rounds:
//!   `s(t) = l(t) · Σ_z E_out·E_in·inner(t, z)` via
//!   [`GruenSplitEqPolynomial::gruen_poly_deg_3`].
//! - **Small fixed K**: after the cycle rounds the state collapses to three
//!   `K = 2^REGISTER_ADDRESS_BITS` dense arrays plus two scalars (bound eq,
//!   bound inc); address rounds cost O(K).
//! - **Direct one-hot claims at extraction**: `rs1_ra(r)`/`rs2_ra(r)` are
//!   computed straight from the per-cycle indices with a 2-way split-eq walk
//!   (legacy's `compute_rs2_ra_claim`, applied to both operands — no γ⁻¹).
//!
//! Not ported: the u16 coefficient lookup table (legacy's
//! `OneHotCoeffLookupTable`) — it halves entry memory for the first ~3
//! rounds only; entries here carry field-element coefficients from round 0.
//! Like the reference kernel, only the default read-write config (phase 1 =
//! all cycle rounds, phase 2 = 0) is supported.

use jolt_claims::protocols::jolt::geometry::registers::rd_inc_read_write;
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltPolynomialId, RegistersReadWritePublic};
use jolt_field::{AdditiveAccumulator, Field, OptimizedMul, RingAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-cycle register activity: operand indices plus the raw values the
/// sparse entries and direct one-hot claims are built from. Hand-implemented
/// bundle — the fields carry no protocol ids, and no atomic witness newtype
/// exposes the operand *indices*.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RegisterCycleRow {
    /// `(register, read value)`.
    pub rs1: Option<(u8, u64)>,
    /// `(register, read value)`.
    pub rs2: Option<(u8, u64)>,
    /// `(register, pre-write value, post-write value)`.
    pub rd: Option<(u8, u64, u64)>,
}

impl WitnessBundle for RegisterCycleRow {
    // `TraceRow` is nameable from this crate only through the doc-hidden
    // re-export the bundle derive uses; jolt-kernels deliberately has no
    // jolt-program dependency.
    fn from_row(
        row: &jolt_witness::__private::TraceRow,
        _next: Option<&jolt_witness::__private::TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self {
            rs1: row.registers.rs1.map(|read| (read.register, read.value)),
            rs2: row.registers.rs2.map(|read| (read.register, read.value)),
            rd: row
                .registers
                .rd
                .map(|write| (write.register, write.pre_value, write.post_value)),
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

/// Cross-member carry: the per-cycle `rd` hot indices, parked by this kernel's
/// `prepare` for the stage-5 val-evaluation kernel (which otherwise re-walks
/// the trace to collect them).
pub(crate) struct SharedRdIndices(pub Vec<Option<u8>>);

/// One non-zero cell of the conceptual `K × T` register matrices: the bound
/// `Val` coefficient plus the γ-combined read and write coefficients of one
/// touched register slice.
///
/// `prev_val`/`next_val` stay raw `u64`s: a register is constant between
/// touches, and a constant slice's bound coefficient is the constant itself,
/// so the values neighboring this entry's slice never need field form until
/// they participate in a merge.
#[derive(Clone, Copy, Debug)]
struct SparseEntry<F> {
    /// Bound `Val(col, row-slice)` coefficient (value *before* the access).
    val: F,
    /// Bound `γ·rs1_ra + γ²·rs2_ra` coefficient.
    ra: F,
    /// Bound `rd_wa` coefficient.
    wa: F,
    /// Register value just before this entry's row slice.
    prev_val: u64,
    /// Register value just after this entry's row slice.
    next_val: u64,
    /// Cycle-domain row index (before binding: the cycle).
    row: usize,
    /// Register index.
    col: u8,
}

impl<F: Field> SparseEntry<F> {
    /// Bind two vertically adjacent cells (rows `2j`/`2j+1`, same column)
    /// with `r`. A missing side is an untouched slice: its `Val` is the
    /// neighbor's raw boundary value and its `ra`/`wa` are zero.
    fn bind(even: Option<&Self>, odd: Option<&Self>, r: F) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                Self {
                    val: even.val + r.mul_0_optimized(odd.val - even.val),
                    ra: even.ra + r.mul_0_optimized(odd.ra - even.ra),
                    wa: even.wa + r.mul_0_optimized(odd.wa - even.wa),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                    row: even.row / 2,
                    col: even.col,
                }
            }
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                let one_minus_r = F::one() - r;
                Self {
                    val: even.val + r.mul_0_optimized(odd_val - even.val),
                    ra: one_minus_r.mul_01_optimized(even.ra),
                    wa: one_minus_r.mul_01_optimized(even.wa),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                    row: even.row / 2,
                    col: even.col,
                }
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                Self {
                    val: even_val + r.mul_0_optimized(odd.val - even_val),
                    ra: r.mul_01_optimized(odd.ra),
                    wa: r.mul_01_optimized(odd.wa),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                    row: odd.row / 2,
                    col: odd.col,
                }
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    /// Accumulate this vertical pair's `[t = 0, t = ∞]` contributions to the
    /// quadratic inner factor: `ra_t·val_t + wa_t·(val_t + inc_t)`.
    fn accumulate_pair_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        acc: &mut [F::Accumulator; 2],
    ) where
        F::Accumulator: RingAccumulator,
    {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                acc[0].fmadd(even.ra, even.val);
                acc[0].fmadd(even.wa, even.val + inc_evals[0]);
                let (ra_m, wa_m, val_m) = (odd.ra - even.ra, odd.wa - even.wa, odd.val - even.val);
                acc[1].fmadd(ra_m, val_m);
                acc[1].fmadd(wa_m, val_m + inc_evals[1]);
            }
            (Some(even), None) => {
                let val_m = F::from_u64(even.next_val) - even.val;
                acc[0].fmadd(even.ra, even.val);
                acc[0].fmadd(even.wa, even.val + inc_evals[0]);
                acc[1].fmadd(-even.ra, val_m);
                acc[1].fmadd(-even.wa, val_m + inc_evals[1]);
            }
            (None, Some(odd)) => {
                // The even side has zero ra/wa, so the t = 0 term vanishes.
                let val_m = odd.val - F::from_u64(odd.prev_val);
                acc[1].fmadd(odd.ra, val_m);
                acc[1].fmadd(odd.wa, val_m + inc_evals[1]);
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }
}

/// Build the (sorted-by-column) sparse entries of one cycle. Returns the
/// filled prefix length (0–3).
fn cycle_entries<F: Field>(
    row: usize,
    cycle: &RegisterCycleRow,
    gamma: F,
    gamma_sq: F,
) -> ([SparseEntry<F>; 3], usize) {
    let empty = SparseEntry {
        val: F::zero(),
        ra: F::zero(),
        wa: F::zero(),
        prev_val: 0,
        next_val: 0,
        row,
        col: 0,
    };
    let mut out = [empty; 3];
    let mut len = 0usize;

    if let Some((rs1, rs1_val)) = cycle.rs1 {
        out[len] = SparseEntry {
            col: rs1,
            prev_val: rs1_val,
            next_val: rs1_val,
            val: F::from_u64(rs1_val),
            ra: gamma,
            ..empty
        };
        len += 1;
    }
    if let Some((rs2, rs2_val)) = cycle.rs2 {
        if let Some(entry) = out[..len].iter_mut().find(|entry| entry.col == rs2) {
            entry.ra = gamma + gamma_sq;
        } else {
            out[len] = SparseEntry {
                col: rs2,
                prev_val: rs2_val,
                next_val: rs2_val,
                val: F::from_u64(rs2_val),
                ra: gamma_sq,
                ..empty
            };
            len += 1;
        }
    }
    if let Some((rd, rd_pre, rd_post)) = cycle.rd {
        if let Some(entry) = out[..len].iter_mut().find(|entry| entry.col == rd) {
            entry.wa = F::one();
            entry.next_val = rd_post;
        } else {
            out[len] = SparseEntry {
                col: rd,
                prev_val: rd_pre,
                next_val: rd_post,
                val: F::from_u64(rd_pre),
                wa: F::one(),
                ..empty
            };
            len += 1;
        }
    }

    // Sort by column; len ≤ 3.
    out[..len].sort_unstable_by_key(|entry| entry.col);
    (out, len)
}

/// Merged length of two adjacent sorted-by-column rows (a bind dry run —
/// the count is value-independent).
fn merge_count<F: Field>(evens: &[SparseEntry<F>], odds: &[SparseEntry<F>]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut produced = 0;
    while i < evens.len() && j < odds.len() {
        match evens[i].col.cmp(&odds[j].col) {
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
        }
        produced += 1;
    }
    produced + (evens.len() - i) + (odds.len() - j)
}

/// Merge-bind two adjacent sorted-by-column rows into `out` (sized by
/// [`merge_count`]), keeping column order.
fn merge_fill<F: Field>(
    evens: &[SparseEntry<F>],
    odds: &[SparseEntry<F>],
    r: F,
    out: &mut [core::mem::MaybeUninit<SparseEntry<F>>],
) {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < evens.len() && j < odds.len() {
        let bound = match evens[i].col.cmp(&odds[j].col) {
            core::cmp::Ordering::Equal => {
                let entry = SparseEntry::bind(Some(&evens[i]), Some(&odds[j]), r);
                i += 1;
                j += 1;
                entry
            }
            core::cmp::Ordering::Less => {
                let entry = SparseEntry::bind(Some(&evens[i]), None, r);
                i += 1;
                entry
            }
            core::cmp::Ordering::Greater => {
                let entry = SparseEntry::bind(None, Some(&odds[j]), r);
                j += 1;
                entry
            }
        };
        out[k] = core::mem::MaybeUninit::new(bound);
        k += 1;
    }
    for even in &evens[i..] {
        out[k] = core::mem::MaybeUninit::new(SparseEntry::bind(Some(even), None, r));
        k += 1;
    }
    for odd in &odds[j..] {
        out[k] = core::mem::MaybeUninit::new(SparseEntry::bind(None, Some(odd), r));
        k += 1;
    }
    debug_assert_eq!(k, out.len());
}

/// Split a row-pair group (entries sharing `row / 2`) into its even and odd
/// rows. Entries are sorted by `(row, col)`, so the evens form the prefix.
fn split_pair_group<F: Field>(group: &[SparseEntry<F>]) -> (&[SparseEntry<F>], &[SparseEntry<F>]) {
    let odd_start = group.partition_point(|entry| entry.row % 2 == 0);
    group.split_at(odd_start)
}

pub struct OptimizedRegistersReadWrite;

impl<F> PrepareKernel<F, RegistersReadWriteChecking<F>> for OptimizedRegistersReadWrite
where
    F: Field,
    F::Accumulator: RingAccumulator,
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let dimensions = inputs.relation.register_dimensions();
        // Same guard as the reference kernel: phase 1 must cover all cycle
        // rounds. The phase-2/phase-3 split of the address rounds is a legacy
        // data-structure choice with no effect on the round polynomials (the
        // default config sets phase 2 = all `log_K` address rounds), so it is
        // deliberately not constrained here.
        if dimensions.phase1_num_rounds() != dimensions.log_t() {
            return Err(KernelError::Unsupported {
                reason: "optimized registers read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds)",
            });
        }
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers read-write checking requires at least one cycle round",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_write_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write input point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;

        let inc_table: Vec<F> = witness.oracle_table(rd_inc_read_write().polynomial_id())?;
        if inc_table.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", rd_inc_read_write()),
                expected: cycles,
                got: inc_table.len(),
            });
        }

        let rows: Vec<RegisterCycleRow> = collect_bundles(witness, cycles)?;
        let gamma = inputs.challenges.gamma;
        let gamma_sq = gamma * gamma;

        // Sparse entry construction: exact per-chunk sizing, one pass.
        let build_chunk = |(chunk_index, chunk): (usize, &[RegisterCycleRow])| {
            let mut entries = Vec::with_capacity(chunk.len() * 3);
            for (offset, cycle) in chunk.iter().enumerate() {
                let row = chunk_index * BUILD_CHUNK + offset;
                let (cells, len) = cycle_entries(row, cycle, gamma, gamma_sq);
                entries.extend_from_slice(&cells[..len]);
            }
            entries
        };
        const BUILD_CHUNK: usize = 1 << 12;
        #[cfg(feature = "parallel")]
        let chunked: Vec<Vec<SparseEntry<F>>> = rows
            .par_chunks(BUILD_CHUNK)
            .enumerate()
            .map(build_chunk)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let chunked: Vec<Vec<SparseEntry<F>>> = rows
            .chunks(BUILD_CHUNK)
            .enumerate()
            .map(build_chunk)
            .collect();
        let mut entries = Vec::with_capacity(chunked.iter().map(Vec::len).sum());
        for chunk in chunked {
            entries.extend_from_slice(&chunk);
        }

        let rs1_indices: Vec<Option<u8>> = rows.iter().map(|row| row.rs1.map(|(k, _)| k)).collect();
        let rs2_indices: Vec<Option<u8>> = rows.iter().map(|row| row.rs2.map(|(k, _)| k)).collect();
        // Park the rd hot indices for the stage-5 val-evaluation kernel.
        session.park(SharedRdIndices(
            rows.iter().map(|row| row.rd.map(|(k, ..)| k)).collect(),
        ));

        Ok(Box::new(ReadWriteKernel {
            log_t,
            log_k,
            entries,
            gruen: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            inc: Polynomial::new(inc_table),
            ra: Vec::new(),
            wa: Vec::new(),
            val: Vec::new(),
            eq_scalar: F::zero(),
            inc_scalar: F::zero(),
            rs1_indices,
            rs2_indices,
            bound_challenges: Vec::with_capacity(log_t + log_k),
            rounds_bound: 0,
        }))
    }
}

struct ReadWriteKernel<F: Field> {
    log_t: usize,
    log_k: usize,
    /// Sparse cycle-major entries, sorted by `(row, col)`; drained at the
    /// cycle→address transition.
    entries: Vec<SparseEntry<F>>,
    gruen: GruenSplitEqPolynomial<F>,
    inc: Polynomial<F>,
    // Address-phase dense state (K-sized), materialized at the transition.
    ra: Vec<F>,
    wa: Vec<F>,
    val: Vec<F>,
    /// Fully bound `eq(r_cycle, ·)` — constant across the address rounds.
    eq_scalar: F,
    /// Fully bound `rd_inc` — constant across the address rounds.
    inc_scalar: F,
    rs1_indices: Vec<Option<u8>>,
    rs2_indices: Vec<Option<u8>>,
    bound_challenges: Vec<F>,
    rounds_bound: usize,
}

impl<F: Field> ReadWriteKernel<F>
where
    F::Accumulator: RingAccumulator,
{
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining cycle domain, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let e_in = self.gruen.e_in_current();
        let e_in_len = e_in.len();
        let in_bits = if e_in_len <= 1 {
            0
        } else {
            e_in_len.trailing_zeros() as usize
        };
        let mask = (1usize << in_bits) - 1;
        let e_out = self.gruen.e_out_current();
        let inc = self.inc.evals();

        let group_contribution = |group: &[SparseEntry<F>]| -> [F; 2] {
            let x_out = (group[0].row / 2) >> in_bits;
            let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
            for pair_group in group.chunk_by(|a, b| a.row / 2 == b.row / 2) {
                let z = pair_group[0].row / 2;
                let e_in_eval = if e_in_len <= 1 {
                    F::one()
                } else {
                    e_in[z & mask]
                };
                let j_prime = 2 * z;
                let inc_0 = inc[j_prime];
                let inc_evals = [inc_0, inc[j_prime + 1] - inc_0];

                let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
                let (evens, odds) = split_pair_group(pair_group);
                let mut i = 0;
                let mut j = 0;
                while i < evens.len() && j < odds.len() {
                    match evens[i].col.cmp(&odds[j].col) {
                        core::cmp::Ordering::Equal => {
                            SparseEntry::accumulate_pair_evals(
                                Some(&evens[i]),
                                Some(&odds[j]),
                                inc_evals,
                                &mut inner,
                            );
                            i += 1;
                            j += 1;
                        }
                        core::cmp::Ordering::Less => {
                            SparseEntry::accumulate_pair_evals(
                                Some(&evens[i]),
                                None,
                                inc_evals,
                                &mut inner,
                            );
                            i += 1;
                        }
                        core::cmp::Ordering::Greater => {
                            SparseEntry::accumulate_pair_evals(
                                None,
                                Some(&odds[j]),
                                inc_evals,
                                &mut inner,
                            );
                            j += 1;
                        }
                    }
                }
                for even in &evens[i..] {
                    SparseEntry::accumulate_pair_evals(Some(even), None, inc_evals, &mut inner);
                }
                for odd in &odds[j..] {
                    SparseEntry::accumulate_pair_evals(None, Some(odd), inc_evals, &mut inner);
                }

                acc[0].fmadd(e_in_eval, inner[0].reduce());
                acc[1].fmadd(e_in_eval, inner[1].reduce());
            }
            let e_out_eval = e_out[x_out];
            [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
        };

        let group_predicate = |a: &SparseEntry<F>, b: &SparseEntry<F>| {
            (a.row / 2) >> in_bits == (b.row / 2) >> in_bits
        };
        #[cfg(feature = "parallel")]
        let quadratic = self
            .entries
            .par_chunk_by(group_predicate)
            .map(group_contribution)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        #[cfg(not(feature = "parallel"))]
        let quadratic = self
            .entries
            .chunk_by(group_predicate)
            .map(group_contribution)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

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

    fn bind_sparse(&mut self, r: F) {
        let pair_predicate = |a: &SparseEntry<F>, b: &SparseEntry<F>| a.row / 2 == b.row / 2;
        let count_group = |group: &[SparseEntry<F>]| -> (usize, usize) {
            let (evens, odds) = split_pair_group(group);
            (group.len(), merge_count(evens, odds))
        };
        // Dry-run pass: per row-pair group, the input length and the exact
        // bound length — sizes the single output allocation.
        #[cfg(feature = "parallel")]
        let group_lengths: Vec<(usize, usize)> = self
            .entries
            .par_chunk_by(pair_predicate)
            .map(count_group)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let group_lengths: Vec<(usize, usize)> = self
            .entries
            .chunk_by(pair_predicate)
            .map(count_group)
            .collect();

        let bound_length = group_lengths.iter().map(|(_, bound)| bound).sum();
        let mut bound: Vec<SparseEntry<F>> = Vec::with_capacity(bound_length);
        let mut out_slices = Vec::with_capacity(group_lengths.len());
        let mut in_slices = Vec::with_capacity(group_lengths.len());
        let mut out_rest = bound.spare_capacity_mut();
        let mut in_rest = self.entries.as_slice();
        for &(unbound_len, bound_len) in &group_lengths {
            let (out_slice, next_out) = out_rest.split_at_mut(bound_len);
            out_rest = next_out;
            out_slices.push(out_slice);
            let (in_slice, next_in) = in_rest.split_at(unbound_len);
            in_rest = next_in;
            in_slices.push(in_slice);
        }

        let fill = |(group, out): (
            &&[SparseEntry<F>],
            &mut [core::mem::MaybeUninit<SparseEntry<F>>],
        )| {
            let (evens, odds) = split_pair_group(group);
            merge_fill(evens, odds, r, out);
        };
        #[cfg(feature = "parallel")]
        in_slices
            .par_iter()
            .zip(out_slices.into_par_iter())
            .for_each(fill);
        #[cfg(not(feature = "parallel"))]
        in_slices.iter().zip(out_slices).for_each(fill);

        // SAFETY: the dry-run pass sized every group's output slice exactly,
        // the slices partition `bound`'s spare capacity up to `bound_length`,
        // and `merge_fill` writes each slot of its slice exactly once.
        unsafe {
            bound.set_len(bound_length);
        }
        self.entries = bound;
    }

    /// Bind the pending challenge: cycle rounds bind eq/inc and merge the
    /// sparse rows; the final cycle bind collapses to the K-sized dense
    /// address state; address rounds bind the three dense arrays.
    fn bind(&mut self, r: F) {
        if self.rounds_bound < self.log_t {
            self.gruen.bind(r);
            self.inc.bind_with_order(r, BindingOrder::LowToHigh);
            self.bind_sparse(r);
        } else {
            for table in [&mut self.ra, &mut self.wa, &mut self.val] {
                let half = table.len() / 2;
                for y in 0..half {
                    let lo = table[2 * y];
                    table[y] = lo + r * (table[2 * y + 1] - lo);
                }
                table.truncate(half);
            }
        }
        self.bound_challenges.push(r);
        self.rounds_bound += 1;

        if self.rounds_bound == self.log_t {
            let k = 1usize << self.log_k;
            let mut ra = vec![F::zero(); k];
            let mut wa = vec![F::zero(); k];
            let mut val = vec![F::zero(); k];
            // `take` frees the entry allocation here rather than at kernel drop.
            for entry in std::mem::take(&mut self.entries) {
                debug_assert_eq!(entry.row, 0);
                ra[entry.col as usize] = entry.ra;
                wa[entry.col as usize] = entry.wa;
                val[entry.col as usize] = entry.val;
            }
            self.ra = ra;
            self.wa = wa;
            self.val = val;
            self.eq_scalar = self.gruen.current_scalar();
            self.inc_scalar = self.inc.evals()[0];
        }
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        let remaining = (self.log_t + self.log_k) - self.rounds_bound;
        if remaining == 0 {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound { remaining })
        }
    }

    /// The bound opening point, split as `(r_address, r_cycle)` — the same
    /// reversal `ReadWriteDimensions::read_write_opening_point` applies under
    /// the default config.
    fn bound_point(&self) -> (Vec<F>, Vec<F>) {
        let r_cycle: Vec<F> = self.bound_challenges[..self.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
        let r_address: Vec<F> = self.bound_challenges[self.log_t..]
            .iter()
            .rev()
            .copied()
            .collect();
        (r_address, r_cycle)
    }
}

/// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the two
/// read operands in one walk — the direct MLE of a one-hot `(K × T)` grid at
/// the bound point.
///
/// Ports legacy `compute_rs2_ra_claim`: a 2-way split over the joint
/// `(cycle ‖ address)` index keeps both eq tables at ~√(K·T). Big-endian
/// joint point `[r_cycle ‖ r_address]`, joint index `(j << addr_bits) | k`.
fn one_hot_operand_claims<F: Field>(
    rs1_indices: &[Option<u8>],
    rs2_indices: &[Option<u8>],
    r_address: &[F],
    r_cycle: &[F],
) -> (F, F) {
    let log_t = r_cycle.len();
    let addr_bits = r_address.len();
    let n = log_t + addr_bits;
    let hi_bits = core::cmp::min(log_t, n.div_ceil(2));

    let r_joint: Vec<F> = r_cycle.iter().chain(r_address.iter()).copied().collect();
    let (r_hi, r_lo) = r_joint.split_at(hi_bits);
    let e_hi = EqPolynomial::<F>::evals(r_hi, None);
    let e_lo = EqPolynomial::<F>::evals(r_lo, None);

    let cycle_bits_in_lo = (n - hi_bits) - addr_bits;
    let cycles_per_block = 1usize << cycle_bits_in_lo;
    let cycle_lo_mask = cycles_per_block - 1;

    let block_contribution = |idx_hi: usize| -> [F; 2] {
        let block_start = idx_hi << cycle_bits_in_lo;
        let block_end = core::cmp::min(block_start + cycles_per_block, rs1_indices.len());
        if block_start >= rs1_indices.len() {
            return [F::zero(); 2];
        }
        let mut sums = [F::Accumulator::default(), F::Accumulator::default()];
        for j in block_start..block_end {
            let j_in_block = (j & cycle_lo_mask) << addr_bits;
            if let Some(rs1) = rs1_indices[j] {
                sums[0].add(e_lo[j_in_block | rs1 as usize]);
            }
            if let Some(rs2) = rs2_indices[j] {
                sums[1].add(e_lo[j_in_block | rs2 as usize]);
            }
        }
        let e_hi_eval = e_hi[idx_hi];
        [e_hi_eval * sums[0].reduce(), e_hi_eval * sums[1].reduce()]
    };

    #[cfg(feature = "parallel")]
    let claims = (0..e_hi.len())
        .into_par_iter()
        .map(block_contribution)
        .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    #[cfg(not(feature = "parallel"))]
    let claims = (0..e_hi.len())
        .map(block_contribution)
        .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

    (claims[0], claims[1])
}

impl<F: Field> ProveRounds<F> for ReadWriteKernel<F>
where
    F::Accumulator: RingAccumulator,
{
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
        if self.rounds_bound < self.log_t {
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

impl<F: Field> SumcheckKernel<F> for ReadWriteKernel<F>
where
    F::Accumulator: RingAccumulator,
{
    type Relation = RegistersReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let (r_address, r_cycle) = self.bound_point();
        let (rs1_ra, rs2_ra) =
            one_hot_operand_claims(&self.rs1_indices, &self.rs2_indices, &r_address, &r_cycle);
        Ok(RegistersReadWriteOutputClaims {
            registers_val: self.val[0],
            rs1_ra,
            rs2_ra,
            rd_wa: self.wa[0],
            rd_inc: self.inc_scalar,
        })
    }

    /// Pin the internally tracked eq factor to the verifier's scalar path:
    /// the fully bound Gruen scalar must equal `derive_output_term(EqCycle)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let id = JoltDerivedId::from(RegistersReadWritePublic::EqCycle);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        if self.eq_scalar != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id,
                expected,
                got: self.eq_scalar,
            });
        }
        Ok(())
    }
}

/// Shared parity-test support for the registers kernel family: a
/// register-consistent synthetic trace behind a full `TraceBackend` witness
/// plane, deterministic challenge sequences, and the engine-mirroring parity
/// driver (bind-then-compute, running claim via `poly.evaluate(challenge)`).
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test support module")]
pub(crate) mod test_support {
    use jolt_claims::protocols::jolt::{JoltChallengeId, JoltOneHotConfig};
    use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{
        JoltProgram, OwnedTrace, RegisterRead, RegisterState, RegisterWrite, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_verifier::stages::relations::{
        ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
        SumcheckOutputClaims,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend};

    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    /// Deterministic nonzero field elements (an LCG over odd u64s), used for
    /// both fixed points and round challenges.
    pub(crate) fn challenge_sequence(len: usize, seed: u64) -> Vec<Fr> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                Fr::from_u64(state | 1)
            })
            .collect()
    }

    /// A register-consistent trace builder: reads return the current register
    /// state, writes advance it, so every witness identity the sumchecks
    /// assume holds by construction.
    pub(crate) struct TraceFixture {
        rows: Vec<TraceRow>,
        state: [u64; 128],
        counter: u64,
        instruction: JoltInstructionRow,
    }

    impl TraceFixture {
        pub(crate) fn new() -> Self {
            Self {
                rows: Vec::new(),
                state: [0; 128],
                counter: 0xDEAD_BEEF_0BAD_F00D,
                instruction: JoltInstructionRow {
                    instruction_kind: JoltInstructionKind::ADDI,
                    address: 0x8000_0000,
                    operands: NormalizedOperands {
                        rd: Some(1),
                        rs1: Some(2),
                        rs2: None,
                        imm: 3,
                    },
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: false,
                },
            }
        }

        pub(crate) fn noop(&mut self) {
            let instruction = self.instruction;
            self.rows.push(TraceRow {
                instruction,
                ..TraceRow::default()
            });
        }

        /// One cycle touching the given operands; the write value is a fresh
        /// pseudo-random u64.
        pub(crate) fn op(&mut self, rd: Option<u8>, rs1: Option<u8>, rs2: Option<u8>) {
            let read = |state: &[u64; 128], register: Option<u8>| {
                register.map(|register| RegisterRead {
                    register,
                    value: state[register as usize],
                })
            };
            let registers = RegisterState {
                rs1: read(&self.state, rs1),
                rs2: read(&self.state, rs2),
                rd: rd.map(|register| {
                    self.counter = self
                        .counter
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let pre_value = self.state[register as usize];
                    let post_value = self.counter;
                    self.state[register as usize] = post_value;
                    RegisterWrite {
                        register,
                        pre_value,
                        post_value,
                    }
                }),
            };
            let instruction = self.instruction;
            self.rows.push(TraceRow {
                instruction,
                registers,
                ..TraceRow::default()
            });
        }

        /// Run `f` against a trace backend padded to `2^log_t` cycles.
        pub(crate) fn with_plane<R>(
            self,
            log_t: usize,
            f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R,
        ) -> R {
            assert!(self.rows.len() <= 1 << log_t, "fixture overflows 2^log_t");
            let preprocessing = JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::preprocess(
                    vec![self.instruction],
                    self.instruction.address as u64,
                    RV64IMAC_JOLT,
                )
                .unwrap(),
                ram: RAMPreprocessing::default(),
                memory_layout: Default::default(),
                max_padded_trace_length: 1 << log_t,
            };
            let program = JoltProgram::default();
            let config = JoltVmWitnessConfig::new(
                log_t,
                64,
                JoltOneHotConfig {
                    log_k_chunk: 4,
                    lookups_ra_virtual_log_k_chunk: 16,
                },
            );
            let inputs = JoltVmWitnessInputs::new(
                &program,
                &preprocessing,
                TraceOutput::new(OwnedTrace::new(self.rows), Default::default(), None),
            );
            let backend = TraceBackend::new(config, inputs);
            f(&backend)
        }
    }

    /// A structured register workload: write-then-read chains, `rs1 == rs2`,
    /// `rd == rs1` in one cycle, repeated writes, high register indices, and
    /// interleaved no-ops. Emits exactly `cycles` rows.
    pub(crate) fn structured_fixture(cycles: usize) -> TraceFixture {
        let mut fixture = TraceFixture::new();
        for step in 0..cycles {
            match step % 8 {
                0 => fixture.op(Some(5), Some(2), None),
                1 => fixture.op(Some(7), Some(5), Some(5)),
                2 => fixture.op(Some(5), Some(5), Some(7)),
                3 => fixture.noop(),
                4 => fixture.op(None, Some(7), Some(100)),
                5 => fixture.op(Some(127), Some(0), Some(5)),
                6 => fixture.op(Some(100), None, None),
                _ => fixture.op(Some(7), Some(127), Some(100)),
            }
        }
        fixture
    }

    /// Prepare the reference and optimized kernels from identical inputs,
    /// drive both through the full round sequence asserting byte-identical
    /// round polynomials, then assert equal typed output claims and run both
    /// kernels' derived-table validation against the relation.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the seam's input decomposition"
    )]
    pub(crate) fn assert_kernel_parity<R>(
        optimized_slot: &dyn PrepareKernel<Fr, R>,
        witness: &dyn JoltWitnessPlane<Fr>,
        relation: &R,
        claims: &SumcheckInputClaims<Fr, R>,
        points: &SumcheckInputPoints<Fr, R>,
        challenges: &ConcreteSumcheckChallenges<Fr, R>,
        input_claim: Fr,
        round_challenges: &[Fr],
    ) where
        R: ConcreteSumcheck<Fr>,
        ReferenceBackend: PrepareKernel<Fr, R>,
        SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
        ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
    {
        assert_kernel_parity_with_session(
            &mut ProofSession::default(),
            optimized_slot,
            witness,
            relation,
            claims,
            points,
            challenges,
            input_claim,
            round_challenges,
        );
    }

    /// [`assert_kernel_parity`] with a caller-supplied session for the
    /// optimized kernel — exercises cross-member session carries.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the seam's input decomposition"
    )]
    pub(crate) fn assert_kernel_parity_with_session<R>(
        optimized_session: &mut ProofSession,
        optimized_slot: &dyn PrepareKernel<Fr, R>,
        witness: &dyn JoltWitnessPlane<Fr>,
        relation: &R,
        claims: &SumcheckInputClaims<Fr, R>,
        points: &SumcheckInputPoints<Fr, R>,
        challenges: &ConcreteSumcheckChallenges<Fr, R>,
        input_claim: Fr,
        round_challenges: &[Fr],
    ) where
        R: ConcreteSumcheck<Fr>,
        ReferenceBackend: PrepareKernel<Fr, R>,
        SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
        ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
    {
        let mut reference_session = ProofSession::default();
        let mut reference = ReferenceBackend
            .prepare(
                &mut reference_session,
                witness,
                ProverInputs {
                    relation,
                    claims,
                    points,
                    challenges,
                },
            )
            .unwrap();
        let mut optimized = optimized_slot
            .prepare(
                optimized_session,
                witness,
                ProverInputs {
                    relation,
                    claims,
                    points,
                    challenges,
                },
            )
            .unwrap();

        let rounds = relation.rounds();
        assert_eq!(reference.num_rounds(), rounds);
        assert_eq!(optimized.num_rounds(), rounds);
        assert_eq!(round_challenges.len(), rounds);

        let mut claim = input_claim;
        for round in 0..rounds {
            let bind = (round > 0).then(|| round_challenges[round - 1]);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly, optimized_poly,
                "round {round} polynomial mismatch"
            );
            assert_eq!(
                optimized_poly.evaluate(Fr::from_u64(0)) + optimized_poly.evaluate(Fr::from_u64(1)),
                claim,
                "round {round} running-claim mismatch"
            );
            claim = reference_poly.evaluate(round_challenges[round]);
        }
        reference
            .finish_rounds(round_challenges[rounds - 1])
            .unwrap();
        optimized
            .finish_rounds(round_challenges[rounds - 1])
            .unwrap();

        let output_points = relation
            .derive_opening_points(round_challenges, points)
            .unwrap();
        reference
            .validate_derived_tables(relation, points, &output_points, challenges)
            .unwrap();
        optimized
            .validate_derived_tables(relation, points, &output_points, challenges)
            .unwrap();

        let reference_outputs = reference.output_claims(claims).unwrap();
        let optimized_outputs = optimized.output_claims(claims).unwrap();
        assert_eq!(
            reference_outputs, optimized_outputs,
            "output claims mismatch"
        );
    }

    /// A fixture guard: an all-zero witness would make parity vacuous, so the
    /// input claim must be a nontrivial field element.
    pub(crate) fn assert_nontrivial(claim: Fr) {
        assert_ne!(
            claim,
            Fr::from_u64(0),
            "degenerate fixture: zero input claim"
        );
        assert_ne!(
            claim,
            Fr::from_u64(1),
            "degenerate fixture: unit input claim"
        );
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;

    use super::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
        TraceFixture,
    };
    use super::OptimizedRegistersReadWrite;

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
        fixture.with_plane(log_t, |backend| {
            let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                log_t,
                0,
            ));
            let r_cycle = challenge_sequence(log_t, seed ^ 0xA5A5);
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&r_cycle)
            };
            let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let input_claim =
                claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
            assert_nontrivial(input_claim);
            let round_challenges = challenge_sequence(log_t + REGISTER_ADDRESS_BITS, seed);
            assert_kernel_parity(
                &OptimizedRegistersReadWrite,
                backend,
                &relation,
                &claims,
                &points,
                &RegistersReadWriteChallenges { gamma },
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 17);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 23);
    }

    #[test]
    fn parity_minimal_padded_trace() {
        // Three real cycles padded to four: exercises the padding rows and
        // registers that are never touched.
        let mut fixture = TraceFixture::new();
        fixture.op(Some(3), Some(1), Some(2));
        fixture.op(Some(3), Some(3), None);
        fixture.op(None, Some(3), Some(3));
        run_parity(fixture, 2, 31);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(9), Some(9), Some(9));
        fixture.op(Some(9), None, Some(9));
        run_parity(fixture, 1, 41);
    }
}
