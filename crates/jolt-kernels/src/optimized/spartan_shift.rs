//! The optimized Spartan shift (stage 3) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/spartan_shift.rs`), which binds seven dense
//! `T`-sized tables (two `eq+1` tables plus the five PC/flag columns) every
//! round.
//!
//! Techniques ported from `jolt-prover-legacy/src/zkvm/spartan/shift.rs`
//! (`ShiftSumcheckProver`, eprint 2025/611 Appendix A):
//!
//! - **`eq+1` prefix–suffix decomposition** for the first half of the rounds:
//!   `eq+1(r, (y_hi, y_lo)) = P_0(y_lo)·S_0(y_hi) + P_1(y_lo)·S_1(y_hi)`
//!   ([`EqPlusOnePrefixSuffix`]), so each of the two summand terms becomes
//!   two rank-1 `P·Q` pairs with
//!   `Q_b(y_lo) = Σ_{y_hi} S_b(y_hi) · v(y_hi ‖ y_lo)` — four √T-sized pairs
//!   per round instead of seven T-sized tables. Suffix-summing commutes with
//!   prefix partial evaluation, so every `P·Q` round polynomial equals the
//!   dense summand's exactly.
//! - **Value fusion in Q**: the outer term's four columns enter linearly, so
//!   they are γ-combined into one scalar per cycle
//!   (`upc + γ·pc + γ²·is_virtual + γ³·is_first`) during the Q build; the
//!   product term's `γ⁴·(1 − is_noop)` folds into its Q the same way.
//! - **Phase-2 regeneration**: once the prefix is bound, the five columns are
//!   folded by `eq(r_prefix)` straight from the raw trace values into
//!   `2^(remaining)`-sized tables, and each `eq+1` table is its suffix pair
//!   recombined with the bound-prefix evaluations
//!   (`P_0(r_prefix)·S_0[j] + P_1(r_prefix)·S_1[j]`) — the exact partial
//!   binds of the dense tables, recomputed instead of carried.
//! - **Raw `u64` trace values with small-scalar deferred-reduction
//!   accumulation** (u32-split `fmadd_u64`) for the phase-2 PC folds, and
//!   ring-accumulator `fmadd` for the field-by-field `Q` products.

use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_field::AkitaField;
use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, SignedScalarAccumulator};
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};
use jolt_witness::witnesses::{InstructionFlag, OpFlag, Pc, UnexpandedPc};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::collect_rows;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::{
    spartan_shift::{SpartanShiftFlagWord, SpartanShiftResidentRows},
    MetalError, SolinasMetal,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-cycle shift columns as native small scalars: the two PCs plus the
/// three flags the summand references (all at cycle `j`, unshifted — the
/// shift lives in the `eq+1` factors).
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct SpartanShiftRow {
    #[opening(UnexpandedPC)]
    unexpanded_pc: UnexpandedPc,
    #[opening(PC)]
    pc: Pc,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    is_virtual: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    is_first_in_sequence: OpFlag,
    #[opening(InstructionFlags(InstructionFlags::IsNoop))]
    is_noop: InstructionFlag,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_spartan_shift_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<SpartanShiftResidentRows, MetalError> {
    context.prepare_spartan_shift_rows_with_fill(cycles, true, |unexpanded_pc, pc, flags| {
        flags.fill(SpartanShiftFlagWord::default());
        if let Some(access) = witness
            .random_access()
            .filter(|access| cycles <= access.cycles())
        {
            #[cfg(feature = "parallel")]
            {
                return unexpanded_pc
                    .par_chunks_mut(32)
                    .zip(pc.par_chunks_mut(32))
                    .zip(flags.par_iter_mut())
                    .enumerate()
                    .try_for_each(|(word_index, ((unexpanded_pc, pc), flags))| {
                        for offset in 0..unexpanded_pc.len() {
                            let row_index = word_index * 32 + offset;
                            let row =
                                access
                                    .window::<SpartanShiftRow>(row_index)
                                    .map_err(|error| MetalError::SpartanShiftRowExtraction {
                                        row: row_index,
                                        message: error.to_string(),
                                    })?;
                            write_spartan_shift_row(
                                row,
                                offset,
                                &mut unexpanded_pc[offset],
                                &mut pc[offset],
                                flags,
                            );
                        }
                        Ok(())
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for row_index in 0..cycles {
                    let row = access
                        .window::<SpartanShiftRow>(row_index)
                        .map_err(|error| MetalError::SpartanShiftRowExtraction {
                            row: row_index,
                            message: error.to_string(),
                        })?;
                    write_spartan_shift_row(
                        row,
                        row_index % 32,
                        &mut unexpanded_pc[row_index],
                        &mut pc[row_index],
                        &mut flags[row_index / 32],
                    );
                }
                return Ok(());
            }
        }

        let mut cursor = 0usize;
        let mut extraction_failure = None;
        let result = witness.visit_chunks(0..cycles, 1 << 12, &mut |rows, next, env| {
            for (offset, current) in rows.iter().enumerate() {
                let row_index = cursor + offset;
                let row =
                    match SpartanShiftRow::from_row(current, rows.get(offset + 1).or(next), env) {
                        Ok(row) => row,
                        Err(error) => {
                            extraction_failure = Some((row_index, error.to_string()));
                            return Err(error);
                        }
                    };
                write_spartan_shift_row(
                    row,
                    row_index % 32,
                    &mut unexpanded_pc[row_index],
                    &mut pc[row_index],
                    &mut flags[row_index / 32],
                );
            }
            cursor += rows.len();
            Ok(())
        });
        if let Err(error) = result {
            let (row, message) = extraction_failure.unwrap_or_else(|| (cursor, error.to_string()));
            return Err(MetalError::SpartanShiftRowExtraction { row, message });
        }
        if cursor != cycles {
            return Err(MetalError::InvalidSpartanShiftState(
                "witness projection did not fill the cycle domain",
            ));
        }
        Ok(())
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn write_spartan_shift_row(
    row: SpartanShiftRow,
    bit: usize,
    unexpanded_pc: &mut u64,
    pc: &mut u64,
    flags: &mut SpartanShiftFlagWord,
) {
    *unexpanded_pc = row.unexpanded_pc.0;
    *pc = row.pc.0;
    let mask = 1u32 << bit;
    flags.is_virtual |= u32::from(row.is_virtual.0) * mask;
    flags.is_first_in_sequence |= u32::from(row.is_first_in_sequence.0) * mask;
    flags.is_noop |= u32::from(row.is_noop.0) * mask;
}

/// Accumulate `eq · F(value)` for a full-range `u64` on the small-scalar
/// accumulator without overflowing it: the accumulator's headroom is one
/// extra limb, which ~4 full-magnitude `field × u64` products exhaust, so the
/// value is split into u32 halves (products ≤ 2^286, headroom ≥ 2^34 terms).
/// `eq_shifted` must be `eq · 2^32`; the two fused adds sum to exactly
/// `eq · F(value)`.
#[inline]
fn fmadd_u64_split<F: Field>(
    accumulator: &mut F::SmallScalarAccumulator,
    eq: F,
    eq_shifted: F,
    value: u64,
) {
    accumulator.fmadd_u64(eq_shifted, value >> 32);
    accumulator.fmadd_u64(eq, value & 0xFFFF_FFFF);
}

pub struct OptimizedSpartanShift;

impl<F: Field> PrepareKernel<F, SpartanShift<F>> for OptimizedSpartanShift {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, SpartanShift<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized Spartan shift requires at least one cycle round",
            });
        }
        let r_outer: &[F] = relation.product_uniskip_tau_low();
        let r_product: &[F] = relation.product_remainder_opening_point();
        if r_outer.len() != log_t || r_product.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan shift eq+1 point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;
        let rows: Vec<SpartanShiftRow> = collect_rows(witness, cycles)?;

        let gamma = inputs.challenges.gamma;
        let mut gamma_powers = [F::one(); 5];
        for i in 1..5 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        let prefix_vars = outer.prefix_0.len().trailing_zeros() as usize;

        // Q_b(y_lo) = Σ_{y_hi} S_b(y_hi) · v(y_hi ‖ y_lo): the outer pair
        // over the γ-combined PC/flag scalar, the product pair over
        // `1 − is_noop` (γ⁴-scaled once at the end).
        const BLOCK: usize = 32;
        let suffix_rows: Vec<[F; 4]> = (0..outer.suffix_0.len())
            .map(|x_hi| {
                [
                    outer.suffix_0[x_hi],
                    outer.suffix_1[x_hi],
                    product.suffix_0[x_hi],
                    product.suffix_1[x_hi],
                ]
            })
            .collect();
        let build_q_block = |(block_index, q_block): (usize, &mut [[F; 4]])| {
            let mut outer_folds = vec![[F::Accumulator::default(); 2]; q_block.len()];
            let mut product_folds = vec![[F::zero(); 2]; q_block.len()];
            for (x_hi, suffix) in suffix_rows.iter().enumerate() {
                let base = x_hi << prefix_vars;
                for (i, (outer_fold, product_fold)) in outer_folds
                    .iter_mut()
                    .zip(product_folds.iter_mut())
                    .enumerate()
                {
                    let x_lo = block_index * BLOCK + i;
                    let row = &rows[base + x_lo];
                    let mut v =
                        F::from_u64(row.unexpanded_pc.0) + gamma_powers[1] * F::from_u64(row.pc.0);
                    if row.is_virtual.0 {
                        v += gamma_powers[2];
                    }
                    if row.is_first_in_sequence.0 {
                        v += gamma_powers[3];
                    }
                    outer_fold[0].fmadd(v, suffix[0]);
                    outer_fold[1].fmadd(v, suffix[1]);
                    if !row.is_noop.0 {
                        product_fold[0] += suffix[2];
                        product_fold[1] += suffix[3];
                    }
                }
            }
            for (q, (outer_fold, product_fold)) in q_block
                .iter_mut()
                .zip(outer_folds.into_iter().zip(product_folds))
            {
                q[0] = outer_fold[0].reduce();
                q[1] = outer_fold[1].reduce();
                q[2] = gamma_powers[4] * product_fold[0];
                q[3] = gamma_powers[4] * product_fold[1];
            }
        };
        let mut q_rows = vec![[F::zero(); 4]; 1 << prefix_vars];
        #[cfg(feature = "parallel")]
        q_rows
            .par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(build_q_block);
        #[cfg(not(feature = "parallel"))]
        q_rows.chunks_mut(BLOCK).enumerate().for_each(build_q_block);

        let [q_0, q_1, q_2, q_3]: [Vec<F>; 4] =
            core::array::from_fn(|pair| q_rows.iter().map(|q| q[pair]).collect());
        let pairs = [
            (outer.prefix_0, q_0),
            (outer.prefix_1, q_1),
            (product.prefix_0, q_2),
            (product.prefix_1, q_3),
        ];

        Ok(Box::new(ShiftKernel {
            log_t,
            gamma_powers,
            r_outer: r_outer.to_vec(),
            r_product: r_product.to_vec(),
            rows,
            phase: Phase::PrefixSuffix { pairs },
            bound_challenges: Vec::with_capacity(log_t),
        }))
    }
}

enum Phase<F> {
    /// First half of the rounds: the four `(P, Q)` pairs over the prefix
    /// variables (outer 0/1, product 0/1 — product Qs carry the γ⁴ scale).
    PrefixSuffix { pairs: [(Vec<F>, Vec<F>); 4] },
    /// Remaining rounds: the two `eq+1` tables and the five columns, dense.
    Dense {
        eq_plus_one_outer: Vec<F>,
        eq_plus_one_product: Vec<F>,
        unexpanded_pc: Vec<F>,
        pc: Vec<F>,
        is_virtual: Vec<F>,
        is_first_in_sequence: Vec<F>,
        is_noop: Vec<F>,
    },
}

struct ShiftKernel<F: Field> {
    log_t: usize,
    gamma_powers: [F; 5],
    /// The two `eq+1` points (big-endian) the summand factors fix.
    r_outer: Vec<F>,
    r_product: Vec<F>,
    /// Raw per-cycle values, kept for the phase-2 regeneration.
    rows: Vec<SpartanShiftRow>,
    phase: Phase<F>,
    bound_challenges: Vec<F>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for ShiftKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, table) in [
            ("r_outer", &self.r_outer),
            ("r_product", &self.r_product),
            ("bound_challenges", &self.bound_challenges),
        ] {
            visitor.visit_simple(allocative::Key::new(key), vec_heap_bytes(table));
        }
        visitor.visit_simple(allocative::Key::new("rows"), vec_heap_bytes(&self.rows));
        let phase_bytes = match &self.phase {
            Phase::PrefixSuffix { pairs } => pairs
                .iter()
                .map(|(p, q)| vec_heap_bytes(p) + vec_heap_bytes(q))
                .sum(),
            Phase::Dense {
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            } => [
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            ]
            .into_iter()
            .map(vec_heap_bytes)
            .sum(),
        };
        visitor.visit_simple(allocative::Key::new("phase"), phase_bytes);
        visitor.exit();
    }
}

impl<F: Field> ShiftKernel<F> {
    fn rounds_bound(&self) -> usize {
        self.bound_challenges.len()
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        let remaining = self.log_t - self.rounds_bound();
        if remaining == 0 {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound { remaining })
        }
    }

    /// Regenerate the dense phase from the raw values: the five columns
    /// folded by `eq(r_prefix)` (their exact partial binds) and each `eq+1`
    /// table recombined from its suffix pair and bound-prefix evaluations.
    fn transition_to_dense(&mut self) {
        let bound = self.rounds_bound();
        let r_prefix: Vec<F> = self.bound_challenges.iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let chunk = eq_prefix.len();
        let remaining = 1usize << (self.log_t - bound);

        let fold_chunk = |rows: &[SpartanShiftRow]| -> [F; 5] {
            let mut pc_folds = [F::SmallScalarAccumulator::default(); 2];
            let mut flag_folds = [F::zero(); 3];
            for (row, (&eq, &eq_shifted)) in rows
                .iter()
                .zip(eq_prefix.iter().zip(eq_prefix_shifted.iter()))
            {
                fmadd_u64_split(&mut pc_folds[0], eq, eq_shifted, row.unexpanded_pc.0);
                fmadd_u64_split(&mut pc_folds[1], eq, eq_shifted, row.pc.0);
                if row.is_virtual.0 {
                    flag_folds[0] += eq;
                }
                if row.is_first_in_sequence.0 {
                    flag_folds[1] += eq;
                }
                if row.is_noop.0 {
                    flag_folds[2] += eq;
                }
            }
            [
                pc_folds[0].reduce(),
                pc_folds[1].reduce(),
                flag_folds[0],
                flag_folds[1],
                flag_folds[2],
            ]
        };
        #[cfg(feature = "parallel")]
        let folds: Vec<[F; 5]> = self.rows.par_chunks(chunk).map(fold_chunk).collect();
        #[cfg(not(feature = "parallel"))]
        let folds: Vec<[F; 5]> = self.rows.chunks(chunk).map(fold_chunk).collect();
        debug_assert_eq!(folds.len(), remaining);

        // The raw values only feed this regeneration; free them now.
        self.rows = Vec::new();

        let recombine = |point: &[F]| -> Vec<F> {
            let split = EqPlusOnePrefixSuffix::new(point);
            let prefix_0_eval = Polynomial::new(split.prefix_0).evaluate(&r_prefix);
            let prefix_1_eval = Polynomial::new(split.prefix_1).evaluate(&r_prefix);
            split
                .suffix_0
                .iter()
                .zip(split.suffix_1.iter())
                .map(|(&suffix_0, &suffix_1)| prefix_0_eval * suffix_0 + prefix_1_eval * suffix_1)
                .collect()
        };

        self.phase = Phase::Dense {
            eq_plus_one_outer: recombine(&self.r_outer),
            eq_plus_one_product: recombine(&self.r_product),
            unexpanded_pc: folds.iter().map(|fold| fold[0]).collect(),
            pc: folds.iter().map(|fold| fold[1]).collect(),
            is_virtual: folds.iter().map(|fold| fold[2]).collect(),
            is_first_in_sequence: folds.iter().map(|fold| fold[3]).collect(),
            is_noop: folds.iter().map(|fold| fold[4]).collect(),
        };
    }

    fn bind(&mut self, r: F) {
        self.bound_challenges.push(r);
        // Last prefix variable: regenerate the dense phase from the raw
        // values instead of binding the exhausted P·Q pairs.
        if matches!(&self.phase, Phase::PrefixSuffix { pairs } if pairs[0].0.len() == 2) {
            self.transition_to_dense();
            return;
        }
        let bind_table = |table: &mut Vec<F>| {
            let half = table.len() / 2;
            for y in 0..half {
                let lo = table[2 * y];
                table[y] = lo + r * (table[2 * y + 1] - lo);
            }
            table.truncate(half);
        };
        match &mut self.phase {
            Phase::PrefixSuffix { pairs } => {
                for (p, q) in pairs {
                    bind_table(p);
                    bind_table(q);
                }
            }
            Phase::Dense {
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            } => {
                for table in [
                    eq_plus_one_outer,
                    eq_plus_one_product,
                    unexpanded_pc,
                    pc,
                    is_virtual,
                    is_first_in_sequence,
                    is_noop,
                ] {
                    bind_table(table);
                }
            }
        }
    }
}

impl<F: Field> ProveRounds<F> for ShiftKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }

        // Degree-2 member: evals at t = 0 and t = 2; s(1) from the hint.
        let evals: [F; 2] = match &self.phase {
            Phase::PrefixSuffix { pairs } => {
                let mut acc = [F::Accumulator::default(); 2];
                for (p, q) in pairs {
                    for y in 0..p.len() / 2 {
                        let (p_0, p_1) = (p[2 * y], p[2 * y + 1]);
                        let (q_0, q_1) = (q[2 * y], q[2 * y + 1]);
                        acc[0].fmadd(p_0, q_0);
                        acc[1].fmadd(p_1 + p_1 - p_0, q_1 + q_1 - q_0);
                    }
                }
                acc.map(F::Accumulator::reduce)
            }
            Phase::Dense {
                eq_plus_one_outer,
                eq_plus_one_product,
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
            } => {
                let mut acc = [F::Accumulator::default(); 2];
                let pair = |table: &[F], y: usize| (table[2 * y], table[2 * y + 1]);
                let extend = |(lo, hi): (F, F)| hi + hi - lo;
                for y in 0..eq_plus_one_outer.len() / 2 {
                    let eq1o = pair(eq_plus_one_outer, y);
                    let eq1p = pair(eq_plus_one_product, y);
                    let upc = pair(unexpanded_pc, y);
                    let pcs = pair(pc, y);
                    let virt = pair(is_virtual, y);
                    let first = pair(is_first_in_sequence, y);
                    let noop = pair(is_noop, y);
                    acc[0].fmadd(
                        eq1o.0,
                        upc.0
                            + self.gamma_powers[1] * pcs.0
                            + self.gamma_powers[2] * virt.0
                            + self.gamma_powers[3] * first.0,
                    );
                    acc[0].fmadd(eq1p.0, self.gamma_powers[4] * (F::one() - noop.0));
                    acc[1].fmadd(
                        extend(eq1o),
                        extend(upc)
                            + self.gamma_powers[1] * extend(pcs)
                            + self.gamma_powers[2] * extend(virt)
                            + self.gamma_powers[3] * extend(first),
                    );
                    acc[1].fmadd(
                        extend(eq1p),
                        self.gamma_powers[4] * (F::one() - extend(noop)),
                    );
                }
                acc.map(F::Accumulator::reduce)
            }
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for ShiftKernel<F> {
    type Relation = SpartanShift<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SpartanShiftOutputClaims<F>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let Phase::Dense {
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        Ok(SpartanShiftOutputClaims {
            unexpanded_pc: unexpanded_pc[0],
            pc: pc[0],
            is_virtual: is_virtual[0],
            is_first_in_sequence: is_first_in_sequence[0],
            is_noop: is_noop[0],
        })
    }

    /// Pin the regenerated `eq+1` tables to the verifier's scalar path: their
    /// fully bound values must equal `derive_output_term` at the bound point.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let Phase::Dense {
            eq_plus_one_outer,
            eq_plus_one_product,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        for (public, got) in [
            (SpartanShiftPublic::EqPlusOneOuter, eq_plus_one_outer[0]),
            (SpartanShiftPublic::EqPlusOneProduct, eq_plus_one_product[0]),
        ] {
            let id = JoltDerivedId::from(public);
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPlusOnePolynomial;
    use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{
        CircuitFlags, InstructionFlags, JoltInstructionKind, JoltInstructionRow,
        NormalizedOperands, RV64IMAC_JOLT,
    };
    use jolt_verifier::stages::stage3::spartan_shift::{
        SpartanShift, SpartanShiftChallenges, SpartanShiftInputClaims,
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use jolt_witness::RowSource;
    use jolt_witness::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, JoltWitnessPlane, TraceBackend,
    };

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence,
    };
    use super::OptimizedSpartanShift;

    fn instruction(
        address: usize,
        virtual_sequence_remaining: Option<u16>,
        first: bool,
    ) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining,
            is_first_in_sequence: first,
            is_compressed: false,
        }
    }

    /// A PC-varied trace: three bytecode addresses, one two-step virtual
    /// sequence (exercising the `is_virtual` / `is_first_in_sequence`
    /// columns), an explicit mid-trace no-op, and no-op padding to `2^log_t`
    /// (exercising `is_noop`).
    fn with_shift_plane<R>(log_t: usize, f: impl FnOnce(&TraceBackend<'_, OwnedTrace>) -> R) -> R {
        let plain_a = instruction(0x8000_0000, None, false);
        let virtual_first = instruction(0x8000_0004, Some(1), true);
        let virtual_last = instruction(0x8000_0004, Some(0), false);
        let plain_b = instruction(0x8000_0008, None, false);
        let noop = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::NoOp,
            ..plain_a
        };
        let bytecode = vec![plain_a, virtual_first, virtual_last, plain_b];

        let script = [plain_a, virtual_first, virtual_last, noop, plain_b, plain_a];
        // At log_t = 1 both cycles must be real instructions: the summand
        // weights only cycle 1 (`eq+1` vanishes at 0), and a no-op there
        // zeroes the input claim.
        let real_rows = if log_t == 1 { 2 } else { (1 << log_t) - 1 };
        let rows: Vec<TraceRow> = script
            .iter()
            .take(real_rows)
            .map(|&instruction| TraceRow {
                instruction,
                ..TraceRow::default()
            })
            .collect();

        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                bytecode,
                plain_a.address as u64,
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
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);
        f(&backend)
    }

    fn run_parity(log_t: usize, seed: u64) {
        with_shift_plane(log_t, |backend| {
            let r_outer = challenge_sequence(log_t, seed ^ 0x07E5);
            let r_product = challenge_sequence(log_t, seed ^ 0xFACE);
            let gamma = Fr::from_u64(0x5EED_0F0F_1234_5678);
            let relation = SpartanShift::<Fr>::new(
                TraceDimensions::new(log_t),
                r_outer.clone(),
                r_product.clone(),
            );

            let table = |polynomial: JoltVirtualPolynomial| -> Vec<Fr> {
                JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap()
            };
            let unexpanded_pc = table(JoltVirtualPolynomial::UnexpandedPC);
            let pc = table(JoltVirtualPolynomial::PC);
            let is_virtual = table(JoltVirtualPolynomial::OpFlags(
                CircuitFlags::VirtualInstruction,
            ));
            let is_first = table(JoltVirtualPolynomial::OpFlags(
                CircuitFlags::IsFirstInSequence,
            ));
            let is_noop = table(JoltVirtualPolynomial::InstructionFlags(
                InstructionFlags::IsNoop,
            ));

            let eq1_outer = EqPlusOnePolynomial::evals(&r_outer, None).1;
            let eq1_product = EqPlusOnePolynomial::evals(&r_product, None).1;
            let gamma_2 = gamma * gamma;
            let gamma_3 = gamma_2 * gamma;
            let gamma_4 = gamma_3 * gamma;
            let input_claim: Fr = (0..1usize << log_t)
                .map(|j| {
                    eq1_outer[j]
                        * (unexpanded_pc[j]
                            + gamma * pc[j]
                            + gamma_2 * is_virtual[j]
                            + gamma_3 * is_first[j])
                        + gamma_4 * eq1_product[j] * (Fr::from_u64(1) - is_noop[j])
                })
                .sum();
            assert_nontrivial(input_claim);

            let round_challenges = challenge_sequence(log_t, seed);
            assert_kernel_parity(
                &OptimizedSpartanShift,
                backend as &dyn JoltWitnessPlane<Fr>,
                &relation,
                &SpartanShiftInputClaims::default(),
                &SpartanShiftInputClaims::<Vec<Fr>>::default(),
                &SpartanShiftChallenges { gamma },
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_even_log_t() {
        run_parity(4, 211);
    }

    #[test]
    fn parity_odd_log_t() {
        run_parity(3, 223);
    }

    #[test]
    fn parity_deep_phase2() {
        run_parity(5, 227);
    }

    #[test]
    fn parity_minimal_single_round() {
        // log_t = 1: the P·Q phase covers the single round and the dense
        // phase materializes inside `finish_rounds`.
        run_parity(1, 229);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_witness_projection_matches_the_prefix_oracle() {
        use crate::metal::solinas::spartan_shift::{
            build_prefix_reference, pack_flag_words, SpartanShiftGeometry,
            SpartanShiftKernelConfig, SpartanShiftNativePlanes,
        };
        use crate::metal::solinas::SolinasMetal;

        use super::{prepare_metal_spartan_shift_witness_rows, SpartanShiftRow};

        with_shift_plane(4, |backend| {
            let context = SolinasMetal::for_akita().unwrap();
            let cycles = 16;
            let geometry = SpartanShiftGeometry::new(cycles).unwrap();
            let resident = prepare_metal_spartan_shift_witness_rows(
                &context,
                backend as &dyn JoltWitnessPlane<jolt_field::AkitaField>,
                cycles,
            )
            .unwrap();

            let access = backend.random_access().unwrap();
            let rows: Vec<SpartanShiftRow> = (0..cycles)
                .map(|index| access.window(index).unwrap())
                .collect();
            let unexpanded_pc: Vec<u64> = rows.iter().map(|row| row.unexpanded_pc.0).collect();
            let pc: Vec<u64> = rows.iter().map(|row| row.pc.0).collect();
            let is_virtual: Vec<bool> = rows.iter().map(|row| row.is_virtual.0).collect();
            let is_first: Vec<bool> = rows.iter().map(|row| row.is_first_in_sequence.0).collect();
            let is_noop: Vec<bool> = rows.iter().map(|row| row.is_noop.0).collect();
            let flags = pack_flag_words(geometry, &is_virtual, &is_first, &is_noop).unwrap();
            let planes =
                SpartanShiftNativePlanes::new(geometry, &unexpanded_pc, &pc, &flags).unwrap();
            let point = |seed: u64| {
                (0..geometry.log_t())
                    .scan(seed, |state, _| {
                        *state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        Some(jolt_field::AkitaField::from_u64(*state | 1))
                    })
                    .collect::<Vec<_>>()
            };
            let r_outer = point(0xA11C_E001);
            let r_product = point(0xB22D_F002);
            let gamma = jolt_field::AkitaField::from_u64(0xC33E_1003);
            let expected =
                build_prefix_reference(geometry, planes, &r_outer, &r_product, gamma).unwrap();
            let observed = context
                .prepare_spartan_shift_prefix(
                    &resident,
                    &r_outer,
                    &r_product,
                    gamma,
                    SpartanShiftKernelConfig {
                        build_threads_per_threadgroup: 32,
                        high_tile_elements: geometry.suffix_elements(),
                        fold_threads_per_threadgroup: 32,
                    },
                )
                .unwrap()
                .execute()
                .unwrap();
            assert_eq!(observed.q, expected.q);
        });
    }
}
