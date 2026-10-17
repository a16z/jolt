//! The optimized Spartan shift (stage 3) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/spartan_shift.rs`), which binds seven dense
//! `T`-sized tables (two `eq+1` tables plus the five PC/flag columns) every
//! round.
//!
//! Uses the prefix-suffix decomposition from eprint 2025/611, Appendix A:
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
//! - **Two task-partitioned row passes** (`task_ranges`): the `Q` build and
//!   the phase-2 fold each read the cycle domain once, every task visiting a
//!   contiguous run of whole `y_hi` rows serially. The `Q` build accumulates
//!   into one caller-allocated `2^prefix`-entry slot per task, merged in place
//!   before reducing; the phase-2 fold writes each completed `y_hi` row into
//!   its task's slices of the tables.

use std::ops::Range;

use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};
use jolt_witness::witnesses::{InstructionFlag, OpFlag, Pc, UnexpandedPc};
use jolt_witness::{JoltWitnessPlane, TraceRows, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_pairs, fmadd_u64_split, gamma_powers_array, map_indices, pin_derived_term, split_at_tasks,
    task_ranges, RoundChallenges, VISIT_ROWS,
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

/// One task's `Q` sums at one `x_lo`: the outer term's two accumulators and
/// the product term's two suffix sums, before the γ⁴ scale.
#[derive(Clone, Copy)]
struct ShiftQPartial<F: JoltField> {
    outer: [F::Accumulator; 2],
    product: [F; 2],
}

impl<F: JoltField> ShiftQPartial<F> {
    fn zero() -> Self {
        Self {
            outer: [F::Accumulator::default(); 2],
            product: [F::zero(); 2],
        }
    }

    fn merge(&mut self, other: &Self) {
        for (sum, term) in self.outer.iter_mut().zip(other.outer) {
            sum.merge(term);
        }
        for (sum, term) in self.product.iter_mut().zip(other.product) {
            *sum += term;
        }
    }
}

pub struct OptimizedSpartanShift;

impl<F: JoltField> PrepareKernel<F, SpartanShift<F>> for OptimizedSpartanShift {
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
        const LABEL: &str = "Spartan shift trace rows";
        let cycles = 1usize << log_t;
        let rows = witness
            .rows()
            .ok_or(WitnessError::UnavailableView { label: LABEL })?;
        if rows.cycles() != cycles {
            return Err(WitnessError::InvalidDimensions {
                label: LABEL,
                reason: format!("{} rows for a 2^{log_t}-cycle stage", rows.cycles()),
            }
            .into());
        }

        let gamma_powers: [F; 5] = gamma_powers_array(inputs.challenges.gamma);

        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        let prefix_vars = outer.prefix_0.len().trailing_zeros() as usize;

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
        // Each task visits a contiguous run of whole `x_hi` rows and adds every
        // row into entry `x_lo` of its own slot, allocated before the pass, so
        // partial state is `tasks · 2^prefix_vars` entries whatever the
        // scheduling. The merge gives each `Q` accumulator exactly the terms
        // one accumulator per `x_lo` would take: its headroom is unchanged.
        let x_lo_mask = (1 << prefix_vars) - 1;
        let ranges = task_ranges(cycles, 1 << prefix_vars);
        let mut slots = map_indices(ranges.len(), |_| {
            vec![ShiftQPartial::<F>::zero(); 1 << prefix_vars]
        });
        let accumulate = |(slot, range): (&mut Vec<ShiftQPartial<F>>, &Range<usize>)| {
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |first, bundles: &[SpartanShiftRow]| {
                    for (cycle, row) in (first..).zip(bundles) {
                        let suffix = &suffix_rows[cycle >> prefix_vars];
                        let partial = &mut slot[cycle & x_lo_mask];
                        let mut v = F::from_u64(row.unexpanded_pc.0)
                            + gamma_powers[1] * F::from_u64(row.pc.0);
                        if row.is_virtual.0 {
                            v += gamma_powers[2];
                        }
                        if row.is_first_in_sequence.0 {
                            v += gamma_powers[3];
                        }
                        partial.outer[0].fmadd(v, suffix[0]);
                        partial.outer[1].fmadd(v, suffix[1]);
                        if !row.is_noop.0 {
                            partial.product[0] += suffix[2];
                            partial.product[1] += suffix[3];
                        }
                    }
                    Ok(())
                },
            )
        };
        #[cfg(feature = "parallel")]
        slots.par_iter_mut().zip(&ranges).try_for_each(accumulate)?;
        #[cfg(not(feature = "parallel"))]
        slots.iter_mut().zip(&ranges).try_for_each(accumulate)?;

        let (merged, rest) = slots.split_at_mut(1);
        let merged = &mut merged[0];
        let rest = &*rest;
        let merge = |(x_lo, partial): (usize, &mut ShiftQPartial<F>)| {
            for slot in rest {
                partial.merge(&slot[x_lo]);
            }
        };
        #[cfg(feature = "parallel")]
        merged.par_iter_mut().enumerate().for_each(merge);
        #[cfg(not(feature = "parallel"))]
        merged.iter_mut().enumerate().for_each(merge);
        let q_outer = |pair: usize| -> Vec<F> {
            merged
                .iter()
                .map(|partial| partial.outer[pair].reduce())
                .collect()
        };
        let q_product = |pair: usize| -> Vec<F> {
            merged
                .iter()
                .map(|partial| gamma_powers[4] * partial.product[pair])
                .collect()
        };
        let pairs = [
            (outer.prefix_0, q_outer(0)),
            (outer.prefix_1, q_outer(1)),
            (product.prefix_0, q_product(0)),
            (product.prefix_1, q_product(1)),
        ];

        Ok(Box::new(ShiftKernel {
            log_t,
            gamma_powers,
            r_outer: r_outer.to_vec(),
            r_product: r_product.to_vec(),
            phase: Phase::PrefixSuffix { pairs, rows },
            challenges: RoundChallenges::new(log_t),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F> {
    /// First half of the rounds: the four `(P, Q)` pairs over the prefix
    /// variables (outer 0/1, product 0/1 — product Qs carry the γ⁴ scale),
    /// and the rows the transition folds.
    PrefixSuffix {
        pairs: [(Vec<F>, Vec<F>); 4],
        /// The witness plane owns these rows; it reports them itself.
        #[cfg_attr(feature = "allocative", allocative(skip))]
        rows: TraceRows,
    },
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

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ShiftKernel<F: JoltField> {
    log_t: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma_powers: [F; 5],
    r_outer: Vec<F>,
    r_product: Vec<F>,
    phase: Phase<F>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ShiftKernel<F> {
    /// The dense phase at the bound prefix: one pass in which every task folds
    /// whole `x_hi` rows by `eq(r_prefix, x_lo)` and writes each row's five
    /// column values into its own slices of the tables.
    fn dense_phase(&self, rows: &TraceRows) -> Result<Phase<F>, WitnessError> {
        let bound = self.challenges.bound();
        let r_prefix: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let cycles = 1usize << self.log_t;
        let x_lo_mask = (1 << bound) - 1;

        let mut columns: [Vec<F>; 5] = core::array::from_fn(|_| vec![F::zero(); cycles >> bound]);
        let ranges = task_ranges(cycles, 1 << bound);
        let slices = split_at_tasks(
            columns.each_mut().map(Vec::as_mut_slice),
            &ranges,
            |cycle| cycle >> bound,
        );
        let fold = |(range, mut outputs): (&Range<usize>, [&mut [F]; 5])| {
            let first_row = range.start >> bound;
            let mut pc_folds = [F::SmallScalarAccumulator::default(); 2];
            let mut flag_folds = [F::zero(); 3];
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |first, bundles: &[SpartanShiftRow]| {
                    for (cycle, row) in (first..).zip(bundles) {
                        let x_lo = cycle & x_lo_mask;
                        let (eq, eq_shifted) = (eq_prefix[x_lo], eq_prefix_shifted[x_lo]);
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
                        if x_lo == x_lo_mask {
                            let entry = (cycle >> bound) - first_row;
                            let [unexpanded_pc, pc] = std::mem::take(&mut pc_folds);
                            let [is_virtual, is_first, is_noop] =
                                std::mem::replace(&mut flag_folds, [F::zero(); 3]);
                            let values = [
                                unexpanded_pc.reduce(),
                                pc.reduce(),
                                is_virtual,
                                is_first,
                                is_noop,
                            ];
                            for (output, value) in outputs.iter_mut().zip(values) {
                                output[entry] = value;
                            }
                        }
                    }
                    Ok(())
                },
            )
        };
        #[cfg(feature = "parallel")]
        ranges.par_iter().zip(slices).try_for_each(fold)?;
        #[cfg(not(feature = "parallel"))]
        ranges.iter().zip(slices).try_for_each(fold)?;

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

        let [unexpanded_pc, pc, is_virtual, is_first_in_sequence, is_noop] = columns;
        Ok(Phase::Dense {
            eq_plus_one_outer: recombine(&self.r_outer),
            eq_plus_one_product: recombine(&self.r_product),
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
        })
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        self.challenges.push(r);
        if let Phase::PrefixSuffix { pairs, rows } = &self.phase {
            if pairs[0].0.len() == 2 {
                self.phase =
                    self.dense_phase(rows)
                        .map_err(|_| SumcheckError::MissingEvaluationSource {
                            kind: "spartan shift row",
                        })?;
                return Ok(());
            }
        }
        match &mut self.phase {
            Phase::PrefixSuffix { pairs, .. } => {
                for (p, q) in pairs {
                    bind_pairs(p, r);
                    bind_pairs(q, r);
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
                    bind_pairs(table, r);
                }
            }
        }
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for ShiftKernel<F> {
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
            self.bind(challenge)?;
        }

        // Degree-2 member: evals at t = 0 and t = 2; s(1) from the hint.
        let evals: [F; 2] = match &self.phase {
            Phase::PrefixSuffix { pairs, .. } => {
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
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for ShiftKernel<F> {
    type Relation = SpartanShift<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SpartanShiftOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
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
        self.challenges.require_complete()?;
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
            pin_derived_term(
                relation,
                JoltDerivedId::from(public),
                input_points,
                output_points,
                challenges,
                got,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId, JoltVirtualPolynomial};
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField, Ring};
    use jolt_poly::EqPlusOnePolynomial;
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{
        CircuitFlags, InstructionFlags, JoltInstructionKind, JoltInstructionRow,
        NormalizedOperands, RV64IMAC_JOLT,
    };
    use jolt_riscv::{JoltTraceRow as TraceRow, RamAccess, RegisterState};
    use jolt_verifier::stages::stage3::spartan_shift::{
        SpartanShift, SpartanShiftChallenges, SpartanShiftInputClaims,
    };
    use jolt_witness::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, JoltWitnessPlane,
        TraceBackend, WitnessError,
    };

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence,
    };
    use super::OptimizedSpartanShift;
    use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

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

    fn with_shift_plane<R>(log_t: usize, f: impl FnOnce(&TraceBackend) -> R) -> R {
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
        // zeroes the input claim. Otherwise the repeated script fills every
        // cycle but the last, so each row-pass task reads real rows.
        let real_rows = if log_t == 1 { 2 } else { (1 << log_t) - 1 };

        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                bytecode,
                plain_a.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 1 << log_t,
        });
        let rows: Vec<TraceRow> = script
            .iter()
            .cycle()
            .take(real_rows)
            .map(|&instruction| {
                TraceRow::new(
                    instruction,
                    RegisterState::default(),
                    RamAccess::NoOp,
                    preprocessing
                        .bytecode
                        .get_pc(&instruction)
                        .unwrap()
                        .try_into()
                        .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let program = Arc::new(JoltProgram::default());
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
            TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);
        f(&backend)
    }

    /// The optimized kernel against the reference tier over `backend`'s rows.
    fn assert_parity<F: JoltField>(backend: &TraceBackend, log_t: usize, seed: u64) {
        let r_outer = challenge_sequence::<F>(log_t, seed ^ 0x07E5);
        let r_product = challenge_sequence::<F>(log_t, seed ^ 0xFACE);
        let gamma = F::from_u64(0x5EED_0F0F_1234_5678);
        let relation = SpartanShift::<F>::new(
            TraceDimensions::new(log_t),
            r_outer.clone(),
            r_product.clone(),
        );

        let table = |polynomial: JoltVirtualPolynomial| -> Vec<F> {
            JoltWitnessOracle::<F>::oracle_table(backend, JoltPolynomialId::Virtual(polynomial))
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
        let input_claim: F = (0..1usize << log_t)
            .map(|j| {
                eq1_outer[j]
                    * (unexpanded_pc[j]
                        + gamma * pc[j]
                        + gamma_2 * is_virtual[j]
                        + gamma_3 * is_first[j])
                    + gamma_4 * eq1_product[j] * (F::from_u64(1) - is_noop[j])
            })
            .sum();
        assert_nontrivial(input_claim);

        let round_challenges = challenge_sequence(log_t, seed);
        assert_kernel_parity(
            &OptimizedSpartanShift,
            backend as &dyn JoltWitnessPlane<F>,
            &relation,
            &SpartanShiftInputClaims::default(),
            &SpartanShiftInputClaims::<Vec<F>>::default(),
            &SpartanShiftChallenges { gamma },
            input_claim,
            &round_challenges,
        );
    }

    fn run_parity(log_t: usize, seed: u64) {
        with_shift_plane(log_t, |backend| assert_parity::<Fr>(backend, log_t, seed));
    }

    /// The optimized kernel prepared over `witness` for an arbitrary relation.
    fn prepare_optimized(
        witness: &TraceBackend,
        log_t: usize,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = SpartanShift<Fr>>>, KernelError<Fr>> {
        let relation = SpartanShift::<Fr>::new(
            TraceDimensions::new(log_t),
            challenge_sequence(log_t, 5),
            challenge_sequence(log_t, 7),
        );
        OptimizedSpartanShift.prepare(
            &mut ProofSession::default(),
            witness,
            ProverInputs {
                relation: &relation,
                claims: &SpartanShiftInputClaims::default(),
                points: &SpartanShiftInputClaims::<Vec<Fr>>::default(),
                challenges: &SpartanShiftChallenges {
                    gamma: Fr::from_u64(3),
                },
            },
        )
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
        run_parity(1, 229);
    }

    /// Reference parity over both row passes' task partitions at three
    /// depths; Akita builds also check the Akita field.
    #[test]
    fn rows_match_reference_across_task_passes() {
        for log_t in [1, 3, 6] {
            with_shift_plane(log_t, |resident| {
                assert_parity::<Fr>(resident, log_t, 239);
                #[cfg(feature = "akita")]
                assert_parity::<Prime128OffsetA7F7>(resident, log_t, 241);
            });
        }
    }

    /// A witness without a valid handle or with one of another domain fails
    /// typed.
    #[test]
    fn missing_or_mismatched_rows_are_typed() {
        with_shift_plane(3, |resident| {
            let with_log_t = |log_t| {
                let mut witness = TraceBackend::new(
                    resident.config.clone(),
                    JoltVmWitnessInputs::new(
                        &resident.program,
                        &resident.preprocessing,
                        resident.trace.clone(),
                    ),
                );
                witness.config.log_t = log_t;
                witness
            };
            assert!(matches!(
                prepare_optimized(&with_log_t(1), 3),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
            assert!(matches!(
                prepare_optimized(&with_log_t(4), 3),
                Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
            ));
        });
    }
}
