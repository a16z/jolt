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
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_pairs, fmadd_u64_split, gamma_powers_array, pin_derived_term, BundleStore, RoundChallenges,
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
        let cycles = 1usize << log_t;
        // Slice-backed witnesses re-extract rows without retaining a vector.
        let rows = BundleStore::<SpartanShiftRow>::resolve(witness, cycles)?;
        let access = rows.access();

        let gamma_powers: [F; 5] = gamma_powers_array(inputs.challenges.gamma);

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
        let build_q_block =
            |(block_index, q_block): (usize, &mut [[F; 4]])| -> Result<(), WitnessError> {
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
                        let row = access.row(base + x_lo)?;
                        let mut v = F::from_u64(row.unexpanded_pc.0)
                            + gamma_powers[1] * F::from_u64(row.pc.0);
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
                Ok(())
            };
        let mut q_rows = vec![[F::zero(); 4]; 1 << prefix_vars];
        #[cfg(feature = "parallel")]
        q_rows
            .par_chunks_mut(BLOCK)
            .enumerate()
            .try_for_each(build_q_block)?;
        #[cfg(not(feature = "parallel"))]
        q_rows
            .chunks_mut(BLOCK)
            .enumerate()
            .try_for_each(build_q_block)?;

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
            challenges: RoundChallenges::new(log_t),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
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

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ShiftKernel<F: JoltField> {
    log_t: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma_powers: [F; 5],
    /// The two `eq+1` points (big-endian) the summand factors fix.
    r_outer: Vec<F>,
    r_product: Vec<F>,
    /// Raw values kept for phase-2 regeneration.
    rows: BundleStore<SpartanShiftRow>,
    phase: Phase<F>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ShiftKernel<F> {
    /// Regenerate the dense phase from the raw values: the five columns
    /// folded by `eq(r_prefix)` (their exact partial binds) and each `eq+1`
    /// table recombined from its suffix pair and bound-prefix evaluations.
    fn transition_to_dense(&mut self) -> Result<(), WitnessError> {
        let bound = self.challenges.bound();
        let r_prefix: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let chunk = eq_prefix.len();
        let remaining = 1usize << (self.log_t - bound);
        let access = self.rows.access();

        let fold_chunk = |chunk_index: usize| -> Result<[F; 5], WitnessError> {
            let base = chunk_index * chunk;
            let mut pc_folds = [F::SmallScalarAccumulator::default(); 2];
            let mut flag_folds = [F::zero(); 3];
            for (offset, (&eq, &eq_shifted)) in
                eq_prefix.iter().zip(eq_prefix_shifted.iter()).enumerate()
            {
                let row = access.row(base + offset)?;
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
            Ok([
                pc_folds[0].reduce(),
                pc_folds[1].reduce(),
                flag_folds[0],
                flag_folds[1],
                flag_folds[2],
            ])
        };
        #[cfg(feature = "parallel")]
        let folds: Vec<[F; 5]> = (0..remaining)
            .into_par_iter()
            .map(fold_chunk)
            .collect::<Result<_, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let folds: Vec<[F; 5]> = (0..remaining).map(fold_chunk).collect::<Result<_, _>>()?;

        // Release retained raw values after regeneration.
        self.rows = BundleStore::Retained(Vec::new());

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
        Ok(())
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        self.challenges.push(r);
        // Last prefix variable: regenerate the dense phase from the raw
        // values instead of binding the exhausted P·Q pairs.
        if matches!(&self.phase, Phase::PrefixSuffix { pairs } if pairs[0].0.len() == 2) {
            return self.transition_to_dense().map_err(|_| {
                SumcheckError::MissingEvaluationSource {
                    kind: "spartan shift row",
                }
            });
        }
        match &mut self.phase {
            Phase::PrefixSuffix { pairs } => {
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
    use jolt_field::{Fr, Ring};
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
    fn with_shift_plane<R>(log_t: usize, f: impl FnOnce(&TraceBackend<OwnedTrace>) -> R) -> R {
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
            .map(|&instruction| TraceRow::from_instruction(instruction).unwrap())
            .collect();

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
}
