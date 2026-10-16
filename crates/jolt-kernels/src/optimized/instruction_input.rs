//! Optimized instruction input virtualization (stage 3).
//!
//! `GruenSplitEqPolynomial` factors the eq term out of
//! `eq(r_product, j) · ((is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc))(j)`.
//!
//! Every pass reads rows through the witness's row handle, split into at most
//! pool-thread-count tasks of whole split-eq groups that each visit their rows
//! once, serially. Round 0 evaluates native integer rows. The eight columns
//! stay separate under every bind: until their dense tables fit
//! `INPUT_DENSE_CAP`, round `j ≥ 1` contracts each column over the low and
//! high halves of its `2^(j+1)`-cycle pairs with `eq(r_1..r_j, ·)` (LowToHigh
//! binding gives `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`), and the first round
//! whose tables fit, round 1 at the earliest, fills them in the pass that
//! yields its message. The bound eq scalar is checked against the verifier.

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly,
};
use jolt_riscv::InstructionFlags;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

use super::support::{
    dense_crossover, gruen_pair_pass, pin_derived_term, task_ranges, try_par_sum_vecs,
    GruenRoundMessage, KernelRows, PairHalves, RoundChallenges, VISIT_ROWS,
};
use crate::mem::purge_retained_memory;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The eight operand/flag tables, in output-claim declaration order:
/// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]`.
const NUM_TABLES: usize = 8;

/// Each round samples `q` at `t = 0, 1, 2, 3`.
const POINTS: usize = 4;

/// Byte cap on the eight dense tables: rounds recompute their pairs from the
/// rows until the current round's tables fit ([`dense_crossover`]).
const INPUT_DENSE_CAP: usize = 4 << 30;

/// One cycle's eight operand/flag values as native scalars.
#[derive(Clone, Copy, Debug, WitnessBundle)]
pub struct InstructionInputRow {
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsRs1Value))]
    pub is_rs1: InstructionFlag,
    pub rs1_value: Rs1Value,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsPC))]
    pub is_pc: InstructionFlag,
    pub unexpanded_pc: UnexpandedPc,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsRs2Value))]
    pub is_rs2: InstructionFlag,
    pub rs2_value: Rs2Value,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsImm))]
    pub is_imm: InstructionFlag,
    pub imm: Imm,
}

impl InstructionInputRow {
    /// The eight column values of this cycle times `weight`, in table order.
    #[inline]
    fn weighted_values<F: JoltField>(&self, weight: F) -> [F; NUM_TABLES] {
        let flag = |set: InstructionFlag| if set.0 { weight } else { F::zero() };
        [
            flag(self.is_rs1),
            weight.mul_u64(self.rs1_value.0),
            flag(self.is_pc),
            weight.mul_u64(self.unexpanded_pc.0),
            flag(self.is_rs2),
            weight.mul_u64(self.rs2_value.0),
            flag(self.is_imm),
            weight.mul_i128(self.imm.0),
        ]
    }
}

pub struct OptimizedInstructionInput;

impl<F: JoltField> PrepareKernel<F, InstructionInput<F>> for OptimizedInstructionInput {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let r_product = inputs.relation.product_remainder_opening_point();
        let rows =
            KernelRows::from_witness(witness, r_product.len(), "instruction input trace rows")?;
        Ok(Box::new(OptimizedInstructionInputKernel::new(
            r_product,
            rows,
            inputs.challenges.gamma,
            INPUT_DENSE_CAP,
        )))
    }
}

/// The eight operand/flag tables over the cycle domain.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum InputTables<F: JoltField> {
    /// Rounds before the crossover recompute their pairs from the rows.
    Streamed,
    /// Filled by the crossover round's pass, bound in place afterwards.
    Dense(Vec<Polynomial<F>>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionInputKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    tables: InputTables<F>,
    /// The round whose pass fills `tables`, at least 1; the round count when
    /// they never fit the cap.
    dense_round: usize,
    purged: bool,
    gruen: GruenSplitEqPolynomial<F>,
    challenges: RoundChallenges<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<InstructionInputRow>,
}

fn row_extraction_error<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "instruction input operand row",
    }
}

#[inline]
fn ext_u64(even: u64, odd: u64) -> (i128, i128) {
    (i128::from(even), i128::from(odd) - i128::from(even))
}

#[inline]
fn ext_flag(even: bool, odd: bool) -> (i64, i64) {
    (i64::from(even), i64::from(odd) - i64::from(even))
}

/// One pair's `q(t)` at `t = 0..POINTS` from each column's `[low, high]`
/// halves, every column linear in `t`.
fn pair_q_evals<F: JoltField>(halves: [[F; 2]; NUM_TABLES], gamma: F) -> [F; POINTS] {
    let mut values = halves.map(|[low, _]| low);
    let steps = halves.map(|[low, high]| high - low);
    let mut q_evals = [F::zero(); POINTS];
    for q in &mut q_evals {
        let right = values[4] * values[5] + values[6] * values[7];
        let left = values[0] * values[1] + values[2] * values[3];
        *q = right + gamma * left;
        for (value, step) in values.iter_mut().zip(&steps) {
            *value += *step;
        }
    }
    q_evals
}

impl<F: JoltField> OptimizedInstructionInputKernel<F> {
    /// `dense_cap` bounds the bytes of the eight dense tables; production
    /// passes [`INPUT_DENSE_CAP`]. Round 0 reads native rows, so the tables
    /// fill at round 1 at the earliest, `T/2` entries each.
    fn new(
        r_product: &[F],
        rows: KernelRows<InstructionInputRow>,
        gamma: F,
        dense_cap: usize,
    ) -> Self {
        let rounds = r_product.len();
        let table_bytes = (NUM_TABLES * size_of::<F>()) << rounds;
        Self {
            gamma,
            tables: InputTables::Streamed,
            dense_round: dense_crossover(table_bytes, dense_cap, rounds).max(1),
            purged: false,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            challenges: RoundChallenges::new(rounds),
            rows,
        }
    }

    /// Round 0's `q` evaluations over native integer row pairs. A callback of
    /// at most `group` aligned rows lies in one `e_out` block and holds whole
    /// even/odd pairs ([`task_ranges`] starts every task on a group boundary
    /// and visits deliver exact `chunk`-row callbacks), so its accumulators
    /// take at most `VISIT_ROWS / 2` terms.
    fn native_q_evals(&self) -> Result<[F; POINTS], WitnessError> {
        type Accumulator<F> = <F as WithAccumulator>::SignedProductAccumulator;
        let e_out = self.gruen.e_out_current();
        let e_in = self.gruen.e_in_current();
        let group = 2 * e_in.len();
        let gamma = self.gamma;
        let ranges = task_ranges(e_out.len() * group, self.rows.segment_len(), group);
        try_par_sum_vecs(ranges.len(), POINTS, |task| {
            let mut q_evals = vec![F::zero(); POINTS];
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(group),
                |start, bundles| {
                    let mut right_acc = [Accumulator::<F>::default(); POINTS];
                    let mut right_high_acc = [Accumulator::<F>::default(); POINTS];
                    let mut left_acc = [Accumulator::<F>::default(); POINTS];
                    for (pair, &weight) in bundles.chunks_exact(2).zip(&e_in[start % group / 2..]) {
                        let (even, odd) = (&pair[0], &pair[1]);
                        let (is_rs1, is_rs1_m) = ext_flag(even.is_rs1.0, odd.is_rs1.0);
                        let (is_pc, is_pc_m) = ext_flag(even.is_pc.0, odd.is_pc.0);
                        let (is_rs2, is_rs2_m) = ext_flag(even.is_rs2.0, odd.is_rs2.0);
                        let (is_imm, is_imm_m) = ext_flag(even.is_imm.0, odd.is_imm.0);
                        let (rs1, rs1_m) = ext_u64(even.rs1_value.0, odd.rs1_value.0);
                        let (upc, upc_m) = ext_u64(even.unexpanded_pc.0, odd.unexpanded_pc.0);
                        let (rs2, rs2_m) = ext_u64(even.rs2_value.0, odd.rs2_value.0);
                        let split = |imm: i128| (imm >> 64, i128::from(imm as u64));
                        let (imm_even_high, imm_even_low) = split(even.imm.0);
                        let (imm_odd_high, imm_odd_low) = split(odd.imm.0);
                        for t in 0..POINTS as i64 {
                            // |f(t)| ≤ 3, |v(t)| < 2^67, and the Lagrange weights
                            // |1 − t|, |t| ≤ 3 keep every low sum below 2^70 and the
                            // high sums below 2^67: exact in `i128` for any immediate.
                            let f_rs1 = is_rs1 + t * is_rs1_m;
                            let f_pc = is_pc + t * is_pc_m;
                            let f_rs2 = is_rs2 + t * is_rs2_m;
                            let f_imm = is_imm + t * is_imm_m;
                            let (w_even, w_odd) =
                                (i128::from(f_imm * (1 - t)), i128::from(f_imm * t));
                            let left = i128::from(f_rs1) * (rs1 + i128::from(t) * rs1_m)
                                + i128::from(f_pc) * (upc + i128::from(t) * upc_m);
                            let right = i128::from(f_rs2) * (rs2 + i128::from(t) * rs2_m)
                                + w_even * imm_even_low
                                + w_odd * imm_odd_low;
                            let right_high = w_even * imm_even_high + w_odd * imm_odd_high;
                            right_acc[t as usize].fmadd_i128(weight, right);
                            if right_high != 0 {
                                right_high_acc[t as usize].fmadd_i128(weight, right_high);
                            }
                            left_acc[t as usize].fmadd_i128(weight, left);
                        }
                    }
                    let e_out = e_out[start / group];
                    for (q, ((right, right_high), left)) in q_evals
                        .iter_mut()
                        .zip(right_acc.into_iter().zip(right_high_acc).zip(left_acc))
                    {
                        let right = right.reduce() + right_high.reduce().mul_pow_2(64);
                        *q += e_out * (right + gamma * left.reduce());
                    }
                    Ok(())
                },
            )?;
            Ok(q_evals)
        })
        .map(|q_evals| core::array::from_fn(|t| q_evals[t]))
    }

    fn dense_q_evals(&self, tables: &[Polynomial<F>]) -> [F; POINTS] {
        self.gruen.par_fold_out_in(
            || [F::zero(); POINTS],
            |q_evals, row, _x_in, e_in| {
                let halves: [[F; 2]; NUM_TABLES] = core::array::from_fn(|table| {
                    let evals = tables[table].evals();
                    [evals[2 * row], evals[2 * row + 1]]
                });
                for (q, pair_q) in q_evals.iter_mut().zip(pair_q_evals(halves, self.gamma)) {
                    *q += e_in * pair_q;
                }
            },
            |_x_out, e_out, mut q_evals| {
                for q in &mut q_evals {
                    *q *= e_out;
                }
                q_evals
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        )
    }

    /// Row weights of the current round `j`'s pass, one per cycle offset `c`
    /// within a pair half: `eq(r_1..r_j, c)`, offset bit `i` pairing
    /// `r_{i+1}`. Passes run for `j ≤ dense_round`, so the `2^j` entries
    /// stay at most `max(2, 2·table_bytes / cap)`.
    fn round_weights(&self) -> Vec<F> {
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&point, None)
    }

    /// One pass of the current round `j ≥ 1` over its pairs of `2^(j+1)`
    /// cycles ([`gruen_pair_pass`]), filling `tables` when given: each row's
    /// eight column values, weighted for its offset within its half, add to
    /// the low or high half of its pair ([`PairHalves`]), and every completed
    /// pair contributes its `q` evaluations.
    fn round_pass(
        &self,
        tables: Option<[&mut [F]; NUM_TABLES]>,
    ) -> Result<[F; POINTS], WitnessError> {
        let weights = self.round_weights();
        let log_half = self.challenges.bound();
        let gamma = self.gamma;
        gruen_pair_pass(
            &self.gruen,
            log_half + 1,
            self.rows.segment_len(),
            tables,
            |halves| pair_q_evals(halves, gamma),
            |range, chunk, sink| {
                let mut halves = PairHalves::new(log_half);
                self.rows.visit(range, chunk, |start, bundles| {
                    for (t, row) in (start..).zip(bundles) {
                        let values = row.weighted_values(weights[halves.offset(t)]);
                        if let Some((pair, pair_halves)) = halves.add(t, values) {
                            sink.emit(pair, pair_halves);
                        }
                    }
                    Ok(())
                })
            },
        )
    }

    /// The current round's pass, filling the eight dense tables: each of
    /// length `T / 2^round`, allocated once at its final capacity and written
    /// in disjoint task slices.
    fn fill_tables(&mut self) -> Result<[F; POINTS], WitnessError> {
        let len = (1usize << self.challenges.total()) >> self.challenges.bound();
        let mut tables: [Vec<F>; NUM_TABLES] =
            core::array::from_fn(|_| unsafe_allocate_zero_vec(len));
        let q_evals = self.round_pass(Some(tables.each_mut().map(|table| table.as_mut_slice())))?;
        self.tables = InputTables::Dense(tables.into_iter().map(Polynomial::new).collect());
        Ok(q_evals)
    }

    fn message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let mut q_evals = if let InputTables::Dense(tables) = &self.tables {
            self.dense_q_evals(tables)
        } else if self.challenges.bound() == 0 {
            self.native_q_evals().map_err(row_extraction_error)?
        } else if self.challenges.bound() == self.dense_round {
            self.fill_tables().map_err(row_extraction_error)?
        } else {
            self.round_pass(None).map_err(row_extraction_error)?
        };
        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    /// Binds the dense tables in place, without half-size copies, one after
    /// another: a shrink holds only one table's old and new buffers beside
    /// the others.
    fn bind(&mut self, challenge: F) {
        if let InputTables::Dense(tables) = &mut self.tables {
            let mut shrunk = false;
            for table in tables.iter_mut() {
                shrunk |= table.bind_low_to_high_in_place(challenge);
            }
            if shrunk && !self.purged {
                self.purged = true;
                purge_retained_memory(self.challenges.total());
            }
        }
        self.gruen.bind(challenge);
        self.challenges.push(challenge);
    }

    /// The eight opening values of never-filled tables: one walk over the
    /// rows with `eq(r_cycle, ·)` as split tables over the bound challenges
    /// reversed into big-endian order.
    fn walked_values(&self) -> Result<[F; NUM_TABLES], WitnessError> {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let weights = TensorEqTable::new(&reversed);
        let (e_out, e_in) = (weights.e_out(), weights.e_in());
        let group = e_in.len();
        let ranges = task_ranges(weights.len(), self.rows.segment_len(), group);
        try_par_sum_vecs(ranges.len(), NUM_TABLES, |task| {
            let mut values = vec![F::zero(); NUM_TABLES];
            // A callback of at most `group` aligned rows lies in one `e_out`
            // block, and its accumulators take at most VISIT_ROWS terms.
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(group),
                |start, bundles| {
                    let mut flags: [<F as WithAccumulator>::SmallScalarAccumulator; 4] =
                        Default::default();
                    let mut words: [<F as WithAccumulator>::SignedProductAccumulator; 4] =
                        Default::default();
                    for (row, &weight) in bundles.iter().zip(&e_in[start % group..]) {
                        flags[0].fmadd_u64(weight, u64::from(row.is_rs1.0));
                        words[0].fmadd_u64(weight, row.rs1_value.0);
                        flags[1].fmadd_u64(weight, u64::from(row.is_pc.0));
                        words[1].fmadd_u64(weight, row.unexpanded_pc.0);
                        flags[2].fmadd_u64(weight, u64::from(row.is_rs2.0));
                        words[2].fmadd_u64(weight, row.rs2_value.0);
                        flags[3].fmadd_u64(weight, u64::from(row.is_imm.0));
                        words[3].fmadd_i128(weight, row.imm.0);
                    }
                    let [is_rs1, is_pc, is_rs2, is_imm] = flags;
                    let [rs1, upc, rs2, imm] = words;
                    let sums = [
                        is_rs1.reduce(),
                        rs1.reduce(),
                        is_pc.reduce(),
                        upc.reduce(),
                        is_rs2.reduce(),
                        rs2.reduce(),
                        is_imm.reduce(),
                        imm.reduce(),
                    ];
                    let scale = e_out[start / group];
                    for (value, sum) in values.iter_mut().zip(sums) {
                        *value += scale * sum;
                    }
                    Ok(())
                },
            )?;
            Ok(values)
        })
        .map(|values| core::array::from_fn(|table| values[table]))
    }

    fn final_values(&self) -> Result<[F; NUM_TABLES], WitnessError> {
        match &self.tables {
            InputTables::Streamed => self.walked_values(),
            InputTables::Dense(tables) => Ok(core::array::from_fn(|i| tables[i].evals()[0])),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionInputKernel<F> {
    fn num_rounds(&self) -> usize {
        self.challenges.total()
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let [left_operand_is_rs1, rs1_value, left_operand_is_pc, unexpanded_pc, right_operand_is_rs2, rs2_value, right_operand_is_imm, imm] =
            self.final_values()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "instruction input row re-extraction failed after the rounds",
                })?;
        Ok(InstructionInputOutputClaims {
            left_operand_is_rs1,
            rs1_value,
            left_operand_is_pc,
            unexpanded_pc,
            right_operand_is_rs2,
            rs2_value,
            right_operand_is_imm,
            imm,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let id = JoltDerivedId::from(InstructionInputPublic::EqProduct);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.gruen.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
        rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::TraceDimensions;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::TraceBackend;

    use super::*;
    use crate::optimized::parity::{probe_input_claim, replayed, run_lockstep, ExceptionalEq};
    use crate::optimized::registers_read_write::test_support::structured_fixture;
    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ReferenceBackend};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x1357_9BDF_2468_ACE0 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Rows no trace produces: random flag combinations, full-range words
    /// and immediates of both signs. Every fifth row sets every flag with
    /// `u64::MAX` words and an `i128::MIN` immediate; the row after it
    /// carries an `i128::MAX` immediate.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<InstructionInputRow> {
        let mut state = seed;
        (0..1usize << log_t)
            .map(|index| {
                let flags = splitmix(&mut state);
                let wide = ((splitmix(&mut state) as i128) << 64) | splitmix(&mut state) as i128;
                let row = InstructionInputRow {
                    is_rs1: InstructionFlag(flags & 1 != 0),
                    rs1_value: Rs1Value(splitmix(&mut state)),
                    is_pc: InstructionFlag(flags & 2 != 0),
                    unexpanded_pc: UnexpandedPc(splitmix(&mut state)),
                    is_rs2: InstructionFlag(flags & 4 != 0),
                    rs2_value: Rs2Value(splitmix(&mut state)),
                    is_imm: InstructionFlag(flags & 8 != 0),
                    imm: Imm(wide),
                };
                match index % 5 {
                    0 => InstructionInputRow {
                        is_rs1: InstructionFlag(true),
                        rs1_value: Rs1Value(u64::MAX),
                        is_pc: InstructionFlag(true),
                        unexpanded_pc: UnexpandedPc(u64::MAX),
                        is_rs2: InstructionFlag(true),
                        rs2_value: Rs2Value(u64::MAX),
                        is_imm: InstructionFlag(true),
                        imm: Imm(i128::MIN),
                    },
                    1 => InstructionInputRow {
                        imm: Imm(i128::MAX),
                        ..row
                    },
                    _ => row,
                }
            })
            .collect()
    }

    /// The row's eight columns through the witness field encoding.
    fn column_values(row: &InstructionInputRow) -> [Fr; NUM_TABLES] {
        [
            row.is_rs1.to_field(),
            row.rs1_value.to_field(),
            row.is_pc.to_field(),
            row.unexpanded_pc.to_field(),
            row.is_rs2.to_field(),
            row.rs2_value.to_field(),
            row.is_imm.to_field(),
            row.imm.to_field(),
        ]
    }

    /// Drives the reference prover over the rows' dense columns and the
    /// optimized kernel, its tables capped at exactly their size at round
    /// `depth` (`1 ≤ depth ≤ log_t`, the round count meaning never), through
    /// every round on `challenges`: equal round polynomials and output
    /// claims, the pinned eq scalar, and tables filled at round `depth` with
    /// `T / 2^depth` entries each.
    fn assert_parity(
        rows: Vec<InstructionInputRow>,
        r_product: &[Fr],
        challenges: &[Fr],
        depth: usize,
    ) {
        let log_t = r_product.len();
        let tables: Vec<Vec<Fr>> = (0..NUM_TABLES)
            .map(|table| rows.iter().map(|row| column_values(row)[table]).collect())
            .collect();
        let gamma = fr(0xDADA_CAFE);
        let relation = InstructionInput::<Fr>::new(TraceDimensions::new(log_t), r_product.to_vec());
        let ids = [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
        ];
        let opening_tables: BTreeMap<_, _> = ids
            .into_iter()
            .zip(&tables)
            .map(|(id, values)| (id, Polynomial::new(values.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionInputPublic::EqProduct),
            Polynomial::new(eq_table(r_product)),
        )]);
        let input_claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let input_points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let gamma_challenge = InstructionInputChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &gamma_challenge,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();
        let cap = ((NUM_TABLES * size_of::<Fr>()) << log_t) >> depth;
        let mut optimized = OptimizedInstructionInputKernel::new(
            r_product,
            KernelRows::Synthetic(rows),
            gamma,
            cap,
        );

        let mut claim = fr(0);
        for (j, weight) in eq_table(r_product).into_iter().enumerate() {
            let right = tables[4][j] * tables[5][j] + tables[6][j] * tables[7][j];
            let left = tables[0][j] * tables[1][j] + tables[2][j] * tables[3][j];
            claim += weight * (right + gamma * left);
        }
        let dense_len = |kernel: &OptimizedInstructionInputKernel<Fr>| match &kernel.tables {
            InputTables::Dense(dense) => Some(dense[0].evals().len()),
            InputTables::Streamed => None,
        };
        let mut filled = None;
        let mut bind = None;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial, log_t {log_t}, depth {depth}"
            );
            filled = filled.or_else(|| dense_len(&optimized).map(|len| (round, len)));
            claim = reference_poly.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference.finish_rounds(last).unwrap();
        optimized.finish_rounds(last).unwrap();
        assert_eq!(
            filled,
            (depth < log_t).then(|| (depth, (1 << log_t) >> depth)),
            "dense tables, log_t {log_t}, depth {depth}"
        );
        assert_eq!(
            optimized.output_claims(&input_claims).unwrap(),
            reference.output_claims(&input_claims).unwrap(),
            "output claims, log_t {log_t}, depth {depth}"
        );
        let output_points = relation
            .derive_opening_points(challenges, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &gamma_challenge)
            .unwrap();
    }

    fn product_point(log_t: usize) -> Vec<Fr> {
        (0..log_t).map(|i| fr(900 + 53 * i as u64)).collect()
    }

    /// Wide synthetic rows at every crossover round, including none. At
    /// 2^13 cycles the never-filled round 12 pairs span two visit callbacks.
    #[test]
    fn synthetic_parity_at_every_crossover() {
        for (log_t, seed) in [
            (1usize, 4242u64),
            (2, 1717),
            (3, 8080),
            (4, 21),
            (5, 55),
            (8, 88),
            (13, 1313),
        ] {
            let challenges: Vec<Fr> = (0..log_t).map(challenge).collect();
            let first_depth = if log_t == 13 { log_t } else { 1 };
            for depth in first_depth..=log_t {
                assert_parity(
                    synthetic_rows(log_t, seed),
                    &product_point(log_t),
                    &challenges,
                    depth,
                );
            }
        }
    }

    /// Exceptional eq points (coordinates 0 or 1, a prefix that vanishes at
    /// the first bind) and zero/one challenges, through round-1 fills,
    /// streamed rounds before a later fill, dense rounds and never-filled
    /// tables.
    #[test]
    fn exceptional_eq_and_zero_one_challenges_match_reference() {
        let log_t = 4;
        let challenges: Vec<Fr> = (0..log_t).map(challenge).collect();
        let zero_one: Vec<Fr> = (0..log_t).map(|round| fr(round as u64 % 2)).collect();
        for depth in [1, 2, log_t] {
            for case in ExceptionalEq::ALL {
                assert_parity(
                    synthetic_rows(log_t, 379),
                    &case.point(log_t, challenges[0]),
                    &challenges,
                    depth,
                );
            }
            assert_parity(
                synthetic_rows(log_t, 379),
                &product_point(log_t),
                &zero_one,
                depth,
            );
        }
    }

    /// One [`trace_rows_match_reference_at_every_crossover`] case: the tables
    /// capped to first fit at round `depth`, or the production cap through
    /// the trait path when `None`.
    fn trace_case<F: JoltField>(
        resident: &TraceBackend,
        witness: &TraceBackend,
        log_t: usize,
        depth: Option<usize>,
    ) {
        let r_product: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(37 + 23 * i as u64))
            .collect();
        let relation = InstructionInput::new(TraceDimensions::new(log_t), r_product.clone());
        let claims = InstructionInputInputClaims {
            right_instruction_input: F::zero(),
            left_instruction_input: F::zero(),
        };
        let points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges {
            gamma: F::from_u64(7019),
        };
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), resident, inputs())
            .unwrap();
        let mut optimized: Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>> = match depth
        {
            Some(depth) => Box::new(OptimizedInstructionInputKernel::new(
                &r_product,
                KernelRows::from_witness::<F>(witness, log_t, "trace rows").unwrap(),
                challenges.gamma,
                ((NUM_TABLES * size_of::<F>()) << log_t) >> depth,
            )),
            None => OptimizedInstructionInput
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap(),
        };
        let input_claim = probe_input_claim(reference.as_mut());
        let round_challenges: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(883 + 29 * i as u64))
            .collect();
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            input_claim,
            &round_challenges,
        );
        assert_eq!(
            optimized.output_claims(&claims).unwrap(),
            reference.output_claims(&claims).unwrap(),
            "output claims, log_t {log_t}, depth {depth:?}"
        );
    }

    /// Every round and the output claims over replayed and resident trace
    /// rows (two-row replay checkpoints, task boundaries and a padded proof
    /// suffix inside the domain) at every crossover round, including none,
    /// and at the production cap, against the reference over the resident
    /// witness, in both fields.
    #[test]
    fn trace_rows_match_reference_at_every_crossover() {
        for log_t in [1, 3, 6] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |resident| {
                let replay = replayed(resident, 2, None);
                for witness in [resident, &replay] {
                    for depth in (1..=log_t).map(Some).chain([None]) {
                        trace_case::<Fr>(resident, witness, log_t, depth);
                        #[cfg(feature = "akita")]
                        trace_case::<Prime128OffsetA7F7>(resident, witness, log_t, depth);
                    }
                }
            });
        }
    }

    /// A witness without a row handle, or with another cycle domain, fails
    /// typed instead of collecting rows; a replay failure surfaces typed
    /// from round 0, a round-1 fill, a streamed round 1 and the opening walk
    /// of never-filled tables.
    #[test]
    fn missing_rows_and_replay_failures_are_typed() {
        let log_t = 3;
        let point = |len: usize| (0..len).map(|i| fr(5 + i as u64)).collect::<Vec<_>>();
        let claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges { gamma: fr(3) };
        let prepare = |witness: &TraceBackend, log_t: usize| {
            let relation = InstructionInput::new(TraceDimensions::new(log_t), point(log_t));
            OptimizedInstructionInput
                .prepare(
                    &mut ProofSession::default(),
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .map(|_| ())
        };
        structured_fixture(7).with_plane(log_t, |resident| {
            // A configured domain shorter than the trace leaves no valid handle.
            let mut shortened = replayed(resident, 2, None);
            shortened.config.log_t = 1;
            assert!(matches!(
                prepare(&shortened, log_t),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
            assert!(matches!(
                prepare(resident, log_t + 1),
                Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
            ));

            let failing = replayed(resident, 2, Some(5));
            let kernel = |depth: usize| {
                OptimizedInstructionInputKernel::new(
                    &point(log_t),
                    KernelRows::from_witness::<Fr>(&failing, log_t, "replayed rows").unwrap(),
                    fr(3),
                    ((NUM_TABLES * size_of::<Fr>()) << log_t) >> depth,
                )
            };
            assert!(matches!(
                kernel(1).prove_round(None, 0, fr(0)),
                Err(SumcheckError::MissingEvaluationSource { .. })
            ));
            for depth in [1, 2] {
                assert!(matches!(
                    kernel(depth).prove_round(Some(fr(9)), 1, fr(0)),
                    Err(SumcheckError::MissingEvaluationSource { .. })
                ));
            }
            let mut never = kernel(log_t);
            for round in 0..log_t {
                never.bind(fr(11 + round as u64));
            }
            assert!(matches!(
                never.output_claims(&claims),
                Err(SumcheckKernelError::InvariantViolation { .. })
            ));
        });
    }
}
