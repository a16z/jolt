//! Optimized instruction input virtualization (stage 3).
//!
//! `GruenSplitEqPolynomial` factors the eq term out of
//! `eq(r_product, j) · ((is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc))(j)`.
//!
//! Round 0 evaluates native integer rows. Early binds retain Boolean flags as
//! packed bit patterns and materialize only operands. The constant suffix
//! contributes its summed equality weight. The bound eq scalar is checked
//! against the verifier.

mod flags;
use flags::PackedFlags;

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::{bind_prefix_in_place, BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_riscv::InstructionFlags;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, ToField, UnexpandedPc};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{pin_derived_term, BundleStore, GruenRoundMessage, RoundProgress};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The eight operand/flag tables, in output-claim declaration order:
/// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]`.
const NUM_TABLES: usize = 8;

use crate::mem::purge_retained_memory;

const LATE_PURGE_ROUNDS: usize = 8;

#[cfg(feature = "parallel")]
const MATERIALIZE_CHUNK: usize = 1 << 12;

/// One cycle's eight operand/flag values as native scalars.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
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
    fn flags(&self) -> [bool; 4] {
        [self.is_rs1.0, self.is_pc.0, self.is_rs2.0, self.is_imm.0]
    }

    fn operands<F: JoltField>(&self) -> [F; 4] {
        [
            self.rs1_value.to_field(),
            self.unexpanded_pc.to_field(),
            self.rs2_value.to_field(),
            self.imm.to_field(),
        ]
    }

    #[inline]
    fn field_values<F: JoltField>(&self) -> [F; NUM_TABLES] {
        [
            self.is_rs1.to_field(),
            self.rs1_value.to_field(),
            self.is_pc.to_field(),
            self.unexpanded_pc.to_field(),
            self.is_rs2.to_field(),
            self.rs2_value.to_field(),
            self.is_imm.to_field(),
            self.imm.to_field(),
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
        let rows = BundleStore::resolve(witness, 1usize << r_product.len())?;
        Ok(Box::new(OptimizedInstructionInputKernel::new(
            r_product,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum InputState<F: JoltField> {
    Native(BundleStore<InstructionInputRow>),
    Packed {
        operands: [Vec<F>; 4],
        flags: PackedFlags<F>,
    },
    Dense(Vec<Vec<F>>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionInputKernel<F: JoltField> {
    progress: RoundProgress,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    state: InputState<F>,
    prefix_len: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    tail: [F; NUM_TABLES],
    gruen: GruenSplitEqPolynomial<F>,
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

impl<F: JoltField> OptimizedInstructionInputKernel<F> {
    pub(crate) fn new(
        r_product: &[F],
        rows: BundleStore<InstructionInputRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = r_product.len();
        if let BundleStore::Retained(rows) = &rows {
            if rows.len() != 1 << log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: "instruction input operand rows".to_owned(),
                    expected: 1 << log_t,
                    got: rows.len(),
                });
            }
        }
        let tail = rows.access().row((1 << log_t) - 1)?;
        let prefix_len = rows.access().constant_suffix_start(1 << log_t)?;
        Ok(Self {
            prefix_len,
            tail: tail.field_values(),
            progress: RoundProgress::new(log_t),
            gamma,
            state: InputState::Native(rows),
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
        })
    }

    /// First-round `q` evaluations over native rows.
    /// The manual fold permits fallible row extraction.
    fn native_q_evals(
        &self,
        rows: &BundleStore<InstructionInputRow>,
    ) -> Result<[F; 4], WitnessError> {
        const POINTS: usize = 4;
        type Accumulator<F> = <F as WithAccumulator>::SignedProductAccumulator;
        let access = rows.access();
        let e_out = self.gruen.e_out_current();
        let e_in = self.gruen.e_in_current();
        let in_len = e_in.len();
        let gamma = self.gamma;
        let pairs = self.prefix_len.div_ceil(2);
        let outer_len = pairs.div_ceil(in_len);

        let block = |x_out: usize| -> Result<[F; POINTS], WitnessError> {
            let mut right_acc = [Accumulator::<F>::default(); POINTS];
            let mut right_high_acc = [Accumulator::<F>::default(); POINTS];
            let mut left_acc = [Accumulator::<F>::default(); POINTS];
            let inner_len = (pairs - x_out * in_len).min(in_len);
            for (x_in, &e_in) in e_in[..inner_len].iter().enumerate() {
                let y = x_out * in_len + x_in;
                let even = access.row(2 * y)?;
                let odd = access.row(2 * y + 1)?;
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
                    let (w_even, w_odd) = (i128::from(f_imm * (1 - t)), i128::from(f_imm * t));
                    let left = i128::from(f_rs1) * (rs1 + i128::from(t) * rs1_m)
                        + i128::from(f_pc) * (upc + i128::from(t) * upc_m);
                    let right = i128::from(f_rs2) * (rs2 + i128::from(t) * rs2_m)
                        + w_even * imm_even_low
                        + w_odd * imm_odd_low;
                    let right_high = w_even * imm_even_high + w_odd * imm_odd_high;
                    right_acc[t as usize].fmadd_i128(e_in, right);
                    if right_high != 0 {
                        right_high_acc[t as usize].fmadd_i128(e_in, right_high);
                    }
                    left_acc[t as usize].fmadd_i128(e_in, left);
                }
            }
            let e_out = e_out[x_out];
            let mut out = [F::zero(); POINTS];
            for (slot, ((right, right_high), left)) in out
                .iter_mut()
                .zip(right_acc.into_iter().zip(right_high_acc).zip(left_acc))
            {
                let right = right.reduce() + right_high.reduce().mul_pow_2(64);
                *slot = e_out * (right + gamma * left.reduce());
            }
            Ok(out)
        };
        let merge = |mut a: [F; POINTS], b: [F; POINTS]| {
            for (a, b) in a.iter_mut().zip(&b) {
                *a += *b;
            }
            a
        };

        #[cfg(feature = "parallel")]
        {
            (0..outer_len)
                .into_par_iter()
                .map(block)
                .try_reduce(|| [F::zero(); POINTS], |a, b| Ok(merge(a, b)))
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut folded = [F::zero(); POINTS];
            for x_out in 0..outer_len {
                folded = merge(folded, block(x_out)?);
            }
            Ok(folded)
        }
    }

    fn operand_product(&self, values: &[F; NUM_TABLES]) -> F {
        let right = values[4] * values[5] + values[6] * values[7];
        let left = values[0] * values[1] + values[2] * values[3];
        right + self.gamma * left
    }

    /// Sum the quadratic's constant and leading coefficients directly. The
    /// current sumcheck claim supplies its remaining coefficient through the
    /// same Gruen reconstruction used by other quadratic kernels.
    fn bound_message(
        &self,
        pair_at: impl Fn(usize, usize) -> (F, F) + Sync,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let pairs = self.prefix_len.div_ceil(2);
        let coefficients = self.gruen.par_fold_out_in_prefix(
            pairs,
            || [F::Accumulator::default(); 4],
            |lanes, row, _x_in, e_in| {
                // Separate left/right lanes let gamma be applied once per
                // block, rather than once per row or evaluation point.
                let mut inner = [F::Accumulator::default(); 4];
                for pair in 0..4 {
                    let (flag_0, flag_1) = pair_at(2 * pair, row);
                    let (value_0, value_1) = pair_at(2 * pair + 1, row);
                    let flag_m = flag_1 - flag_0;
                    let value_m = value_1 - value_0;
                    let offset = 2 * (pair / 2);
                    inner[offset].fmadd(flag_0, value_0);
                    inner[offset + 1].fmadd(flag_m, value_m);
                }
                for (lane, inner) in lanes.iter_mut().zip(inner) {
                    lane.fmadd(e_in, inner.reduce());
                }
            },
            |_x_out, e_out, lanes| {
                let [left_0, left_2, right_0, right_2] = lanes;
                [
                    e_out * (right_0.reduce() + self.gamma * left_0.reduce()),
                    e_out * (right_2.reduce() + self.gamma * left_2.reduce()),
                ]
            },
            |a, b| [a[0] + b[0], a[1] + b[1]],
        );
        let suffix = self.gruen.suffix_weight(pairs) * self.operand_product(&self.tail);
        self.gruen.checked_cubic(
            coefficients[0] + suffix,
            coefficients[1],
            previous_claim,
            round,
            || {
                // Exceptional equality challenges require q(1) explicitly.
                suffix
                    + self.gruen.par_fold_out_in_prefix(
                        pairs,
                        F::Accumulator::default,
                        |sum, row, _x_in, e_in| {
                            let values = core::array::from_fn(|i| pair_at(i, row).1);
                            sum.fmadd(e_in, self.operand_product(&values));
                        },
                        |_x_out, e_out, sum| e_out * sum.reduce(),
                        |a, b| a + b,
                    )
            },
        )
    }

    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let mut q_evals = match &self.state {
            InputState::Native(rows) => self.native_q_evals(rows).map_err(row_extraction_error)?,
            InputState::Dense(tables) => {
                return self.bound_message(
                    |i, row| {
                        (
                            tables[i][2 * row],
                            tables[i].get(2 * row + 1).copied().unwrap_or(self.tail[i]),
                        )
                    },
                    round,
                    previous_claim,
                )
            }
            InputState::Packed { operands, flags } => {
                return self.bound_message(
                    |i, row| {
                        if i % 2 == 0 {
                            (flags.value(i / 2, 2 * row), flags.value(i / 2, 2 * row + 1))
                        } else {
                            (
                                operands[i / 2][2 * row],
                                operands[i / 2]
                                    .get(2 * row + 1)
                                    .copied()
                                    .unwrap_or(self.tail[i]),
                            )
                        }
                    },
                    round,
                    previous_claim,
                )
            }
        };
        let suffix = self.gruen.suffix_weight(self.prefix_len.div_ceil(2))
            * self.operand_product(&self.tail);
        for value in &mut q_evals {
            *value += suffix;
        }
        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn materialize_half(
        rows: &BundleStore<InstructionInputRow>,
        challenge: F,
        half: usize,
        tail: [bool; 4],
    ) -> Result<InputState<F>, WitnessError> {
        let access = rows.access();
        let cell = |y: usize| -> Result<([F; 4], u64), WitnessError> {
            let even_row = access.row(2 * y)?;
            let odd_row = access.row(2 * y + 1)?;
            let word = PackedFlags::<F>::encode(even_row.flags(), odd_row.flags());
            let even = even_row.operands::<F>();
            let odd = odd_row.operands::<F>();
            Ok((
                core::array::from_fn(|i| even[i] + challenge * (odd[i] - even[i])),
                word,
            ))
        };

        let mut tables: [Vec<F>; 4] = core::array::from_fn(|_| unsafe_allocate_zero_vec(half));
        let mut words = vec![0u64; half];
        #[cfg(feature = "parallel")]
        {
            let [t0, t1, t2, t3] = &mut tables;
            (
                t0.par_chunks_mut(MATERIALIZE_CHUNK),
                t1.par_chunks_mut(MATERIALIZE_CHUNK),
                t2.par_chunks_mut(MATERIALIZE_CHUNK),
                t3.par_chunks_mut(MATERIALIZE_CHUNK),
                words.par_chunks_mut(MATERIALIZE_CHUNK),
            )
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_index, chunks)| -> Result<(), WitnessError> {
                    let base = chunk_index * MATERIALIZE_CHUNK;
                    let (c0, c1, c2, c3, codes) = chunks;
                    for offset in 0..c0.len() {
                        let (values, word) = cell(base + offset)?;
                        c0[offset] = values[0];
                        c1[offset] = values[1];
                        c2[offset] = values[2];
                        c3[offset] = values[3];
                        codes[offset] = word;
                    }
                    Ok(())
                })?;
        }
        #[cfg(not(feature = "parallel"))]
        for y in 0..half {
            let (values, word) = cell(y)?;
            for (table, value) in tables.iter_mut().zip(values) {
                table[y] = value;
            }
            words[y] = word;
        }
        Ok(InputState::Packed {
            operands: tables,
            flags: PackedFlags::new(words, tail, challenge),
        })
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.gruen.bind(challenge);
        match &mut self.state {
            InputState::Native(rows) => {
                self.state = Self::materialize_half(
                    rows,
                    challenge,
                    self.prefix_len.div_ceil(2),
                    core::array::from_fn(|i| self.tail[2 * i] == F::one()),
                )
                .map_err(row_extraction_error)?;
                purge_retained_memory(self.progress.total());
            }
            InputState::Packed { operands, flags } => {
                for (i, table) in operands.iter_mut().enumerate() {
                    let _ = bind_prefix_in_place(table, challenge, self.tail[2 * i + 1]);
                }
                if flags.saturated() {
                    let flag_tables = flags.materialize_bound(challenge);
                    let values = core::mem::take(operands);
                    self.state = InputState::Dense(
                        flag_tables
                            .into_iter()
                            .zip(values)
                            .flat_map(|(flag, value)| [flag, value])
                            .collect(),
                    );
                } else {
                    flags.bind(challenge);
                }
            }
            InputState::Dense(tables) => {
                for (table, tail) in tables.iter_mut().zip(self.tail) {
                    let _ = bind_prefix_in_place(table, challenge, tail);
                }
                if self.progress.bound() + 1 == LATE_PURGE_ROUNDS {
                    purge_retained_memory(self.progress.total());
                }
            }
        }
        self.prefix_len = self.prefix_len.div_ceil(2);
        self.progress.advance();
        Ok(())
    }

    fn final_values(&self) -> Result<[F; NUM_TABLES], WitnessError> {
        match &self.state {
            InputState::Native(rows) => Ok(rows.access().row(0)?.field_values()),
            InputState::Packed { operands, flags } => Ok(core::array::from_fn(|i| {
                if i % 2 == 0 {
                    flags.value(i / 2, 0)
                } else {
                    operands[i / 2].first().copied().unwrap_or(self.tail[i])
                }
            })),
            InputState::Dense(tables) => Ok(core::array::from_fn(|i| {
                tables[i].first().copied().unwrap_or(self.tail[i])
            })),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionInputKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
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
        self.progress.require_complete()?;
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
    use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId, TraceDimensions};
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage3::outputs::InstructionInput;
    use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::{BundleStore, InstructionInputRow, OptimizedInstructionInputKernel};

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn assert_parity(log_t: usize, seed: u64, prefix: usize, boolean_challenges: bool) {
        assert_parity_field::<Fr>(log_t, seed, prefix, boolean_challenges, 0xDADA_CAFE);
    }

    fn assert_parity_field<F: JoltField>(
        log_t: usize,
        seed: u64,
        prefix: usize,
        boolean_challenges: bool,
        gamma_value: u64,
    ) {
        let fr = F::from_u64;
        let mut state = seed;
        let mut rows: Vec<InstructionInputRow> = (0..1usize << log_t)
            .map(|index| {
                let raw = splitmix(&mut state);
                let wide = ((splitmix(&mut state) as i128) << 64) | splitmix(&mut state) as i128;
                InstructionInputRow {
                    is_rs1: InstructionFlag(raw & 1 != 0),
                    rs1_value: Rs1Value(splitmix(&mut state)),
                    is_pc: InstructionFlag(raw & 2 != 0),
                    unexpanded_pc: UnexpandedPc(splitmix(&mut state)),
                    is_rs2: InstructionFlag(raw & 4 != 0),
                    rs2_value: Rs2Value(splitmix(&mut state)),
                    is_imm: InstructionFlag(raw & 8 != 0),
                    imm: Imm(if index % 3 == 0 { -wide } else { wide }),
                }
            })
            .collect();
        let tail = rows[rows.len() - 1];
        rows[prefix..].fill(tail);
        let challenge = |round| {
            if boolean_challenges {
                fr((round % 2) as u64)
            } else {
                F::from_u64(
                    0x1357_9BDF_2468_ACE0
                        ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ 0x55,
                )
            }
        };
        let tables: Vec<Vec<F>> = (0..8)
            .map(|table| {
                rows.iter()
                    .map(|row| row.field_values::<F>()[table])
                    .collect()
            })
            .collect();
        let r_product: Vec<F> = (0..log_t)
            .map(|i| {
                if boolean_challenges {
                    fr((i % 2) as u64)
                } else {
                    fr(900 + 53 * i as u64)
                }
            })
            .collect();
        let gamma = fr(gamma_value);
        let relation = InstructionInput::<F>::new(TraceDimensions::new(log_t), r_product.clone());

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
            Polynomial::new(eq_table(&r_product)),
        )]);
        let input_claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let input_points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &challenges,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();

        let mut optimized =
            OptimizedInstructionInputKernel::new(&r_product, BundleStore::Retained(rows), gamma)
                .unwrap();

        let eq = eq_table(&r_product);
        let mut claim = fr(0);
        for j in 0..1usize << log_t {
            let right = tables[4][j] * tables[5][j] + tables[6][j] * tables[7][j];
            let left = tables[0][j] * tables[1][j] + tables[2][j] * tables[3][j];
            claim += eq[j] * (right + gamma * left);
        }

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let reference_outputs = reference.output_claims(&input_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);

        let sumcheck_point: Vec<F> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn parity_even_log_t() {
        for prefix in 0..=16 {
            for boolean_challenges in [false, true] {
                assert_parity(4, 21, prefix, boolean_challenges);
            }
        }
    }

    #[test]
    fn parity_odd_log_t() {
        assert_parity(3, 8080, 5, false);
    }

    #[test]
    fn parity_single_round() {
        assert_parity(1, 4242, 1, false);
    }

    #[test]
    fn parity_two_rounds() {
        assert_parity(2, 1717, 4, false);
    }

    #[test]
    fn parity_through_packed_to_dense_transition() {
        for prefix in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64] {
            for exceptional in [false, true] {
                assert_parity(6, 919, prefix, exceptional);
            }
        }
    }

    #[cfg(feature = "akita")]
    #[test]
    fn parity_fp128_with_constant_tails_and_exceptional_challenges() {
        for prefix in [0, 1, 2, 15, 16, 17, 63, 64] {
            for exceptional in [false, true] {
                for gamma in [0, 1, 0xDADA_CAFE] {
                    assert_parity_field::<Prime128OffsetA7F7>(6, 717, prefix, exceptional, gamma);
                }
            }
        }
    }
}
