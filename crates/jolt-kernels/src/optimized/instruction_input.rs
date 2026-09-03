//! Optimized instruction input virtualization (stage 3).
//!
//! `GruenSplitEqPolynomial` factors the eq term out of
//! `eq(r_product, j) · ((is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc))(j)`.
//!
//! Round 0 evaluates native integer rows. The first bind materializes all
//! eight columns at `T/2`. The bound eq scalar is checked against the verifier.

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::signed::{S192, S256, S64};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
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

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::support::collect_rows;
use super::support::{pin_derived_term, BundleStore, GruenRoundMessage, RoundProgress};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The eight operand/flag tables, in output-claim declaration order:
/// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]`.
const NUM_TABLES: usize = 8;

use crate::mem::purge_retained_memory;

/// Bind count that triggers the late allocator purge.
const LATE_PURGE_ROUNDS: usize = 8;

/// The parallel scatter grain of the `T/2` table materialization.
#[cfg(feature = "parallel")]
const MATERIALIZE_CHUNK: usize = 1 << 12;

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

#[cfg(all(feature = "metal", target_os = "macos"))]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct PreparedInstructionInputRows {
    rows: Vec<InstructionInputRow>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl PreparedInstructionInputRows {
    pub(crate) const fn len(&self) -> usize {
        self.rows.len()
    }

    fn into_rows(self) -> Vec<InstructionInputRow> {
        self.rows
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_instruction_input_rows<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    trace_elements: usize,
) -> Result<(), KernelError<F>> {
    let rows = collect_rows(witness, trace_elements)?;
    session.park(PreparedInstructionInputRows { rows });
    Ok(())
}

impl InstructionInputRow {
    /// Field values in table order.
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

/// Optimized [`PrepareKernel`] implementor for the `instruction_input` slot.
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

/// Native rows through round 0; eight dense tables afterward.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum InputState<F: JoltField> {
    Native(BundleStore<InstructionInputRow>),
    Dense(Vec<Polynomial<F>>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Offloaded,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionInputKernel<F: JoltField> {
    progress: RoundProgress,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    state: InputState<F>,
    gruen: GruenSplitEqPolynomial<F>,
}

fn row_extraction_error<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "instruction input operand row",
    }
}

/// Exact `(value at t = 0, step)` for a linear extension.
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
        Ok(Self {
            progress: RoundProgress::new(log_t),
            gamma,
            state: InputState::Native(rows),
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
        })
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn new_offloaded(r_product: &[F], gamma: F) -> Self {
        Self {
            progress: RoundProgress::new(r_product.len()),
            gamma,
            state: InputState::Offloaded,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
        }
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

        let block = |x_out: usize| -> Result<[F; POINTS], WitnessError> {
            let mut right_acc = [Accumulator::<F>::default(); POINTS];
            let mut left_acc = [Accumulator::<F>::default(); POINTS];
            for (x_in, &e_in) in e_in.iter().enumerate() {
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
                let imm_even = S192::from_i128(even.imm.0);
                let imm_odd = S192::from_i128(odd.imm.0);
                for t in 0..POINTS as i64 {
                    // |f(t)| ≤ 3; |v(t)| < 2^67; products < 2^70.
                    let f_rs1 = is_rs1 + t * is_rs1_m;
                    let f_pc = is_pc + t * is_pc_m;
                    let f_rs2 = is_rs2 + t * is_rs2_m;
                    let f_imm = is_imm + t * is_imm_m;
                    let left = i128::from(f_rs1) * (rs1 + i128::from(t) * rs1_m)
                        + i128::from(f_pc) * (upc + i128::from(t) * upc_m);
                    // Lagrange coefficients are ≤ 12; products fit `S256`.
                    let mut right =
                        S256::from_i128(i128::from(f_rs2) * (rs2 + i128::from(t) * rs2_m));
                    S64::from_i64(f_imm * (1 - t)).fmadd_trunc::<3, 4>(&imm_even, &mut right);
                    S64::from_i64(f_imm * t).fmadd_trunc::<3, 4>(&imm_odd, &mut right);
                    right_acc[t as usize].fmadd_s256(e_in, &right);
                    left_acc[t as usize].fmadd_s256(e_in, &S256::from_i128(left));
                }
            }
            let e_out = e_out[x_out];
            let mut out = [F::zero(); POINTS];
            for (slot, (right, left)) in out.iter_mut().zip(right_acc.into_iter().zip(left_acc)) {
                *slot = e_out * (right.reduce() + gamma * left.reduce());
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
            (0..e_out.len())
                .into_par_iter()
                .map(block)
                .try_reduce(|| [F::zero(); POINTS], |a, b| Ok(merge(a, b)))
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut folded = [F::zero(); POINTS];
            for x_out in 0..e_out.len() {
                folded = merge(folded, block(x_out)?);
            }
            Ok(folded)
        }
    }

    /// The bound rounds' `q` evaluations over the eight dense tables.
    fn dense_q_evals(&self, tables: &[Polynomial<F>]) -> [F; 4] {
        const POINTS: usize = 4;
        self.gruen.par_fold_out_in(
            || {
                (
                    [F::zero(); POINTS],
                    [F::zero(); NUM_TABLES],
                    [F::zero(); NUM_TABLES],
                )
            },
            |(acc, evals, steps), row, _x_in, e_in| {
                for ((table, eval), step) in
                    tables.iter().zip(evals.iter_mut()).zip(steps.iter_mut())
                {
                    let table = table.evals();
                    let lo = table[2 * row];
                    *eval = lo;
                    *step = table[2 * row + 1] - lo;
                }
                for value in acc.iter_mut() {
                    let right = evals[4] * evals[5] + evals[6] * evals[7];
                    let left = evals[0] * evals[1] + evals[2] * evals[3];
                    *value += e_in * (right + self.gamma * left);
                    for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                        *eval += *step;
                    }
                }
            },
            |_x_out, e_out, (mut acc, _, _)| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        )
    }

    /// `s(t) = ℓ(t) · Σ_y E(y) · q(t, y)` at `t = 0..=3`, with
    /// `q = (is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc)`.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let mut q_evals = match &self.state {
            InputState::Native(rows) => self.native_q_evals(rows).map_err(row_extraction_error)?,
            InputState::Dense(tables) => self.dense_q_evals(tables),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => {
                return Err(instruction_input_state_error(
                    "CPU message requested while instruction input is resident on Metal",
                ));
            }
        };
        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    /// Materializes the first bind directly at half size.
    fn materialize_half(
        rows: &BundleStore<InstructionInputRow>,
        challenge: F,
        half: usize,
    ) -> Result<Vec<Polynomial<F>>, WitnessError> {
        let access = rows.access();
        let cell = |y: usize| -> Result<[F; NUM_TABLES], WitnessError> {
            let even = access.row(2 * y)?.field_values::<F>();
            let odd = access.row(2 * y + 1)?.field_values::<F>();
            Ok(core::array::from_fn(|i| {
                even[i] + challenge * (odd[i] - even[i])
            }))
        };

        // Fill all eight tables in one row pass.
        let mut tables: [Vec<F>; NUM_TABLES] =
            core::array::from_fn(|_| unsafe_allocate_zero_vec(half));
        #[cfg(feature = "parallel")]
        {
            let [t0, t1, t2, t3, t4, t5, t6, t7] = &mut tables;
            (
                t0.par_chunks_mut(MATERIALIZE_CHUNK),
                t1.par_chunks_mut(MATERIALIZE_CHUNK),
                t2.par_chunks_mut(MATERIALIZE_CHUNK),
                t3.par_chunks_mut(MATERIALIZE_CHUNK),
                t4.par_chunks_mut(MATERIALIZE_CHUNK),
                t5.par_chunks_mut(MATERIALIZE_CHUNK),
                t6.par_chunks_mut(MATERIALIZE_CHUNK),
                t7.par_chunks_mut(MATERIALIZE_CHUNK),
            )
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_index, chunks)| -> Result<(), WitnessError> {
                    let base = chunk_index * MATERIALIZE_CHUNK;
                    let (c0, c1, c2, c3, c4, c5, c6, c7) = chunks;
                    for offset in 0..c0.len() {
                        let values = cell(base + offset)?;
                        c0[offset] = values[0];
                        c1[offset] = values[1];
                        c2[offset] = values[2];
                        c3[offset] = values[3];
                        c4[offset] = values[4];
                        c5[offset] = values[5];
                        c6[offset] = values[6];
                        c7[offset] = values[7];
                    }
                    Ok(())
                })?;
        }
        #[cfg(not(feature = "parallel"))]
        for y in 0..half {
            let values = cell(y)?;
            for (table, value) in tables.iter_mut().zip(values) {
                table[y] = value;
            }
        }
        Ok(tables.into_iter().map(Polynomial::new).collect())
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.gruen.bind(challenge);
        match &mut self.state {
            InputState::Native(rows) => {
                let tables =
                    Self::materialize_half(rows, challenge, 1 << (self.progress.total() - 1))
                        .map_err(row_extraction_error)?;
                self.state = InputState::Dense(tables);
                // Return row and preparation pages before dense rounds.
                purge_retained_memory(self.progress.total());
            }
            InputState::Dense(tables) => {
                // In-place binds avoid eight dead half-size generations.
                for table in tables.iter_mut() {
                    let _ = table.bind_low_to_high_in_place(challenge);
                }
                // Return shrink tails once the live tables are small.
                if self.progress.bound() + 1 == LATE_PURGE_ROUNDS {
                    purge_retained_memory(self.progress.total());
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => unreachable!("offloaded instruction input binds on Metal"),
        }
        self.progress.advance();
        Ok(())
    }

    /// The eight fully bound table values, table order.
    fn final_values(&self) -> Result<[F; NUM_TABLES], WitnessError> {
        match &self.state {
            // Only for `log_t = 0`.
            InputState::Native(rows) => Ok(rows.access().row(0)?.field_values()),
            InputState::Dense(tables) => Ok(core::array::from_fn(|i| tables[i].evals()[0])),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => Err(WitnessError::UnavailableView {
                label: "instruction input state is offloaded to Metal",
            }),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn metal_restore_dense(
        &mut self,
        flat_tables: &[F],
        elements: usize,
    ) -> Result<(), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "instruction input restore requested while the CPU still owns the state",
            ));
        }
        if elements == 0
            || !elements.is_power_of_two()
            || self.progress.bound() > self.progress.total()
        {
            return Err(instruction_input_state_error(
                "invalid instruction input dense-tail geometry",
            ));
        }
        let expected_elements = 1usize
            .checked_shl((self.progress.total() - self.progress.bound()) as u32)
            .ok_or_else(|| {
                instruction_input_state_error("instruction input table length overflow")
            })?;
        if elements != expected_elements {
            return Err(instruction_input_state_error(format!(
                "instruction input dense tail has {elements} elements per table; expected {expected_elements}"
            )));
        }
        let expected_values = NUM_TABLES.checked_mul(elements).ok_or_else(|| {
            instruction_input_state_error("instruction input readback length overflow")
        })?;
        if flat_tables.len() != expected_values {
            return Err(instruction_input_state_error(format!(
                "instruction input dense tail has {} values; expected {expected_values}",
                flat_tables.len()
            )));
        }
        self.state = InputState::Dense(
            flat_tables
                .chunks_exact(elements)
                .map(|values| Polynomial::new(values.to_vec()))
                .collect(),
        );
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn instruction_input_state_error<F: JoltField>(message: impl Into<String>) -> SumcheckError<F> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
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

    /// Check the bound Gruen scalar against the verifier.
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
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage3::outputs::InstructionInput;
    use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::{BundleStore, InstructionInputRow, OptimizedInstructionInputKernel};

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

    fn assert_parity(log_t: usize, seed: u64) {
        let mut state = seed;
        // Covers full-width values and signed immediates.
        let rows: Vec<InstructionInputRow> = (0..1usize << log_t)
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
        let tables: Vec<Vec<Fr>> = (0..8)
            .map(|table| {
                rows.iter()
                    .map(|row| row.field_values::<Fr>()[table])
                    .collect()
            })
            .collect();
        let r_product: Vec<Fr> = (0..log_t).map(|i| fr(900 + 53 * i as u64)).collect();
        let gamma = fr(0xDADA_CAFE);
        let relation = InstructionInput::<Fr>::new(TraceDimensions::new(log_t), r_product.clone());

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

        // Direct hypercube sum.
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

        let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn parity_even_log_t() {
        assert_parity(4, 21);
    }

    #[test]
    fn parity_odd_log_t() {
        assert_parity(3, 8080);
    }

    #[test]
    fn parity_single_round() {
        // Fully bound during the deferred first bind.
        assert_parity(1, 4242);
    }

    #[test]
    fn parity_two_rounds() {
        // Materializes length-one tables in `finish_rounds`.
        assert_parity(2, 1717);
    }
}
