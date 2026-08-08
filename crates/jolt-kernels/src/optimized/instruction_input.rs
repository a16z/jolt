//! Optimized instruction input-virtualization (stage 3) kernel.
//!
//! The summand is
//! `eq(r_product, j) · ((is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc))(j)`
//! — degree 3, with the eq weight factored out via `GruenSplitEqPolynomial`:
//! each round emits `s(t) = ℓ(t) · Σ_y E_out·E_in · q(t, y)` at the naive
//! prover's `t = 0..=3` sample points through the same `from_evals`
//! constructor — byte-identical round polynomials and output claims, with no
//! `T`-sized eq table materialized or bound.
//!
//! The eight operand/flag columns stay **native scalars** until the first
//! bind (the legacy prover's compact-polynomial shape): one typed-bundle
//! collection replaces eight dense `oracle_table` materializations, and the
//! first round's `q(t, y)` runs on an exact integer pipeline — flag and u64
//! lanes extend linearly over `t` as `i64`/`i128`, the `i128` immediate lane
//! through small Lagrange coefficients into `S256` (`imm(t)·f(t) =
//! f(t)(1−t)·e₀ + f(t)t·e₁`, coefficients ≤ 12) — folded into the field
//! through the signed-product accumulator. The field is a ring homomorphism
//! from the integers, so every `q(t, y)` equals the dense-table computation
//! exactly. The first bind materializes the eight dense tables at `T/2`.
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables)
//! (Gruen scalar vs `derive_output_term(EqProduct)`).

use std::mem::size_of;

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::signed::{S192, S256, S64};
use jolt_field::{Field, SignedProductAccumulator as _, WithSignedProductAccumulator};
use jolt_poly::thread::unsafe_allocate_zero_vec;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_riscv::InstructionFlags;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, ToField, UnexpandedPc};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::collect_rows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The eight operand/flag tables, in output-claim declaration order:
/// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]`.
const NUM_TABLES: usize = 8;

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

pub(crate) struct PreparedInstructionInputRows {
    rows: Vec<InstructionInputRow>,
}

impl PreparedInstructionInputRows {
    pub(crate) const fn len(&self) -> usize {
        self.rows.len()
    }

    fn into_rows(self) -> Vec<InstructionInputRow> {
        self.rows
    }

    fn storage_id(&self) -> usize {
        self.rows.as_ptr() as usize
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PreparedInstructionInputRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("rows"),
            crate::backend::vec_heap_bytes(&self.rows),
        );
        visitor.exit();
    }
}

pub(crate) fn prepare_instruction_input_rows<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    trace_elements: usize,
) -> Result<(), KernelError<F>> {
    let row_bytes = trace_elements
        .checked_mul(size_of::<InstructionInputRow>())
        .ok_or(KernelError::InvariantViolation {
            reason: "InstructionInput prepared-row byte length overflow",
        })?;
    let rows = collect_rows(witness, trace_elements)?;
    let prepared = PreparedInstructionInputRows { rows };
    let cpu_rows_storage_id = prepared.storage_id();
    let _span = tracing::info_span!(
        "OptimizedInstructionInput::rows_prepare",
        cpu_rows_storage_id,
        cpu_rows = trace_elements,
        cpu_row_bytes = row_bytes / trace_elements
    )
    .entered();
    session.park(prepared);
    Ok(())
}

impl InstructionInputRow {
    /// The row's values as field elements, in table order — exactly the
    /// entries the dense tables hold (same `ToField` conversions as the
    /// oracle walk).
    #[inline]
    fn field_values<F: Field>(&self) -> [F; NUM_TABLES] {
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

impl<F: Field> PrepareKernel<F, InstructionInput<F>> for OptimizedInstructionInput {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let r_product = inputs.relation.product_remainder_opening_point();
        let trace_elements = 1usize << r_product.len();
        let prepared = session.take::<PreparedInstructionInputRows>();
        let cpu_rows_storage_id = prepared.as_ref().map_or(0, |rows| rows.storage_id());
        let _span = tracing::info_span!(
            "OptimizedInstructionInput::rows_stage3_use",
            cpu_rows_storage_id,
            cpu_rows = trace_elements
        )
        .entered();
        let rows = match prepared {
            Some(prepared) if prepared.len() == trace_elements => prepared.into_rows(),
            Some(_) => {
                return Err(KernelError::InvariantViolation {
                    reason: "prepared InstructionInput row count disagrees with the relation",
                });
            }
            None => collect_rows(witness, trace_elements)?,
        };
        Ok(Box::new(OptimizedInstructionInputKernel::new(
            r_product,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// The column state: native rows until the first bind, eight dense tables
/// afterwards.
enum InputState<F: Field> {
    Native(Vec<InstructionInputRow>),
    Dense(Vec<Polynomial<F>>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Offloaded,
}

pub struct OptimizedInstructionInputKernel<F: Field> {
    log_t: usize,
    gamma: F,
    state: InputState<F>,
    gruen: GruenSplitEqPolynomial<F>,
    bind_scratch: Vec<F>,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for OptimizedInstructionInputKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{gruen_heap_bytes, polys_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        let state_bytes = match &self.state {
            InputState::Native(rows) => vec_heap_bytes(rows),
            InputState::Dense(polys) => polys_heap_bytes(polys),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => 0,
        };
        visitor.visit_simple(allocative::Key::new("state"), state_bytes);
        visitor.visit_simple(allocative::Key::new("gruen"), gruen_heap_bytes(&self.gruen));
        visitor.visit_simple(
            allocative::Key::new("bind_scratch"),
            vec_heap_bytes(&self.bind_scratch),
        );
        visitor.exit();
    }
}

/// `(value at t = 0, step)` of the pair's linear extension, as exact
/// integers.
#[inline]
fn ext_u64(even: u64, odd: u64) -> (i128, i128) {
    (i128::from(even), i128::from(odd) - i128::from(even))
}

#[inline]
fn ext_flag(even: bool, odd: bool) -> (i64, i64) {
    (i64::from(even), i64::from(odd) - i64::from(even))
}

impl<F: Field> OptimizedInstructionInputKernel<F> {
    pub fn new(
        r_product: &[F],
        rows: Vec<InstructionInputRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = r_product.len();
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction input operand rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        Ok(Self {
            log_t,
            gamma,
            state: InputState::Native(rows),
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        })
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn new_offloaded(r_product: &[F], gamma: F) -> Self {
        Self {
            log_t: r_product.len(),
            gamma,
            state: InputState::Offloaded,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        }
    }

    /// The first round's `q` evaluations over the native rows: per pair and
    /// sample point, `left`/`right` are exact integers folded into the field
    /// through the signed-product accumulator; `Σ e_in·right + γ·Σ e_in·left`
    /// equals the dense per-point evaluation by distributivity.
    fn native_q_evals(&self, rows: &[InstructionInputRow]) -> [F; 4] {
        const POINTS: usize = 4;
        type Accumulator<F> = <F as WithSignedProductAccumulator>::SignedProductAccumulator;
        self.gruen.par_fold_out_in(
            || {
                (
                    [Accumulator::<F>::default(); POINTS],
                    [Accumulator::<F>::default(); POINTS],
                )
            },
            |(right_acc, left_acc), y, _x_in, e_in| {
                let even = &rows[2 * y];
                let odd = &rows[2 * y + 1];
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
                    // Flag lanes stay tiny (|f(t)| ≤ 3); u64 lanes fit i128
                    // (|v(t)| < 2^67); products < 2^70.
                    let f_rs1 = is_rs1 + t * is_rs1_m;
                    let f_pc = is_pc + t * is_pc_m;
                    let f_rs2 = is_rs2 + t * is_rs2_m;
                    let f_imm = is_imm + t * is_imm_m;
                    let left = i128::from(f_rs1) * (rs1 + i128::from(t) * rs1_m)
                        + i128::from(f_pc) * (upc + i128::from(t) * upc_m);
                    // `f(t)·imm(t) = f(t)(1−t)·e₀ + f(t)t·e₁`: coefficients
                    // ≤ 12, so the products stay under 2^131 inside `S256`
                    // even for full-range `i128` immediates.
                    let mut right =
                        S256::from_i128(i128::from(f_rs2) * (rs2 + i128::from(t) * rs2_m));
                    S64::from_i64(f_imm * (1 - t)).fmadd_trunc::<3, 4>(&imm_even, &mut right);
                    S64::from_i64(f_imm * t).fmadd_trunc::<3, 4>(&imm_odd, &mut right);
                    right_acc[t as usize].fmadd_s256(e_in, &right);
                    left_acc[t as usize].fmadd_s256(e_in, &S256::from_i128(left));
                }
            },
            |_x_out, e_out, (right_acc, left_acc)| {
                let mut out = [F::zero(); POINTS];
                for (slot, (right, left)) in out.iter_mut().zip(right_acc.into_iter().zip(left_acc))
                {
                    *slot = e_out * (right.reduce() + self.gamma * left.reduce());
                }
                out
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        )
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
        let q_evals = match &self.state {
            InputState::Native(rows) => self.native_q_evals(rows),
            InputState::Dense(tables) => self.dense_q_evals(tables),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => {
                return Err(instruction_input_state_error(
                    "CPU message requested while instruction input is resident on Metal",
                ));
            }
        };

        self.message_from_q_evals(q_evals, round, previous_claim)
    }

    fn message_from_q_evals(
        &self,
        q_evals: [F; 4],
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let (l_at_0, l_at_1) = self.gruen.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        let mut evals = [F::zero(); 4];
        for (eval, q) in evals.iter_mut().zip(&q_evals) {
            *eval = l_eval * *q;
            l_eval += l_step;
        }

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

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if self.rounds_bound >= self.log_t {
            return Err(instruction_input_state_error(
                "instruction input received more binds than cycle variables",
            ));
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "CPU bind requested while instruction input is resident on Metal",
            ));
        }
        self.gruen.bind(challenge);
        match &mut self.state {
            InputState::Native(rows) => {
                // First bind: materialize the eight dense tables at `T/2` —
                // the same `v₀ + r·(v₁ − v₀)` a dense-table bind computes.
                let half = rows.len() / 2;
                let materialize = |table: usize| -> Polynomial<F> {
                    let mut values: Vec<F> = unsafe_allocate_zero_vec(half);
                    let fill = |(y, slot): (usize, &mut F)| {
                        let even = rows[2 * y].field_values::<F>()[table];
                        let odd = rows[2 * y + 1].field_values::<F>()[table];
                        *slot = even + challenge * (odd - even);
                    };
                    #[cfg(feature = "parallel")]
                    values.par_iter_mut().enumerate().for_each(|(y, slot)| {
                        fill((y, slot));
                    });
                    #[cfg(not(feature = "parallel"))]
                    values.iter_mut().enumerate().for_each(|(y, slot)| {
                        fill((y, slot));
                    });
                    Polynomial::new(values)
                };
                let tables = (0..NUM_TABLES).map(materialize).collect();
                self.state = InputState::Dense(tables);
            }
            InputState::Dense(tables) => {
                for table in tables.iter_mut() {
                    table.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => unreachable!("offloaded state returned above"),
        }
        self.rounds_bound += 1;
        Ok(())
    }

    /// The eight fully bound table values, table order.
    fn final_values(&self) -> Option<[F; NUM_TABLES]> {
        match &self.state {
            // Bindless extraction happens only for `log_t = 0` geometries.
            InputState::Native(rows) => Some(rows[0].field_values()),
            InputState::Dense(tables) => Some(core::array::from_fn(|i| tables[i].evals()[0])),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InputState::Offloaded => None,
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<F: Field> OptimizedInstructionInputKernel<F> {
    pub(crate) fn metal_copy_dense_tables(
        &self,
        table_ids: [usize; 2],
        expected_rounds_bound: usize,
        expected_elements: usize,
    ) -> Result<[Vec<F>; 2], SumcheckError<F>> {
        if self.rounds_bound != expected_rounds_bound {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot has the wrong bind count",
            ));
        }
        let InputState::Dense(tables) = &self.state else {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot requires host dense tables",
            ));
        };
        if table_ids
            .iter()
            .any(|&table| table >= tables.len() || tables[table].len() != expected_elements)
        {
            return Err(instruction_input_state_error(
                "instruction input alias snapshot has the wrong table geometry",
            ));
        }
        Ok(table_ids.map(|table| tables[table].evals().to_vec()))
    }

    pub(crate) fn metal_weights(&self) -> Result<(&[F], &[F]), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal weights requested after instruction input returned to the CPU",
            ));
        }
        Ok((self.gruen.e_in_current(), self.gruen.e_out_current()))
    }

    pub(crate) fn metal_bind_offloaded(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal bind requested after instruction input returned to the CPU",
            ));
        }
        if self.rounds_bound >= self.log_t {
            return Err(instruction_input_state_error(
                "instruction input received more binds than cycle variables",
            ));
        }
        self.gruen.bind(challenge);
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_message(
        &self,
        q_coefficients: [F; 3],
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "Metal message supplied after instruction input returned to the CPU",
            ));
        }
        let [q_at_0, q_at_1, q_quadratic] = q_coefficients;
        let twice_quadratic = q_quadratic + q_quadratic;
        let q_at_2 = q_at_1 + q_at_1 - q_at_0 + twice_quadratic;
        let q_at_3 = q_at_2 + q_at_1 - q_at_0 + twice_quadratic + twice_quadratic;
        self.message_from_q_evals([q_at_0, q_at_1, q_at_2, q_at_3], round, previous_claim)
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        flat_tables: &[F],
        elements: usize,
    ) -> Result<(), SumcheckError<F>> {
        if !matches!(self.state, InputState::Offloaded) {
            return Err(instruction_input_state_error(
                "instruction input dense tables restored more than once",
            ));
        }
        if elements == 0 || !elements.is_power_of_two() || self.rounds_bound > self.log_t {
            return Err(instruction_input_state_error(
                "invalid instruction input dense-tail geometry",
            ));
        }
        let expected_elements = 1usize
            .checked_shl((self.log_t - self.rounds_bound) as u32)
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

fn instruction_input_state_error<F: Field>(message: impl Into<String>) -> SumcheckError<F> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

impl<F: Field> ProveRounds<F> for OptimizedInstructionInputKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let [left_operand_is_rs1, rs1_value, left_operand_is_pc, unexpanded_pc, right_operand_is_rs2, rs2_value, right_operand_is_imm, imm] =
            self.final_values()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "instruction input output requested while tables remain on Metal",
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

    /// Pin the fully-bound Gruen scalar to the verifier's
    /// `derive_output_term(EqProduct)`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(InstructionInputPublic::EqProduct);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use jolt_field::Invertible;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage3::outputs::InstructionInput;
    use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::{InstructionInputRow, OptimizedInstructionInputKernel};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x1357_9BDF_2468_ACE0 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn quadratic_descriptors(evals: [Fr; 4]) -> [Fr; 3] {
        let inverse_two = fr(2).inverse().unwrap();
        [
            evals[0],
            evals[1],
            (evals[2] + evals[0]) * inverse_two - evals[1],
        ]
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
        // Native rows with Boolean flags, full-range u64 values, and signed
        // immediates spanning the i128 lane (including a negative one wider
        // than u64 — the S256 Lagrange path).
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

        let mut optimized = OptimizedInstructionInputKernel::new(&r_product, rows, gamma).unwrap();

        // True input claim: the full hypercube sum of the summand.
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_offload_roundtrip_matches_cpu() {
        let log_t = 4;
        let mut state = 0xA11C_E55E_D00D_F00D;
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
        let r_product: Vec<Fr> = (0..log_t).map(|i| fr(700 + 29 * i as u64)).collect();
        let gamma = fr(0xC001_CAFE);
        let eq = eq_table(&r_product);
        let mut claim = fr(0);
        for (weight, row) in eq.iter().zip(&rows) {
            let values = row.field_values::<Fr>();
            let right = values[4] * values[5] + values[6] * values[7];
            let left = values[0] * values[1] + values[2] * values[3];
            claim += *weight * (right + gamma * left);
        }

        let mut cpu =
            OptimizedInstructionInputKernel::new(&r_product, rows.clone(), gamma).unwrap();
        let mut offloaded = OptimizedInstructionInputKernel::new_offloaded(&r_product, gamma);

        let q_coefficients = quadratic_descriptors(cpu.native_q_evals(&rows));
        let cpu_poly = cpu.prove_round(None, 0, claim).unwrap();
        let metal_poly = offloaded.metal_message(q_coefficients, 0, claim).unwrap();
        assert_eq!(metal_poly.coefficients(), cpu_poly.coefficients());
        claim = cpu_poly.evaluate(challenge(0));

        let cpu_poly = cpu.prove_round(Some(challenge(0)), 1, claim).unwrap();
        offloaded.metal_bind_offloaded(challenge(0)).unwrap();
        let tables = match &cpu.state {
            super::InputState::Dense(tables) => Some(tables),
            super::InputState::Native(_) | super::InputState::Offloaded => None,
        }
        .unwrap();
        let q_coefficients = quadratic_descriptors(cpu.dense_q_evals(tables));
        let elements = tables[0].len();
        let flat_tables: Vec<Fr> = tables
            .iter()
            .flat_map(|table| table.evals().iter().copied())
            .collect();
        let metal_poly = offloaded.metal_message(q_coefficients, 1, claim).unwrap();
        assert_eq!(metal_poly.coefficients(), cpu_poly.coefficients());
        offloaded
            .metal_restore_dense(&flat_tables, elements)
            .unwrap();
        claim = cpu_poly.evaluate(challenge(1));

        for round in 2..log_t {
            let bind = challenge(round - 1);
            let cpu_poly = cpu.prove_round(Some(bind), round, claim).unwrap();
            let restored_poly = offloaded.prove_round(Some(bind), round, claim).unwrap();
            assert_eq!(restored_poly.coefficients(), cpu_poly.coefficients());
            claim = cpu_poly.evaluate(challenge(round));
        }
        let final_bind = challenge(log_t - 1);
        cpu.finish_rounds(final_bind).unwrap();
        offloaded.finish_rounds(final_bind).unwrap();
        let input_claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        assert_eq!(
            offloaded.output_claims(&input_claims).unwrap(),
            cpu.output_claims(&input_claims).unwrap()
        );
    }
}
