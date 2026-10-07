//! Optimized stage-1 Spartan outer kernels with the reference kernels' exact
//! wire behavior:
//!
//! - **Typed small-scalar row evaluation**: the 19 eq-conditional constraint
//!   rows are evaluated per cycle as integers (`i64` guards, `i128` B values,
//!   the second-group values split as `hi·2^64 + lo`) straight off a
//!   typed witness bundle — the ordinary R1CS input tables are never
//!   materialized as field vectors (`R1CSEval::{eval_az,eval_bz}_*_group`).
//! - **Univariate skip over the centered integer domain**: the first-round
//!   polynomial needs only `DOMAIN − 1` extended-node evaluations (in-domain
//!   nodes vanish). Finite differences extend the row values exactly.
//!   Ordinary cycles use integer additions plus one field × `i128` fmadd
//!   per `(cycle, stream, node)` (two for the split second stream). Active
//!   field-inline cycles extend the B values in the field.
//! - **Unreduced accumulation**: field × wide-integer products accumulate
//!   through `jolt-field`'s specialized accumulators and reduce once per block
//!   (`FullAccumS`/`SmallAccumU`/`WideAccumS` + `barrett_reduce`).
//! - **Split-eq (Gruen/Dao-Thaler) factoring**: `eq(τ_low, ·)` is held as an
//!   `E_out ⊗ E_in` tensor and a per-round linear factor
//!   ([`GruenSplitEqPolynomial`]); round polynomials come from the two
//!   endpoints `q(0)`, `q(∞)` and the running claim (`gruen_poly_deg_3`),
//!   never from four full-domain evaluation sweeps.
//! - **Fused round-0 materialization**: the bound `Az`/`Bz` tables over the
//!   joint `(cycle ‖ stream)` domain and the first round's endpoints are
//!   produced by one pass over the typed rows
//!   (`OuterLinearStage::fused_materialise_polynomials_round_zero`).
//! - **In-place binding**: `Az`/`Bz` bind without swap buffers; the ordinary input
//!   tables are never bound.
//! - **Post-hoc opening evaluation**: the ordinary produced opening claims come
//!   from one final eq-weighted walk over the typed rows
//!   (`R1CSEval::compute_claimed_inputs`), not from binding all input polynomials
//!   through every round.
//!
//! Byte parity with the reference kernels holds because every step computes
//! the same field values by exact integer/field algebra (the integer
//! extension equals the field Lagrange extension at integer nodes; ring
//! homomorphism does the rest), and the wire assembly reuses the
//! reference's own `jolt-poly` interpolation path.

#[cfg(feature = "field-inline")]
use core::cmp::Ordering;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedOpeningId;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::spartan::outer_output_openings as field_outer_output_openings;
use jolt_verifier::stages::relations::OpeningIdOf;
use std::collections::BTreeMap;
use std::ops::{Add, Sub};

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::r1cs::field_constraints::limb_radix;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::r1cs::field_constraints::{
    ROW_ADVICE_LIMB, ROW_ASSERT_EQ, ROW_ASSERT_ZERO, ROW_FADD, ROW_FINV, ROW_FMUL, ROW_FSUB,
    ROW_LOAD_ACCUMULATE_FROM_MEMORY, ROW_LOAD_ACCUMULATE_FROM_REGISTER, ROW_LOAD_IMM,
};
use jolt_claims::protocols::composed::r1cs::rv64::NUM_EQ_CONSTRAINTS as RV64_NUM_EQ_CONSTRAINTS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::r1cs::{
    SPARTAN_OUTER_FIRST_GROUP_ROWS, SPARTAN_OUTER_SECOND_GROUP_ROWS,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT;
use jolt_claims::protocols::jolt::geometry::spartan::{
    outer_opening, SpartanOuterDimensions, SPARTAN_OUTER_R1CS_INPUTS,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, JoltPolynomialId, SpartanOuterPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::{S128, S256, S64};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
// The COMPOSED R1CS shapes (feature-aware): identical to the rv64-only constants
// without field-inline, the field-inline-extended row/column composition under
// `field-inline` — the same sources the reference kernel folds with.
use jolt_claims::protocols::composed::r1cs::{
    spartan_outer_constraints, spartan_outer_opening_columns, spartan_outer_row_weights,
    SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
};
use jolt_riscv::{CircuitFlags, InstructionFlags, JoltTraceRow as TraceRow};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
#[cfg(feature = "field-inline")]
use jolt_witness::field_inline::FieldInlineSpartanRow;
use jolt_witness::witnesses::{
    lookup_values, Imm, LeftInstructionInput, LeftLookupOperand, LookupOutput,
    NextIsFirstInSequence, NextIsVirtual, NextPc, NextUnexpandedPc, OpFlag, Pc, Product,
    RamAddress, RamReadValue, RamWriteValue, RdWriteValue, RightInstructionInput,
    RightLookupOperand, Rs1Value, Rs2Value, ShouldBranch, ShouldJump, UnexpandedPc, WitnessEnv,
};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "field-inline")]
use super::support::map_reduce_chunks;
use super::support::{
    pin_derived_term_if_derived, try_par_sum_vecs, BundleAccess, BundleStore, GruenRoundMessage,
    RoundChallenges,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
const SECOND_GROUP_LEN: usize = SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const EXTENDED_NODE_COUNT: usize = DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);
/// The rv64 prefixes of the composed stream groups
/// (`SPARTAN_OUTER_{FIRST,SECOND}_GROUP_ROWS` order): field-inline rows append behind
/// them under `field-inline`; the complete order lives in those row arrays.
const RV64_FIRST_GROUP_LEN: usize = RV64_NUM_EQ_CONSTRAINTS.div_ceil(2);
const RV64_SECOND_GROUP_LEN: usize = RV64_NUM_EQ_CONSTRAINTS / 2;

#[cfg(feature = "field-inline")]
const _: () = {
    let first = [
        ROW_FADD,
        ROW_FSUB,
        ROW_FMUL,
        ROW_FINV,
        ROW_LOAD_ACCUMULATE_FROM_MEMORY,
    ];
    let second = [
        ROW_ASSERT_EQ,
        ROW_LOAD_ACCUMULATE_FROM_REGISTER,
        ROW_ASSERT_ZERO,
        ROW_LOAD_IMM,
        ROW_ADVICE_LIMB,
    ];
    let mut i = 0;
    while i < first.len() {
        assert!(
            SPARTAN_OUTER_FIRST_GROUP_ROWS[RV64_FIRST_GROUP_LEN + i]
                == RV64_NUM_EQ_CONSTRAINTS + first[i]
        );
        assert!(
            SPARTAN_OUTER_SECOND_GROUP_ROWS[RV64_SECOND_GROUP_LEN + i]
                == RV64_NUM_EQ_CONSTRAINTS + second[i]
        );
        i += 1;
    }
};

#[derive(Clone, Copy, Debug)]
struct SpartanOuterRow {
    left_instruction_input: LeftInstructionInput,
    right_instruction_input: RightInstructionInput,
    product: Product,
    should_branch: ShouldBranch,
    pc: Pc,
    unexpanded_pc: UnexpandedPc,
    imm: Imm,
    ram_address: RamAddress,
    rs1_value: Rs1Value,
    rs2_value: Rs2Value,
    rd_write_value: RdWriteValue,
    ram_read_value: RamReadValue,
    ram_write_value: RamWriteValue,
    left_lookup_operand: LeftLookupOperand,
    right_lookup_operand: RightLookupOperand,
    next_unexpanded_pc: NextUnexpandedPc,
    next_pc: NextPc,
    next_is_virtual: NextIsVirtual,
    next_is_first_in_sequence: NextIsFirstInSequence,
    lookup_output: LookupOutput,
    should_jump: ShouldJump,
    add_operands: OpFlag,
    subtract_operands: OpFlag,
    multiply_operands: OpFlag,
    load: OpFlag,
    store: OpFlag,
    jump: OpFlag,
    write_lookup_output_to_rd: OpFlag,
    virtual_instruction: OpFlag,
    assert_flag: OpFlag,
    do_not_update_unexpanded_pc: OpFlag,
    advice: OpFlag,
    is_compressed: OpFlag,
    is_first_in_sequence: OpFlag,
    is_last_in_sequence: OpFlag,
    #[cfg(feature = "field-inline")]
    field_add: OpFlag,
    #[cfg(feature = "field-inline")]
    field_sub: OpFlag,
    #[cfg(feature = "field-inline")]
    field_mul: OpFlag,
    #[cfg(feature = "field-inline")]
    field_inv: OpFlag,
    #[cfg(feature = "field-inline")]
    field_assert_eq: OpFlag,
    #[cfg(feature = "field-inline")]
    field_load_accumulate_from_register: OpFlag,
    #[cfg(feature = "field-inline")]
    field_assert_zero: OpFlag,
    #[cfg(feature = "field-inline")]
    field_load_imm: OpFlag,
    #[cfg(feature = "field-inline")]
    field_load_accumulate_from_memory: OpFlag,
    #[cfg(feature = "field-inline")]
    field_advice_limb: OpFlag,
}

impl WitnessBundle for SpartanOuterRow {
    #[inline]
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let circuit_flags = row.circuit_flags();
        let instruction_flags = row.instruction_flags();
        let (
            (left_instruction_input, right_instruction_input),
            (left_lookup_operand, right_lookup_operand),
            lookup_output,
        ) = lookup_values(row);
        let next_flags = next.map(TraceRow::circuit_flags);
        let flag = |flag| OpFlag(circuit_flags[flag]);

        Ok(Self {
            left_instruction_input: LeftInstructionInput(left_instruction_input),
            right_instruction_input: RightInstructionInput(right_instruction_input),
            product: Product(
                S64::from_u64(left_instruction_input)
                    .mul_trunc::<2, 2>(&S128::from_i128(right_instruction_input)),
            ),
            should_branch: ShouldBranch(
                instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
            ),
            pc: Pc(row.pc()),
            unexpanded_pc: UnexpandedPc(row.unexpanded_pc()),
            imm: Imm(row.imm()),
            ram_address: RamAddress(row.ram_address()),
            rs1_value: Rs1Value(row.rs1_value()),
            rs2_value: Rs2Value(row.rs2_value()),
            rd_write_value: RdWriteValue(row.rd_write_value()),
            ram_read_value: RamReadValue(row.ram_read_value()),
            ram_write_value: RamWriteValue(row.ram_write_value()),
            left_lookup_operand: LeftLookupOperand(left_lookup_operand),
            right_lookup_operand: RightLookupOperand(right_lookup_operand),
            next_unexpanded_pc: NextUnexpandedPc(next.map_or(0, TraceRow::unexpanded_pc)),
            next_pc: NextPc(next.map_or(0, TraceRow::pc)),
            next_is_virtual: NextIsVirtual(
                next_flags.is_some_and(|flags| flags[CircuitFlags::VirtualInstruction]),
            ),
            next_is_first_in_sequence: NextIsFirstInSequence(
                next_flags.is_some_and(|flags| flags[CircuitFlags::IsFirstInSequence]),
            ),
            lookup_output: LookupOutput(lookup_output),
            should_jump: ShouldJump(
                circuit_flags[CircuitFlags::Jump] && !next.is_some_and(|row| row.is_noop()),
            ),
            add_operands: flag(CircuitFlags::AddOperands),
            subtract_operands: flag(CircuitFlags::SubtractOperands),
            multiply_operands: flag(CircuitFlags::MultiplyOperands),
            load: flag(CircuitFlags::Load),
            store: flag(CircuitFlags::Store),
            jump: flag(CircuitFlags::Jump),
            write_lookup_output_to_rd: flag(CircuitFlags::WriteLookupOutputToRD),
            virtual_instruction: flag(CircuitFlags::VirtualInstruction),
            assert_flag: flag(CircuitFlags::Assert),
            do_not_update_unexpanded_pc: flag(CircuitFlags::DoNotUpdateUnexpandedPC),
            advice: flag(CircuitFlags::Advice),
            is_compressed: flag(CircuitFlags::IsCompressed),
            is_first_in_sequence: flag(CircuitFlags::IsFirstInSequence),
            is_last_in_sequence: flag(CircuitFlags::IsLastInSequence),
            #[cfg(feature = "field-inline")]
            field_add: flag(CircuitFlags::FieldAdd),
            #[cfg(feature = "field-inline")]
            field_sub: flag(CircuitFlags::FieldSub),
            #[cfg(feature = "field-inline")]
            field_mul: flag(CircuitFlags::FieldMul),
            #[cfg(feature = "field-inline")]
            field_inv: flag(CircuitFlags::FieldInv),
            #[cfg(feature = "field-inline")]
            field_assert_eq: flag(CircuitFlags::FieldAssertEq),
            #[cfg(feature = "field-inline")]
            field_load_accumulate_from_register: flag(
                CircuitFlags::FieldLoadAccumulateFromRegister,
            ),
            #[cfg(feature = "field-inline")]
            field_assert_zero: flag(CircuitFlags::FieldAssertZero),
            #[cfg(feature = "field-inline")]
            field_load_imm: flag(CircuitFlags::FieldLoadImm),
            #[cfg(feature = "field-inline")]
            field_load_accumulate_from_memory: flag(CircuitFlags::FieldLoadAccumulateFromMemory),
            #[cfg(feature = "field-inline")]
            field_advice_limb: flag(CircuitFlags::FieldAdviceLimb),
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .map(JoltPolynomialId::Virtual)
            .collect()
    }
}

/// One cycle's integer values of the composed eq-conditional rows, split into
/// the two uni-skip stream groups. A-side guards satisfy `|a| ≤ 3`, with at
/// most two per group above 1 in magnitude (one ≤ 2, one ≤ 3); first-group B
/// values satisfy `|b| ≤ 2^64`. Second-group B magnitudes are below 2^129 (the
/// `RightLookupOperand`/`Product`/`Imm`-bearing rows), so each is carried as
/// `hi·2^64 + lo` with `|lo|, |hi| < 2^65`. Under `field-inline` the arrays
/// span the composed groups including field-inline rows with inactive
/// columns; active field-inline cycles use [`FieldGroupValues`] instead.
/// These bounds follow from the bundle's scalar types, without flag exclusivity:
/// first-group exceptions are `Add + Sub + Mul` and its complement; second-group
/// exceptions are `Load + Store` and `1 − Add − Sub − Mul − Advice`. Appended
/// guards are booleans; their integer B passengers are `−1`, negated words,
/// and the two halves of `−Imm`, including `Imm = i128::MIN`.
struct RowGroupValues {
    a_first: [i64; DOMAIN],
    a_second: [i64; SECOND_GROUP_LEN],
    b_first: [i128; DOMAIN],
    b_second_lo: [i128; SECOND_GROUP_LEN],
    b_second_hi: [i128; SECOND_GROUP_LEN],
}

/// One active field-inline cycle's composed group values, in field form: the
/// field-inline magnitudes are full field elements, so the integer pipeline cannot
/// carry them. Its work is proportional to active cycles, which may occupy
/// the entire trace.
#[cfg(feature = "field-inline")]
struct FieldGroupValues<F> {
    integer: RowGroupValues,
    b_first: [F; DOMAIN - RV64_FIRST_GROUP_LEN],
    b_second: [F; SECOND_GROUP_LEN - RV64_SECOND_GROUP_LEN],
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> FieldGroupValues<F> {
    fn extended_products(&self) -> [(F, F); EXTENDED_NODE_COUNT] {
        let az_first = extend(&self.integer.a_first);
        let az_second = extend(&self.integer.a_second);
        let mut bz_first = self.integer.b_first.map(F::from_i128);
        let mut bz_second: [F; SECOND_GROUP_LEN] = std::array::from_fn(|i| {
            F::from_i128(self.integer.b_second_lo[i])
                + F::from_i128(self.integer.b_second_hi[i]).mul_pow_2(64)
        });
        for (b, correction) in bz_first[RV64_FIRST_GROUP_LEN..]
            .iter_mut()
            .zip(self.b_first)
        {
            *b += correction;
        }
        for (b, correction) in bz_second[RV64_SECOND_GROUP_LEN..]
            .iter_mut()
            .zip(self.b_second)
        {
            *b += correction;
        }
        let bz_first = extend(&bz_first);
        let bz_second = extend(&bz_second);
        std::array::from_fn(|slot| {
            (
                F::from_i64(az_first[slot]) * bz_first[slot],
                F::from_i64(az_second[slot]) * bz_second[slot],
            )
        })
    }

    fn fold_first(&self, weights: &[F]) -> (F, F) {
        let (a, b) = fold_group(
            weights,
            &self.integer.a_first,
            &self.integer.b_first,
            &[0; DOMAIN],
        );
        let correction: F = weights[RV64_FIRST_GROUP_LEN..]
            .iter()
            .zip(self.b_first)
            .map(|(c, b)| *c * b)
            .sum();
        (a, b + correction)
    }

    fn fold_second(&self, weights: &[F]) -> (F, F) {
        let (a, b) = fold_group(
            weights,
            &self.integer.a_second,
            &self.integer.b_second_lo,
            &self.integer.b_second_hi,
        );
        let correction: F = weights[RV64_SECOND_GROUP_LEN..SECOND_GROUP_LEN]
            .iter()
            .zip(self.b_second)
            .map(|(c, b)| *c * b)
            .sum();
        (a, b + correction)
    }
}

/// Transfers the extracted field rows from the outer kernel to the product kernel.
#[cfg(feature = "field-inline")]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) struct FieldSpartanCarry<F: JoltField>(
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    pub(crate) Vec<(usize, FieldInlineSpartanRow<F>)>,
);

#[cfg(feature = "field-inline")]
pub(crate) struct FieldInlineRowCursor<'a, F> {
    rows: &'a [(usize, FieldInlineSpartanRow<F>)],
    next: usize,
}

#[cfg(feature = "field-inline")]
impl<'a, F> FieldInlineRowCursor<'a, F> {
    /// A cursor positioned at the first row with cycle ≥ `start` — each
    /// parallel block seeks independently.
    pub(crate) fn seek(rows: &'a [(usize, FieldInlineSpartanRow<F>)], start: usize) -> Self {
        Self {
            rows,
            next: rows.partition_point(|&(cycle, _)| cycle < start),
        }
    }

    /// The field-inline row at cycle `t`, if any; `t` must be non-decreasing across
    /// calls on one cursor.
    pub(crate) fn advance(&mut self, t: usize) -> Option<&'a FieldInlineSpartanRow<F>> {
        while let Some(&(cycle, ref row)) = self.rows.get(self.next) {
            match cycle.cmp(&t) {
                Ordering::Less => self.next += 1,
                Ordering::Equal => {
                    self.next += 1;
                    return Some(row);
                }
                Ordering::Greater => return None,
            }
        }
        None
    }
}

impl SpartanOuterRow {
    /// Evaluate the ordinary constraint rows and inactive field rows with exact integer
    /// arithmetic. Formulas transcribe `jolt-claims`'s `rv64_eq_constraint_rows`
    /// verbatim (matrix semantics, not satisfied-witness shortcuts), grouped as
    /// `SPARTAN_OUTER_{FIRST,SECOND}_GROUP_ROWS` orders them.
    fn group_values(&self) -> RowGroupValues {
        let flag = |value: bool| i64::from(value);
        let load = flag(self.load.0);
        let store = flag(self.store.0);
        let add = flag(self.add_operands.0);
        let sub = flag(self.subtract_operands.0);
        let mul = flag(self.multiply_operands.0);
        let jump = flag(self.jump.0);
        let should_branch = flag(self.should_branch.0);

        // Rows 1, 2, 3, 4, 5, 6, 11, 14, 17, 18.
        let rv64_a_first = [
            1 - load - store,
            load,
            load,
            store,
            add + sub + mul,
            1 - add - sub - mul,
            flag(self.assert_flag.0),
            flag(self.should_jump.0),
            flag(self.virtual_instruction.0) - flag(self.is_last_in_sequence.0),
            flag(self.next_is_virtual.0) - flag(self.next_is_first_in_sequence.0),
        ];
        // Rows 0, 7, 8, 9, 10, 12, 13, 15, 16.
        let rv64_a_second = [
            load + store,
            add,
            sub,
            mul,
            1 - add - sub - mul - flag(self.advice.0),
            flag(self.write_lookup_output_to_rd.0),
            jump,
            should_branch,
            1 - should_branch - jump,
        ];

        let word = |value: u64| i128::from(value);
        let diff = |a: u64, b: u64| word(a) - word(b);
        let rv64_b_first = [
            word(self.ram_address.0),
            diff(self.ram_read_value.0, self.ram_write_value.0),
            diff(self.ram_read_value.0, self.rd_write_value.0),
            diff(self.rs2_value.0, self.ram_write_value.0),
            word(self.left_lookup_operand.0),
            diff(self.left_lookup_operand.0, self.left_instruction_input.0),
            word(self.lookup_output.0) - 1,
            diff(self.next_unexpanded_pc.0, self.lookup_output.0),
            diff(self.next_pc.0, self.pc.0) - 1,
            i128::from(1 - flag(self.do_not_update_unexpanded_pc.0)),
        ];

        let halves = |value: i128| (value >> 64, word(value as u64));
        let right_lookup = self.right_lookup_operand.0;
        let (right_lookup_hi, right_lookup_lo) =
            (word((right_lookup >> 64) as u64), word(right_lookup as u64));
        let (right_input_hi, right_input_lo) = halves(self.right_instruction_input.0);
        let (imm_hi, imm_lo) = halves(self.imm.0);
        let [product_lo, product_hi] = self.product.0.magnitude_limbs();
        let product_sign: i128 = if self.product.0.is_positive { 1 } else { -1 };
        let (product_hi, product_lo) = (
            product_sign * word(product_hi),
            product_sign * word(product_lo),
        );
        let left_input = word(self.left_instruction_input.0);
        let compressed = 2 * i128::from(self.is_compressed.0);
        let pc_step = diff(self.next_unexpanded_pc.0, self.unexpanded_pc.0);
        let rv64_b_second_lo = [
            diff(self.ram_address.0, self.rs1_value.0) - imm_lo,
            right_lookup_lo - left_input - right_input_lo,
            right_lookup_lo - left_input + right_input_lo,
            right_lookup_lo - product_lo,
            right_lookup_lo - right_input_lo,
            diff(self.rd_write_value.0, self.lookup_output.0),
            diff(self.rd_write_value.0, self.unexpanded_pc.0) - 4 + compressed,
            pc_step - imm_lo,
            pc_step - 4 + 4 * i128::from(self.do_not_update_unexpanded_pc.0) + compressed,
        ];
        let rv64_b_second_hi = [
            -imm_hi,
            right_lookup_hi - right_input_hi,
            right_lookup_hi + right_input_hi - 1,
            right_lookup_hi - product_hi,
            right_lookup_hi - right_input_hi,
            0,
            0,
            -imm_hi,
            0,
        ];

        let mut values = RowGroupValues {
            a_first: [0; DOMAIN],
            a_second: [0; SECOND_GROUP_LEN],
            b_first: [0; DOMAIN],
            b_second_lo: [0; SECOND_GROUP_LEN],
            b_second_hi: [0; SECOND_GROUP_LEN],
        };
        values.a_first[..RV64_FIRST_GROUP_LEN].copy_from_slice(&rv64_a_first);
        values.a_second[..RV64_SECOND_GROUP_LEN].copy_from_slice(&rv64_a_second);
        values.b_first[..RV64_FIRST_GROUP_LEN].copy_from_slice(&rv64_b_first);
        values.b_second_lo[..RV64_SECOND_GROUP_LEN].copy_from_slice(&rv64_b_second_lo);
        values.b_second_hi[..RV64_SECOND_GROUP_LEN].copy_from_slice(&rv64_b_second_hi);

        // Field rows with zero field values still use their ordinary op flags.
        // Nonzero field magnitudes are supplied by `field_group_values` below.
        #[cfg(feature = "field-inline")]
        {
            values.a_first[RV64_FIRST_GROUP_LEN..].copy_from_slice(&[
                flag(self.field_add.0),
                flag(self.field_sub.0),
                flag(self.field_mul.0),
                flag(self.field_inv.0),
                flag(self.field_load_accumulate_from_memory.0),
            ]);
            values.a_second[RV64_SECOND_GROUP_LEN..].copy_from_slice(&[
                flag(self.field_assert_eq.0),
                flag(self.field_load_accumulate_from_register.0),
                flag(self.field_assert_zero.0),
                flag(self.field_load_imm.0),
                flag(self.field_advice_limb.0),
            ]);
            let rd_write_value = word(self.rd_write_value.0);
            values.b_first[RV64_FIRST_GROUP_LEN + 3] = -1;
            values.b_first[RV64_FIRST_GROUP_LEN + 4] = -rd_write_value;
            values.b_second_lo[RV64_SECOND_GROUP_LEN + 1] = -word(self.rs1_value.0);
            values.b_second_lo[RV64_SECOND_GROUP_LEN + 3] = -imm_lo;
            values.b_second_hi[RV64_SECOND_GROUP_LEN + 3] = -imm_hi;
            values.b_second_lo[RV64_SECOND_GROUP_LEN + 4] = -rd_write_value;
        }

        values
    }

    /// The composed group values of one active field-inline cycle, in field form: the
    /// rv64 guards/magnitudes promoted plus the field-inline rows' native field values
    /// (`jolt-claims`'s `field_eq_constraint_rows` transcribed at the composed group
    /// positions). Exact — the integer pipeline and this one compute the same field
    /// elements, so routing a cycle either way is wire-invisible; the integer path
    /// simply cannot represent an active cycle's field magnitudes.
    #[cfg(feature = "field-inline")]
    fn field_group_values<F: JoltField>(
        &self,
        field_row: &FieldInlineSpartanRow<F>,
    ) -> FieldGroupValues<F> {
        // The integer rows already include constants and RV64 passengers of the
        // appended constraints. Only their field-value corrections belong here.
        FieldGroupValues {
            integer: self.group_values(),
            b_first: [
                field_row.rs1_value + field_row.rs2_value - field_row.rd_value,
                field_row.rs1_value - field_row.rs2_value - field_row.rd_value,
                field_row.product - field_row.rd_value,
                field_row.inv_product,
                field_row.rd_value - limb_radix::<F>() * field_row.rs1_value,
            ],
            b_second: [
                field_row.rs1_value - field_row.rs2_value,
                field_row.rd_value - limb_radix::<F>() * field_row.rs1_value,
                field_row.rs1_value,
                field_row.rd_value,
                field_row.rs1_value - limb_radix::<F>() * field_row.rd_value,
            ],
        }
    }
}

const LEFT_NODE_COUNT: usize = (DOMAIN_START - EXTENDED_START) as usize;

/// Each extended node's position in the `2·DOMAIN − 1` window, in
/// [`extend`]'s slot order.
const EXTENDED_POSITIONS: [usize; EXTENDED_NODE_COUNT] = {
    let mut positions = [0; EXTENDED_NODE_COUNT];
    let mut slot = 0;
    while slot < EXTENDED_NODE_COUNT {
        positions[slot] = if slot < LEFT_NODE_COUNT {
            slot
        } else {
            slot + DOMAIN
        };
        slot += 1;
    }
    positions
};

/// `(Λ, max |L_i(node)|)` over the extended nodes, from the Lagrange product
/// formula. The assertions below pin the `i128` exactness argument of
/// [`extend`] and [`NodeProducts`] at compile time for the selected domain.
const EXTENSION_GAIN: (u128, u128) = {
    let (mut gain, mut max_coefficient) = (0u128, 0u128);
    let mut slot = 0;
    while slot < EXTENDED_NODE_COUNT {
        let node = EXTENDED_START + EXTENDED_POSITIONS[slot] as i64;
        let mut sum = 0u128;
        let mut i = 0;
        while i < DOMAIN {
            let (mut numerator, mut denominator) = (1i128, 1i128);
            let mut j = 0;
            while j < DOMAIN {
                if j != i {
                    numerator *= (node - DOMAIN_START - j as i64) as i128;
                    denominator *= i as i128 - j as i128;
                }
                j += 1;
            }
            assert!(numerator % denominator == 0);
            let coefficient = (numerator / denominator).unsigned_abs();
            sum += coefficient;
            if coefficient > max_coefficient {
                max_coefficient = coefficient;
            }
            i += 1;
        }
        if sum > gain {
            gain = sum;
        }
        slot += 1;
    }
    (gain, max_coefficient)
};

const _: () = {
    let (gain, max_coefficient) = EXTENSION_GAIN;
    let difference_growth = 1u128 << (DOMAIN - 1);
    let az = gain + 3 * max_coefficient;
    let bz = gain * (1 << 65);
    assert!(az < 1 << 63);
    assert!(3 * gain * difference_growth < 1 << 63);
    assert!(bz * difference_growth < 1 << 127);
    assert!(az * bz < 1 << 126);
};

/// Exact extension of the degree-`< DOMAIN` interpolant of `values`
/// (on the base window's consecutive nodes; missing top values are zero) to
/// every extended node, in [`EXTENDED_POSITIONS`] order. Finite differences
/// replace the Lagrange dot products: the base window's difference table
/// yields the forward diagonal at its first node and the backward diagonal
/// at its last, and since `Δ^DOMAIN` vanishes, each step outward costs
/// `DOMAIN − 1` additions. For integer inputs, every intermediate is a `k`-th
/// difference over the extended window, bounded by `2^(DOMAIN−1) · max|P|`,
/// and `max|P| ≤ Λ · max|value|` with `Λ = max_node Σ_i |L_i(node)|`:
/// `Λ < 2^20` on the 10-node rv64 domain, `Λ < 2^30` on `field-inline`'s 15.
/// Thus guard intermediates are below `3·2^44 < 2^46`, within `i64`.
fn extend<T>(values: &[T]) -> [T; EXTENDED_NODE_COUNT]
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
    let mut row = [T::default(); DOMAIN];
    row[..values.len()].copy_from_slice(values);
    let mut forward = row;
    let mut backward = row;
    backward[0] = row[DOMAIN - 1];
    for order in 1..DOMAIN {
        for j in 0..DOMAIN - order {
            row[j] = row[j + 1] - row[j];
        }
        forward[order] = row[0];
        backward[order] = row[DOMAIN - 1 - order];
    }
    let mut extended = [T::default(); EXTENDED_NODE_COUNT];
    let (left, right) = extended.split_at_mut(LEFT_NODE_COUNT);
    for value in left.iter_mut().rev() {
        for order in (0..DOMAIN - 1).rev() {
            forward[order] = forward[order] - forward[order + 1];
        }
        *value = forward[0];
    }
    for value in right {
        for order in (0..DOMAIN - 1).rev() {
            backward[order] = backward[order] + backward[order + 1];
        }
        *value = backward[0];
    }
    extended
}

/// One extended node's `Az·Bz` for one cycle: the first stream's product and
/// the second stream's as `hi·2^64 + lo`. With the [`RowGroupValues`] ranges,
/// `|az| ≤ Λ + 3·max|L_i|` and every extended B value (or half) is below
/// `Λ·2^65`. On the rv64 domain that is `|az| < 2^20`, B below `2^85`
/// (difference-table intermediates below `2^94`) and products below `2^105`;
/// on `field-inline`'s domain `|az| < 2^31`, B below `2^95` (intermediates
/// below `2^109`) and products below `2^126` — exact `i128` arithmetic
/// throughout, pinned by the [`EXTENSION_GAIN`] assertions.
#[derive(Clone, Copy)]
struct NodeProducts {
    first: i128,
    second_lo: i128,
    second_hi: i128,
}

fn extended_products(values: &RowGroupValues) -> [NodeProducts; EXTENDED_NODE_COUNT] {
    let az_first = extend(&values.a_first);
    let az_second = extend(&values.a_second);
    let bz_first = extend(&values.b_first);
    let bz_second_lo = extend(&values.b_second_lo);
    let bz_second_hi = extend(&values.b_second_hi);
    std::array::from_fn(|slot| NodeProducts {
        first: i128::from(az_first[slot]) * bz_first[slot],
        second_lo: i128::from(az_second[slot]) * bz_second_lo[slot],
        second_hi: i128::from(az_second[slot]) * bz_second_hi[slot],
    })
}

fn wide_value(top: i128, low: u64) -> S256 {
    let is_positive = top >= 0;
    let (top, low) = if is_positive {
        (top.unsigned_abs(), low)
    } else {
        // −(top·2^64 + low) = (|top| − [low ≠ 0])·2^64 + (−low mod 2^64).
        (
            top.unsigned_abs() - u128::from(low != 0),
            low.wrapping_neg(),
        )
    };
    S256::new([low, top as u64, (top >> 64) as u64, 0], is_positive)
}

fn fold_group<F: JoltField>(weights: &[F], guards: &[i64], lo: &[i128], hi: &[i128]) -> (F, F) {
    let mut az = <F as WithAccumulator>::SmallScalarAccumulator::default();
    let mut bz = <F as WithAccumulator>::SignedProductAccumulator::default();
    for (((&weight, &guard), &lo), &hi) in weights.iter().zip(guards).zip(lo).zip(hi) {
        az.fmadd_i64(weight, guard);
        let (top, low) = (hi + (lo >> 64), lo as u64);
        match top {
            0 => bz.fmadd_signed_u64(weight, low, true),
            -1 if low != 0 => bz.fmadd_signed_u64(weight, low.wrapping_neg(), false),
            _ => bz.fmadd_s256(weight, &wide_value(top, low)),
        }
    }
    (az.reduce(), bz.reduce())
}

/// The uni-skip carry: everything the uni-skip front computes that the
/// remainder slot reclaims — the typed-row store (reused for
/// materialization and the final opening walk), the stage challenge vector,
/// and the extended-node evaluations of `t1`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SpartanOuterCarry<F: JoltField> {
    log_t: usize,
    tau: Vec<F>,
    /// Typed-row store: slice-backed witnesses stay unmaterialized (the
    /// ~176 B × T row vector is the prover's peak allocation at large scale).
    rows: BundleStore<SpartanOuterRow>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    /// All `2·DOMAIN − 1` node values of `t1`; in-domain nodes stay zero (a
    /// satisfying witness vanishes there), matching the reference layout.
    t1_values: Vec<F>,
}

pub struct OptimizedOuterUniskip;

impl OptimizedOuterUniskip {
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: Vec<SpartanOuterRow>,
        #[cfg(feature = "field-inline")] field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer row count disagrees with log_t",
            });
        }
        Self::prepare_from_store(
            session,
            log_t,
            tau,
            BundleStore::Retained(rows),
            #[cfg(feature = "field-inline")]
            field_rows,
        )
    }

    fn prepare_from_store<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: BundleStore<SpartanOuterRow>,
        #[cfg(feature = "field-inline")] field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer tau must carry log_t + 2 challenges",
            });
        }
        let (tau_low, _) = tau.split_at(log_t + 1);
        let t1_values = Self::extended_t1_values(
            &rows.access(),
            tau_low,
            #[cfg(feature = "field-inline")]
            &field_rows,
        )?;
        session.park(SpartanOuterCarry {
            log_t,
            tau: tau.to_vec(),
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            t1_values,
        });
        Ok(())
    }

    fn extended_t1_values<F: JoltField>(
        rows: &BundleAccess<'_, SpartanOuterRow>,
        tau_low: &[F],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<F>)],
    ) -> Result<Vec<F>, WitnessError> {
        let split = tau_low.len() / 2;
        let (out_point, in_point) = tau_low.split_at(split);
        let e_out = EqPolynomial::<F>::evals(out_point, None);
        let e_in = EqPolynomial::<F>::evals(in_point, None);
        // `in_point` always covers the stream bit (τ_low's last entry), so every
        // (cycle, stream) pair sits inside one `x_out` block.
        let pairs_per_block = e_in.len() / 2;
        let two_pow_64 = F::from_u128(1 << 64);

        let extended = try_par_sum_vecs(e_out.len(), EXTENDED_NODE_COUNT, |x_out| {
            // At most e_in.len() fmadds per accumulator: on 64-bit hosts the
            // split gives ≤ 2^32 terms. BN254 slots grow by < 2^66 per i128
            // term; fp128 slots by < 2^65, leaving ample carry headroom.
            let mut sums = [(
                <F as WithAccumulator>::SignedProductAccumulator::default(),
                <F as WithAccumulator>::SignedProductAccumulator::default(),
            ); EXTENDED_NODE_COUNT];
            #[cfg(feature = "field-inline")]
            let mut field_sums = [F::zero(); EXTENDED_NODE_COUNT];
            #[cfg(feature = "field-inline")]
            let mut field_cursor = FieldInlineRowCursor::seek(field_rows, x_out * pairs_per_block);
            for pair in 0..pairs_per_block {
                let t = x_out * pairs_per_block + pair;
                let row = rows.row(t)?;
                let (e_first, e_second) = (e_in[2 * pair], e_in[2 * pair + 1]);
                #[cfg(feature = "field-inline")]
                if let Some(field_row) = field_cursor.advance(t) {
                    let values = row.field_group_values(field_row);
                    let products = values.extended_products();
                    for (sum, (first, second)) in field_sums.iter_mut().zip(&products) {
                        *sum += e_first * *first + e_second * *second;
                    }
                    continue;
                }
                let products = extended_products(&row.group_values());
                for ((low, high), product) in sums.iter_mut().zip(products) {
                    low.fmadd_i128(e_first, product.first);
                    low.fmadd_i128(e_second, product.second_lo);
                    high.fmadd_i128(e_second, product.second_hi);
                }
            }
            #[cfg(feature = "field-inline")]
            return Ok(sums
                .into_iter()
                .zip(field_sums)
                .map(|((low, high), field_sum)| {
                    e_out[x_out] * (low.reduce() + two_pow_64 * high.reduce() + field_sum)
                })
                .collect());
            #[cfg(not(feature = "field-inline"))]
            Ok(sums
                .into_iter()
                .map(|(low, high)| e_out[x_out] * (low.reduce() + two_pow_64 * high.reduce()))
                .collect())
        })?;

        let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
        for (position, value) in EXTENDED_POSITIONS.into_iter().zip(extended) {
            t1_values[position] = value;
        }
        Ok(t1_values)
    }
}

impl<F: JoltField> UniskipKernel<F, OuterRemainder<F>> for OptimizedOuterUniskip {
    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let rows = BundleStore::resolve(witness, 1usize << log_t)?;
        #[cfg(feature = "field-inline")]
        let field_rows = witness
            .field_inline()
            .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                label: "composed Spartan outer field-inline oracle",
            }))?
            .field_inline_spartan_rows()?;
        Self::prepare_from_store(
            session,
            log_t,
            tau,
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
        )
    }

    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        _late_tau: &[F],
        _inputs: &(),
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let carry =
            session
                .state::<SpartanOuterCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason:
                        "the outer uni-skip slot parked no carry for the first-round polynomial",
                })?;
        let tau_high = carry.tau[carry.log_t + 1];
        let kernel_values = centered_lagrange_evals::<F>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &carry.t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

/// The stage-1 remainder slot: reclaims the uni-skip carry and builds the
/// linear-time round kernel.
pub struct OptimizedOuterRemainder;

impl<F: JoltField> PrepareKernel<F, OuterRemainder<F>> for OptimizedOuterRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        let carry =
            session
                .take::<SpartanOuterCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the outer uni-skip slot parked no carry for the remainder member",
                })?;
        Ok(Box::new(OuterRemainderKernel::prepare(carry, &inputs)?))
    }
}

/// The `Az`/`Bz` linear forms folded at both stream values — the closed forms
/// of the relation's derived leaves after the stream bind, kept for
/// [`SumcheckKernel::validate_derived_tables`].
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct DerivedWeights<F> {
    az_weights: [Vec<F>; 2],
    bz_weights: [Vec<F>; 2],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    az_constant: [F; 2],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    bz_constant: [F; 2],
}

/// The linear-time outer remainder rounds over the joint `(cycle ‖ stream)`
/// domain (stream = index LSB, bound `LowToHigh`).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct OuterRemainderKernel<F: JoltField> {
    az: Polynomial<F>,
    bz: Polynomial<F>,
    purged: bool,
    split_eq: GruenSplitEqPolynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pending_endpoints: Option<(F, F)>,
    challenges: RoundChallenges<F>,
    rows: BundleStore<SpartanOuterRow>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    opening_ids: Vec<JoltOpeningId>,
    derived: DerivedWeights<F>,
}
impl<F: JoltField> OuterRemainderKernel<F> {
    fn prepare(
        carry: SpartanOuterCarry<F>,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Self, KernelError<F>> {
        let SpartanOuterCarry {
            log_t,
            tau,
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            ..
        } = carry;
        let rounds = inputs.relation.rounds();
        if rounds != log_t + 1 {
            return Err(KernelError::InvariantViolation {
                reason: "outer remainder rounds disagree with the uni-skip carry's log_t",
            });
        }
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let tau_high = tau[log_t + 1];
        let tau_low = &tau[..=log_t];
        let lagrange_r0 = centered_lagrange_evals::<F>(DOMAIN, uniskip_challenge)?;
        let kernel = centered_lagrange_kernel::<F>(DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            tau_low,
            BindingOrder::LowToHigh,
            Some(kernel),
        );

        let dimensions = SpartanOuterDimensions::rv64(log_t);
        let opening_ids: Vec<JoltOpeningId> = dimensions
            .variables()
            .iter()
            .map(|&variable| outer_opening(variable))
            .collect();
        let derived = Self::derived_weights(uniskip_challenge)?;

        let cycles = 1usize << log_t;
        let mut az: Vec<F> = unsafe_allocate_zero_vec(2 * cycles);
        let mut bz: Vec<F> = unsafe_allocate_zero_vec(2 * cycles);
        let e_out = split_eq.e_out_current();
        let e_in = split_eq.e_in_current();
        let in_len = e_in.len();
        let width = 2 * in_len;
        let access = rows.access();
        let lagrange = &lagrange_r0;
        #[cfg(feature = "field-inline")]
        let field_rows_ref: &[(usize, FieldInlineSpartanRow<F>)] = &field_rows;
        let block = |x_out: usize,
                     az_chunk: &mut [F],
                     bz_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            #[cfg(feature = "field-inline")]
            let mut field_cursor = FieldInlineRowCursor::seek(field_rows_ref, x_out * in_len);
            for x_in in 0..in_len {
                let t = x_out * in_len + x_in;
                let row = access.row(t)?;
                let integer_fold = || {
                    let values = row.group_values();
                    let (az_zero, bz_zero) =
                        fold_group(lagrange, &values.a_first, &values.b_first, &[0; DOMAIN]);
                    let (az_one, bz_one) = fold_group(
                        lagrange,
                        &values.a_second,
                        &values.b_second_lo,
                        &values.b_second_hi,
                    );
                    (az_zero, bz_zero, az_one, bz_one)
                };
                #[cfg(feature = "field-inline")]
                let (az_zero, bz_zero, az_one, bz_one) =
                    if let Some(field_row) = field_cursor.advance(t) {
                        let values = row.field_group_values(field_row);
                        let (az_zero, bz_zero) = values.fold_first(lagrange);
                        let (az_one, bz_one) = values.fold_second(lagrange);
                        (az_zero, bz_zero, az_one, bz_one)
                    } else {
                        integer_fold()
                    };
                #[cfg(not(feature = "field-inline"))]
                let (az_zero, bz_zero, az_one, bz_one) = integer_fold();
                az_chunk[2 * x_in] = az_zero;
                az_chunk[2 * x_in + 1] = az_one;
                bz_chunk[2 * x_in] = bz_zero;
                bz_chunk[2 * x_in + 1] = bz_one;
                let e = e_in[x_in];
                inner_zero += e * (az_zero * bz_zero);
                inner_infinity += e * ((az_one - az_zero) * (bz_one - bz_zero));
            }
            Ok((e_out[x_out] * inner_zero, e_out[x_out] * inner_infinity))
        };
        let add = |left: (F, F), right: (F, F)| (left.0 + right.0, left.1 + right.1);

        #[cfg(feature = "parallel")]
        let endpoints = az
            .par_chunks_mut(width)
            .zip(bz.par_chunks_mut(width))
            .enumerate()
            .map(|(x_out, (az_chunk, bz_chunk))| block(x_out, az_chunk, bz_chunk))
            .try_reduce(
                || (F::zero(), F::zero()),
                |left, right| Ok(add(left, right)),
            )?;
        #[cfg(not(feature = "parallel"))]
        let endpoints = {
            let mut folded = (F::zero(), F::zero());
            for (x_out, (az_chunk, bz_chunk)) in
                az.chunks_mut(width).zip(bz.chunks_mut(width)).enumerate()
            {
                folded = add(folded, block(x_out, az_chunk, bz_chunk)?);
            }
            folded
        };
        Ok(Self {
            az: Polynomial::new(az),
            bz: Polynomial::new(bz),
            purged: false,
            split_eq,
            pending_endpoints: Some(endpoints),
            challenges: RoundChallenges::new(rounds),
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            opening_ids,
            derived,
        })
    }

    /// Az/Bz column weights at both stream values over the composed opening-column
    /// selection, from the same `jolt-claims` sources the verifier's coefficient build
    /// uses (35 rv64 columns without field-inline; the non-contiguous 45 + 5 selection
    /// under `field-inline`).
    fn derived_weights(uniskip_challenge: F) -> Result<DerivedWeights<F>, KernelError<F>> {
        let matrices = spartan_outer_constraints::<F>();
        let columns: Vec<usize> = spartan_outer_opening_columns();
        let mut az_weights = [Vec::new(), Vec::new()];
        let mut bz_weights = [Vec::new(), Vec::new()];
        let mut az_constant = [F::zero(); 2];
        let mut bz_constant = [F::zero(); 2];
        for (index, stream) in [F::zero(), F::one()].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(uniskip_challenge, stream)?;
            let weighted = matrices.weighted_columns(&weights, &columns)?;
            az_weights[index] = weighted.a;
            bz_weights[index] = weighted.b;
            let constants = matrices.public_column_contributions(&weights, 0, F::one())?;
            az_constant[index] = constants.a;
            bz_constant[index] = constants.b;
        }
        Ok(DerivedWeights {
            az_weights,
            bz_weights,
            az_constant,
            bz_constant,
        })
    }

    fn bind(&mut self, challenge: F) {
        let shrunk = self.az.bind_low_to_high_in_place(challenge);
        let _ = self.bz.bind_low_to_high_in_place(challenge);
        if shrunk && !self.purged {
            self.purged = true;
            crate::mem::purge_retained_memory(self.challenges.total() - 1);
        }
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    fn cycle_weights(&self) -> Vec<F> {
        let reversed: Vec<F> = self.challenges.as_slice()[1..]
            .iter()
            .rev()
            .copied()
            .collect();
        let _span = tracing::info_span!("SpartanOuter::claimed_input_weights").entered();
        EqPolynomial::<F>::evals(&reversed, None)
    }

    #[tracing::instrument(skip_all, name = "SpartanOuter::claimed_inputs")]
    fn claimed_inputs(&self, weights: &[F]) -> Result<Vec<F>, WitnessError> {
        let cycles = weights.len();
        let access = self.rows.access();

        let block_size = 1usize << 12;
        let blocks = cycles.div_ceil(block_size);
        let block = |index: usize| -> Result<Vec<F>, WitnessError> {
            let start = index * block_size;
            let end = (start + block_size).min(cycles);
            let mut accumulator = ClaimAccumulator::<F>::default();
            for (t, &weight) in (start..end).zip(&weights[start..end]) {
                let row = access.row(t)?;
                accumulator.add_row(weight, &row);
            }
            Ok(accumulator.finish())
        };
        let claimed = {
            let _span = tracing::info_span!("SpartanOuter::claimed_input_walk").entered();
            try_par_sum_vecs(blocks, VARIABLE_COUNT, block)
        };
        claimed
    }

    /// The field-inline opening values at the bound cycle point: one eq-weighted walk
    /// over the sparse field-inline rows (columns in
    /// `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` order — the appendage order the
    /// composed remainder relation folds).
    #[cfg(feature = "field-inline")]
    fn field_claimed_inputs(&self, weights: &[F]) -> Vec<F> {
        map_reduce_chunks(
            self.field_rows.len(),
            1 << 12,
            |range| {
                let mut values = [F::zero(); FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT];
                for (cycle, row) in &self.field_rows[range] {
                    let weight = weights[*cycle];
                    for (value, column) in values.iter_mut().zip(row.columns()) {
                        *value += weight * column;
                    }
                }
                values
            },
            |a, b| std::array::from_fn(|i| a[i] + b[i]),
            || [F::zero(); FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT],
        )
        .to_vec()
    }
}

const VARIABLE_COUNT: usize = SPARTAN_OUTER_R1CS_INPUTS.len();

/// Which canonical inputs are boolean-valued: those stay on the small-scalar
/// accumulator, whose 5-limb (320-bit) window only has headroom when the
/// scalar sum stays tiny (Σ ≤ block size for 0/1 scalars). Word-valued
/// columns would overflow it — a full-range `u64` scalar puts a single term
/// at ~2^318 — so they go through the signed-product path instead.
const BOOLEAN_INPUT: [bool; VARIABLE_COUNT] = {
    let mut mask = [false; VARIABLE_COUNT];
    mask[3] = true; // ShouldBranch
    mask[17] = true; // NextIsVirtual
    mask[18] = true; // NextIsFirstInSequence
    mask[20] = true; // ShouldJump
    let mut flag = 21; // the circuit flags
    while flag < VARIABLE_COUNT {
        mask[flag] = true;
        flag += 1;
    }
    mask
};

struct ClaimAccumulator<F: JoltField> {
    small: Vec<<F as WithAccumulator>::SmallScalarAccumulator>,
    wide: Vec<<F as WithAccumulator>::SignedProductAccumulator>,
}

impl<F: JoltField> Default for ClaimAccumulator<F> {
    fn default() -> Self {
        Self {
            small: vec![Default::default(); VARIABLE_COUNT],
            wide: vec![Default::default(); VARIABLE_COUNT],
        }
    }
}

impl<F: JoltField> ClaimAccumulator<F> {
    fn add_row(&mut self, weight: F, row: &SpartanOuterRow) {
        let mut flag = |index: usize, value: bool| {
            self.small[index].fmadd_u64(weight, u64::from(value));
        };
        flag(3, row.should_branch.0);
        flag(17, row.next_is_virtual.0);
        flag(18, row.next_is_first_in_sequence.0);
        flag(20, row.should_jump.0);
        flag(21, row.add_operands.0);
        flag(22, row.subtract_operands.0);
        flag(23, row.multiply_operands.0);
        flag(24, row.load.0);
        flag(25, row.store.0);
        flag(26, row.jump.0);
        flag(27, row.write_lookup_output_to_rd.0);
        flag(28, row.virtual_instruction.0);
        flag(29, row.assert_flag.0);
        flag(30, row.do_not_update_unexpanded_pc.0);
        flag(31, row.advice.0);
        flag(32, row.is_compressed.0);
        flag(33, row.is_first_in_sequence.0);
        flag(34, row.is_last_in_sequence.0);
        #[cfg(feature = "field-inline")]
        {
            flag(35, row.field_add.0);
            flag(36, row.field_sub.0);
            flag(37, row.field_mul.0);
            flag(38, row.field_inv.0);
            flag(39, row.field_assert_eq.0);
            flag(40, row.field_load_accumulate_from_register.0);
            flag(41, row.field_assert_zero.0);
            flag(42, row.field_load_imm.0);
            flag(43, row.field_load_accumulate_from_memory.0);
            flag(44, row.field_advice_limb.0);
        }

        let mut word = |index: usize, magnitude: u128, is_positive: bool| {
            if let Ok(magnitude) = u64::try_from(magnitude) {
                self.wide[index].fmadd_signed_u64(weight, magnitude, is_positive);
            } else {
                self.wide[index].fmadd_s256(
                    weight,
                    &S256::new(
                        [magnitude as u64, (magnitude >> 64) as u64, 0, 0],
                        is_positive,
                    ),
                );
            }
        };
        word(0, u128::from(row.left_instruction_input.0), true);
        word(4, u128::from(row.pc.0), true);
        word(5, u128::from(row.unexpanded_pc.0), true);
        word(7, u128::from(row.ram_address.0), true);
        word(8, u128::from(row.rs1_value.0), true);
        word(9, u128::from(row.rs2_value.0), true);
        word(10, u128::from(row.rd_write_value.0), true);
        word(11, u128::from(row.ram_read_value.0), true);
        word(12, u128::from(row.ram_write_value.0), true);
        word(13, u128::from(row.left_lookup_operand.0), true);
        word(15, u128::from(row.next_unexpanded_pc.0), true);
        word(16, u128::from(row.next_pc.0), true);
        word(19, u128::from(row.lookup_output.0), true);

        let product_limbs = row.product.0.magnitude_limbs();
        word(
            1,
            row.right_instruction_input.0.unsigned_abs(),
            row.right_instruction_input.0 >= 0,
        );
        word(
            2,
            (u128::from(product_limbs[1]) << 64) | u128::from(product_limbs[0]),
            row.product.0.is_positive,
        );
        word(6, row.imm.0.unsigned_abs(), row.imm.0 >= 0);
        word(14, row.right_lookup_operand.0, true);
    }

    fn finish(self) -> Vec<F> {
        self.small
            .into_iter()
            .zip(self.wide)
            .zip(BOOLEAN_INPUT)
            .map(|((small, wide), boolean)| {
                if boolean {
                    small.reduce()
                } else {
                    wide.reduce()
                }
            })
            .collect()
    }
}

impl<F: JoltField> ProveRounds<F> for OuterRemainderKernel<F> {
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
        let (q_zero, q_infinity) = match self.pending_endpoints.take() {
            Some(endpoints) => endpoints,
            None => self.split_eq.product_endpoints(&self.az, &self.bz),
        };
        self.split_eq
            .checked_cubic(q_zero, q_infinity, previous_claim, round, || {
                self.split_eq.product_at_one(&self.az, &self.bz)
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OuterRemainderKernel<F> {
    type Relation = OuterRemainder<F>;

    #[cfg(feature = "field-inline")]
    fn park_residue(mut self: Box<Self>, session: &mut ProofSession) {
        session.park(FieldSpartanCarry(std::mem::take(&mut self.field_rows)));
        crate::mem::drop_in_background_thread(self);
    }

    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects composed claims and opening ids"
        )
    )]
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let weights = self.cycle_weights();
        let claimed =
            self.claimed_inputs(&weights)
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "outer opening walk re-extraction failed after the rounds",
                })?;
        let claims: BTreeMap<OpeningIdOf<F, Self::Relation>, F> = self
            .opening_ids
            .iter()
            .copied()
            .map(Into::into)
            .zip(claimed)
            .collect();
        #[cfg(feature = "field-inline")]
        let claims: BTreeMap<_, _> = claims
            .into_iter()
            .chain(
                field_outer_output_openings()
                    .into_iter()
                    .map(ComposedOpeningId::from)
                    .zip(self.field_claimed_inputs(&weights)),
            )
            .collect();
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        // The stream challenge binds the per-stream weight pairs; the split-eq
        // scalar is the fully bound TauKernel — both from the kernel's own
        // state, cross-checked against the verifier's coefficient build.
        let stream = self.challenges.as_slice()[0];
        let blend = |pair: [&F; 2]| *pair[0] + stream * (*pair[1] - *pair[0]);
        let variable_count = self.derived.az_weights[0].len();
        let ids = std::iter::once(SpartanOuterPublic::TauKernel)
            .chain((0..variable_count).map(SpartanOuterPublic::AzWeight))
            .chain((0..variable_count).map(SpartanOuterPublic::BzWeight))
            .chain([
                SpartanOuterPublic::AzConstant,
                SpartanOuterPublic::BzConstant,
            ]);
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let got = match public_id {
                SpartanOuterPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanOuterPublic::AzWeight(index) => blend([
                    &self.derived.az_weights[0][index],
                    &self.derived.az_weights[1][index],
                ]),
                SpartanOuterPublic::BzWeight(index) => blend([
                    &self.derived.bz_weights[0][index],
                    &self.derived.bz_weights[1][index],
                ]),
                SpartanOuterPublic::AzConstant => {
                    blend([&self.derived.az_constant[0], &self.derived.az_constant[1]])
                }
                SpartanOuterPublic::BzConstant => {
                    blend([&self.derived.bz_constant[0], &self.derived.bz_constant[1]])
                }
            };
            pin_derived_term_if_derived(
                relation,
                id,
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
    use crate::optimized::parity::ExceptionalEq;
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::{
        geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS, FieldInlinePolynomialId,
    };
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::jolt::JoltPolynomialId;
    use jolt_claims::NoChallenges;
    use jolt_field::signed::S128;
    use jolt_field::{Fr, Ring};
    use jolt_program::execution::OwnedTrace;
    use jolt_verifier::stages::stage1::outer_remainder::outer_remainder_input_values_from_uniskip_output;
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    #[cfg(feature = "field-inline")]
    use jolt_witness::FixedFieldInline;
    use jolt_witness::{BundleSource, JoltWitnessOracle};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape, TraceBackend};

    use super::*;
    use crate::reference::spartan_outer::{ReferenceOuterRemainder, SpartanOuterKernel};
    use crate::ReferenceBackend;

    fn variable_field_value(row: &SpartanOuterRow, index: usize) -> Fr {
        match index {
            0 => row.left_instruction_input.to_field(),
            1 => row.right_instruction_input.to_field(),
            2 => row.product.to_field(),
            3 => row.should_branch.to_field(),
            4 => row.pc.to_field(),
            5 => row.unexpanded_pc.to_field(),
            6 => row.imm.to_field(),
            7 => row.ram_address.to_field(),
            8 => row.rs1_value.to_field(),
            9 => row.rs2_value.to_field(),
            10 => row.rd_write_value.to_field(),
            11 => row.ram_read_value.to_field(),
            12 => row.ram_write_value.to_field(),
            13 => row.left_lookup_operand.to_field(),
            14 => row.right_lookup_operand.to_field(),
            15 => row.next_unexpanded_pc.to_field(),
            16 => row.next_pc.to_field(),
            17 => row.next_is_virtual.to_field(),
            18 => row.next_is_first_in_sequence.to_field(),
            19 => row.lookup_output.to_field(),
            20 => row.should_jump.to_field(),
            21 => row.add_operands.to_field(),
            22 => row.subtract_operands.to_field(),
            23 => row.multiply_operands.to_field(),
            24 => row.load.to_field(),
            25 => row.store.to_field(),
            26 => row.jump.to_field(),
            27 => row.write_lookup_output_to_rd.to_field(),
            28 => row.virtual_instruction.to_field(),
            29 => row.assert_flag.to_field(),
            30 => row.do_not_update_unexpanded_pc.to_field(),
            31 => row.advice.to_field(),
            32 => row.is_compressed.to_field(),
            33 => row.is_first_in_sequence.to_field(),
            34 => row.is_last_in_sequence.to_field(),
            #[cfg(feature = "field-inline")]
            35 => row.field_add.to_field(),
            #[cfg(feature = "field-inline")]
            36 => row.field_sub.to_field(),
            #[cfg(feature = "field-inline")]
            37 => row.field_mul.to_field(),
            #[cfg(feature = "field-inline")]
            38 => row.field_inv.to_field(),
            #[cfg(feature = "field-inline")]
            39 => row.field_assert_eq.to_field(),
            #[cfg(feature = "field-inline")]
            40 => row.field_load_accumulate_from_register.to_field(),
            #[cfg(feature = "field-inline")]
            41 => row.field_assert_zero.to_field(),
            #[cfg(feature = "field-inline")]
            42 => row.field_load_imm.to_field(),
            #[cfg(feature = "field-inline")]
            43 => row.field_load_accumulate_from_memory.to_field(),
            #[cfg(feature = "field-inline")]
            44 => row.field_advice_limb.to_field(),
            _ => unreachable!("canonical R1CS inputs"),
        }
    }

    /// Structured pseudo-random rows: full-range `u64`s, mixed-sign `i128`s,
    /// two-limb `u128`/`S128` values (both wide B-row paths), diverse flags.
    /// No satisfying-witness structure — parity must hold pointwise on any
    /// witness.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<SpartanOuterRow> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..1usize << log_t)
            .map(|_| {
                let mut bit = {
                    let value = next();
                    let mut position = 0;
                    move || {
                        position += 1;
                        (value >> position) & 1 == 1
                    }
                };
                let wide = |low: u64, high: u64| (u128::from(high) << 64) | u128::from(low);
                let signed = |value: u64| match value & 3 {
                    0 => i128::MIN,
                    1 => i128::MAX,
                    2 => -(i128::from(value) << 63),
                    _ => i128::from(value) << 63,
                };
                SpartanOuterRow {
                    left_instruction_input: LeftInstructionInput(next()),
                    right_instruction_input: RightInstructionInput(signed(next())),
                    product: Product(S128::new([next() | 1, next()], next() & 1 == 1)),
                    should_branch: ShouldBranch(bit()),
                    pc: Pc(next() >> 20),
                    unexpanded_pc: UnexpandedPc(next()),
                    imm: Imm(signed(next())),
                    ram_address: RamAddress(next()),
                    rs1_value: Rs1Value(next()),
                    rs2_value: Rs2Value(next()),
                    rd_write_value: RdWriteValue(next()),
                    ram_read_value: RamReadValue(next()),
                    ram_write_value: RamWriteValue(next()),
                    left_lookup_operand: LeftLookupOperand(next()),
                    right_lookup_operand: RightLookupOperand(wide(next(), next())),
                    next_unexpanded_pc: NextUnexpandedPc(next()),
                    next_pc: NextPc(next() >> 20),
                    next_is_virtual: NextIsVirtual(bit()),
                    next_is_first_in_sequence: NextIsFirstInSequence(bit()),
                    lookup_output: LookupOutput(next()),
                    should_jump: ShouldJump(bit()),
                    add_operands: OpFlag(bit()),
                    subtract_operands: OpFlag(bit()),
                    multiply_operands: OpFlag(bit()),
                    load: OpFlag(bit()),
                    store: OpFlag(bit()),
                    jump: OpFlag(bit()),
                    write_lookup_output_to_rd: OpFlag(bit()),
                    virtual_instruction: OpFlag(bit()),
                    assert_flag: OpFlag(bit()),
                    do_not_update_unexpanded_pc: OpFlag(bit()),
                    advice: OpFlag(bit()),
                    is_compressed: OpFlag(bit()),
                    is_first_in_sequence: OpFlag(bit()),
                    is_last_in_sequence: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_add: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_sub: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_mul: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_inv: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_assert_eq: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_load_accumulate_from_register: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_assert_zero: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_load_imm: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_load_accumulate_from_memory: OpFlag(bit()),
                    #[cfg(feature = "field-inline")]
                    field_advice_limb: OpFlag(bit()),
                }
            })
            .collect()
    }

    #[cfg(feature = "field-inline")]
    fn synthetic_field_rows(log_t: usize, seed: u64) -> Vec<(usize, FieldInlineSpartanRow<Fr>)> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let base = Fr::from_u64(state);
            base * base + base
        };
        (0..1usize << log_t)
            .filter(|cycle| cycle % 3 == 1 || log_t == 1)
            .map(|cycle| {
                (
                    cycle,
                    FieldInlineSpartanRow {
                        rs1_value: next(),
                        rs2_value: next(),
                        rd_value: next(),
                        product: next(),
                        inv_product: next(),
                    },
                )
            })
            .collect()
    }

    /// The dense image of the sparse field-inline rows for one field-inline column
    /// index (the `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` position) — what the fixed
    /// backend's field-inline view serves the reference kernel.
    #[cfg(feature = "field-inline")]
    fn field_column_table(
        field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
        cycles: usize,
        index: usize,
    ) -> Vec<Fr> {
        let mut table = vec![Fr::from_u64(0); cycles];
        for (cycle, row) in field_rows {
            table[*cycle] = row.columns()[index];
        }
        table
    }

    fn fixed_backend_from_rows(
        log_t: usize,
        rows: &[SpartanOuterRow],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
    ) -> FixedBackend<Fr> {
        let mut backend = FixedBackend::new();
        for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
            let values: Vec<Fr> = rows
                .iter()
                .map(|row| variable_field_value(row, index))
                .collect();
            backend
                .insert(
                    JoltPolynomialId::Virtual(*variable),
                    Shape::new(log_t, PolynomialEncoding::Dense),
                    values,
                )
                .unwrap();
        }
        #[cfg(feature = "field-inline")]
        {
            let mut field_inline = FixedFieldInline::default();
            for (index, id) in FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
                field_inline
                    .insert(
                        FieldInlinePolynomialId::Virtual(*id),
                        Shape::new(log_t, PolynomialEncoding::Dense),
                        field_column_table(field_rows, 1 << log_t, index),
                    )
                    .unwrap();
            }
            backend.set_field_inline(field_inline);
        }
        backend
    }

    /// The remainder's true input claim
    /// `Σ_{t,s} kernel · eq(τ_low, (t,s)) · Az(t,s) · Bz(t,s)`, computed
    /// through the public `jolt-claims` column-weight path over the COMPOSED
    /// opening selection (independent of both kernels' row-value pipelines).
    fn true_input_claim(
        rows: &[SpartanOuterRow],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
        tau: &[Fr],
        r0: Fr,
        log_t: usize,
    ) -> Fr {
        let tau_low = &tau[..=log_t];
        let tau_high = tau[log_t + 1];
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let kernel = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
        let matrices = spartan_outer_constraints::<Fr>();
        let columns: Vec<usize> = spartan_outer_opening_columns();
        let value = |t: usize, position: usize| -> Fr {
            if position < VARIABLE_COUNT {
                return variable_field_value(&rows[t], position);
            }
            #[cfg(feature = "field-inline")]
            {
                field_rows
                    .iter()
                    .find(|(cycle, _)| *cycle == t)
                    .map_or_else(
                        || Fr::from_u64(0),
                        |(_, row)| row.columns()[position - VARIABLE_COUNT],
                    )
            }
            #[cfg(not(feature = "field-inline"))]
            unreachable!("the rv64 selection has exactly 35 columns")
        };
        let mut total = Fr::from_u64(0);
        for (s, stream) in [Fr::from_u64(0), Fr::from_u64(1)].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(r0, stream).unwrap();
            let weighted = matrices.weighted_columns(&weights, &columns).unwrap();
            let constants = matrices
                .public_column_contributions(&weights, 0, Fr::from_u64(1))
                .unwrap();
            for t in 0..rows.len() {
                let mut az = constants.a;
                let mut bz = constants.b;
                for (index, (&a, &b)) in weighted.a.iter().zip(&weighted.b).enumerate() {
                    let value = value(t, index);
                    az += a * value;
                    bz += b * value;
                }
                total += eq[(t << 1) | s] * az * bz;
            }
        }
        kernel * total
    }

    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        parity_case_with_eq(dummy_plane, log_t, seed, None, false);
    }

    fn parity_case_with_eq(
        dummy_plane: &dyn JoltWitnessPlane<Fr>,
        log_t: usize,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
        zero_scale: bool,
    ) {
        let rows = synthetic_rows(log_t, seed);
        #[cfg(feature = "field-inline")]
        let field_rows = synthetic_field_rows(log_t, seed ^ 0xF1E1D);
        let mut tau: Vec<Fr> = exceptional.map_or_else(
            || {
                (0..log_t + 2)
                    .map(|i| Fr::from_u64(3 + seed + 7 * i as u64))
                    .collect()
            },
            |case| {
                let mut point = case.point(log_t + 1, Fr::from_u64(7919 + seed));
                point.push(Fr::from_u64(3 + seed + 7 * (log_t + 1) as u64));
                point
            },
        );
        if zero_scale {
            tau[log_t + 1] = Fr::from_u64(0);
        }
        let backend = fixed_backend_from_rows(
            log_t,
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
        );

        let mut reference_session = ProofSession::default();
        reference_session.park(SpartanOuterKernel::<Fr>::prepare(log_t, &tau, &backend).unwrap());
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[],
                &(),
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedOuterUniskip::prepare_from_rows(
            &mut optimized_session,
            log_t,
            &tau,
            rows.clone(),
            #[cfg(feature = "field-inline")]
            field_rows.clone(),
        )
        .unwrap();
        let optimized_uniskip =
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &OptimizedOuterUniskip,
                &mut optimized_session,
                &[],
                &(),
            )
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "uni-skip first-round polynomial, log_t = {log_t}"
        );

        let r0 = Fr::from_u64(if zero_scale { 1 } else { 40961 + seed });
        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
            &tau,
            r0,
            log_t,
        );
        let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
        let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
        let points = SumcheckInputPoints::<Fr, OuterRemainder<Fr>>::default();
        let no_challenges = NoChallenges::<Fr>::default();

        let mut reference_kernel = ReferenceOuterRemainder
            .prepare(
                &mut reference_session,
                dummy_plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();
        let mut optimized_kernel = OptimizedOuterRemainder
            .prepare(
                &mut optimized_session,
                dummy_plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();

        let rounds = log_t + 1;
        let challenges: Vec<Fr> = (0..rounds)
            .map(|i| Fr::from_u64(7919 + seed + 31 * i as u64))
            .collect();
        let mut bind = None;
        let mut previous = input_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
            let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
            assert_eq!(
                optimized_round, reference_round,
                "round {round} polynomial, log_t = {log_t}"
            );
            previous = reference_round.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference_kernel.finish_rounds(last).unwrap();
        optimized_kernel.finish_rounds(last).unwrap();

        let reference_outputs = reference_kernel.output_claims(&claims).unwrap();
        let optimized_outputs = optimized_kernel.output_claims(&claims).unwrap();
        assert_eq!(
            optimized_outputs, reference_outputs,
            "typed output claims, log_t = {log_t}"
        );

        let output_points = relation
            .derive_opening_points(&challenges, &points)
            .unwrap();
        reference_kernel
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .unwrap();
        optimized_kernel
            .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
            .unwrap();
    }

    #[test]
    fn remainder_matches_reference_at_exceptional_eq_and_zero_scaling() {
        with_sample_backend(|dummy| {
            for case in ExceptionalEq::ALL {
                parity_case_with_eq(dummy, 4, 389, Some(case), false);
            }
            parity_case_with_eq(dummy, 4, 389, None, true);
        });
    }

    #[test]
    fn synthetic_parity_with_reference_kernels() {
        with_sample_backend(|dummy| {
            for (log_t, seed) in [(1usize, 111u64), (2, 222), (3, 333), (4, 444)] {
                parity_case(dummy, log_t, seed);
            }
        });
    }

    /// The trait-path parity body over a real trace backend: the optimized
    /// bundle walk against the reference's oracle tables, with the remainder
    /// driven by the true joint-domain sum. The trace fixtures are
    /// witness-extraction fixtures, not constraint-satisfying traces, so the
    /// uni-skip reduction at r0 need not equal the joint-domain sum here; the
    /// remainder runs on the true sum, which is what the naive reference
    /// self-checks against.
    fn sample_case(backend: &TraceBackend<OwnedTrace>, log_t: usize) {
        let tau: Vec<Fr> = (0..log_t + 2)
            .map(|i| Fr::from_u64(29 + 13 * i as u64))
            .collect();

        let mut reference_session = ProofSession::default();
        <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
            &ReferenceBackend,
            &mut reference_session,
            log_t,
            &tau,
            backend,
        )
        .unwrap();
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[],
                &(),
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::prepare(
            &OptimizedOuterUniskip,
            &mut optimized_session,
            log_t,
            &tau,
            backend,
        )
        .unwrap();
        let optimized_uniskip =
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &OptimizedOuterUniskip,
                &mut optimized_session,
                &[],
                &(),
            )
            .unwrap();
        assert_eq!(optimized_uniskip, reference_uniskip);

        let r0 = Fr::from_u64(9173);
        let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
        #[cfg(feature = "field-inline")]
        let field_rows = JoltWitnessOracle::<Fr>::field_inline(backend)
            .unwrap()
            .field_inline_spartan_rows()
            .unwrap();
        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
            &tau,
            r0,
            log_t,
        );

        let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
        let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
        let points = SumcheckInputPoints::<Fr, OuterRemainder<Fr>>::default();
        let no_challenges = NoChallenges::<Fr>::default();
        let mut reference_kernel = ReferenceOuterRemainder
            .prepare(
                &mut reference_session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();
        let mut optimized_kernel = OptimizedOuterRemainder
            .prepare(
                &mut optimized_session,
                backend,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                },
            )
            .unwrap();

        let challenges: Vec<Fr> = (0..=log_t)
            .map(|i| Fr::from_u64(523 + 17 * i as u64))
            .collect();
        let mut bind = None;
        let mut previous = input_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
            let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
            assert_eq!(optimized_round, reference_round, "round {round}");
            previous = reference_round.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference_kernel.finish_rounds(last).unwrap();
        optimized_kernel.finish_rounds(last).unwrap();
        assert_eq!(
            optimized_kernel.output_claims(&claims).unwrap(),
            reference_kernel.output_claims(&claims).unwrap()
        );
    }

    /// Full trait-path parity: without field-inline on the canned sample trace; with
    /// field-inline enabled over a field-inline fixture trace (the sample backend
    /// carries no field-inline view), exercising the trace-backed sparse field-inline
    /// row seam.
    #[test]
    fn sample_trace_parity_through_the_trait_path() {
        #[cfg(not(feature = "field-inline"))]
        with_sample_backend(|backend| sample_case(backend, 2));
        #[cfg(feature = "field-inline")]
        crate::optimized::field_registers_testing::structured_field_register_fixture(12)
            .with_plane(4, |backend| sample_case(backend, 4));
    }

    /// The integer extension of each base-window basis vector is exactly the
    /// field Lagrange basis evaluated at every extended node — by linearity,
    /// the fact that ties the integer pipeline to the reference's field
    /// pipeline.
    #[test]
    fn extension_matches_field_lagrange() {
        for i in 0..DOMAIN {
            let mut basis = [0i128; DOMAIN];
            basis[i] = 1;
            for (position, value) in EXTENDED_POSITIONS.into_iter().zip(extend(&basis)) {
                let node = EXTENDED_START + position as i64;
                let expected = centered_lagrange_evals::<Fr>(DOMAIN, Fr::from_i64(node)).unwrap();
                assert_eq!(Fr::from_i128(value), expected[i], "node {node}, basis {i}");
            }
        }
    }

    /// The [`RowGroupValues`] ranges the [`EXTENSION_GAIN`] assertions
    /// assume: guards of magnitude ≤ 1 except one ≤ 2 and one ≤ 3 per group,
    /// B values (or halves) below 2^65.
    #[test]
    fn group_values_within_extension_ranges() {
        for row in synthetic_rows(8, 0xB0_0D) {
            let values = row.group_values();
            for guards in [values.a_first.as_slice(), values.a_second.as_slice()] {
                let excess: u64 = guards
                    .iter()
                    .map(|guard| guard.unsigned_abs().saturating_sub(1))
                    .sum();
                assert!(excess <= 3);
            }
            assert!(values.b_first.iter().all(|b| b.unsigned_abs() <= 1 << 64));
            assert!(values
                .b_second_lo
                .iter()
                .chain(&values.b_second_hi)
                .all(|b| b.unsigned_abs() < 1 << 65));
        }
    }

    /// The typed bundle's columns equal the oracle tables the reference
    /// kernel materializes — the two witness paths meeting at the shared
    /// `Extract` impls, for all ordinary R1CS inputs.
    #[test]
    fn bundle_columns_match_oracle_tables() {
        with_sample_backend(|backend| {
            let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            for (index, variable) in SPARTAN_OUTER_R1CS_INPUTS.iter().enumerate() {
                let table: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(*variable),
                )
                .unwrap();
                let column: Vec<Fr> = rows
                    .iter()
                    .map(|row| variable_field_value(row, index))
                    .collect();
                assert_eq!(column, table, "{variable:?}");
            }
        });
    }
}
