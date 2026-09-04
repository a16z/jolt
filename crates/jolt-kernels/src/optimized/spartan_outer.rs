//! Optimized stage-1 Spartan outer kernels: the legacy prover's algorithms
//! behind the reference kernels' exact wire behavior.
//!
//! Techniques ported from `jolt-prover-legacy`'s `zkvm/spartan/outer.rs` +
//! `r1cs/evaluation.rs`:
//!
//! - **Typed small-scalar row evaluation**: the 19 eq-conditional constraint
//!   rows are evaluated per cycle as integers (`i64` guards, `S192`
//!   magnitudes) straight off a typed witness bundle — the 35 R1CS input
//!   tables are never materialized as field vectors
//!   (`R1CSEval::{eval_az,eval_bz}_*_group`).
//! - **Univariate skip over the centered integer domain**: the first-round
//!   polynomial needs only the 9 extended-node evaluations (in-domain nodes
//!   vanish); each is an integer Lagrange extension of the row values
//!   (`COEFFS_PER_J` / `extended_azbz_product_*`), so the whole pass costs 9
//!   integer dot products and one field fmadd per `(cycle, stream)` instead
//!   of per-row field multiplies.
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
//! - **In-place binding**: `Az`/`Bz` bind without swap buffers; the 35 input
//!   tables are never bound.
//! - **Post-hoc opening evaluation**: the 35 produced opening claims come
//!   from one final eq-weighted walk over the typed rows
//!   (`R1CSEval::compute_claimed_inputs`), not from binding 35 polynomials
//!   through every round.
//!
//! Byte parity with the reference kernels holds because every step computes
//! the same field values by exact integer/field algebra (the integer Lagrange
//! extension coefficients equal the field Lagrange evaluations at integer
//! nodes; ring homomorphism does the rest), and the wire assembly reuses the
//! reference's own `jolt-poly` interpolation path.

#[cfg(feature = "field-inline")]
use core::cmp::Ordering;
use std::collections::BTreeMap;

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineOpFlag;
use jolt_claims::protocols::jolt::geometry::spartan::{
    outer_opening, SpartanOuterDimensions, SPARTAN_OUTER_R1CS_INPUTS,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, JoltPolynomialId, SpartanOuterPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::{S128, S192, S256, S64};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
// The COMPOSED jolt-r1cs shapes (feature-aware): identical to the rv64-only
// constants FR-off, the FR-extended row/column composition under
// `field-inline` — the same sources the reference kernel folds with.
use jolt_r1cs::constraints::jolt::{
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
/// (`SPARTAN_OUTER_{FIRST,SECOND}_GROUP_ROWS` order): FR rows append behind
/// them under `field-inline`, in [FADD, FSUB, FMUL, FINV] /
/// [ASSERT_EQ, LOAD_FROM_X, STORE_TO_X, LOAD_IMM] order.
const RV64_FIRST_GROUP_LEN: usize = 10;
const RV64_SECOND_GROUP_LEN: usize = 9;

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
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .map(JoltPolynomialId::Virtual)
            .collect()
    }
}

/// One cycle's integer values of the composed eq-conditional rows, split
/// into the two uni-skip stream groups (A-side guards as `i64`, B-side
/// magnitudes as `S192` — wide enough for the `RightLookupOperand`-bearing
/// rows, whose values reach ±2^130, times the composed 14-node extension
/// coefficients ≤ 2^26). Under `field-inline` the arrays span the composed
/// groups with the FR rows at their FR-INACTIVE values; FR-active cycles use
/// [`FieldGroupValues`] instead.
struct RowGroupValues {
    a_first: [i64; DOMAIN],
    a_second: [i64; SECOND_GROUP_LEN],
    b_first: [S192; DOMAIN],
    b_second: [S192; SECOND_GROUP_LEN],
}

/// One FR-ACTIVE cycle's composed group values, in field form: the FR
/// magnitudes are full field elements, so the integer pipeline cannot carry
/// them. FR-active cycles are rare (bounded by the FR instruction count), so
/// the field path's extra cost stays proportional to FR activity.
#[cfg(feature = "field-inline")]
struct FieldGroupValues<F> {
    a_first: [F; DOMAIN],
    a_second: [F; SECOND_GROUP_LEN],
    b_first: [F; DOMAIN],
    b_second: [F; SECOND_GROUP_LEN],
}

/// The field image of an `S192` magnitude, through the same deferred
/// accumulator path the integer pipeline reduces with.
#[cfg(feature = "field-inline")]
fn s192_to_field<F: JoltField>(value: &S192) -> F {
    let mut accumulator = <F as WithAccumulator>::SignedProductAccumulator::default();
    accumulator.fmadd_s256(F::one(), &widen(value));
    accumulator.reduce()
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> FieldGroupValues<F> {
    /// `Az·Bz` at every extended node for one cycle, per stream — the field
    /// twin of [`extended_products`], over the coefficient
    /// field images (equal by the ring homomorphism).
    fn extended_products(
        &self,
        coefficients: &[(usize, [F; DOMAIN]); EXTENDED_NODE_COUNT],
    ) -> [(F, F); EXTENDED_NODE_COUNT] {
        let mut out = [(F::zero(), F::zero()); EXTENDED_NODE_COUNT];
        for (slot, (_, coefficients)) in coefficients.iter().enumerate() {
            let mut az_first = F::zero();
            let mut az_second = F::zero();
            let mut bz_first = F::zero();
            let mut bz_second = F::zero();
            for (i, &c) in coefficients.iter().enumerate() {
                az_first += c * self.a_first[i];
                bz_first += c * self.b_first[i];
                if i < SECOND_GROUP_LEN {
                    az_second += c * self.a_second[i];
                    bz_second += c * self.b_second[i];
                }
            }
            out[slot] = (az_first * bz_first, az_second * bz_second);
        }
        out
    }

    /// The bound `Az`/`Bz` values of the first stream group under the
    /// uni-skip challenge's Lagrange weights — the field twin of
    /// [`fold_group`].
    fn fold_first(&self, weights: &[F]) -> (F, F) {
        fold_field_group(weights, &self.a_first, &self.b_first)
    }

    /// The field twin of [`fold_group`].
    fn fold_second(&self, weights: &[F]) -> (F, F) {
        fold_field_group(&weights[..SECOND_GROUP_LEN], &self.a_second, &self.b_second)
    }
}

#[cfg(feature = "field-inline")]
fn fold_field_group<F: JoltField>(weights: &[F], guards: &[F], magnitudes: &[F]) -> (F, F) {
    let mut az = F::zero();
    let mut bz = F::zero();
    for ((&weight, &guard), &magnitude) in weights.iter().zip(guards).zip(magnitudes) {
        az += weight * guard;
        bz += weight * magnitude;
    }
    (az, bz)
}

/// The field images of [`extension_coefficients`] — what ties the FR-active
/// field path to the same Lagrange extension the integer pipeline uses.
#[cfg(feature = "field-inline")]
fn extension_coefficient_fields<F: JoltField>() -> [(usize, [F; DOMAIN]); EXTENDED_NODE_COUNT] {
    extension_coefficients()
        .map(|(position, coefficients)| (position, coefficients.map(F::from_i64)))
}

/// The sorted sparse FR spartan rows plus a moving cursor, so cycle-ordered
/// block walks can route FR-active cycles to the field path in O(1) per
/// cycle. Shared with the product kernel, whose lanes walk the same rows.
#[cfg(feature = "field-inline")]
pub(crate) struct FrRowCursor<'a, F> {
    rows: &'a [(usize, FieldInlineSpartanRow<F>)],
    next: usize,
}

#[cfg(feature = "field-inline")]
impl<'a, F> FrRowCursor<'a, F> {
    /// A cursor positioned at the first row with cycle ≥ `start` — each
    /// parallel block seeks independently.
    pub(crate) fn seek(rows: &'a [(usize, FieldInlineSpartanRow<F>)], start: usize) -> Self {
        Self {
            rows,
            next: rows.partition_point(|&(cycle, _)| cycle < start),
        }
    }

    /// The FR row at cycle `t`, if any; `t` must be non-decreasing across
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
    /// Evaluate the 19 constraint rows at one cycle with exact integer
    /// arithmetic. Formulas transcribe `jolt-r1cs`'s `rv64_eq_constraint_rows`
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

        let diff = |a: u64, b: u64| S192::from_i128(i128::from(a) - i128::from(b));
        let rv64_b_first = [
            S192::from_u64(self.ram_address.0),
            diff(self.ram_read_value.0, self.ram_write_value.0),
            diff(self.ram_read_value.0, self.rd_write_value.0),
            diff(self.rs2_value.0, self.ram_write_value.0),
            S192::from_u64(self.left_lookup_operand.0),
            diff(self.left_lookup_operand.0, self.left_instruction_input.0),
            S192::from_i128(i128::from(self.lookup_output.0) - 1),
            diff(self.next_unexpanded_pc.0, self.lookup_output.0),
            S192::from_i128(i128::from(self.next_pc.0) - i128::from(self.pc.0) - 1),
            S192::from_i64(1 - flag(self.do_not_update_unexpanded_pc.0)),
        ];

        let flag_i128 = |value: bool| i128::from(value);
        let right_lookup = S192::from_u128(self.right_lookup_operand.0);
        let right_input = S192::from_i128(self.right_instruction_input.0);
        let left_input = S192::from_u64(self.left_instruction_input.0);
        let imm = S192::from_i128(self.imm.0);
        let product_limbs = self.product.0.magnitude_limbs();
        let product = S192::new(
            [product_limbs[0], product_limbs[1], 0],
            self.product.0.is_positive,
        );
        let two_pow_64 = S192::new([0, 1, 0], true);
        let rv64_b_second = [
            S192::from_i128(i128::from(self.ram_address.0) - i128::from(self.rs1_value.0)) - imm,
            right_lookup - left_input - right_input,
            right_lookup - left_input + right_input - two_pow_64,
            right_lookup - product,
            right_lookup - right_input,
            S192::from_i128(i128::from(self.rd_write_value.0) - i128::from(self.lookup_output.0)),
            S192::from_i128(
                i128::from(self.rd_write_value.0) - i128::from(self.unexpanded_pc.0) - 4
                    + 2 * flag_i128(self.is_compressed.0),
            ),
            S192::from_i128(
                i128::from(self.next_unexpanded_pc.0) - i128::from(self.unexpanded_pc.0),
            ) - imm,
            S192::from_i128(
                i128::from(self.next_unexpanded_pc.0) - i128::from(self.unexpanded_pc.0) - 4
                    + 4 * flag_i128(self.do_not_update_unexpanded_pc.0)
                    + 2 * flag_i128(self.is_compressed.0),
            ),
        ];

        let mut values = RowGroupValues {
            a_first: [0; DOMAIN],
            a_second: [0; SECOND_GROUP_LEN],
            b_first: [S192::zero(); DOMAIN],
            b_second: [S192::zero(); SECOND_GROUP_LEN],
        };
        values.a_first[..RV64_FIRST_GROUP_LEN].copy_from_slice(&rv64_a_first);
        values.a_second[..RV64_SECOND_GROUP_LEN].copy_from_slice(&rv64_a_second);
        values.b_first[..RV64_FIRST_GROUP_LEN].copy_from_slice(&rv64_b_first);
        values.b_second[..RV64_SECOND_GROUP_LEN].copy_from_slice(&rv64_b_second);

        // The FR rows at their FR-INACTIVE values (all FR columns zero) —
        // still integers, off the shared rv64 columns the bridge rows reuse.
        // FR-active cycles go through `field_group_values` instead; calling
        // this on one is a routing bug the parity tests would surface as a
        // wrong t1 value.
        // First group [FADD, FSUB, FMUL, FINV]: guards zero; magnitudes
        // zero except FINV's `inv_product − 1 = −1`.
        // Second group [ASSERT_EQ, LOAD_FROM_X, STORE_TO_X, LOAD_IMM,
        // STORE_TO_X_LOOKUP]: guards zero; magnitudes `0`,
        // `frd − Rs1Value = −Rs1Value`, `RdWriteValue − frs1 = RdWriteValue`,
        // `frd − Imm = −Imm`, `RightLookupOperand − frs1 = RightLookupOperand`.
        #[cfg(feature = "field-inline")]
        {
            values.b_first[DOMAIN - 1] = S192::from_i64(-1);
            values.b_second[RV64_SECOND_GROUP_LEN + 1] =
                S192::zero() - S192::from_u64(self.rs1_value.0);
            values.b_second[RV64_SECOND_GROUP_LEN + 2] = S192::from_u64(self.rd_write_value.0);
            values.b_second[RV64_SECOND_GROUP_LEN + 3] = S192::zero() - imm;
            values.b_second[RV64_SECOND_GROUP_LEN + 4] = right_lookup;
        }

        values
    }

    /// The composed group values of one FR-ACTIVE cycle, in field form: the
    /// rv64 guards/magnitudes promoted plus the FR rows' native field values
    /// (`jolt-r1cs`'s `field_eq_constraint_rows` transcribed at the composed
    /// group positions). Exact — the integer pipeline and this one compute
    /// the same field elements, so routing a cycle either way is
    /// wire-invisible; the integer path simply cannot represent an active
    /// cycle's field magnitudes.
    #[cfg(feature = "field-inline")]
    fn field_group_values<F: JoltField>(
        &self,
        fr: &FieldInlineSpartanRow<F>,
    ) -> FieldGroupValues<F> {
        let integer = self.group_values();
        let mut values = FieldGroupValues {
            a_first: integer.a_first.map(F::from_i64),
            a_second: integer.a_second.map(F::from_i64),
            b_first: integer.b_first.map(|value| s192_to_field(&value)),
            b_second: integer.b_second.map(|value| s192_to_field(&value)),
        };
        let flag = |flag: FieldInlineOpFlag| fr.flags[flag as usize];
        values.a_first[RV64_FIRST_GROUP_LEN] = flag(FieldInlineOpFlag::Add);
        values.a_first[RV64_FIRST_GROUP_LEN + 1] = flag(FieldInlineOpFlag::Sub);
        values.a_first[RV64_FIRST_GROUP_LEN + 2] = flag(FieldInlineOpFlag::Mul);
        values.a_first[RV64_FIRST_GROUP_LEN + 3] = flag(FieldInlineOpFlag::Inv);
        values.b_first[RV64_FIRST_GROUP_LEN] = fr.rs1_value + fr.rs2_value - fr.rd_value;
        values.b_first[RV64_FIRST_GROUP_LEN + 1] = fr.rs1_value - fr.rs2_value - fr.rd_value;
        values.b_first[RV64_FIRST_GROUP_LEN + 2] = fr.product - fr.rd_value;
        values.b_first[RV64_FIRST_GROUP_LEN + 3] = fr.inv_product - F::one();
        values.a_second[RV64_SECOND_GROUP_LEN] = flag(FieldInlineOpFlag::AssertEq);
        values.a_second[RV64_SECOND_GROUP_LEN + 1] = flag(FieldInlineOpFlag::LoadFromX);
        values.a_second[RV64_SECOND_GROUP_LEN + 2] = flag(FieldInlineOpFlag::StoreToX);
        values.a_second[RV64_SECOND_GROUP_LEN + 3] = flag(FieldInlineOpFlag::LoadImm);
        values.b_second[RV64_SECOND_GROUP_LEN] = fr.rs1_value - fr.rs2_value;
        values.b_second[RV64_SECOND_GROUP_LEN + 1] = fr.rd_value - F::from_u64(self.rs1_value.0);
        values.b_second[RV64_SECOND_GROUP_LEN + 2] =
            F::from_u64(self.rd_write_value.0) - fr.rs1_value;
        values.b_second[RV64_SECOND_GROUP_LEN + 3] = fr.rd_value - F::from_i128(self.imm.0);
        values.a_second[RV64_SECOND_GROUP_LEN + 4] = flag(FieldInlineOpFlag::StoreToX);
        values.b_second[RV64_SECOND_GROUP_LEN + 4] =
            F::from_u128(self.right_lookup_operand.0) - fr.rs1_value;
        values
    }
}

/// The exact integer Lagrange extension coefficients from the 10-node base
/// window to each out-of-domain extended node: `coeffs[i] = L_i(node)`.
/// Consecutive-integer domains make these integers (legacy's `COEFFS_PER_J`);
/// their field images equal `centered_lagrange_evals` at the node, which is
/// what ties the integer pipeline to the reference's field pipeline.
fn extension_coefficients() -> [(usize, [i64; DOMAIN]); EXTENDED_NODE_COUNT] {
    let mut out = [(0usize, [0i64; DOMAIN]); EXTENDED_NODE_COUNT];
    let mut slot = 0;
    for position in 0..EXTENDED_SIZE {
        let node = EXTENDED_START + position as i64;
        if node >= DOMAIN_START && node < DOMAIN_START + DOMAIN as i64 {
            continue;
        }
        let mut coefficients = [0i64; DOMAIN];
        for (i, coefficient) in coefficients.iter_mut().enumerate() {
            let mut numerator: i128 = 1;
            let mut denominator: i128 = 1;
            for j in 0..DOMAIN {
                if j == i {
                    continue;
                }
                numerator *= i128::from(node - (DOMAIN_START + j as i64));
                denominator *= i128::from(i as i64 - j as i64);
            }
            debug_assert_eq!(numerator % denominator, 0);
            *coefficient = (numerator / denominator) as i64;
        }
        out[slot] = (position, coefficients);
        slot += 1;
    }
    debug_assert_eq!(slot, EXTENDED_NODE_COUNT);
    out
}

/// `Az·Bz` at every extended node for one cycle, per stream: integer Lagrange
/// extension of the group row values, then one wide integer product. Ranges:
/// `|az| < 2^22`, `|bz| < 2^152`, product `< 2^174` — inside `S256`.
fn extended_products(
    values: &RowGroupValues,
    coefficients: &[(usize, [i64; DOMAIN]); EXTENDED_NODE_COUNT],
) -> [(S256, S256); EXTENDED_NODE_COUNT] {
    let mut out = [(S256::zero(), S256::zero()); EXTENDED_NODE_COUNT];
    for (slot, (_, coefficients)) in coefficients.iter().enumerate() {
        let mut az_first: i64 = 0;
        let mut az_second: i64 = 0;
        let mut bz_first = S192::zero();
        let mut bz_second = S192::zero();
        for (i, &c) in coefficients.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let a_first = values.a_first[i];
            if a_first != 0 {
                az_first += c * a_first;
            }
            let b_first = &values.b_first[i];
            if b_first.magnitude_limbs() != [0; 3] {
                S64::from_i64(c).fmadd_trunc::<3, 3>(b_first, &mut bz_first);
            }
            if i < SECOND_GROUP_LEN {
                let a_second = values.a_second[i];
                if a_second != 0 {
                    az_second += c * a_second;
                }
                let b_second = &values.b_second[i];
                if b_second.magnitude_limbs() != [0; 3] {
                    S64::from_i64(c).fmadd_trunc::<3, 3>(b_second, &mut bz_second);
                }
            }
        }
        out[slot] = (
            S64::from_i64(az_first).mul_trunc::<3, 4>(&bz_first),
            S64::from_i64(az_second).mul_trunc::<3, 4>(&bz_second),
        );
    }
    out
}

fn widen(value: &S192) -> S256 {
    let limbs = value.magnitude_limbs();
    S256::new([limbs[0], limbs[1], limbs[2], 0], value.is_positive)
}

/// Fold group row values with the uni-skip challenge's Lagrange weights into
/// the bound `Az`/`Bz` values for one `(cycle, stream)` cell, through the
/// unreduced accumulators.
fn fold_group<F: JoltField>(weights: &[F], guards: &[i64], magnitudes: &[S192]) -> (F, F) {
    let mut az = <F as WithAccumulator>::SmallScalarAccumulator::default();
    let mut bz = <F as WithAccumulator>::SignedProductAccumulator::default();
    for ((&weight, &guard), magnitude) in weights.iter().zip(guards).zip(magnitudes) {
        az.fmadd_i64(weight, guard);
        let limbs = magnitude.magnitude_limbs();
        if limbs[1] == 0 && limbs[2] == 0 {
            bz.fmadd_signed_u64(weight, limbs[0], magnitude.is_positive);
        } else {
            bz.fmadd_s256(weight, &widen(magnitude));
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
    /// The FR-active cycles' composed column values, sparse and sorted by
    /// cycle (the witness seam's direct walk — the 13 dense FR tables never
    /// materialize).
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    /// All `2·DOMAIN − 1` node values of `t1`; in-domain nodes stay zero (a
    /// satisfying witness vanishes there), matching the reference layout.
    t1_values: Vec<F>,
}

/// The stage-1 uni-skip front: typed-row collection, the extended-node
/// evaluation pass, and the first-round polynomial assembly.
pub struct OptimizedOuterUniskip;

impl OptimizedOuterUniskip {
    /// The post-collection half of [`UniskipKernel::prepare`], for the
    /// in-module parity tests (which construct rows — and FR-on, the sparse
    /// FR rows — directly).
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: Vec<SpartanOuterRow>,
        #[cfg(feature = "field-inline")] fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
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
            fr_rows,
        )
    }

    /// The store-generic half of `prepare`.
    fn prepare_from_store<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: BundleStore<SpartanOuterRow>,
        #[cfg(feature = "field-inline")] fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
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
            &fr_rows,
        )?;
        session.park(SpartanOuterCarry {
            log_t,
            tau: tau.to_vec(),
            rows,
            #[cfg(feature = "field-inline")]
            fr_rows,
            t1_values,
        });
        Ok(())
    }

    /// Extended-node evaluations of
    /// `t1(Y) = Σ_{t,s} eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)`, with the eq
    /// table factored as `E_out ⊗ E_in` and the per-cycle products from the
    /// integer extension pipeline.
    fn extended_t1_values<F: JoltField>(
        rows: &BundleAccess<'_, SpartanOuterRow>,
        tau_low: &[F],
        #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<F>)],
    ) -> Result<Vec<F>, WitnessError> {
        let split = tau_low.len() / 2;
        let (out_point, in_point) = tau_low.split_at(split);
        let e_out = EqPolynomial::<F>::evals(out_point, None);
        let e_in = EqPolynomial::<F>::evals(in_point, None);
        // `in_point` always covers the stream bit (τ_low's last entry), so every
        // (cycle, stream) pair sits inside one `x_out` block.
        let pairs_per_block = e_in.len() / 2;
        let coefficients = extension_coefficients();
        #[cfg(feature = "field-inline")]
        let field_coefficients = extension_coefficient_fields::<F>();

        let extended = try_par_sum_vecs(e_out.len(), EXTENDED_NODE_COUNT, |x_out| {
            let mut accumulators: Vec<<F as WithAccumulator>::SignedProductAccumulator> =
                vec![Default::default(); EXTENDED_NODE_COUNT];
            #[cfg(feature = "field-inline")]
            let mut field_sums = vec![F::zero(); EXTENDED_NODE_COUNT];
            #[cfg(feature = "field-inline")]
            let mut fr_cursor = FrRowCursor::seek(fr_rows, x_out * pairs_per_block);
            for pair in 0..pairs_per_block {
                let t = x_out * pairs_per_block + pair;
                let row = rows.row(t)?;
                #[cfg(feature = "field-inline")]
                if let Some(fr) = fr_cursor.advance(t) {
                    let values = row.field_group_values(fr);
                    let products = values.extended_products(&field_coefficients);
                    for (sum, (first, second)) in field_sums.iter_mut().zip(&products) {
                        *sum += e_in[2 * pair] * *first + e_in[2 * pair + 1] * *second;
                    }
                    continue;
                }
                let values = row.group_values();
                let products = extended_products(&values, &coefficients);
                for (accumulator, (first, second)) in accumulators.iter_mut().zip(&products) {
                    accumulator.fmadd_s256(e_in[2 * pair], first);
                    accumulator.fmadd_s256(e_in[2 * pair + 1], second);
                }
            }
            #[cfg(feature = "field-inline")]
            return Ok(accumulators
                .into_iter()
                .zip(field_sums)
                .map(|(accumulator, field_sum)| e_out[x_out] * (accumulator.reduce() + field_sum))
                .collect());
            #[cfg(not(feature = "field-inline"))]
            Ok(accumulators
                .into_iter()
                .map(|accumulator| e_out[x_out] * accumulator.reduce())
                .collect())
        })?;

        let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
        for ((position, _), value) in extension_coefficients().iter().zip(extended) {
            t1_values[*position] = value;
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
        let fr_rows = witness
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
            fr_rows,
        )
    }

    #[tracing::instrument(skip_all, name = "SpartanOuterUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        _late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let carry =
            session
                .state::<SpartanOuterCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason:
                        "the outer uni-skip slot parked no carry for the first-round polynomial",
                })?;
        // The reference's exact assembly path, fed the same t1 node values.
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
    /// `(Az, Bz)` over the joint domain.
    az: Polynomial<F>,
    bz: Polynomial<F>,
    /// Whether the first-shrink purge ran.
    purged: bool,
    split_eq: GruenSplitEqPolynomial<F>,
    /// Round-0 endpoints, fused into the materialization pass.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pending_endpoints: Option<(F, F)>,
    challenges: RoundChallenges<F>,
    rows: BundleStore<SpartanOuterRow>,
    /// The Arc-shared relation cell: the FR opening appendage publishes on
    /// it at extraction (the driver's curated absorb and the stage-1 recipe
    /// read it there).
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(skip))]
    relation: OuterRemainder<F>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
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
            fr_rows,
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

        // Fused round-0 materialization: one pass over the typed rows writes
        // the bound Az/Bz tables and accumulates the first round's endpoints
        // q(0) = Σ_t E(t)·az₀·bz₀ and q(∞) = Σ_t E(t)·(az₁−az₀)(bz₁−bz₀).
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
        let fr_rows_ref: &[(usize, FieldInlineSpartanRow<F>)] = &fr_rows;
        let block = |x_out: usize,
                     az_chunk: &mut [F],
                     bz_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            #[cfg(feature = "field-inline")]
            let mut fr_cursor = FrRowCursor::seek(fr_rows_ref, x_out * in_len);
            for x_in in 0..in_len {
                let t = x_out * in_len + x_in;
                let row = access.row(t)?;
                #[cfg(feature = "field-inline")]
                let (az_zero, bz_zero, az_one, bz_one) = if let Some(fr) = fr_cursor.advance(t) {
                    let values = row.field_group_values(fr);
                    let (az_zero, bz_zero) = values.fold_first(lagrange);
                    let (az_one, bz_one) = values.fold_second(lagrange);
                    (az_zero, bz_zero, az_one, bz_one)
                } else {
                    let values = row.group_values();
                    let (az_zero, bz_zero) = fold_group(lagrange, &values.a_first, &values.b_first);
                    let (az_one, bz_one) = fold_group(
                        &lagrange[..SECOND_GROUP_LEN],
                        &values.a_second,
                        &values.b_second,
                    );
                    (az_zero, bz_zero, az_one, bz_one)
                };
                #[cfg(not(feature = "field-inline"))]
                let (az_zero, bz_zero, az_one, bz_one) = {
                    let values = row.group_values();
                    let (az_zero, bz_zero) = fold_group(lagrange, &values.a_first, &values.b_first);
                    let (az_one, bz_one) = fold_group(
                        &lagrange[..SECOND_GROUP_LEN],
                        &values.a_second,
                        &values.b_second,
                    );
                    (az_zero, bz_zero, az_one, bz_one)
                };
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
            relation: inputs.relation.clone(),
            #[cfg(feature = "field-inline")]
            fr_rows,
            opening_ids,
            derived,
        })
    }

    /// Az/Bz column weights at both stream values over the composed
    /// opening-column selection, from the same `jolt-r1cs` sources the
    /// verifier's coefficient build uses (35 rv64 columns FR-off; the
    /// non-contiguous 35 + 13 selection under `field-inline`).
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
        // Purge once after the first shrink.
        if shrunk && !self.purged {
            self.purged = true;
            // `rounds = log_t + 1`.
            crate::mem::purge_retained_memory(self.challenges.total() - 1);
        }
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    /// The bound cycle point's eq table (the stream challenge excluded).
    fn cycle_weights(&self) -> Vec<F> {
        let reversed: Vec<F> = self.challenges.as_slice()[1..]
            .iter()
            .rev()
            .copied()
            .collect();
        let _span = tracing::info_span!("SpartanOuter::claimed_input_weights").entered();
        EqPolynomial::<F>::evals(&reversed, None)
    }

    /// The 35 produced opening values at the bound cycle point: one
    /// eq-weighted walk over the typed rows (`compute_claimed_inputs`),
    /// mixed-width accumulators per input.
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

    /// The 13 FR opening values at the bound cycle point: one eq-weighted
    /// walk over the sparse FR rows (columns in
    /// `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` order — the appendage order
    /// `set_field_inline_outputs` expects).
    #[cfg(feature = "field-inline")]
    fn fr_claimed_inputs(&self, weights: &[F]) -> Vec<F> {
        let mut values = vec![F::zero(); 13];
        for (cycle, row) in &self.fr_rows {
            let weight = weights[*cycle];
            for (value, column) in values.iter_mut().zip(row.columns()) {
                *value += weight * column;
            }
        }
        values
    }
}

const VARIABLE_COUNT: usize = 35;

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
    let mut flag = 21; // the 14 circuit flags
    while flag < VARIABLE_COUNT {
        mask[flag] = true;
        flag += 1;
    }
    mask
};

/// Mixed-width claim accumulators for the final opening walk: boolean inputs
/// through the small-scalar path, word/wide inputs through the signed-product
/// path.
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
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let (q_zero, q_infinity) = match self.pending_endpoints.take() {
            Some(endpoints) => endpoints,
            None => self.split_eq.product_endpoints(&self.az, &self.bz),
        };
        Ok(self
            .split_eq
            .gruen_poly_deg_3(q_zero, q_infinity, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OuterRemainderKernel<F> {
    type Relation = OuterRemainder<F>;

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
        // Publish the FR appendage on the Arc-shared relation cell: the
        // driver's curated absorb, its composed expected-output fold, and
        // the stage-1 recipe's claim assembly all read it from there.
        #[cfg(feature = "field-inline")]
        self.relation
            .set_field_inline_outputs(self.fr_claimed_inputs(&weights))
            .map_err(SumcheckKernelError::Verifier)?;
        let claims: BTreeMap<JoltOpeningId, F> =
            self.opening_ids.iter().copied().zip(claimed).collect();
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
        // The composed selection width (48 FR-on), not the 35 ordinary ids.
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

/// Byte parity against the reference kernels: identical uni-skip first-round
/// polynomials, identical remainder round polynomials at every round,
/// identical typed output claims — from identical `ProverInputs`, over
/// synthetic structured witnesses (both groups' wide integer paths exercised)
/// and over the real sample trace through the full trait path.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
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
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    #[cfg(feature = "field-inline")]
    use jolt_witness::FixedFieldInline;
    use jolt_witness::{BundleSource, JoltWitnessOracle};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape, TraceBackend};

    use super::*;
    use crate::reference::spartan_outer::{ReferenceOuterRemainder, SpartanOuterKernel};
    use crate::ReferenceBackend;

    /// The `ToField` image of one canonical R1CS input, straight off the
    /// typed row — the single conversion source for backend columns and
    /// consistency checks.
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
            _ => unreachable!("35 canonical R1CS inputs"),
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
                let signed = |value: u64| {
                    let magnitude = i128::from(value >> 1) << 33;
                    if value & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    }
                };
                SpartanOuterRow {
                    left_instruction_input: LeftInstructionInput(next()),
                    right_instruction_input: RightInstructionInput(signed(next())),
                    product: Product(S128::new([next() | 1, next()], next() & 1 == 1)),
                    should_branch: ShouldBranch(bit()),
                    pc: Pc(next() >> 20),
                    unexpanded_pc: UnexpandedPc(next()),
                    imm: Imm(signed(next()) >> 40),
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
                }
            })
            .collect()
    }

    /// Sparse synthetic FR rows on roughly a third of the cycles, with
    /// pseudo-random FULL-FIELD values in every FR column — flags included
    /// (the composed matrices are linear in the flag columns, so parity must
    /// hold pointwise on arbitrary flag values too).
    #[cfg(feature = "field-inline")]
    fn synthetic_fr_rows(log_t: usize, seed: u64) -> Vec<(usize, FieldInlineSpartanRow<Fr>)> {
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
                        flags: core::array::from_fn(|_| next()),
                    },
                )
            })
            .collect()
    }

    /// The dense image of the sparse FR rows for one FR column index (the
    /// `FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS` position) — what the fixed
    /// backend's FR view serves the reference kernel.
    #[cfg(feature = "field-inline")]
    fn fr_column_table(
        fr_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
        cycles: usize,
        index: usize,
    ) -> Vec<Fr> {
        let mut table = vec![Fr::from_u64(0); cycles];
        for (cycle, row) in fr_rows {
            table[*cycle] = row.columns()[index];
        }
        table
    }

    fn fixed_backend_from_rows(
        log_t: usize,
        rows: &[SpartanOuterRow],
        #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
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
                        fr_column_table(fr_rows, 1 << log_t, index),
                    )
                    .unwrap();
            }
            backend.set_field_inline(field_inline);
        }
        backend
    }

    /// The remainder's true input claim
    /// `Σ_{t,s} kernel · eq(τ_low, (t,s)) · Az(t,s) · Bz(t,s)`, computed
    /// through the public `jolt-r1cs` column-weight path over the COMPOSED
    /// opening selection (independent of both kernels' row-value pipelines).
    fn true_input_claim(
        rows: &[SpartanOuterRow],
        #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
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
        // Selection position → column value at cycle `t`: rv64 typed row
        // fields for the first 35 positions, the sparse FR rows behind them.
        let value = |t: usize, position: usize| -> Fr {
            if position < VARIABLE_COUNT {
                return variable_field_value(&rows[t], position);
            }
            #[cfg(feature = "field-inline")]
            {
                fr_rows.iter().find(|(cycle, _)| *cycle == t).map_or_else(
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

    /// One full parity case: uni-skip polynomial, every remainder round
    /// polynomial, typed output claims, and both kernels' derived-table
    /// validation — reference and optimized fed identical `ProverInputs`
    /// (FR-on: the composed 48-column selection over synthetic FR rows too).
    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        let rows = synthetic_rows(log_t, seed);
        #[cfg(feature = "field-inline")]
        let fr_rows = synthetic_fr_rows(log_t, seed ^ 0xF1E1D);
        let tau: Vec<Fr> = (0..log_t + 2)
            .map(|i| Fr::from_u64(3 + seed + 7 * i as u64))
            .collect();
        let backend = fixed_backend_from_rows(
            log_t,
            &rows,
            #[cfg(feature = "field-inline")]
            &fr_rows,
        );

        let mut reference_session = ProofSession::default();
        reference_session.park(SpartanOuterKernel::<Fr>::prepare(log_t, &tau, &backend).unwrap());
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[],
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedOuterUniskip::prepare_from_rows(
            &mut optimized_session,
            log_t,
            &tau,
            rows.clone(),
            #[cfg(feature = "field-inline")]
            fr_rows.clone(),
        )
        .unwrap();
        let optimized_uniskip =
            <OptimizedOuterUniskip as UniskipKernel<Fr, OuterRemainder<Fr>>>::first_round_poly(
                &OptimizedOuterUniskip,
                &mut optimized_session,
                &[],
            )
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "uni-skip first-round polynomial, log_t = {log_t}"
        );

        let r0 = Fr::from_u64(40961 + seed);
        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &fr_rows,
            &tau,
            r0,
            log_t,
        );
        let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
        let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
        let points = OuterRemainderInputClaims::<Vec<Fr>>::default();
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

    /// Synthetic parity across sizes spanning the uni-skip boundary and
    /// degenerate small domains. The sample backend only supplies the (never
    /// read) witness-plane argument of the remainder `prepare` calls.
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
            )
            .unwrap();
        assert_eq!(optimized_uniskip, reference_uniskip);

        let r0 = Fr::from_u64(9173);
        let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
        #[cfg(feature = "field-inline")]
        let fr_rows = JoltWitnessOracle::<Fr>::field_inline(backend)
            .unwrap()
            .field_inline_spartan_rows()
            .unwrap();
        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &fr_rows,
            &tau,
            r0,
            log_t,
        );

        let relation = OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
        let claims = outer_remainder_input_values_from_uniskip_output(input_claim);
        let points = OuterRemainderInputClaims::<Vec<Fr>>::default();
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

    /// Full trait-path parity: FR-off on the canned sample trace; FR-on over
    /// an FR-profile fixture trace (the sample backend carries no
    /// field-inline view), exercising the trace-backed sparse FR row seam.
    #[test]
    fn sample_trace_parity_through_the_trait_path() {
        #[cfg(not(feature = "field-inline"))]
        with_sample_backend(|backend| sample_case(backend, 2));
        #[cfg(feature = "field-inline")]
        crate::optimized::field_registers_testing::structured_fr_fixture(12)
            .with_plane(4, |backend| sample_case(backend, 4));
    }

    /// The integer extension coefficients are exactly the field Lagrange
    /// basis evaluations at the extended nodes — the fact that ties the
    /// integer pipeline to the reference's field pipeline.
    #[test]
    fn extension_coefficients_match_field_lagrange() {
        for (position, coefficients) in extension_coefficients() {
            let node = EXTENDED_START + position as i64;
            let expected = centered_lagrange_evals::<Fr>(DOMAIN, Fr::from_i64(node)).unwrap();
            for (i, &coefficient) in coefficients.iter().enumerate() {
                assert_eq!(
                    Fr::from_i64(coefficient),
                    expected[i],
                    "node {node}, basis {i}"
                );
            }
        }
    }

    /// The typed bundle's columns equal the oracle tables the reference
    /// kernel materializes — the two witness paths meeting at the shared
    /// `Extract` impls, for all 35 R1CS inputs.
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
