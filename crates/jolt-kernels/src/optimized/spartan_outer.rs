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

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
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
use jolt_r1cs::constraints::jolt::{spartan_outer_constraints, spartan_outer_row_weights};
use jolt_riscv::{CircuitFlags, InstructionFlags, JoltTraceRow as TraceRow};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
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
use crate::mem::PURGE_MIN_LOG_T;
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
const SECOND_GROUP_LEN: usize = DOMAIN - 1;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const EXTENDED_NODE_COUNT: usize = DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

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

/// One cycle's integer values of the 19 eq-conditional rows, split into the
/// two uni-skip stream groups (A-side guards as `i64`, B-side magnitudes as
/// `S192` — wide enough for the `RightLookupOperand`-bearing rows, whose
/// values reach ±2^130).
struct RowGroupValues {
    a_first: [i64; DOMAIN],
    a_second: [i64; SECOND_GROUP_LEN],
    b_first: [S192; DOMAIN],
    b_second: [S192; SECOND_GROUP_LEN],
}

/// Evaluate the 19 constraint rows at one cycle with exact integer
/// arithmetic. Formulas transcribe `jolt-r1cs`'s `rv64_eq_constraint_rows`
/// verbatim (matrix semantics, not satisfied-witness shortcuts), grouped as
/// `SPARTAN_OUTER_{FIRST,SECOND}_GROUP_ROWS` orders them.
fn row_group_values(row: &SpartanOuterRow) -> RowGroupValues {
    let flag = |value: bool| i64::from(value);
    let load = flag(row.load.0);
    let store = flag(row.store.0);
    let add = flag(row.add_operands.0);
    let sub = flag(row.subtract_operands.0);
    let mul = flag(row.multiply_operands.0);
    let jump = flag(row.jump.0);
    let should_branch = flag(row.should_branch.0);

    // Rows 1, 2, 3, 4, 5, 6, 11, 14, 17, 18.
    let a_first = [
        1 - load - store,
        load,
        load,
        store,
        add + sub + mul,
        1 - add - sub - mul,
        flag(row.assert_flag.0),
        flag(row.should_jump.0),
        flag(row.virtual_instruction.0) - flag(row.is_last_in_sequence.0),
        flag(row.next_is_virtual.0) - flag(row.next_is_first_in_sequence.0),
    ];
    // Rows 0, 7, 8, 9, 10, 12, 13, 15, 16.
    let a_second = [
        load + store,
        add,
        sub,
        mul,
        1 - add - sub - mul - flag(row.advice.0),
        flag(row.write_lookup_output_to_rd.0),
        jump,
        should_branch,
        1 - should_branch - jump,
    ];

    let diff = |a: u64, b: u64| S192::from_i128(i128::from(a) - i128::from(b));
    let b_first = [
        S192::from_u64(row.ram_address.0),
        diff(row.ram_read_value.0, row.ram_write_value.0),
        diff(row.ram_read_value.0, row.rd_write_value.0),
        diff(row.rs2_value.0, row.ram_write_value.0),
        S192::from_u64(row.left_lookup_operand.0),
        diff(row.left_lookup_operand.0, row.left_instruction_input.0),
        S192::from_i128(i128::from(row.lookup_output.0) - 1),
        diff(row.next_unexpanded_pc.0, row.lookup_output.0),
        S192::from_i128(i128::from(row.next_pc.0) - i128::from(row.pc.0) - 1),
        S192::from_i64(1 - flag(row.do_not_update_unexpanded_pc.0)),
    ];

    let flag_i128 = |value: bool| i128::from(value);
    let right_lookup = S192::from_u128(row.right_lookup_operand.0);
    let right_input = S192::from_i128(row.right_instruction_input.0);
    let left_input = S192::from_u64(row.left_instruction_input.0);
    let imm = S192::from_i128(row.imm.0);
    let product_limbs = row.product.0.magnitude_limbs();
    let product = S192::new(
        [product_limbs[0], product_limbs[1], 0],
        row.product.0.is_positive,
    );
    let two_pow_64 = S192::new([0, 1, 0], true);
    let b_second = [
        S192::from_i128(i128::from(row.ram_address.0) - i128::from(row.rs1_value.0)) - imm,
        right_lookup - left_input - right_input,
        right_lookup - left_input + right_input - two_pow_64,
        right_lookup - product,
        right_lookup - right_input,
        S192::from_i128(i128::from(row.rd_write_value.0) - i128::from(row.lookup_output.0)),
        S192::from_i128(
            i128::from(row.rd_write_value.0) - i128::from(row.unexpanded_pc.0) - 4
                + 2 * flag_i128(row.is_compressed.0),
        ),
        S192::from_i128(i128::from(row.next_unexpanded_pc.0) - i128::from(row.unexpanded_pc.0))
            - imm,
        S192::from_i128(
            i128::from(row.next_unexpanded_pc.0) - i128::from(row.unexpanded_pc.0) - 4
                + 4 * flag_i128(row.do_not_update_unexpanded_pc.0)
                + 2 * flag_i128(row.is_compressed.0),
        ),
    ];

    RowGroupValues {
        a_first,
        a_second,
        b_first,
        b_second,
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
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct SpartanOuterCarry<F: JoltField> {
    log_t: usize,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    tau: Vec<F>,
    rows: BundleStore<SpartanOuterRow>,
    /// All `2·DOMAIN − 1` node values of `t1`; in-domain nodes stay zero (a
    /// satisfying witness vanishes there), matching the reference layout.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    t1_values: Vec<F>,
}

/// Extended-node evaluations of
/// `t1(Y) = Σ_{t,s} eq(τ_low, (t,s)) · Az(Y,s,t) · Bz(Y,s,t)`, with the eq
/// table factored as `E_out ⊗ E_in` and the per-cycle products from the
/// integer extension pipeline.
fn extended_t1_values<F: JoltField>(
    rows: &BundleAccess<'_, SpartanOuterRow>,
    tau_low: &[F],
) -> Result<Vec<F>, WitnessError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<F>::evals(out_point, None);
    let e_in = EqPolynomial::<F>::evals(in_point, None);
    // `in_point` always covers the stream bit (τ_low's last entry), so every
    // (cycle, stream) pair sits inside one `x_out` block.
    let pairs_per_block = e_in.len() / 2;
    let coefficients = extension_coefficients();

    let block = |x_out: usize| -> Result<Vec<F>, WitnessError> {
        let mut accumulators: Vec<<F as WithAccumulator>::SignedProductAccumulator> =
            vec![Default::default(); EXTENDED_NODE_COUNT];
        for pair in 0..pairs_per_block {
            let t = x_out * pairs_per_block + pair;
            let row = rows.row(t)?;
            let values = row_group_values(&row);
            let products = extended_products(&values, &coefficients);
            for (accumulator, (first, second)) in accumulators.iter_mut().zip(&products) {
                accumulator.fmadd_s256(e_in[2 * pair], first);
                accumulator.fmadd_s256(e_in[2 * pair + 1], second);
            }
        }
        Ok(accumulators
            .into_iter()
            .map(|accumulator| e_out[x_out] * accumulator.reduce())
            .collect())
    };
    let extended = try_par_sum_vecs(e_out.len(), EXTENDED_NODE_COUNT, block)?;

    let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
    for ((position, _), value) in extension_coefficients().iter().zip(extended) {
        t1_values[*position] = value;
    }
    Ok(t1_values)
}

/// The stage-1 uni-skip front: typed-row collection, the extended-node
/// evaluation pass, and the first-round polynomial assembly.
pub struct OptimizedOuterUniskip;

impl OptimizedOuterUniskip {
    /// The post-collection half of [`UniskipKernel::prepare`], for the
    /// in-module parity tests (which construct rows directly).
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: Vec<SpartanOuterRow>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer row count disagrees with log_t",
            });
        }
        Self::prepare_from_store(session, log_t, tau, BundleStore::Retained(rows))
    }

    /// The store-generic half of `prepare`.
    fn prepare_from_store<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        rows: BundleStore<SpartanOuterRow>,
    ) -> Result<(), KernelError<F>> {
        if tau.len() != log_t + 2 {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan outer tau must carry log_t + 2 challenges",
            });
        }
        let (tau_low, _) = tau.split_at(log_t + 1);
        let t1_values = extended_t1_values(&rows.access(), tau_low)?;
        session.park(SpartanOuterCarry {
            log_t,
            tau: tau.to_vec(),
            rows,
            t1_values,
        });
        Ok(())
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
        Self::prepare_from_store(session, log_t, tau, rows)
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
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
struct DerivedWeights<F> {
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
    az_weights: [Vec<F>; 2],
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
    bz_weights: [Vec<F>; 2],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    az_constant: [F; 2],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    bz_constant: [F; 2],
}

/// The linear-time outer remainder rounds over the joint `(cycle ‖ stream)`
/// domain (stream = index LSB, bound `LowToHigh`).
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
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
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    opening_ids: Vec<JoltOpeningId>,
    derived: DerivedWeights<F>,
}
impl<F: JoltField> OuterRemainderKernel<F> {
    fn prepare(
        carry: SpartanOuterCarry<F>,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Self, KernelError<F>> {
        let SpartanOuterCarry {
            log_t, tau, rows, ..
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
        let derived = Self::derived_weights(uniskip_challenge, opening_ids.len())?;

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
        let block = |x_out: usize,
                     az_chunk: &mut [F],
                     bz_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            for x_in in 0..in_len {
                let t = x_out * in_len + x_in;
                let row = access.row(t)?;
                let values = row_group_values(&row);
                let (az_zero, bz_zero) = fold_group(lagrange, &values.a_first, &values.b_first);
                let (az_one, bz_one) = fold_group(
                    &lagrange[..SECOND_GROUP_LEN],
                    &values.a_second,
                    &values.b_second,
                );
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
            opening_ids,
            derived,
        })
    }

    /// Az/Bz column weights at both stream values, from the same `jolt-r1cs`
    /// sources the verifier's coefficient build uses.
    fn derived_weights(
        uniskip_challenge: F,
        variable_count: usize,
    ) -> Result<DerivedWeights<F>, KernelError<F>> {
        let matrices = spartan_outer_constraints::<F>();
        let columns: Vec<usize> = (1..=variable_count).collect();
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
        self.az.bind_low_to_high_in_place(challenge);
        self.bz.bind_low_to_high_in_place(challenge);
        // Shrink dominant tails; purge once after the first shrink.
        if self.az.capacity() >= 8 * self.az.len().max(1) {
            self.az.shrink_to_fit();
            self.bz.shrink_to_fit();
            // `rounds = log_t + 1`; compare against log_t.
            if !self.purged && self.challenges.total() > PURGE_MIN_LOG_T {
                self.purged = true;
                let _ = crate::mem::release_retained_memory();
            }
        }
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    /// The 35 produced opening values at the bound cycle point: one
    /// eq-weighted walk over the typed rows (`compute_claimed_inputs`),
    /// mixed-width accumulators per input.
    #[tracing::instrument(skip_all, name = "SpartanOuter::claimed_inputs")]
    fn claimed_inputs(&self) -> Result<Vec<F>, WitnessError> {
        let reversed: Vec<F> = self.challenges.as_slice()[1..]
            .iter()
            .rev()
            .copied()
            .collect();
        let weights = {
            let _span = tracing::info_span!("SpartanOuter::claimed_input_weights").entered();
            EqPolynomial::<F>::evals(&reversed, None)
        };
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
        let claimed =
            self.claimed_inputs()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "outer opening walk re-extraction failed after the rounds",
                })?;
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
        let variable_count = self.opening_ids.len();
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
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::jolt::JoltPolynomialId;
    use jolt_claims::NoChallenges;
    use jolt_field::signed::S128;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage1::outer_remainder::{
        outer_remainder_input_values_from_uniskip_output, OuterRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::{BundleSource, FixedBackend, JoltWitnessOracle, PolynomialEncoding, Shape};

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

    fn fixed_backend_from_rows(log_t: usize, rows: &[SpartanOuterRow]) -> FixedBackend<Fr> {
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
        backend
    }

    /// The remainder's true input claim
    /// `Σ_{t,s} kernel · eq(τ_low, (t,s)) · Az(t,s) · Bz(t,s)`, computed
    /// through the public `jolt-r1cs` column-weight path (independent of both
    /// kernels' row-value pipelines).
    fn true_input_claim(rows: &[SpartanOuterRow], tau: &[Fr], r0: Fr, log_t: usize) -> Fr {
        let tau_low = &tau[..=log_t];
        let tau_high = tau[log_t + 1];
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let kernel = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
        let matrices = spartan_outer_constraints::<Fr>();
        let columns: Vec<usize> = (1..=VARIABLE_COUNT).collect();
        let mut total = Fr::from_u64(0);
        for (s, stream) in [Fr::from_u64(0), Fr::from_u64(1)].into_iter().enumerate() {
            let weights = spartan_outer_row_weights(r0, stream).unwrap();
            let weighted = matrices.weighted_columns(&weights, &columns).unwrap();
            let constants = matrices
                .public_column_contributions(&weights, 0, Fr::from_u64(1))
                .unwrap();
            for (t, row) in rows.iter().enumerate() {
                let mut az = constants.a;
                let mut bz = constants.b;
                for (index, (&a, &b)) in weighted.a.iter().zip(&weighted.b).enumerate() {
                    let value = variable_field_value(row, index);
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
    /// validation — reference and optimized fed identical `ProverInputs`.
    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        let rows = synthetic_rows(log_t, seed);
        let tau: Vec<Fr> = (0..log_t + 2)
            .map(|i| Fr::from_u64(3 + seed + 7 * i as u64))
            .collect();
        let backend = fixed_backend_from_rows(log_t, &rows);

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
        OptimizedOuterUniskip::prepare_from_rows(&mut optimized_session, log_t, &tau, rows.clone())
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
        let input_claim = true_input_claim(&rows, &tau, r0, log_t);
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

    /// Full trait-path parity on the real sample trace: the optimized bundle
    /// walk against the reference's oracle tables, with the genuine uni-skip
    /// output claim feeding the remainder (a satisfying witness, so the
    /// uni-skip reduction and the joint-domain sum agree).
    #[test]
    fn sample_trace_parity_through_the_trait_path() {
        with_sample_backend(|backend| {
            let log_t = 2usize;
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
            let optimized_uniskip = <OptimizedOuterUniskip as UniskipKernel<
                Fr,
                OuterRemainder<Fr>,
            >>::first_round_poly(
                &OptimizedOuterUniskip, &mut optimized_session, &[]
            )
            .unwrap();
            assert_eq!(optimized_uniskip, reference_uniskip);

            // The sample fixture is a witness-extraction fixture, not a
            // constraint-satisfying trace (its second row RAM-writes without
            // the Store flag), so the uni-skip reduction at r0 need not equal
            // the joint-domain sum here; the remainder is driven by the true
            // sum, which is what the naive reference self-checks against.
            let r0 = Fr::from_u64(9173);
            let rows: Vec<SpartanOuterRow> = backend.bundles().unwrap();
            let input_claim = true_input_claim(&rows, &tau, r0, log_t);

            let relation =
                OuterRemainder::new(SpartanOuterDimensions::rv64(log_t), tau.clone(), r0);
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
        });
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
