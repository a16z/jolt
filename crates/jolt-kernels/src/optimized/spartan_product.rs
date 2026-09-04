//! Optimized stage-2 Spartan product-virtualization kernels: the same
//! technique set as [`super::spartan_outer`] (typed small-scalar rows,
//! integer Lagrange extension over the centered uni-skip domain, unreduced
//! accumulation, split-eq/Gruen rounds, fused round-0 materialization,
//! in-place binding, post-hoc opening walk), on the much smaller product
//! geometry: a 3-node uni-skip window over three factor lanes, a remainder
//! over the plain cycle domain, and eight per-cycle witness columns.
//!
//! Unlike the outer uni-skip, the in-domain `t1` nodes do not vanish (they
//! are the three stage-1 product claims), so all `2·3 − 1` node values are
//! computed; the extension coefficients at in-domain nodes are the 0/1
//! Lagrange selectors, so one integer pipeline serves every node.

use std::collections::BTreeMap;

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::product::{
    composed_remainder_factor_contributions, FieldProductLaneFactors,
};
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, SpartanProductVirtualizationPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::{S128, S192, S256};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
// The COMPOSED jolt-r1cs lane domain (feature-aware): 3 rv64 lanes FR-off,
// the FR-extended 5-lane domain under `field-inline` — the same source the
// reference kernel folds with.
#[cfg(feature = "field-inline")]
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_BASE_LANES;
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
#[cfg(feature = "field-inline")]
use jolt_witness::field_inline::FieldInlineSpartanRow;
use jolt_witness::witnesses::{
    InstructionFlag, LeftInstructionInput, LookupOutput, NextIsNoop, OpFlag, RightInstructionInput,
};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "field-inline")]
use super::spartan_outer::FrRowCursor;
use super::support::{
    pin_derived_term_if_derived, try_par_sum_vecs, BundleAccess, BundleStore, GruenRoundMessage,
    RoundChallenges,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

/// The per-cycle product-virtualization witness: the three left/right factor
/// lanes plus the two wire passengers, as native small scalars.
#[derive(Clone, Copy, Debug, WitnessBundle)]
pub struct SpartanProductRow {
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: LeftInstructionInput,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: RightInstructionInput,
    #[opening(OpFlags(CircuitFlags::Jump))]
    pub jump_flag: OpFlag,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    pub write_lookup_output_to_rd: OpFlag,
    #[opening(LookupOutput)]
    pub lookup_output: LookupOutput,
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    pub branch_flag: InstructionFlag,
    #[opening(NextIsNoop)]
    pub next_is_noop: NextIsNoop,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: OpFlag,
}

/// The exact integer Lagrange coefficients `L_i(node)` of the 3-node base
/// window at every node of the extended window (in-domain nodes included —
/// there they are the 0/1 selectors).
fn extension_coefficients() -> [[i64; DOMAIN]; EXTENDED_SIZE] {
    let mut out = [[0i64; DOMAIN]; EXTENDED_SIZE];
    for (position, coefficients) in out.iter_mut().enumerate() {
        let node = EXTENDED_START + position as i64;
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
    }
    out
}

/// `left(node) · right(node)` for one cycle at every extended node, as exact
/// integers: `|left| < 2^74` (two u64 lanes and a flag, times composed
/// coefficients ≤ 2^7), `|right| < 2^137` (the i128 lane), product
/// `< 2^211` — inside `S256`. FR-INACTIVE cycles only: the FR lanes'
/// factors are all zero there, so the three rv64 coefficient slots below
/// are the whole composed sum; FR-active cycles route through
/// [`field_extended_products`].
/// `coefficients` is [`extension_coefficients`], hoisted out of the per-cycle
/// loop (its integer Lagrange build is not free at `2^23` calls).
fn extended_products(
    row: &SpartanProductRow,
    coefficients: &[[i64; DOMAIN]; EXTENDED_SIZE],
) -> [S256; EXTENDED_SIZE] {
    let mut out = [S256::zero(); EXTENDED_SIZE];
    let left_lanes = [
        i128::from(row.left_instruction_input.0),
        i128::from(row.lookup_output.0),
        i128::from(row.jump_flag.0),
    ];
    let right_wide = S192::from_i128(row.right_instruction_input.0);
    let right_flags = [
        i64::from(row.branch_flag.0),
        1 - i64::from(row.next_is_noop.0),
    ];
    for (slot, coefficients) in out.iter_mut().zip(coefficients) {
        let mut left: i128 = 0;
        for (lane, &c) in left_lanes.iter().zip(coefficients) {
            left += i128::from(c) * lane;
        }
        let mut right = S192::from_i64(coefficients[0]).mul_trunc::<3, 3>(&right_wide);
        right +=
            S192::from_i64(coefficients[1] * right_flags[0] + coefficients[2] * right_flags[1]);
        *slot = S128::from_i128(left).mul_trunc::<3, 4>(&right);
    }
    out
}
/// The field twin of [`extended_products`] for FR-ACTIVE cycles:
/// the composed left/right factor forms with the FR lane contributions
/// from the pinned jolt-claims composed-lane helper (the same fold the
/// verifier's composed checks perform).
#[cfg(feature = "field-inline")]
fn field_extended_products<F: JoltField>(
    row: &SpartanProductRow,
    fr: &FieldInlineSpartanRow<F>,
    coefficients: &[[F; DOMAIN]; EXTENDED_SIZE],
) -> [F; EXTENDED_SIZE] {
    let factors = FieldProductLaneFactors {
        rs1_value: fr.rs1_value,
        rs2_value: fr.rs2_value,
        rd_value: fr.rd_value,
    };
    let left_lanes = [
        F::from_u64(row.left_instruction_input.0),
        F::from_u64(row.lookup_output.0),
        F::from_bool(row.jump_flag.0),
    ];
    let right_lanes = [
        F::from_i128(row.right_instruction_input.0),
        F::from_bool(row.branch_flag.0),
        F::one() - F::from_bool(row.next_is_noop.0),
    ];
    let mut out = [F::zero(); EXTENDED_SIZE];
    for (slot, weights) in out.iter_mut().zip(coefficients) {
        let mut left = F::zero();
        let mut right = F::zero();
        for ((&weight, &left_lane), &right_lane) in
            weights.iter().zip(&left_lanes).zip(&right_lanes)
        {
            left += weight * left_lane;
            right += weight * right_lane;
        }
        #[expect(clippy::expect_used, reason = "the composed weights span DOMAIN nodes")]
        let (fr_left, fr_right) =
            composed_remainder_factor_contributions(weights, SPARTAN_PRODUCT_BASE_LANES, &factors)
                .expect("composed product weights cover the FR lanes");
        *slot = (left + fr_left) * (right + fr_right);
    }
    out
}

/// The field images of [`extension_coefficients`], for the FR-active cycles'
/// field path.
#[cfg(feature = "field-inline")]
fn extension_coefficient_fields<F: JoltField>() -> [[F; DOMAIN]; EXTENDED_SIZE] {
    extension_coefficients().map(|coefficients| coefficients.map(F::from_i64))
}

/// The uni-skip carry: the typed rows (reused by the remainder), the low
/// challenge vector, and all extended-node values of `t1`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SpartanProductCarry<F: JoltField> {
    log_t: usize,
    tau_low: Vec<F>,
    rows: BundleStore<SpartanProductRow>,
    /// The FR-active cycles' composed column values, sparse and sorted by
    /// cycle (only the three value columns feed the product lanes).
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    t1_values: Vec<F>,
}

/// Extended-node evaluations of
/// `t1(Y) = Σ_j eq(τ_low, j) · left_Y(j) · right_Y(j)`, split-eq factored.
fn extended_t1_values<F: JoltField>(
    rows: &BundleAccess<'_, SpartanProductRow>,
    tau_low: &[F],
    #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<F>)],
) -> Result<Vec<F>, WitnessError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<F>::evals(out_point, None);
    let e_in = EqPolynomial::<F>::evals(in_point, None);
    let in_len = e_in.len();
    let coefficients = extension_coefficients();
    #[cfg(feature = "field-inline")]
    let field_coefficients = extension_coefficient_fields::<F>();

    let block = |x_out: usize| -> Result<Vec<F>, WitnessError> {
        let mut accumulators: Vec<<F as WithAccumulator>::SignedProductAccumulator> =
            vec![Default::default(); EXTENDED_SIZE];
        #[cfg(feature = "field-inline")]
        let mut field_sums = vec![F::zero(); EXTENDED_SIZE];
        #[cfg(feature = "field-inline")]
        let mut fr_cursor = FrRowCursor::seek(fr_rows, x_out * in_len);
        for (x_in, &e) in e_in.iter().enumerate() {
            let t = x_out * in_len + x_in;
            let row = rows.row(t)?;
            #[cfg(feature = "field-inline")]
            if let Some(fr) = fr_cursor.advance(t) {
                let products = field_extended_products(&row, fr, &field_coefficients);
                for (sum, product) in field_sums.iter_mut().zip(&products) {
                    *sum += e * *product;
                }
                continue;
            }
            let products = extended_products(&row, &coefficients);
            for (accumulator, product) in accumulators.iter_mut().zip(&products) {
                accumulator.fmadd_s256(e, product);
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
    };
    try_par_sum_vecs(e_out.len(), EXTENDED_SIZE, block)
}

/// The stage-2 product uni-skip front. `prepare` runs on `τ_low` only;
/// `τ_high` arrives as the single late challenge of `first_round_poly`.
pub struct OptimizedProductUniskip;

impl OptimizedProductUniskip {
    /// The post-collection half of [`UniskipKernel::prepare`], shared with
    /// the in-module parity tests (which construct rows — and FR-on, the
    /// sparse FR rows — directly).
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: Vec<SpartanProductRow>,
        #[cfg(feature = "field-inline")] fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product row count disagrees with log_t",
            });
        }
        Self::prepare_from_store(
            session,
            log_t,
            tau_low,
            BundleStore::Retained(rows),
            #[cfg(feature = "field-inline")]
            fr_rows,
        )
    }

    /// The store-generic half of `prepare`.
    fn prepare_from_store<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: BundleStore<SpartanProductRow>,
        #[cfg(feature = "field-inline")] fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product tau_low must carry log_t challenges",
            });
        }
        let t1_values = extended_t1_values(
            &rows.access(),
            tau_low,
            #[cfg(feature = "field-inline")]
            &fr_rows,
        )?;
        session.park(SpartanProductCarry {
            log_t,
            tau_low: tau_low.to_vec(),
            rows,
            #[cfg(feature = "field-inline")]
            fr_rows,
            t1_values,
        });
        Ok(())
    }
}

impl<F: JoltField> UniskipKernel<F, ProductRemainder<F>> for OptimizedProductUniskip {
    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let rows = BundleStore::resolve(witness, 1usize << log_t)?;
        #[cfg(feature = "field-inline")]
        let fr_rows = witness
            .field_inline()
            .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                label: "composed Spartan product field-inline oracle",
            }))?
            .field_inline_spartan_rows()?;
        Self::prepare_from_store(
            session,
            log_t,
            tau_low,
            rows,
            #[cfg(feature = "field-inline")]
            fr_rows,
        )
    }

    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let &[tau_high] = late_tau else {
            return Err(KernelError::InvariantViolation {
                reason:
                    "the product uni-skip first-round polynomial expects exactly one late challenge",
            });
        };
        let carry =
            session
                .state::<SpartanProductCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason:
                        "the product uni-skip slot parked no carry for the first-round polynomial",
                })?;
        let kernel_values = centered_lagrange_evals::<F>(DOMAIN, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(DOMAIN_START, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(EXTENDED_START, &carry.t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }
}

/// The stage-2 product remainder slot: reclaims the uni-skip carry and builds
/// the linear-time round kernel.
pub struct OptimizedProductRemainder;

impl<F: JoltField> PrepareKernel<F, ProductRemainder<F>> for OptimizedProductRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        let carry =
            session
                .take::<SpartanProductCarry<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the product uni-skip slot parked no carry for the remainder member",
                })?;
        Ok(Box::new(ProductRemainderKernel::prepare(carry, &inputs)?))
    }
}

/// The linear-time product remainder rounds over the cycle domain
/// (bound `LowToHigh`).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ProductRemainderKernel<F: JoltField> {
    left: Polynomial<F>,
    right: Polynomial<F>,
    /// Whether the first-shrink purge ran.
    purged: bool,
    split_eq: GruenSplitEqPolynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pending_endpoints: Option<(F, F)>,
    challenges: RoundChallenges<F>,
    rows: BundleStore<SpartanProductRow>,
    /// The Arc-shared relation cell: the FR product appendage publishes on
    /// it at extraction.
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(skip))]
    relation: ProductRemainder<F>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    fr_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    /// `L_i(r₀)` — the values of the constant `LagrangeWeight(i)` leaves.
    lagrange_weights: Vec<F>,
}
impl<F: JoltField> ProductRemainderKernel<F> {
    fn prepare(
        carry: SpartanProductCarry<F>,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Self, KernelError<F>> {
        let SpartanProductCarry {
            log_t,
            tau_low,
            rows,
            #[cfg(feature = "field-inline")]
            fr_rows,
            ..
        } = carry;
        let rounds = inputs.relation.rounds();
        if rounds != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "product remainder rounds disagree with the uni-skip carry's log_t",
            });
        }
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let tau_high = inputs.relation.tau_high();
        let weights = centered_lagrange_evals::<F>(DOMAIN, uniskip_challenge)?;
        let scale = centered_lagrange_kernel::<F>(DOMAIN, tau_high, uniskip_challenge)?;
        let split_eq = GruenSplitEqPolynomial::new_with_scaling(
            &tau_low,
            BindingOrder::LowToHigh,
            Some(scale),
        );

        // Fused round-0 materialization: one pass over the typed rows writes
        // the weighted left/right tables and accumulates the first round's
        // Gruen endpoints. Left folds through the WIDE accumulator: two
        // full-u64 lanes exceed the small-scalar accumulator's Barrett
        // window (`reduce_nplus1` needs the magnitude under 2^318; two
        // full-range terms reach ~2^318.6, a latent wrong-answer for unlucky
        // weight draws that the composed 5-node Lagrange weights actually
        // hit). Right's i128 lane goes through the signed-product path.
        let cycles = 1usize << log_t;
        let mut left: Vec<F> = unsafe_allocate_zero_vec(cycles);
        let mut right: Vec<F> = unsafe_allocate_zero_vec(cycles);
        let e_out = split_eq.e_out_current();
        let e_in = split_eq.e_in_current();
        let in_len = e_in.len();
        let width = 2 * in_len;
        let access = rows.access();
        let weights_ref = &weights;
        let cell = |row: &SpartanProductRow| -> (F, F) {
            let mut left_acc = F::Accumulator::default();
            left_acc.fmadd_u64(weights_ref[0], row.left_instruction_input.0);
            left_acc.fmadd_u64(weights_ref[1], row.lookup_output.0);
            left_acc.fmadd_u64(weights_ref[2], u64::from(row.jump_flag.0));
            let mut right_acc = <F as WithAccumulator>::SignedProductAccumulator::default();
            right_acc.fmadd_s256(
                weights_ref[0],
                &S256::from_i128(row.right_instruction_input.0),
            );
            right_acc.fmadd_s256(
                weights_ref[1],
                &S256::from_u64(u64::from(row.branch_flag.0)),
            );
            right_acc.fmadd_s256(
                weights_ref[2],
                &S256::from_u64(1 - u64::from(row.next_is_noop.0)),
            );
            (left_acc.reduce(), right_acc.reduce())
        };
        #[cfg(feature = "field-inline")]
        let fr_rows_ref: &[(usize, FieldInlineSpartanRow<F>)] = &fr_rows;
        #[cfg(feature = "field-inline")]
        let fr_factors = |fr: &FieldInlineSpartanRow<F>| -> (F, F) {
            #[expect(clippy::expect_used, reason = "the composed weights span DOMAIN nodes")]
            composed_remainder_factor_contributions(
                weights_ref,
                SPARTAN_PRODUCT_BASE_LANES,
                &FieldProductLaneFactors {
                    rs1_value: fr.rs1_value,
                    rs2_value: fr.rs2_value,
                    rd_value: fr.rd_value,
                },
            )
            .expect("composed product weights cover the FR lanes")
        };
        let block = |x_out: usize,
                     left_chunk: &mut [F],
                     right_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            #[cfg(feature = "field-inline")]
            let mut fr_cursor = FrRowCursor::seek(fr_rows_ref, 2 * x_out * in_len);
            #[cfg(feature = "field-inline")]
            let cell =
                |t: usize, fr_cursor: &mut FrRowCursor<'_, F>| -> Result<(F, F), WitnessError> {
                    let (left, right) = cell(&access.row(t)?);
                    Ok(match fr_cursor.advance(t) {
                        Some(fr) => {
                            let (fr_left, fr_right) = fr_factors(fr);
                            (left + fr_left, right + fr_right)
                        }
                        None => (left, right),
                    })
                };
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = x_out * in_len + x_in;
                #[cfg(feature = "field-inline")]
                let (left_low, right_low) = cell(2 * pair, &mut fr_cursor)?;
                #[cfg(feature = "field-inline")]
                let (left_high, right_high) = cell(2 * pair + 1, &mut fr_cursor)?;
                #[cfg(not(feature = "field-inline"))]
                let (left_low, right_low) = cell(&access.row(2 * pair)?);
                #[cfg(not(feature = "field-inline"))]
                let (left_high, right_high) = cell(&access.row(2 * pair + 1)?);
                left_chunk[2 * x_in] = left_low;
                left_chunk[2 * x_in + 1] = left_high;
                right_chunk[2 * x_in] = right_low;
                right_chunk[2 * x_in + 1] = right_high;
                inner_zero += e * (left_low * right_low);
                inner_infinity += e * ((left_high - left_low) * (right_high - right_low));
            }
            Ok((e_out[x_out] * inner_zero, e_out[x_out] * inner_infinity))
        };
        let add = |left: (F, F), right: (F, F)| (left.0 + right.0, left.1 + right.1);

        #[cfg(feature = "parallel")]
        let endpoints = left
            .par_chunks_mut(width)
            .zip(right.par_chunks_mut(width))
            .enumerate()
            .map(|(x_out, (left_chunk, right_chunk))| block(x_out, left_chunk, right_chunk))
            .try_reduce(
                || (F::zero(), F::zero()),
                |left, right| Ok(add(left, right)),
            )?;
        #[cfg(not(feature = "parallel"))]
        let endpoints = {
            let mut folded = (F::zero(), F::zero());
            for (x_out, (left_chunk, right_chunk)) in left
                .chunks_mut(width)
                .zip(right.chunks_mut(width))
                .enumerate()
            {
                folded = add(folded, block(x_out, left_chunk, right_chunk)?);
            }
            folded
        };

        Ok(Self {
            left: Polynomial::new(left),
            right: Polynomial::new(right),
            purged: false,
            split_eq,
            pending_endpoints: Some(endpoints),
            challenges: RoundChallenges::new(rounds),
            rows,
            #[cfg(feature = "field-inline")]
            relation: inputs.relation.clone(),
            #[cfg(feature = "field-inline")]
            fr_rows,
            lagrange_weights: weights,
        })
    }

    fn bind(&mut self, challenge: F) {
        let shrunk = self.left.bind_low_to_high_in_place(challenge);
        let _ = self.right.bind_low_to_high_in_place(challenge);
        // Purge once after the first shrink.
        if shrunk && !self.purged {
            self.purged = true;
            crate::mem::purge_retained_memory(self.challenges.total());
        }
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    /// `eq(r_cycle, ·)` over the bound cycle point, shared by the base and
    /// FR opening walks below (one `2^log_T` table per extraction).
    fn cycle_weights(&self) -> Vec<F> {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&reversed, None)
    }

    /// The eight produced opening values at the bound cycle point: one
    /// eq-weighted walk over the typed rows, in the output claims' canonical
    /// field order.
    fn claimed_inputs(&self, weights: &[F]) -> Result<Vec<F>, WitnessError> {
        let cycles = weights.len();
        let access = self.rows.access();

        let block_size = 1usize << 12;
        let blocks = cycles.div_ceil(block_size);
        let block = |index: usize| -> Result<Vec<F>, WitnessError> {
            let start = index * block_size;
            let end = (start + block_size).min(cycles);
            let mut words: [<F as WithAccumulator>::SignedProductAccumulator; 3] =
                [Default::default(), Default::default(), Default::default()];
            let mut flags: [<F as WithAccumulator>::SmallScalarAccumulator; 5] = Default::default();
            for (t, &weight) in (start..end).zip(&weights[start..end]) {
                let row = access.row(t)?;
                words[0].fmadd_s256(weight, &S256::from_u64(row.left_instruction_input.0));
                words[1].fmadd_s256(weight, &S256::from_i128(row.right_instruction_input.0));
                words[2].fmadd_s256(weight, &S256::from_u64(row.lookup_output.0));
                flags[0].fmadd_u64(weight, u64::from(row.jump_flag.0));
                flags[1].fmadd_u64(weight, u64::from(row.write_lookup_output_to_rd.0));
                flags[2].fmadd_u64(weight, u64::from(row.branch_flag.0));
                flags[3].fmadd_u64(weight, u64::from(row.next_is_noop.0));
                flags[4].fmadd_u64(weight, u64::from(row.virtual_instruction.0));
            }
            let [left_input, right_input, lookup_output] = words;
            let [jump, write_lookup, branch, noop, virtual_instruction] = flags;
            Ok(vec![
                left_input.reduce(),
                right_input.reduce(),
                jump.reduce(),
                write_lookup.reduce(),
                lookup_output.reduce(),
                branch.reduce(),
                noop.reduce(),
                virtual_instruction.reduce(),
            ])
        };
        try_par_sum_vecs(blocks, 8, block)
    }

    /// The three FR factor opening values at the bound cycle point
    /// (`selected_product_remainder_output_openings` order: rs1, rs2, rd) —
    /// one eq-weighted walk over the sparse FR rows.
    #[cfg(feature = "field-inline")]
    fn fr_claimed_inputs(&self, weights: &[F]) -> [F; 3] {
        let mut values = [F::zero(); 3];
        for (cycle, row) in &self.fr_rows {
            let weight = weights[*cycle];
            values[0] += weight * row.rs1_value;
            values[1] += weight * row.rs2_value;
            values[2] += weight * row.rd_value;
        }
        values
    }
}

impl<F: JoltField> ProveRounds<F> for ProductRemainderKernel<F> {
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
            None => self.split_eq.product_endpoints(&self.left, &self.right),
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

impl<F: JoltField> SumcheckKernel<F> for ProductRemainderKernel<F> {
    type Relation = ProductRemainder<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let weights = self.cycle_weights();
        // Publish the FR product appendage on the Arc-shared relation cell:
        // the driver's curated absorb, its composed expected-output fold, and
        // the stage-2 recipe's claim assembly all read it from there.
        #[cfg(feature = "field-inline")]
        {
            use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;

            let [rs1_value, rs2_value, rd_value] = self.fr_claimed_inputs(&weights);
            self.relation
                .set_field_inline_outputs(FieldRegistersProductOutputClaims {
                    rs1_value,
                    rs2_value,
                    rd_value,
                })
                .map_err(SumcheckKernelError::Verifier)?;
        }
        let ids = [
            left_instruction_input_product(),
            right_instruction_input_product(),
            jump_flag_product(),
            write_lookup_output_to_rd_product(),
            lookup_output_product(),
            branch_flag_product(),
            next_is_noop_product(),
            virtual_instruction_product(),
        ];
        let claims: BTreeMap<JoltOpeningId, F> = ids
            .into_iter()
            .zip(self.claimed_inputs(&weights).map_err(|_| {
                SumcheckKernelError::InvariantViolation {
                    reason: "product opening walk re-extraction failed after the rounds",
                }
            })?)
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
        let ids = std::iter::once(SpartanProductVirtualizationPublic::TauKernel)
            .chain((0..DOMAIN).map(SpartanProductVirtualizationPublic::LagrangeWeight));
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let got = match public_id {
                SpartanProductVirtualizationPublic::TauKernel => self.split_eq.current_scalar(),
                SpartanProductVirtualizationPublic::LagrangeWeight(index) => {
                    self.lagrange_weights[index]
                }
                SpartanProductVirtualizationPublic::UniskipLagrangeWeight(_) => continue,
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

/// Byte parity against the reference product kernels, mirroring the outer
/// module's test structure: synthetic wide-value witnesses across sizes plus
/// the real sample trace through the full trait path.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::{
        FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    };
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::{CanonicalBytes, Fr, Ring};
    use jolt_program::execution::OwnedTrace;
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    #[cfg(feature = "field-inline")]
    use jolt_witness::FixedFieldInline;
    use jolt_witness::{BundleSource, JoltWitnessOracle};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape, TraceBackend};

    use super::*;
    use crate::reference::spartan_product::{ReferenceProductRemainder, SpartanProductKernel};
    use crate::ReferenceBackend;

    /// The eight product columns in the output claims' canonical order.
    const COLUMNS: [JoltVirtualPolynomial; 8] = [
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltVirtualPolynomial::RightInstructionInput,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
        JoltVirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        JoltVirtualPolynomial::LookupOutput,
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        JoltVirtualPolynomial::NextIsNoop,
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
    ];

    fn column_field_value(row: &SpartanProductRow, index: usize) -> Fr {
        match index {
            0 => row.left_instruction_input.to_field(),
            1 => row.right_instruction_input.to_field(),
            2 => row.jump_flag.to_field(),
            3 => row.write_lookup_output_to_rd.to_field(),
            4 => row.lookup_output.to_field(),
            5 => row.branch_flag.to_field(),
            6 => row.next_is_noop.to_field(),
            7 => row.virtual_instruction.to_field(),
            _ => unreachable!("8 product columns"),
        }
    }

    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<SpartanProductRow> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..1usize << log_t)
            .map(|_| {
                let bits = next();
                let signed = {
                    let value = next();
                    let magnitude = i128::from(value >> 1) << 60;
                    if value & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    }
                };
                SpartanProductRow {
                    left_instruction_input: LeftInstructionInput(next()),
                    right_instruction_input: RightInstructionInput(signed),
                    jump_flag: OpFlag(bits & 1 == 1),
                    write_lookup_output_to_rd: OpFlag(bits & 2 == 2),
                    lookup_output: LookupOutput(next()),
                    branch_flag: InstructionFlag(bits & 4 == 4),
                    next_is_noop: NextIsNoop(bits & 8 == 8),
                    virtual_instruction: OpFlag(bits & 16 == 16),
                }
            })
            .collect()
    }

    /// Sparse synthetic FR rows with pseudo-random full-field values in the
    /// three lane factor columns (the composed product lanes read nothing
    /// else off the FR rows).
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
                        ..FieldInlineSpartanRow::default()
                    },
                )
            })
            .collect()
    }

    fn fixed_backend_from_rows(
        log_t: usize,
        rows: &[SpartanProductRow],
        #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
    ) -> FixedBackend<Fr> {
        let mut backend = FixedBackend::new();
        for (index, variable) in COLUMNS.iter().enumerate() {
            let values: Vec<Fr> = rows
                .iter()
                .map(|row| column_field_value(row, index))
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
            for (id, value) in [
                (FieldInlineVirtualPolynomial::FieldRs1Value, 0usize),
                (FieldInlineVirtualPolynomial::FieldRs2Value, 1),
                (FieldInlineVirtualPolynomial::FieldRdValue, 2),
            ] {
                let mut table = vec![Fr::from_u64(0); 1 << log_t];
                for (cycle, row) in fr_rows {
                    table[*cycle] = [row.rs1_value, row.rs2_value, row.rd_value][value];
                }
                field_inline
                    .insert(
                        FieldInlinePolynomialId::Virtual(id),
                        Shape::new(log_t, PolynomialEncoding::Dense),
                        table,
                    )
                    .unwrap();
            }
            backend.set_field_inline(field_inline);
        }
        backend
    }

    /// The remainder's true input claim
    /// `scale · Σ_j eq(τ_low, j) · left(j) · right(j)` over the composed lane
    /// selection, straight field math.
    fn true_input_claim(
        rows: &[SpartanProductRow],
        #[cfg(feature = "field-inline")] fr_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
        tau_low: &[Fr],
        tau_high: Fr,
        r0: Fr,
    ) -> Fr {
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let weights = centered_lagrange_evals::<Fr>(DOMAIN, r0).unwrap();
        let scale = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
        let one = Fr::from_u64(1);
        let mut total = Fr::from_u64(0);
        for (j, (row, &eq_value)) in rows.iter().zip(&eq).enumerate() {
            #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
            let mut left = weights[0] * column_field_value(row, 0)
                + weights[1] * column_field_value(row, 4)
                + weights[2] * column_field_value(row, 2);
            #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
            let mut right = weights[0] * column_field_value(row, 1)
                + weights[1] * column_field_value(row, 5)
                + weights[2] * (one - column_field_value(row, 6));
            #[cfg(feature = "field-inline")]
            if let Some((_, fr)) = fr_rows.iter().find(|(cycle, _)| *cycle == j) {
                let (fr_left, fr_right) = composed_remainder_factor_contributions(
                    &weights,
                    SPARTAN_PRODUCT_BASE_LANES,
                    &FieldProductLaneFactors {
                        rs1_value: fr.rs1_value,
                        rs2_value: fr.rs2_value,
                        rd_value: fr.rd_value,
                    },
                )
                .unwrap();
                left += fr_left;
                right += fr_right;
            }
            #[cfg(not(feature = "field-inline"))]
            let _ = j;
            total += eq_value * left * right;
        }
        scale * total
    }

    struct ParityInputs {
        log_t: usize,
        seed: u64,
        rows: Vec<SpartanProductRow>,
        #[cfg(feature = "field-inline")]
        fr_rows: Vec<(usize, FieldInlineSpartanRow<Fr>)>,
        tau_low: Vec<Fr>,
        tau_high: Fr,
        r0: Fr,
    }

    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        parity_run(
            dummy_plane,
            ParityInputs {
                log_t,
                seed,
                rows: synthetic_rows(log_t, seed),
                #[cfg(feature = "field-inline")]
                fr_rows: synthetic_fr_rows(log_t, seed ^ 0xF1E1D),
                tau_low: (0..log_t)
                    .map(|i| Fr::from_u64(5 + seed + 11 * i as u64))
                    .collect(),
                tau_high: Fr::from_u64(6007 + seed),
                r0: Fr::from_u64(31337 + seed),
            },
        );
    }

    fn parity_run(dummy_plane: &dyn JoltWitnessPlane<Fr>, inputs: ParityInputs) {
        let ParityInputs {
            log_t,
            seed,
            rows,
            #[cfg(feature = "field-inline")]
            fr_rows,
            tau_low,
            tau_high,
            r0,
        } = inputs;
        let backend = fixed_backend_from_rows(
            log_t,
            &rows,
            #[cfg(feature = "field-inline")]
            &fr_rows,
        );

        let mut reference_session = ProofSession::default();
        reference_session
            .park(SpartanProductKernel::<Fr>::prepare(log_t, &tau_low, &backend).unwrap());
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, ProductRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[tau_high],
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedProductUniskip::prepare_from_rows(
            &mut optimized_session,
            log_t,
            &tau_low,
            rows.clone(),
            #[cfg(feature = "field-inline")]
            fr_rows.clone(),
        )
        .unwrap();
        let optimized_uniskip =
            <OptimizedProductUniskip as UniskipKernel<Fr, ProductRemainder<Fr>>>::first_round_poly(
                &OptimizedProductUniskip,
                &mut optimized_session,
                &[tau_high],
            )
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "product uni-skip first-round polynomial, log_t = {log_t}"
        );

        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &fr_rows,
            &tau_low,
            tau_high,
            r0,
        );
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            r0,
            tau_high,
            tau_low.clone(),
        );
        let claims = product_remainder_input_values_from_uniskip_output(input_claim);
        let points = ProductRemainderInputClaims::<Vec<Fr>>::default();
        let no_challenges = NoChallenges::<Fr>::default();

        let mut reference_kernel = ReferenceProductRemainder
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
        let mut optimized_kernel = OptimizedProductRemainder
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

        let challenges: Vec<Fr> = (0..log_t)
            .map(|i| Fr::from_u64(1201 + seed + 43 * i as u64))
            .collect();
        let mut bind = None;
        let mut previous = input_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_round = reference_kernel.prove_round(bind, round, previous).unwrap();
            let optimized_round = optimized_kernel.prove_round(bind, round, previous).unwrap();
            assert_eq!(
                optimized_round, reference_round,
                "product round {round} polynomial, log_t = {log_t}"
            );
            previous = reference_round.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = *challenges.last().unwrap();
        reference_kernel.finish_rounds(last).unwrap();
        optimized_kernel.finish_rounds(last).unwrap();

        let reference_outputs = reference_kernel.output_claims(&claims).unwrap();
        let optimized_outputs = optimized_kernel.output_claims(&claims).unwrap();
        assert_eq!(
            optimized_outputs, reference_outputs,
            "product output claims, log_t = {log_t}"
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
    fn synthetic_parity_with_reference_kernels() {
        with_sample_backend(|dummy| {
            for (log_t, seed) in [(1usize, 71u64), (2, 72), (3, 73), (4, 74)] {
                parity_case(dummy, log_t, seed);
            }
        });
    }

    /// Canonical value at or above `0.7·p` (top limb of the BN254 scalar
    /// modulus is `0x3064_4E72_E131_A029`).
    fn heavy(value: Fr) -> bool {
        let mut bytes_le = [0u8; 32];
        value.to_bytes_le(&mut bytes_le);
        u64::from_le_bytes(bytes_le[24..].try_into().unwrap()) >= 0x2200_0000_0000_0000
    }

    /// Round-0 materialization at the small-scalar accumulator's Barrett
    /// boundary: both full-u64 left lanes at `u64::MAX` under uni-skip
    /// weights whose canonical values exceed `0.7·p`, where two
    /// `fmadd_u64` terms reach ~2^318.6 and leave the `reduce_nplus1`
    /// window (2^318). The wide accumulator path must still match the
    /// reference kernel's straight field arithmetic round for round.
    #[test]
    fn full_range_left_lanes_under_heavy_uniskip_weights_match_reference() {
        let log_t = 2;
        let seed = 91;
        let rows: Vec<SpartanProductRow> = synthetic_rows(log_t, seed)
            .into_iter()
            .map(|row| SpartanProductRow {
                left_instruction_input: LeftInstructionInput(u64::MAX),
                lookup_output: LookupOutput(u64::MAX),
                jump_flag: OpFlag(true),
                ..row
            })
            .collect();
        // At integer points the centered Lagrange weights are small (signed)
        // integers, and small-integer points keep the low-degree weights
        // structured, so walk a squaring iteration (full-field after two
        // steps) until both full-u64 lane weights are heavy.
        let mut r0 = Fr::from_u64(0x9E37_79B9_7F4A_7C15);
        let r0 = (1u64..=4096)
            .map(|k| {
                r0 = r0 * r0 + Fr::from_u64(k);
                r0
            })
            .find(|&r0| {
                let weights = centered_lagrange_evals::<Fr>(DOMAIN, r0).unwrap();
                heavy(weights[0]) && heavy(weights[1])
            })
            .unwrap();
        with_sample_backend(|dummy| {
            parity_run(
                dummy,
                ParityInputs {
                    log_t,
                    seed,
                    rows,
                    #[cfg(feature = "field-inline")]
                    fr_rows: Vec::new(),
                    tau_low: vec![Fr::from_u64(5), Fr::from_u64(16)],
                    tau_high: Fr::from_u64(6007),
                    r0,
                },
            );
        });
    }

    /// The trait-path parity body over a real trace backend, with the
    /// remainder driven by the true joint-domain sum (the trace fixtures are
    /// not constraint-satisfying; see the outer module's twin test).
    fn sample_case(backend: &TraceBackend<OwnedTrace>, log_t: usize) {
        {
            let tau_low: Vec<Fr> = (0..log_t)
                .map(|i| Fr::from_u64(41 + 19 * i as u64))
                .collect();
            let tau_high = Fr::from_u64(7211);

            let mut reference_session = ProofSession::default();
            <ReferenceBackend as UniskipKernel<Fr, ProductRemainder<Fr>>>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                log_t,
                &tau_low,
                backend,
            )
            .unwrap();
            let reference_uniskip =
                <ReferenceBackend as UniskipKernel<Fr, ProductRemainder<Fr>>>::first_round_poly(
                    &ReferenceBackend,
                    &mut reference_session,
                    &[tau_high],
                )
                .unwrap();

            let mut optimized_session = ProofSession::default();
            <OptimizedProductUniskip as UniskipKernel<Fr, ProductRemainder<Fr>>>::prepare(
                &OptimizedProductUniskip,
                &mut optimized_session,
                log_t,
                &tau_low,
                backend,
            )
            .unwrap();
            let optimized_uniskip = <OptimizedProductUniskip as UniskipKernel<
                Fr,
                ProductRemainder<Fr>,
            >>::first_round_poly(
                &OptimizedProductUniskip,
                &mut optimized_session,
                &[tau_high],
            )
            .unwrap();
            assert_eq!(optimized_uniskip, reference_uniskip);

            let r0 = Fr::from_u64(15013);
            let rows: Vec<SpartanProductRow> = backend.bundles().unwrap();
            #[cfg(feature = "field-inline")]
            let fr_rows = JoltWitnessOracle::<Fr>::field_inline(backend)
                .unwrap()
                .field_inline_spartan_rows()
                .unwrap();
            let input_claim = true_input_claim(
                &rows,
                #[cfg(feature = "field-inline")]
                &fr_rows,
                &tau_low,
                tau_high,
                r0,
            );

            let relation = ProductRemainder::new(
                SpartanProductDimensions::new(log_t),
                r0,
                tau_high,
                tau_low.clone(),
            );
            let claims = product_remainder_input_values_from_uniskip_output(input_claim);
            let points = ProductRemainderInputClaims::<Vec<Fr>>::default();
            let no_challenges = NoChallenges::<Fr>::default();
            let mut reference_kernel = ReferenceProductRemainder
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
            let mut optimized_kernel = OptimizedProductRemainder
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

            let challenges: Vec<Fr> = (0..log_t)
                .map(|i| Fr::from_u64(883 + 29 * i as u64))
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
            let last = *challenges.last().unwrap();
            reference_kernel.finish_rounds(last).unwrap();
            optimized_kernel.finish_rounds(last).unwrap();
            assert_eq!(
                optimized_kernel.output_claims(&claims).unwrap(),
                reference_kernel.output_claims(&claims).unwrap()
            );
        }
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

    /// The integer extension coefficients equal the field Lagrange basis
    /// evaluations at every extended node (in-domain selectors included).
    #[test]
    fn extension_coefficients_match_field_lagrange() {
        for (position, coefficients) in extension_coefficients().iter().enumerate() {
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

    /// The typed bundle's columns equal the oracle tables for all eight
    /// product openings.
    #[test]
    fn bundle_columns_match_oracle_tables() {
        with_sample_backend(|backend| {
            let rows: Vec<SpartanProductRow> = backend.bundles().unwrap();
            for (index, variable) in COLUMNS.iter().enumerate() {
                let table: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(*variable),
                )
                .unwrap();
                let column: Vec<Fr> = rows
                    .iter()
                    .map(|row| column_field_value(row, index))
                    .collect();
                assert_eq!(column, table, "{variable:?}");
            }
        });
    }
}
