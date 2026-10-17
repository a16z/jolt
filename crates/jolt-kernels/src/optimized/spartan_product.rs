//! Optimized stage-2 Spartan product-virtualization kernels: the same
//! technique set as [`super::spartan_outer`] (typed small-scalar rows,
//! integer Lagrange extension over the centered uni-skip domain, unreduced
//! accumulation, split-eq/Gruen rounds, a streamed prefix before the fused
//! materialization, in-place binding, post-hoc split-eq opening walk), on the
//! much smaller product geometry: a 3-node uni-skip window over three factor
//! lanes, a remainder over the plain cycle domain, and eight per-cycle
//! witness columns.
//!
//! Unlike the outer uni-skip, the in-domain `t1` nodes do not vanish (they
//! are the three stage-1 product claims), so all `2·3 − 1` node values are
//! computed; the extension coefficients at in-domain nodes are the 0/1
//! Lagrange selectors, so one integer pipeline serves every node.
//!
//! Every pass reads rows through the witness's row handle, split into tasks of
//! whole Gruen groups, at most `TASKS_PER_THREAD` per pool thread
//! (`task_ranges`), each visiting its rows once, serially. While the two dense
//! left/right tables exceed `PRODUCT_DENSE_CAP`, round `j` recomputes its
//! pairs from the rows with the uni-skip weights scaled by `eq(r_1..r_j, ·)`
//! (LowToHigh binding gives `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`); the
//! first round whose tables fit fills them in the same pass that yields its
//! endpoints.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedOpeningId;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::product::selected_product_remainder_output_openings;
use jolt_verifier::stages::relations::OpeningIdOf;
use std::collections::BTreeMap;
use std::ops::Range;

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::geometry::SPARTAN_PRODUCT_BASE_LANES;
use jolt_claims::protocols::composed::geometry::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::product::{
    composed_remainder_factor_contributions, FieldProductLaneFactors,
};
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanProductVirtualizationPublic};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::signed::{S128, S192, S256};
use jolt_field::{Accumulator as _, JoltField, WithAccumulator};
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly,
};
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::stages::stage2::product_uniskip::ProductUniskip;
#[cfg(feature = "field-inline")]
use jolt_witness::field_inline::FieldInlineSpartanRow;
use jolt_witness::witnesses::{
    InstructionFlag, LeftInstructionInput, LookupOutput, NextIsNoop, OpFlag, RightInstructionInput,
};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

#[cfg(feature = "field-inline")]
use super::spartan_outer::{FieldInlineRowCursor, FieldSpartanCarry};
#[cfg(feature = "field-inline")]
use super::support::map_reduce_chunks;
use super::support::{
    dense_crossover, gruen_pair_pass, pin_derived_term_if_derived, product_endpoint_terms,
    task_ranges, try_par_sum_vecs, GruenRoundMessage, KernelRows, PairHalves, RoundChallenges,
    VISIT_ROWS,
};
use crate::uniskip::UniskipKernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
const EXTENDED_SIZE: usize = 2 * DOMAIN - 1;
const DOMAIN_START: i64 = -((DOMAIN as i64 - 1) / 2);
const EXTENDED_START: i64 = -((EXTENDED_SIZE as i64 - 1) / 2);

/// Byte cap on the two dense left/right tables: the remainder recomputes each
/// round's pairs from the rows until the current round's tables fit
/// ([`dense_crossover`]).
const PRODUCT_DENSE_CAP: usize = 4 << 30;

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

impl SpartanProductRow {
    /// The `[left, right]` factor forms of this cycle under per-lane
    /// `weights`, plus an active field-inline cycle's composed lane
    /// contributions. Left folds through the WIDE accumulator: two full-u64
    /// lanes exceed the small-scalar accumulator's Barrett window
    /// (`reduce_nplus1` needs the magnitude under 2^318; two full-range terms
    /// reach ~2^318.6, a latent wrong-answer for unlucky weight draws that the
    /// composed 5-node Lagrange weights actually hit). Right's i128 lane goes
    /// through the signed-product path.
    fn factors<F: JoltField>(
        &self,
        #[cfg(feature = "field-inline")] field_row: Option<&FieldInlineSpartanRow<F>>,
        weights: &[F; DOMAIN],
    ) -> [F; 2] {
        let mut left = F::Accumulator::default();
        left.fmadd_u64(weights[0], self.left_instruction_input.0);
        left.fmadd_u64(weights[1], self.lookup_output.0);
        left.fmadd_u64(weights[2], u64::from(self.jump_flag.0));
        let mut right = <F as WithAccumulator>::SignedProductAccumulator::default();
        right.fmadd_i128(weights[0], self.right_instruction_input.0);
        right.fmadd_u64(weights[1], u64::from(self.branch_flag.0));
        right.fmadd_u64(weights[2], 1 - u64::from(self.next_is_noop.0));
        let factors = [left.reduce(), right.reduce()];
        #[cfg(feature = "field-inline")]
        if let Some(field_row) = field_row {
            #[expect(clippy::expect_used, reason = "the composed weights span DOMAIN nodes")]
            let (field_left, field_right) = composed_remainder_factor_contributions(
                weights,
                SPARTAN_PRODUCT_BASE_LANES,
                &FieldProductLaneFactors {
                    rs1_value: field_row.rs1_value,
                    rs2_value: field_row.rs2_value,
                    rd_value: field_row.rd_value,
                },
            )
            .expect("composed product weights cover the field-inline lanes");
            return [factors[0] + field_left, factors[1] + field_right];
        }
        factors
    }
}

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

/// `left(node) · right(node)` for one cycle at every extended node, as exact integers:
/// `|left| < 2^74` (two u64 lanes and a flag, times composed coefficients ≤ 2^7),
/// `|right| < 2^137` (the i128 lane), product `< 2^211` — inside `S256`. For cycles
/// without field-inline activity: the field-inline lanes' factors are all zero there,
/// so the three rv64 coefficient slots below are the whole composed sum; active
/// field-inline cycles route through [`field_extended_products`]. `coefficients` is
/// [`extension_coefficients`], hoisted out of the per-cycle loop (its integer Lagrange
/// build is not free at `2^23` calls).
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
/// The field twin of [`extended_products`] for active field-inline cycles: the composed
/// left/right factor forms with the field-inline lane contributions from the pinned
/// jolt-claims composed-lane helper (the same fold the verifier's composed checks
/// perform).
#[cfg(feature = "field-inline")]
fn field_extended_products<F: JoltField>(
    row: &SpartanProductRow,
    field_row: &FieldInlineSpartanRow<F>,
    coefficients: &[[F; DOMAIN]; EXTENDED_SIZE],
) -> [F; EXTENDED_SIZE] {
    let factors = FieldProductLaneFactors {
        rs1_value: field_row.rs1_value,
        rs2_value: field_row.rs2_value,
        rd_value: field_row.rd_value,
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
        let (field_left, field_right) =
            composed_remainder_factor_contributions(weights, SPARTAN_PRODUCT_BASE_LANES, &factors)
                .expect("composed product weights cover the field-inline lanes");
        *slot = (left + field_left) * (right + field_right);
    }
    out
}

#[cfg(feature = "field-inline")]
fn extension_coefficient_fields<F: JoltField>() -> [[F; DOMAIN]; EXTENDED_SIZE] {
    extension_coefficients().map(|coefficients| coefficients.map(F::from_i64))
}

/// The uni-skip carry: the row handle (revisited by the remainder's passes
/// and the final opening walk), `τ_low`, and the extended-node evaluations of
/// `t1`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SpartanProductCarry<F: JoltField> {
    log_t: usize,
    tau_low: Vec<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<SpartanProductRow>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    t1_values: Vec<F>,
}

fn extended_t1_values<F: JoltField>(
    rows: &KernelRows<SpartanProductRow>,
    cycles: usize,
    tau_low: &[F],
    #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<F>)],
) -> Result<Vec<F>, WitnessError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<F>::evals(out_point, None);
    let e_in = EqPolynomial::<F>::evals(in_point, None);
    let group = e_in.len();
    let coefficients = extension_coefficients();
    #[cfg(feature = "field-inline")]
    let field_coefficients = extension_coefficient_fields::<F>();
    let ranges = task_ranges(cycles, group);

    try_par_sum_vecs(ranges.len(), EXTENDED_SIZE, |task| {
        let range = ranges[task].clone();
        let mut t1_values = vec![F::zero(); EXTENDED_SIZE];
        #[cfg(feature = "field-inline")]
        let mut field_cursor = FieldInlineRowCursor::seek(field_rows, range.start);
        // A callback of at most `group` aligned rows lies in one `e_out`
        // block, and its accumulators take at most VISIT_ROWS terms.
        rows.visit(range, VISIT_ROWS.min(group), |start, bundles| {
            let mut accumulators =
                [<F as WithAccumulator>::SignedProductAccumulator::default(); EXTENDED_SIZE];
            #[cfg(feature = "field-inline")]
            let mut field_sums = [F::zero(); EXTENDED_SIZE];
            for ((t, row), &e) in (start..).zip(bundles).zip(&e_in[start % group..]) {
                #[cfg(feature = "field-inline")]
                if let Some(field_row) = field_cursor.advance(t) {
                    let products = field_extended_products(row, field_row, &field_coefficients);
                    for (sum, product) in field_sums.iter_mut().zip(&products) {
                        *sum += e * *product;
                    }
                    continue;
                }
                #[cfg(not(feature = "field-inline"))]
                let _ = t;
                let products = extended_products(row, &coefficients);
                for (accumulator, product) in accumulators.iter_mut().zip(&products) {
                    accumulator.fmadd_s256(e, product);
                }
            }
            let scale = e_out[start / group];
            for (value, accumulator) in t1_values.iter_mut().zip(accumulators) {
                *value += scale * accumulator.reduce();
            }
            #[cfg(feature = "field-inline")]
            for (value, field_sum) in t1_values.iter_mut().zip(field_sums) {
                *value += scale * field_sum;
            }
            Ok(())
        })?;
        Ok(t1_values)
    })
}

/// The stage-2 product uni-skip front. `prepare` runs on `τ_low` only;
/// `τ_high` arrives as the single late challenge of `first_round_poly`.
pub struct OptimizedProductUniskip;

impl OptimizedProductUniskip {
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: Vec<SpartanProductRow>,
        #[cfg(feature = "field-inline")] field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product row count disagrees with log_t",
            });
        }
        Self::prepare_with_rows(
            session,
            log_t,
            tau_low,
            KernelRows::Synthetic(rows),
            #[cfg(feature = "field-inline")]
            field_rows,
        )
    }

    fn prepare_with_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: KernelRows<SpartanProductRow>,
        #[cfg(feature = "field-inline")] field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    ) -> Result<(), KernelError<F>> {
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product tau_low must carry log_t challenges",
            });
        }
        let t1_values = extended_t1_values(
            &rows,
            1 << log_t,
            tau_low,
            #[cfg(feature = "field-inline")]
            &field_rows,
        )?;
        session.park(SpartanProductCarry {
            log_t,
            tau_low: tau_low.to_vec(),
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            t1_values,
        });
        Ok(())
    }
}

impl<F: JoltField> UniskipKernel<F, ProductRemainder<F>, SumcheckInputClaims<F, ProductUniskip<F>>>
    for OptimizedProductUniskip
{
    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        let rows = KernelRows::from_witness(witness, log_t, "Spartan product trace rows")?;
        #[cfg(feature = "field-inline")]
        let field_rows = if let Some(carry) = session.take::<FieldSpartanCarry<F>>() {
            carry.0
        } else {
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "composed Spartan product field-inline oracle",
                }))?
                .field_inline_spartan_rows()?
        };
        Self::prepare_with_rows(
            session,
            log_t,
            tau_low,
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
        )
    }

    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[F],
        _inputs: &SumcheckInputClaims<F, ProductUniskip<F>>,
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
        Ok(Box::new(ProductRemainderKernel::prepare(
            carry,
            &inputs,
            PRODUCT_DENSE_CAP,
        )?))
    }
}

/// The left/right factor tables over the cycle domain.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum ProductTables<F: JoltField> {
    /// Rounds before the crossover recompute their pairs from the rows.
    Streamed,
    /// Filled by the crossover round's pass, bound in place afterwards.
    Dense {
        left: Polynomial<F>,
        right: Polynomial<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ProductRemainderKernel<F: JoltField> {
    tables: ProductTables<F>,
    /// The round whose pass fills `tables`; the round count when they never
    /// fit the cap.
    dense_round: usize,
    purged: bool,
    split_eq: GruenSplitEqPolynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pending_endpoints: Option<(F, F)>,
    challenges: RoundChallenges<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<SpartanProductRow>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    field_rows: Vec<(usize, FieldInlineSpartanRow<F>)>,
    lagrange_weights: Vec<F>,
}
impl<F: JoltField> ProductRemainderKernel<F> {
    /// `dense_cap` bounds the bytes of the two dense tables; production passes
    /// [`PRODUCT_DENSE_CAP`].
    fn prepare(
        carry: SpartanProductCarry<F>,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
        dense_cap: usize,
    ) -> Result<Self, KernelError<F>> {
        let SpartanProductCarry {
            log_t,
            tau_low,
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
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
        let table_bytes = (2 * size_of::<F>()) << log_t;
        let mut kernel = Self {
            tables: ProductTables::Streamed,
            dense_round: dense_crossover(table_bytes, dense_cap, rounds),
            purged: false,
            split_eq,
            pending_endpoints: None,
            challenges: RoundChallenges::new(rounds),
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            lagrange_weights: weights,
        };
        if kernel.dense_round == 0 {
            kernel.pending_endpoints = Some(kernel.fill_tables()?);
        }
        Ok(kernel)
    }

    fn cycles(&self) -> usize {
        1 << self.challenges.total()
    }

    /// The current round's pass, filling the dense tables: each of length
    /// `T / 2^round`, allocated once at its final capacity and written in
    /// disjoint task slices.
    fn fill_tables(&mut self) -> Result<(F, F), WitnessError> {
        let len = self.cycles() >> self.challenges.bound();
        let mut left: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut right: Vec<F> = unsafe_allocate_zero_vec(len);
        let [zero, infinity] = self.round_pass(
            Some([&mut left[..], &mut right[..]]),
            product_endpoint_terms,
        )?;
        self.tables = ProductTables::Dense {
            left: Polynomial::new(left),
            right: Polynomial::new(right),
        };
        Ok((zero, infinity))
    }

    /// Lane weights of the current round `j`'s pass, one per cycle offset `c`
    /// within a pair half: the uni-skip Lagrange weights times
    /// `eq(r_1..r_j, c)`, offset bit `i` pairing `r_{i+1}`. Before the
    /// crossover, round `j`'s `2^j` offsets stay below `table_bytes / cap`.
    fn round_weights(&self) -> Vec<[F; DOMAIN]> {
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&point, None)
            .into_iter()
            .map(|eq| std::array::from_fn(|lane| eq * self.lagrange_weights[lane]))
            .collect()
    }

    /// One pass of the current round `j` over its pairs of `2^(j+1)` cycles
    /// ([`gruen_pair_pass`]), filling `tables` when given.
    fn round_pass<const N: usize>(
        &self,
        tables: Option<[&mut [F]; 2]>,
        terms: impl Fn([[F; 2]; 2]) -> [F; N] + Sync,
    ) -> Result<[F; N], WitnessError> {
        let weights = self.round_weights();
        gruen_pair_pass(
            &self.split_eq,
            self.challenges.bound() + 1,
            tables,
            terms,
            |range, chunk, sink| {
                self.walk_pairs(range, chunk, &weights, |pair, halves| {
                    sink.emit(pair, halves);
                })
            },
        )
    }

    /// One task's serial walk of `range` in the current round `j`: each row's
    /// factor forms, weighted for its offset within its half, add to the low
    /// or high half of its `2^(j+1)`-cycle pair ([`PairHalves`]), and `emit`
    /// receives every completed pair's `[left, right]` halves in order.
    fn walk_pairs(
        &self,
        range: Range<usize>,
        chunk: usize,
        weights: &[[F; DOMAIN]],
        mut emit: impl FnMut(usize, [[F; 2]; 2]),
    ) -> Result<(), WitnessError> {
        #[cfg(feature = "field-inline")]
        let mut field_cursor = FieldInlineRowCursor::seek(&self.field_rows, range.start);
        let mut halves = PairHalves::new(self.challenges.bound());
        self.rows.visit(range, chunk, |start, bundles| {
            for (t, row) in (start..).zip(bundles) {
                let factors = row.factors(
                    #[cfg(feature = "field-inline")]
                    field_cursor.advance(t),
                    &weights[halves.offset(t)],
                );
                if let Some((pair, pair_halves)) = halves.add(t, factors) {
                    emit(pair, pair_halves);
                }
            }
            Ok(())
        })
    }

    fn bind(&mut self, challenge: F) {
        if let ProductTables::Dense { left, right } = &mut self.tables {
            let shrunk = left.bind_low_to_high_in_place(challenge);
            let _ = right.bind_low_to_high_in_place(challenge);
            if shrunk && !self.purged {
                self.purged = true;
                crate::mem::purge_retained_memory(self.challenges.total());
            }
        }
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
        self.pending_endpoints = None;
    }

    /// `eq(r_cycle, ·)` as split tables over the cycle bits: the bound
    /// challenges reversed into big-endian order.
    fn claimed_input_weights(&self) -> TensorEqTable<F> {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        TensorEqTable::new(&reversed)
    }

    /// The eight produced opening values at the bound cycle point: one
    /// split-eq-weighted walk over the typed rows, in the output claims'
    /// canonical field order.
    fn claimed_inputs(&self, weights: &TensorEqTable<F>) -> Result<Vec<F>, WitnessError> {
        let (e_out, e_in) = (weights.e_out(), weights.e_in());
        let group = e_in.len();
        let ranges = task_ranges(weights.len(), group);
        try_par_sum_vecs(ranges.len(), 8, |task| {
            let mut claimed = vec![F::zero(); 8];
            // A callback of at most `group` aligned rows lies in one `e_out`
            // block, and its accumulators take at most VISIT_ROWS terms.
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(group),
                |start, bundles| {
                    let mut words: [<F as WithAccumulator>::SignedProductAccumulator; 3] =
                        Default::default();
                    let mut flags: [<F as WithAccumulator>::SmallScalarAccumulator; 5] =
                        Default::default();
                    for (row, &weight) in bundles.iter().zip(&e_in[start % group..]) {
                        words[0].fmadd_u64(weight, row.left_instruction_input.0);
                        words[1].fmadd_i128(weight, row.right_instruction_input.0);
                        words[2].fmadd_u64(weight, row.lookup_output.0);
                        flags[0].fmadd_u64(weight, u64::from(row.jump_flag.0));
                        flags[1].fmadd_u64(weight, u64::from(row.write_lookup_output_to_rd.0));
                        flags[2].fmadd_u64(weight, u64::from(row.branch_flag.0));
                        flags[3].fmadd_u64(weight, u64::from(row.next_is_noop.0));
                        flags[4].fmadd_u64(weight, u64::from(row.virtual_instruction.0));
                    }
                    let [left_input, right_input, lookup_output] = words;
                    let [jump, write_lookup, branch, noop, virtual_instruction] = flags;
                    let sums = [
                        left_input.reduce(),
                        right_input.reduce(),
                        jump.reduce(),
                        write_lookup.reduce(),
                        lookup_output.reduce(),
                        branch.reduce(),
                        noop.reduce(),
                        virtual_instruction.reduce(),
                    ];
                    let scale = e_out[start / group];
                    for (value, sum) in claimed.iter_mut().zip(sums) {
                        *value += scale * sum;
                    }
                    Ok(())
                },
            )?;
            Ok(claimed)
        })
    }

    /// The three field-inline factor opening values at the bound cycle point
    /// (`selected_product_remainder_output_openings` order: rs1, rs2, rd) — one
    /// eq-weighted walk over the sparse field-inline rows.
    #[cfg(feature = "field-inline")]
    fn field_claimed_inputs(&self, weights: &TensorEqTable<F>) -> [F; 3] {
        map_reduce_chunks(
            self.field_rows.len(),
            1 << 12,
            |range| {
                let mut values = [F::zero(); 3];
                for (cycle, row) in &self.field_rows[range] {
                    let weight = weights.evaluate_index(*cycle);
                    values[0] += weight * row.rs1_value;
                    values[1] += weight * row.rs2_value;
                    values[2] += weight * row.rd_value;
                }
                values
            },
            |a, b| std::array::from_fn(|i| a[i] + b[i]),
            || [F::zero(); 3],
        )
    }
}

/// A re-extraction failure inside a round, after `prepare` visited the rows.
fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "Spartan product trace rows",
    }
}

impl<F: JoltField> ProveRounds<F> for ProductRemainderKernel<F> {
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
        let (q_zero, q_infinity) = if let Some(endpoints) = self.pending_endpoints.take() {
            endpoints
        } else if let ProductTables::Dense { left, right } = &self.tables {
            self.split_eq.product_endpoints(left, right)
        } else if self.challenges.bound() == self.dense_round {
            self.fill_tables().map_err(rows_unavailable)?
        } else {
            let [zero, infinity] = self
                .round_pass(None, product_endpoint_terms)
                .map_err(rows_unavailable)?;
            (zero, infinity)
        };
        // A streamed round's `q(1)` fallback is one more pass, which can fail:
        // the failure is held and returned below, so its placeholder zero never
        // reaches a round message.
        let mut failure = None;
        let message =
            self.split_eq
                .checked_cubic(q_zero, q_infinity, previous_claim, round, || {
                    match &self.tables {
                        ProductTables::Dense { left, right } => {
                            self.split_eq.product_at_one(left, right)
                        }
                        ProductTables::Streamed => self
                            .round_pass(None, |[left, right]| [left[1] * right[1]])
                            .map_or_else(
                                |error| {
                                    failure = Some(error);
                                    F::zero()
                                },
                                |[at_one]| at_one,
                            ),
                    }
                });
        match failure {
            Some(error) => Err(rows_unavailable(error)),
            None => message,
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for ProductRemainderKernel<F> {
    type Relation = ProductRemainder<F>;

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
        let weights = self.claimed_input_weights();
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
        let claims: BTreeMap<OpeningIdOf<F, Self::Relation>, F> = ids
            .into_iter()
            .map(Into::into)
            .zip(self.claimed_inputs(&weights).map_err(|_| {
                SumcheckKernelError::InvariantViolation {
                    reason: "product opening walk re-extraction failed after the rounds",
                }
            })?)
            .collect();
        #[cfg(feature = "field-inline")]
        let claims: BTreeMap<_, _> = claims
            .into_iter()
            .chain(
                selected_product_remainder_output_openings()
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use crate::optimized::parity::ExceptionalEq;
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::composed::{ComposedClaims, FieldProductUniskipInputs};
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::{
        FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    };
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{CanonicalBytes, Fr, Ring};
    use jolt_verifier::stages::stage2::product_remainder::product_remainder_input_values_from_uniskip_output;
    use jolt_verifier::stages::stage2::product_uniskip::ProductUniskipInputClaims;
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    #[cfg(feature = "field-inline")]
    use jolt_witness::FixedFieldInline;
    use jolt_witness::{BundleSource, JoltWitnessOracle};
    use jolt_witness::{
        FixedBackend, JoltVmWitnessInputs, PolynomialEncoding, Shape, TraceBackend,
    };

    use super::*;
    use crate::optimized::registers_read_write::test_support::structured_fixture;
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

    fn column_field_value<F: JoltField>(row: &SpartanProductRow, index: usize) -> F {
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
                        ..FieldInlineSpartanRow::default()
                    },
                )
            })
            .collect()
    }

    fn fixed_backend_from_rows(
        log_t: usize,
        rows: &[SpartanProductRow],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<Fr>)],
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
                for (cycle, row) in field_rows {
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

    fn true_input_claim<F: JoltField>(
        rows: &[SpartanProductRow],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<F>)],
        tau_low: &[F],
        tau_high: F,
        r0: F,
    ) -> F {
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let weights = centered_lagrange_evals::<F>(DOMAIN, r0).unwrap();
        let scale = centered_lagrange_kernel::<F>(DOMAIN, tau_high, r0).unwrap();
        let mut total = F::zero();
        for (j, (row, &eq_value)) in rows.iter().zip(&eq).enumerate() {
            let value = |index| column_field_value::<F>(row, index);
            #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
            let mut left = weights[0] * value(0) + weights[1] * value(4) + weights[2] * value(2);
            #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
            let mut right =
                weights[0] * value(1) + weights[1] * value(5) + weights[2] * (F::one() - value(6));
            #[cfg(feature = "field-inline")]
            if let Some((_, field_row)) = field_rows.iter().find(|(cycle, _)| *cycle == j) {
                let (field_left, field_right) = composed_remainder_factor_contributions(
                    &weights,
                    SPARTAN_PRODUCT_BASE_LANES,
                    &FieldProductLaneFactors {
                        rs1_value: field_row.rs1_value,
                        rs2_value: field_row.rs2_value,
                        rd_value: field_row.rd_value,
                    },
                )
                .unwrap();
                left += field_left;
                right += field_right;
            }
            #[cfg(not(feature = "field-inline"))]
            let _ = j;
            total += eq_value * left * right;
        }
        scale * total
    }

    fn uniskip_input_claims<F: JoltField>(
        rows: &[SpartanProductRow],
        #[cfg(feature = "field-inline")] field_rows: &[(usize, FieldInlineSpartanRow<F>)],
        tau_low: &[F],
    ) -> SumcheckInputClaims<F, ProductUniskip<F>> {
        let at = |lane| {
            let node = F::from_i64(DOMAIN_START + lane);
            true_input_claim(
                rows,
                #[cfg(feature = "field-inline")]
                field_rows,
                tau_low,
                node,
                node,
            )
        };
        let inputs = ProductUniskipInputClaims {
            product: at(0),
            should_branch: at(1),
            should_jump: at(2),
        };
        #[cfg(feature = "field-inline")]
        let inputs = ComposedClaims {
            base: inputs,
            field_inline: FieldProductUniskipInputs {
                product: at(SPARTAN_PRODUCT_BASE_LANES as i64),
                inv_product: at(SPARTAN_PRODUCT_BASE_LANES as i64 + 1),
            },
        };
        inputs
    }

    struct ParityInputs {
        log_t: usize,
        seed: u64,
        rows: Vec<SpartanProductRow>,
        #[cfg(feature = "field-inline")]
        field_rows: Vec<(usize, FieldInlineSpartanRow<Fr>)>,
        tau_low: Vec<Fr>,
        tau_high: Fr,
        r0: Fr,
    }

    /// The optimized remainder over the parked uni-skip carry, its two dense
    /// tables capped at exactly their size at round `depth`, so they first fit
    /// there (and fit no round when `depth` is the round count).
    fn capped_remainder<F: JoltField>(
        session: &mut ProofSession,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
        log_t: usize,
        depth: usize,
    ) -> ProductRemainderKernel<F> {
        let carry = session.take::<SpartanProductCarry<F>>().unwrap();
        let cap = ((2 * size_of::<F>()) << log_t) >> depth;
        ProductRemainderKernel::prepare(carry, inputs, cap).unwrap()
    }

    /// Drives both remainders through every round on the reference's
    /// messages, asserting equal round polynomials and that the optimized
    /// tables materialize at round `depth` with `T / 2^depth` entries each,
    /// or never when `depth` is the round count.
    fn lockstep_rounds<F: JoltField>(
        reference: &mut dyn SumcheckKernel<F, Relation = ProductRemainder<F>>,
        optimized: &mut ProductRemainderKernel<F>,
        challenges: &[F],
        input_claim: F,
        depth: usize,
    ) {
        let dense_len = |kernel: &ProductRemainderKernel<F>| match &kernel.tables {
            ProductTables::Dense { left, right } => {
                assert_eq!(left.evals().len(), right.evals().len());
                Some(left.evals().len())
            }
            ProductTables::Streamed => None,
        };
        let mut materialized = dense_len(optimized).map(|len| (0, len));
        let mut bind = None;
        let mut previous = input_claim;
        for (round, &challenge) in challenges.iter().enumerate() {
            let reference_round = reference.prove_round(bind, round, previous).unwrap();
            let optimized_round = optimized.prove_round(bind, round, previous).unwrap();
            assert_eq!(
                optimized_round, reference_round,
                "round {round} polynomial, depth {depth}"
            );
            materialized = materialized.or_else(|| dense_len(optimized).map(|len| (round, len)));
            previous = reference_round.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference.finish_rounds(last).unwrap();
        optimized.finish_rounds(last).unwrap();
        let rounds = challenges.len();
        let expected = (depth < rounds).then(|| (depth, (1 << rounds) >> depth));
        assert_eq!(materialized, expected, "dense tables, depth {depth}");
    }

    fn parity_case_with_eq(
        dummy_plane: &dyn JoltWitnessPlane<Fr>,
        log_t: usize,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
        zero_scale: bool,
        depth: usize,
    ) {
        parity_run(
            dummy_plane,
            ParityInputs {
                log_t,
                seed,
                rows: synthetic_rows(log_t, seed),
                #[cfg(feature = "field-inline")]
                field_rows: synthetic_field_rows(log_t, seed ^ 0xF1E1D),
                tau_low: exceptional.map_or_else(
                    || {
                        (0..log_t)
                            .map(|i| Fr::from_u64(5 + seed + 11 * i as u64))
                            .collect()
                    },
                    |case| case.point(log_t, Fr::from_u64(1201 + seed)),
                ),
                tau_high: Fr::from_u64(if zero_scale { 0 } else { 6007 + seed }),
                r0: Fr::from_u64(if zero_scale { 1 } else { 31337 + seed }),
            },
            depth,
        );
    }

    fn parity_run(dummy_plane: &dyn JoltWitnessPlane<Fr>, inputs: ParityInputs, depth: usize) {
        let ParityInputs {
            log_t,
            seed,
            rows,
            #[cfg(feature = "field-inline")]
            field_rows,
            tau_low,
            tau_high,
            r0,
        } = inputs;
        let backend = fixed_backend_from_rows(
            log_t,
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
        );
        let uniskip_inputs = uniskip_input_claims(
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
            &tau_low,
        );

        let mut reference_session = ProofSession::default();
        reference_session
            .park(SpartanProductKernel::<Fr>::prepare(log_t, &tau_low, &backend).unwrap());
        let reference_uniskip = ReferenceBackend
            .first_round_poly(&mut reference_session, &[tau_high], &uniskip_inputs)
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedProductUniskip::prepare_from_rows(
            &mut optimized_session,
            log_t,
            &tau_low,
            rows.clone(),
            #[cfg(feature = "field-inline")]
            field_rows.clone(),
        )
        .unwrap();
        let optimized_uniskip = OptimizedProductUniskip
            .first_round_poly(&mut optimized_session, &[tau_high], &uniskip_inputs)
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "product uni-skip first-round polynomial, log_t = {log_t}"
        );

        let input_claim = true_input_claim(
            &rows,
            #[cfg(feature = "field-inline")]
            &field_rows,
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
        let points = SumcheckInputPoints::<Fr, ProductRemainder<Fr>>::default();
        let no_challenges = NoChallenges::<Fr>::default();
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &no_challenges,
        };

        let mut reference_kernel = ReferenceProductRemainder
            .prepare(&mut reference_session, dummy_plane, inputs())
            .unwrap();
        let mut optimized_kernel =
            capped_remainder(&mut optimized_session, &inputs(), log_t, depth);

        let challenges: Vec<Fr> = (0..log_t)
            .map(|i| Fr::from_u64(1201 + seed + 43 * i as u64))
            .collect();
        lockstep_rounds(
            reference_kernel.as_mut(),
            &mut optimized_kernel,
            &challenges,
            input_claim,
            depth,
        );

        let reference_outputs = reference_kernel.output_claims(&claims).unwrap();
        let optimized_outputs = optimized_kernel.output_claims(&claims).unwrap();
        assert_eq!(
            optimized_outputs, reference_outputs,
            "product output claims, log_t = {log_t}, depth {depth}"
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

    /// Every Gruen exceptional endpoint, in dense rounds, in a streamed prefix
    /// with a mid-domain crossover, and with every round streamed.
    #[test]
    fn remainder_matches_reference_at_exceptional_eq_and_zero_scaling() {
        with_sample_backend(|dummy| {
            for depth in [0, 2, 4] {
                for case in ExceptionalEq::ALL {
                    parity_case_with_eq(dummy, 4, 379, Some(case), false, depth);
                }
                parity_case_with_eq(dummy, 4, 379, None, true, depth);
            }
        });
    }

    /// Wide synthetic rows at every crossover round, including none.
    #[test]
    fn synthetic_parity_with_reference_kernels() {
        with_sample_backend(|dummy| {
            for (log_t, seed) in [(1usize, 71u64), (2, 72), (3, 73), (6, 76)] {
                for depth in 0..=log_t {
                    parity_case_with_eq(dummy, log_t, seed, None, false, depth);
                }
            }
        });
    }

    fn heavy(value: Fr) -> bool {
        let mut bytes_le = [0u8; 32];
        value.to_bytes_le(&mut bytes_le);
        u64::from_le_bytes(bytes_le[24..].try_into().unwrap()) >= 0x2200_0000_0000_0000
    }

    /// Factor folds at the small-scalar accumulator's Barrett
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
            for depth in 0..=log_t {
                parity_run(
                    dummy,
                    ParityInputs {
                        log_t,
                        seed,
                        rows: rows.clone(),
                        #[cfg(feature = "field-inline")]
                        field_rows: Vec::new(),
                        tau_low: vec![Fr::from_u64(5), Fr::from_u64(16)],
                        tau_high: Fr::from_u64(6007),
                        r0,
                    },
                    depth,
                );
            }
        });
    }

    /// The trait-path parity body over a real trace backend, with the
    /// remainder driven by the true joint-domain sum (the trace fixtures are
    /// not constraint-satisfying; see the outer module's twin test).
    fn sample_case(backend: &TraceBackend, log_t: usize) {
        {
            let tau_low: Vec<Fr> = (0..log_t)
                .map(|i| Fr::from_u64(41 + 19 * i as u64))
                .collect();
            let tau_high = Fr::from_u64(7211);
            let rows: Vec<SpartanProductRow> = backend.bundles().unwrap();
            #[cfg(feature = "field-inline")]
            let field_rows = JoltWitnessOracle::<Fr>::field_inline(backend)
                .unwrap()
                .field_inline_spartan_rows()
                .unwrap();
            let uniskip_inputs = uniskip_input_claims(
                &rows,
                #[cfg(feature = "field-inline")]
                &field_rows,
                &tau_low,
            );

            let mut reference_session = ProofSession::default();
            <ReferenceBackend as UniskipKernel<
                Fr,
                ProductRemainder<Fr>,
                SumcheckInputClaims<Fr, ProductUniskip<Fr>>,
            >>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                log_t,
                &tau_low,
                backend,
            )
            .unwrap();
            let reference_uniskip = ReferenceBackend
                .first_round_poly(&mut reference_session, &[tau_high], &uniskip_inputs)
                .unwrap();

            let mut optimized_session = ProofSession::default();
            <OptimizedProductUniskip as UniskipKernel<
                Fr,
                ProductRemainder<Fr>,
                SumcheckInputClaims<Fr, ProductUniskip<Fr>>,
            >>::prepare(
                &OptimizedProductUniskip,
                &mut optimized_session,
                log_t,
                &tau_low,
                backend,
            )
            .unwrap();
            let optimized_uniskip = OptimizedProductUniskip
                .first_round_poly(&mut optimized_session, &[tau_high], &uniskip_inputs)
                .unwrap();
            assert_eq!(optimized_uniskip, reference_uniskip);

            let r0 = Fr::from_u64(15013);
            let input_claim = true_input_claim(
                &rows,
                #[cfg(feature = "field-inline")]
                &field_rows,
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
            let points = SumcheckInputPoints::<Fr, ProductRemainder<Fr>>::default();
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

    /// Uni-skip, every remainder round and the output claims over the
    /// witness's row handle, the tables first fitting at round `depth`,
    /// against the reference.
    #[cfg(not(feature = "field-inline"))]
    fn trace_rows_case<F: JoltField>(backend: &TraceBackend, log_t: usize, depth: usize) {
        let tau_low: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(37 + 23 * i as u64))
            .collect();
        let tau_high = F::from_u64(7019);
        let rows: Vec<SpartanProductRow> = backend.bundles().unwrap();
        let uniskip_inputs = uniskip_input_claims(&rows, &tau_low);
        let mut reference_session = ProofSession::default();
        <ReferenceBackend as UniskipKernel<
            F,
            ProductRemainder<F>,
            SumcheckInputClaims<F, ProductUniskip<F>>,
        >>::prepare(
            &ReferenceBackend,
            &mut reference_session,
            log_t,
            &tau_low,
            backend,
        )
        .unwrap();
        let mut optimized_session = ProofSession::default();
        OptimizedProductUniskip
            .prepare(&mut optimized_session, log_t, &tau_low, backend)
            .unwrap();
        assert_eq!(
            OptimizedProductUniskip
                .first_round_poly(&mut optimized_session, &[tau_high], &uniskip_inputs)
                .unwrap(),
            ReferenceBackend
                .first_round_poly(&mut reference_session, &[tau_high], &uniskip_inputs)
                .unwrap(),
            "uni-skip first-round polynomial, log_t = {log_t}"
        );

        let r0 = F::from_u64(15013);
        let input_claim = true_input_claim(&rows, &tau_low, tau_high, r0);
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            r0,
            tau_high,
            tau_low.clone(),
        );
        let claims = product_remainder_input_values_from_uniskip_output(input_claim);
        let points = SumcheckInputPoints::<F, ProductRemainder<F>>::default();
        let no_challenges = NoChallenges::<F>::default();
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &no_challenges,
        };
        let mut reference_kernel = ReferenceProductRemainder
            .prepare(&mut reference_session, backend, inputs())
            .unwrap();
        let mut optimized_kernel =
            capped_remainder(&mut optimized_session, &inputs(), log_t, depth);
        let challenges: Vec<F> = (0..log_t)
            .map(|i| F::from_u64(883 + 29 * i as u64))
            .collect();
        lockstep_rounds(
            reference_kernel.as_mut(),
            &mut optimized_kernel,
            &challenges,
            input_claim,
            depth,
        );
        assert_eq!(
            optimized_kernel.output_claims(&claims).unwrap(),
            reference_kernel.output_claims(&claims).unwrap(),
            "typed output claims, log_t = {log_t}, depth {depth}"
        );
    }

    /// Task boundaries and a padded proof suffix fall inside the domain;
    /// every crossover round, including none, in both fields.
    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn streamed_rounds_over_trace_rows_match_reference() {
        for log_t in [1, 3, 6] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |backend| {
                for depth in 0..=log_t {
                    trace_rows_case::<Fr>(backend, log_t, depth);
                    #[cfg(feature = "akita")]
                    trace_rows_case::<Prime128OffsetA7F7>(backend, log_t, depth);
                }
            });
        }
    }

    /// A witness without a row handle fails typed instead of collecting rows.
    #[test]
    fn missing_rows_are_typed() {
        let log_t = 3;
        let tau_low: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64(5 + i as u64)).collect();
        structured_fixture(7).with_plane(log_t, |backend| {
            // A configured domain shorter than the trace leaves no valid handle.
            let mut shortened = TraceBackend::new(
                backend.config.clone(),
                JoltVmWitnessInputs::new(
                    &backend.program,
                    &backend.preprocessing,
                    backend.trace.clone(),
                ),
            );
            shortened.config.log_t = 1;
            assert!(matches!(
                OptimizedProductUniskip.prepare(
                    &mut ProofSession::default(),
                    log_t,
                    &tau_low,
                    &shortened
                ),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
        });
    }

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
