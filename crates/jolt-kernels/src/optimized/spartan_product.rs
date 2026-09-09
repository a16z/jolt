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

use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
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
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::witnesses::{
    InstructionFlag, LeftInstructionInput, LookupOutput, NextIsNoop, OpFlag, RightInstructionInput,
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

const DOMAIN: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;
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
/// integers: `|left| < 2^67` (two u64 lanes and a flag), `|right| < 2^129`
/// (the i128 lane), product `< 2^196` — inside `S256`.
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

/// The uni-skip carry: the typed rows (reused by the remainder), the low
/// challenge vector, and all extended-node values of `t1`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct SpartanProductCarry<F: JoltField> {
    log_t: usize,
    tau_low: Vec<F>,
    rows: BundleStore<SpartanProductRow>,
    t1_values: Vec<F>,
}

/// Extended-node evaluations of
/// `t1(Y) = Σ_j eq(τ_low, j) · left_Y(j) · right_Y(j)`, split-eq factored.
fn extended_t1_values<F: JoltField>(
    rows: &BundleAccess<'_, SpartanProductRow>,
    tau_low: &[F],
) -> Result<Vec<F>, WitnessError> {
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<F>::evals(out_point, None);
    let e_in = EqPolynomial::<F>::evals(in_point, None);
    let in_len = e_in.len();
    let coefficients = extension_coefficients();

    let block = |x_out: usize| -> Result<Vec<F>, WitnessError> {
        let mut accumulators: Vec<<F as WithAccumulator>::SignedProductAccumulator> =
            vec![Default::default(); EXTENDED_SIZE];
        for (x_in, &e) in e_in.iter().enumerate() {
            let row = rows.row(x_out * in_len + x_in)?;
            let products = extended_products(&row, &coefficients);
            for (accumulator, product) in accumulators.iter_mut().zip(&products) {
                accumulator.fmadd_s256(e, product);
            }
        }
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
    /// the in-module parity tests.
    #[cfg(test)]
    fn prepare_from_rows<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: Vec<SpartanProductRow>,
    ) -> Result<(), KernelError<F>> {
        if rows.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product row count disagrees with log_t",
            });
        }
        Self::prepare_from_store(session, log_t, tau_low, BundleStore::Retained(rows))
    }

    /// The store-generic half of `prepare`.
    fn prepare_from_store<F: JoltField>(
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        rows: BundleStore<SpartanProductRow>,
    ) -> Result<(), KernelError<F>> {
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan product tau_low must carry log_t challenges",
            });
        }
        let t1_values = extended_t1_values(&rows.access(), tau_low)?;
        session.park(SpartanProductCarry {
            log_t,
            tau_low: tau_low.to_vec(),
            rows,
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
        Self::prepare_from_store(session, log_t, tau_low, rows)
    }

    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::first_round_poly")]
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[F],
        _known_values: &[F],
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
        // Gruen endpoints. Left folds through the small-scalar accumulator —
        // its 5-limb window holds exactly this shape (two full-u64 lanes and
        // a flag stay under 2^319); right's i128 lane goes through the
        // signed-product path.
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
            let mut left_acc = <F as WithAccumulator>::SmallScalarAccumulator::default();
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
        let block = |x_out: usize,
                     left_chunk: &mut [F],
                     right_chunk: &mut [F]|
         -> Result<(F, F), WitnessError> {
            let mut inner_zero = F::zero();
            let mut inner_infinity = F::zero();
            for (x_in, &e) in e_in.iter().enumerate() {
                let pair = x_out * in_len + x_in;
                let (left_low, right_low) = cell(&access.row(2 * pair)?);
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

    /// The eight produced opening values at the bound cycle point: one
    /// eq-weighted walk over the typed rows, in the output claims' canonical
    /// field order.
    fn claimed_inputs(&self) -> Result<Vec<F>, WitnessError> {
        let reversed: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let weights = EqPolynomial::<F>::evals(&reversed, None);
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
        let claims: BTreeMap<JoltOpeningId, F> =
            ids.into_iter()
                .zip(self.claimed_inputs().map_err(|_| {
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
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::{BundleSource, FixedBackend, JoltWitnessOracle, PolynomialEncoding, Shape};

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

    fn fixed_backend_from_rows(log_t: usize, rows: &[SpartanProductRow]) -> FixedBackend<Fr> {
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
        backend
    }

    /// The remainder's true input claim
    /// `scale · Σ_j eq(τ_low, j) · left(j) · right(j)`, straight field math.
    fn true_input_claim(rows: &[SpartanProductRow], tau_low: &[Fr], tau_high: Fr, r0: Fr) -> Fr {
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let weights = centered_lagrange_evals::<Fr>(DOMAIN, r0).unwrap();
        let scale = centered_lagrange_kernel::<Fr>(DOMAIN, tau_high, r0).unwrap();
        let one = Fr::from_u64(1);
        let mut total = Fr::from_u64(0);
        for (row, &eq_value) in rows.iter().zip(&eq) {
            let left = weights[0] * column_field_value(row, 0)
                + weights[1] * column_field_value(row, 4)
                + weights[2] * column_field_value(row, 2);
            let right = weights[0] * column_field_value(row, 1)
                + weights[1] * column_field_value(row, 5)
                + weights[2] * (one - column_field_value(row, 6));
            total += eq_value * left * right;
        }
        scale * total
    }

    fn parity_case(dummy_plane: &dyn JoltWitnessPlane<Fr>, log_t: usize, seed: u64) {
        let rows = synthetic_rows(log_t, seed);
        let tau_low: Vec<Fr> = (0..log_t)
            .map(|i| Fr::from_u64(5 + seed + 11 * i as u64))
            .collect();
        let tau_high = Fr::from_u64(6007 + seed);
        let backend = fixed_backend_from_rows(log_t, &rows);

        let mut reference_session = ProofSession::default();
        reference_session
            .park(SpartanProductKernel::<Fr>::prepare(log_t, &tau_low, &backend).unwrap());
        let reference_uniskip =
            <ReferenceBackend as UniskipKernel<Fr, ProductRemainder<Fr>>>::first_round_poly(
                &ReferenceBackend,
                &mut reference_session,
                &[tau_high],
                &[],
            )
            .unwrap();

        let mut optimized_session = ProofSession::default();
        OptimizedProductUniskip::prepare_from_rows(
            &mut optimized_session,
            log_t,
            &tau_low,
            rows.clone(),
        )
        .unwrap();
        let optimized_uniskip =
            <OptimizedProductUniskip as UniskipKernel<Fr, ProductRemainder<Fr>>>::first_round_poly(
                &OptimizedProductUniskip,
                &mut optimized_session,
                &[tau_high],
                &[],
            )
            .unwrap();
        assert_eq!(
            optimized_uniskip, reference_uniskip,
            "product uni-skip first-round polynomial, log_t = {log_t}"
        );

        let r0 = Fr::from_u64(31337 + seed);
        let input_claim = true_input_claim(&rows, &tau_low, tau_high, r0);
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

    /// Full trait-path parity on the real sample trace, with the remainder
    /// driven by the true joint-domain sum (the sample fixture is not a
    /// constraint-satisfying trace; see the outer module's twin test).
    #[test]
    fn sample_trace_parity_through_the_trait_path() {
        with_sample_backend(|backend| {
            let log_t = 2usize;
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
                    &[],
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
                &[],
            )
            .unwrap();
            assert_eq!(optimized_uniskip, reference_uniskip);

            let r0 = Fr::from_u64(15013);
            let rows: Vec<SpartanProductRow> = backend.bundles().unwrap();
            let input_claim = true_input_claim(&rows, &tau_low, tau_high, r0);

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
        });
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
