//! The Spartan product-virtualization (stage 2) kernels: the product uni-skip
//! first-round polynomial and the product-remainder batch member.
//!
//! The uni-skip row polynomial
//! `t1(Y) = Σ_j eq(τ_low, j) · left_Y(j) · right_Y(j)` — with `left_Y`/`right_Y`
//! the centered-Lagrange-weighted combinations of the selected left/right
//! factor columns (the three rv64 lanes, plus the two FR lanes under
//! `field-inline`) — is brute-forced at every node of the extended centered
//! window over the COMPOSED lane domain. Unlike stage 1's outer uni-skip, the
//! in-domain values do not vanish: they equal the per-lane stage-1 claims,
//! and the engine's round-sum check pins them against the folded input claim.
//! The transmitted polynomial is `LK(τ_high, ·) × t1`.
//!
//! The rv64 remainder member needs no composite treatment: every leaf of the
//! product-remainder `Expr` is multilinear over the cycle domain (the Lagrange
//! weights are scalars — there is no stage-1-style quadratic stream
//! coefficient), so it is a plain [`NaiveSumcheckProver`], bound `LowToHigh`.
//! FR-on the member is the composed kernel at the bottom of this file.

use std::collections::BTreeMap;

#[cfg(all(feature = "allocative", feature = "field-inline"))]
use allocative::{Allocative, Key, Visitor};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::geometry::product::{
    composed_remainder_factor_contributions, FieldProductLaneFactors,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{FieldInlinePolynomialId, FieldInlineVirtualPolynomial};
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanProductVirtualizationPublic};
use jolt_field::JoltField;
use jolt_poly::lagrange::{
    centered_lagrange_evals, centered_lagrange_kernel, interpolate_to_coeffs, poly_mul,
};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
// The COMPOSED jolt-r1cs lane domain (feature-aware): identical to the
// jolt-claims RV64-only constant FR-off, the FR-extended 5-lane domain under
// `field-inline` — the shape the composed verifier checks.
#[cfg(feature = "field-inline")]
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_BASE_LANES;
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
#[cfg(feature = "field-inline")]
use jolt_sumcheck::{ProveRounds, SumcheckError};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
#[cfg(feature = "field-inline")]
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessOracle;
#[cfg(feature = "field-inline")]
use jolt_witness::WitnessError;

use super::views::{dense_view, eq_table};
use crate::uniskip::UniskipKernel;
#[cfg(not(feature = "field-inline"))]
use crate::NaiveSumcheckProver;
use crate::ProverInputs;
#[cfg(feature = "field-inline")]
use crate::SumcheckKernelError;
use crate::{KernelError, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel};
use jolt_witness::JoltWitnessPlane;
impl<F: JoltField> UniskipKernel<F, ProductRemainder<F>> for ReferenceBackend {
    /// Runs on `tau_low` only — `τ_high` is drawn after this call and reaches
    /// the slot as the single `late_tau` entry of
    /// [`first_round_poly`](UniskipKernel::first_round_poly).
    // The backend-neutral `SpartanProductUniskip::*` spans live at the
    // stage-2 call boundary (`crates/jolt-prover/src/stages/stage2.rs`), so
    // every `UniskipKernel` implementation inherits them — see the
    // taxonomy's kernel-seam contract.
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        session.park(SpartanProductKernel::prepare(log_t, tau_low, witness)?);
        Ok(())
    }

    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let &[tau_high] = late_tau else {
            return Err(KernelError::InvariantViolation {
                reason: "the product uni-skip first-round polynomial expects exactly one late challenge (τ_high)",
            });
        };
        session
            .state::<SpartanProductKernel<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "the product uni-skip slot parked no kernel for the first-round polynomial",
            })?
            .uniskip_first_round_poly(tau_high)
    }
}

/// The stage-2 remainder slot server: reclaims the [`SpartanProductKernel`]
/// the uni-skip slot parked and binds it into the batch member.
pub struct ReferenceProductRemainder;

impl<F: JoltField> PrepareKernel<F, ProductRemainder<F>> for ReferenceProductRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        session
            .take::<SpartanProductKernel<F>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "the product uni-skip slot parked no kernel for the remainder member",
            })?
            .into_remainder(&inputs)
    }
}

/// The shared product compute state: the eight cycle-indexed factor/wire
/// tables and `eq(τ_low, ·)` — everything the uni-skip polynomial and the
/// remainder member both consume.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub struct SpartanProductKernel<F: JoltField> {
    log_t: usize,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    eq_cycle: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    left_instruction_input: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    lookup_output: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    jump_flag: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    right_instruction_input: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    branch_flag: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    next_is_noop: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    write_lookup_output_to_rd: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    virtual_instruction: Vec<F>,
    /// The FR lane factor columns (`FieldRs1Value`, `FieldRs2Value`,
    /// `FieldRdValue`), cycle-indexed — the composed lanes' left/right
    /// factors per `FieldRegistersProductLane::factor_openings`.
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    field_rs1_value: Vec<F>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    field_rs2_value: Vec<F>,
    #[cfg(feature = "field-inline")]
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    field_rd_value: Vec<F>,
}

impl<F: JoltField> SpartanProductKernel<F> {
    pub fn prepare(
        log_t: usize,
        tau_low: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Self, KernelError<F>> {
        #[cfg(feature = "field-inline")]
        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "composed Spartan product field-inline oracle",
                }))?;
        #[cfg(feature = "field-inline")]
        let field_table = |polynomial: FieldInlineVirtualPolynomial| {
            field_inline.oracle_table(FieldInlinePolynomialId::Virtual(polynomial))
        };
        Ok(Self {
            log_t,
            eq_cycle: eq_table(tau_low),
            left_instruction_input: dense_view(witness, left_instruction_input_product())?,
            lookup_output: dense_view(witness, lookup_output_product())?,
            jump_flag: dense_view(witness, jump_flag_product())?,
            right_instruction_input: dense_view(witness, right_instruction_input_product())?,
            branch_flag: dense_view(witness, branch_flag_product())?,
            next_is_noop: dense_view(witness, next_is_noop_product())?,
            write_lookup_output_to_rd: dense_view(witness, write_lookup_output_to_rd_product())?,
            virtual_instruction: dense_view(witness, virtual_instruction_product())?,
            #[cfg(feature = "field-inline")]
            field_rs1_value: field_table(FieldInlineVirtualPolynomial::FieldRs1Value)?,
            #[cfg(feature = "field-inline")]
            field_rs2_value: field_table(FieldInlineVirtualPolynomial::FieldRs2Value)?,
            #[cfg(feature = "field-inline")]
            field_rd_value: field_table(FieldInlineVirtualPolynomial::FieldRdValue)?,
        })
    }

    /// The composed left/right factor values at cycle `j` under `weights`
    /// (the centered-Lagrange weights over the composed lane domain): the
    /// three ordinary lanes, plus (under `field-inline`) the FR lanes via the
    /// jolt-claims composed-lane helper — the same helper the verifier's
    /// composed checks fold with, so the lane order cannot drift.
    fn composed_lane_factors(&self, weights: &[F], j: usize) -> Result<(F, F), KernelError<F>> {
        let left = weights[0] * self.left_instruction_input[j]
            + weights[1] * self.lookup_output[j]
            + weights[2] * self.jump_flag[j];
        let right = weights[0] * self.right_instruction_input[j]
            + weights[1] * self.branch_flag[j]
            + weights[2] * (F::one() - self.next_is_noop[j]);
        #[cfg(feature = "field-inline")]
        {
            let (field_left, field_right) = composed_remainder_factor_contributions(
                weights,
                SPARTAN_PRODUCT_BASE_LANES,
                &FieldProductLaneFactors {
                    rs1_value: self.field_rs1_value[j],
                    rs2_value: self.field_rs2_value[j],
                    rd_value: self.field_rd_value[j],
                },
            )
            .ok_or(KernelError::InvariantViolation {
                reason: "composed product weights do not cover the FR lanes",
            })?;
            Ok((left + field_left, right + field_right))
        }
        #[cfg(not(feature = "field-inline"))]
        Ok((left, right))
    }

    fn uniskip_first_round_poly(&self, tau_high: F) -> Result<UnivariatePoly<F>, KernelError<F>> {
        let extended_size = 2 * SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1;
        let domain_start = -((SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
        let extended_start = -((extended_size as i64 - 1) / 2);
        let cycles = 1usize << self.log_t;

        let mut t1_values = vec![F::zero(); extended_size];
        for (position, value) in t1_values.iter_mut().enumerate() {
            let node = extended_start + position as i64;
            let node_field = if node >= 0 {
                F::from_u64(node as u64)
            } else {
                -F::from_u64(node.unsigned_abs())
            };
            let weights =
                centered_lagrange_evals::<F>(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, node_field)?;
            let mut sum = F::zero();
            for j in 0..cycles {
                let (left, right) = self.composed_lane_factors(&weights, j)?;
                sum += self.eq_cycle[j] * left * right;
            }
            *value = sum;
        }

        let kernel_values =
            centered_lagrange_evals::<F>(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)?;
        let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
        let t1_coefficients = interpolate_to_coeffs(extended_start, &t1_values);
        Ok(UnivariatePoly::new(poly_mul(
            &kernel_coefficients,
            &t1_coefficients,
        )))
    }

    /// The remainder member: the naive prover over the expanded product form.
    /// Each `LagrangeWeight(i)` leaf is the SCALAR `L_i(r₀)` (a constant
    /// table); `TauKernel` is the `LK(τ_high, r₀)`-scaled eq-cycle table.
    fn into_remainder(
        self,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        let tau_high = inputs.relation.tau_high();
        let uniskip_challenge = inputs.relation.uniskip_challenge();
        let cycles = 1usize << self.log_t;
        let weights =
            centered_lagrange_evals::<F>(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, uniskip_challenge)?;
        let scale = centered_lagrange_kernel::<F>(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            tau_high,
            uniskip_challenge,
        )?;

        // The composed member: the rv64 symbolic expression cannot name the
        // FR lane factors (separate id family), so the FR-on kernel
        // materializes the two composed weighted factor forms directly (the
        // weights are scalars, so both forms are plain multilinears).
        #[cfg(feature = "field-inline")]
        {
            let mut left_table = vec![F::zero(); cycles];
            let mut right_table = vec![F::zero(); cycles];
            for (j, (left_slot, right_slot)) in left_table
                .iter_mut()
                .zip(right_table.iter_mut())
                .enumerate()
            {
                let (left, right) = self.composed_lane_factors(&weights, j)?;
                *left_slot = left;
                *right_slot = right;
            }
            let tau_kernel_table = self
                .eq_cycle
                .iter()
                .map(|&eq| eq * scale)
                .collect::<Vec<F>>();
            let opening_tables = BTreeMap::from([
                (
                    left_instruction_input_product(),
                    Polynomial::new(self.left_instruction_input),
                ),
                (lookup_output_product(), Polynomial::new(self.lookup_output)),
                (jump_flag_product(), Polynomial::new(self.jump_flag)),
                (
                    right_instruction_input_product(),
                    Polynomial::new(self.right_instruction_input),
                ),
                (branch_flag_product(), Polynomial::new(self.branch_flag)),
                (next_is_noop_product(), Polynomial::new(self.next_is_noop)),
                (
                    write_lookup_output_to_rd_product(),
                    Polynomial::new(self.write_lookup_output_to_rd),
                ),
                (
                    virtual_instruction_product(),
                    Polynomial::new(self.virtual_instruction),
                ),
            ]);
            Ok(Box::new(ComposedProductRemainderKernel {
                relation: inputs.relation.clone(),
                tau_kernel: Polynomial::new(tau_kernel_table),
                left: Polynomial::new(left_table),
                right: Polynomial::new(right_table),
                opening_tables,
                field_inline_tables: [
                    Polynomial::new(self.field_rs1_value),
                    Polynomial::new(self.field_rs2_value),
                    Polynomial::new(self.field_rd_value),
                ],
                rounds_bound: 0,
            }))
        }

        #[cfg(not(feature = "field-inline"))]
        {
            let mut derived_tables = BTreeMap::new();
            let _ = derived_tables.insert(
                JoltDerivedId::from(SpartanProductVirtualizationPublic::TauKernel),
                Polynomial::new(
                    self.eq_cycle
                        .iter()
                        .map(|&eq| eq * scale)
                        .collect::<Vec<F>>(),
                ),
            );
            for (index, &weight) in weights.iter().enumerate() {
                let _ = derived_tables.insert(
                    JoltDerivedId::from(SpartanProductVirtualizationPublic::LagrangeWeight(index)),
                    Polynomial::new(vec![weight; cycles]),
                );
            }

            let opening_tables = BTreeMap::from([
                (
                    left_instruction_input_product(),
                    Polynomial::new(self.left_instruction_input),
                ),
                (lookup_output_product(), Polynomial::new(self.lookup_output)),
                (jump_flag_product(), Polynomial::new(self.jump_flag)),
                (
                    right_instruction_input_product(),
                    Polynomial::new(self.right_instruction_input),
                ),
                (branch_flag_product(), Polynomial::new(self.branch_flag)),
                (next_is_noop_product(), Polynomial::new(self.next_is_noop)),
                (
                    write_lookup_output_to_rd_product(),
                    Polynomial::new(self.write_lookup_output_to_rd),
                ),
                (
                    virtual_instruction_product(),
                    Polynomial::new(self.virtual_instruction),
                ),
            ]);

            Ok(Box::new(NaiveSumcheckProver::new(
                inputs,
                opening_tables,
                derived_tables,
                BindingOrder::LowToHigh,
            )?))
        }
    }
}

/// The composed (field-inline) stage-2 product-remainder member.
///
/// Proves `TauKernel · LEFT · RIGHT` over the cycle domain with the two
/// weighted factor forms spanning the composed 5-lane selection (3 ordinary +
/// 2 FR lanes). The rv64 symbolic expression cannot name the FR lane factors
/// (a separate id family, per the protocol ruling), so this kernel
/// materializes `LEFT`/`RIGHT` as dense tables — exact, because the Lagrange
/// weights are scalars, so both forms are plain multilinears and their bound
/// values equal the verifier's weight-folded openings (tied down per proof by
/// [`SumcheckKernel::validate_derived_tables`] and the driver's composed
/// expected-output fold).
///
/// The eight ordinary opening tables and the three FR factor tables ride
/// along (bound in lockstep) for the typed extraction and the FR product
/// appendage: once fully bound, the kernel publishes the three FR opening
/// values on the (`Arc`-shared) relation cell the driver's curated absorb and
/// the stage-2 recipe read.
#[cfg(feature = "field-inline")]
struct ComposedProductRemainderKernel<F: JoltField> {
    relation: ProductRemainder<F>,
    tau_kernel: Polynomial<F>,
    left: Polynomial<F>,
    right: Polynomial<F>,
    opening_tables: BTreeMap<JoltOpeningId, Polynomial<F>>,
    /// The FR factor tables, in `selected_product_remainder_output_openings`
    /// order: `FieldRs1Value`, `FieldRs2Value`, `FieldRdValue`.
    field_inline_tables: [Polynomial<F>; 3],
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(all(feature = "allocative", feature = "field-inline"))]
impl<F: JoltField> Allocative for ComposedProductRemainderKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            Key::new("tau_kernel"),
            self.tau_kernel.len() * size_of::<F>(),
        );
        visitor.visit_simple(Key::new("left"), self.left.len() * size_of::<F>());
        visitor.visit_simple(Key::new("right"), self.right.len() * size_of::<F>());
        visitor.visit_simple(
            Key::new("opening_tables"),
            self.opening_tables
                .values()
                .map(|table| table.len() * size_of::<F>())
                .sum::<usize>(),
        );
        visitor.visit_simple(
            Key::new("field_inline_tables"),
            self.field_inline_tables
                .iter()
                .map(|table| table.len() * size_of::<F>())
                .sum::<usize>(),
        );
        visitor.exit();
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> ComposedProductRemainderKernel<F> {
    fn remaining_rounds(&self) -> usize {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        self.tau_kernel
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.left
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.right
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        for table in self.opening_tables.values_mut() {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        for table in &mut self.field_inline_tables {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(SumcheckKernelError::NotFullyBound { remaining }),
        }
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> ProveRounds<F> for ComposedProductRemainderKernel<F> {
    fn num_rounds(&self) -> usize {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;

        if let Some(challenge) = bind {
            self.bind_tables(challenge);
        }
        let half = (1usize << self.remaining_rounds()) / 2;
        let degree = self.relation.degree();
        let order = BindingOrder::LowToHigh;
        let mut evals = Vec::with_capacity(degree + 1);
        for sample in 0..=degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    self.tau_kernel
                        .sumcheck_round_eval_with_order(y, point, order)
                        * self.left.sumcheck_round_eval_with_order(y, point, order)
                        * self.right.sumcheck_round_eval_with_order(y, point, order)
                })
                .sum::<F>();
            evals.push(sum);
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

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind_tables(bind);
        Ok(())
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> SumcheckKernel<F> for ComposedProductRemainderKernel<F> {
    type Relation = ProductRemainder<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, ProductRemainder<F>>,
    ) -> Result<SumcheckOutputClaims<F, ProductRemainder<F>>, SumcheckKernelError<F>> {
        use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;
        use jolt_claims::{InputClaims as _, OutputClaims as _};

        self.require_fully_bound()?;
        // Publish the FR product appendage on the Arc-shared relation cell:
        // the driver's curated absorb, its composed expected-output fold, and
        // the stage-2 recipe's claim assembly all read it from there.
        let [rs1_value, rs2_value, rd_value] = &self.field_inline_tables;
        self.relation
            .set_field_inline_outputs(FieldRegistersProductOutputClaims {
                rs1_value: rs1_value.evals()[0],
                rs2_value: rs2_value.evals()[0],
                rd_value: rd_value.evals()[0],
            })
            .map_err(SumcheckKernelError::Verifier)?;

        let opening_tables = &self.opening_tables;
        SumcheckOutputClaims::<F, ProductRemainder<F>>::from_opening_values(|id| {
            opening_tables
                .get(id)
                .map(|table| table.evals()[0])
                .or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    /// Ties the materialized tables to the verifier's scalar path: the bound
    /// `TauKernel` must equal `derive_output_term(TauKernel)`, and the bound
    /// `LEFT`/`RIGHT` factor forms must equal the verifier's Lagrange-weight
    /// scalars folded over the bound lane columns — ordinary lanes plus the
    /// jolt-claims composed-lane helper's FR contributions, the same fold the
    /// composed expected-output check performs.
    fn validate_derived_tables(
        &self,
        relation: &ProductRemainder<F>,
        input_points: &SumcheckInputPoints<F, ProductRemainder<F>>,
        output_points: &SumcheckOutputPoints<F, ProductRemainder<F>>,
        challenges: &ConcreteSumcheckChallenges<F, ProductRemainder<F>>,
    ) -> Result<(), SumcheckKernelError<F>> {
        use jolt_verifier::stages::relations::ConcreteSumcheck as _;

        self.require_fully_bound()?;
        let resolve = |public: SpartanProductVirtualizationPublic| {
            relation.derive_output_term(
                &JoltDerivedId::from(public),
                input_points,
                output_points,
                challenges,
            )
        };
        let expected_tau_kernel = resolve(SpartanProductVirtualizationPublic::TauKernel)?;
        let got_tau_kernel = self.tau_kernel.evals()[0];
        if got_tau_kernel != expected_tau_kernel {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id: JoltDerivedId::from(SpartanProductVirtualizationPublic::TauKernel),
                expected: expected_tau_kernel,
                got: got_tau_kernel,
            });
        }

        let weights = (0..SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE)
            .map(|index| resolve(SpartanProductVirtualizationPublic::LagrangeWeight(index)))
            .collect::<Result<Vec<F>, _>>()?;
        let bound = |id: JoltOpeningId| -> Result<F, SumcheckKernelError<F>> {
            self.opening_tables
                .get(&id)
                .map(|table| table.evals()[0])
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "composed product kernel is missing a lane opening table",
                })
        };
        let [rs1_value, rs2_value, rd_value] = &self.field_inline_tables;
        let (field_left, field_right) = composed_remainder_factor_contributions(
            &weights,
            SPARTAN_PRODUCT_BASE_LANES,
            &FieldProductLaneFactors {
                rs1_value: rs1_value.evals()[0],
                rs2_value: rs2_value.evals()[0],
                rd_value: rd_value.evals()[0],
            },
        )
        .ok_or(SumcheckKernelError::InvariantViolation {
            reason: "composed product weights do not cover the FR lanes",
        })?;
        let expected_left = weights[0] * bound(left_instruction_input_product())?
            + weights[1] * bound(lookup_output_product())?
            + weights[2] * bound(jump_flag_product())?
            + field_left;
        let expected_right = weights[0] * bound(right_instruction_input_product())?
            + weights[1] * bound(branch_flag_product())?
            + weights[2] * (F::one() - bound(next_is_noop_product())?)
            + field_right;
        for (label, expected, got) in [
            ("LEFT", expected_left, self.left.evals()[0]),
            ("RIGHT", expected_right, self.right.evals()[0]),
        ] {
            if got != expected {
                return Err(SumcheckKernelError::Verifier(
                    VerifierError::StageClaimSumcheckFailed {
                        stage: "SpartanProductVirtualization".to_string(),
                        reason: format!(
                            "composed {label} factor form bound to {got:?}, but the \
                             verifier's weight fold gives {expected:?}"
                        ),
                    },
                ));
            }
        }
        Ok(())
    }
}
