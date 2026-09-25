//! The carry claim-reduction (stage 6b) kernel: a naive member over the
//! cycle domain (implicit-carry).
//!
//! The summand is
//! `(eq(r_product, j) + γ·eq(r_shift, j) + γ²·eq(0, j)) · Carry(j)`
//! — reducing the two upstream committed `Carry` openings (product
//! virtualization, shift) together with the `carry_init` public pair (the
//! all-zeros point, claim 0) to one fresh opening at one cycle point. The
//! carry table is the committed dense trace view; each eq leaf is one
//! multilinear over its source point, the zero selector over the all-zeros
//! point.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::spartan::carry_reduced;
use jolt_claims::protocols::jolt::{CarryClaimReductionPublic, JoltDerivedId};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::carry_claim_reduction::CarryClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, CarryClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, CarryClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = CarryClaimReduction<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let cycle_points = relation.cycle_points();
        for point in cycle_points {
            if point.len() != relation.rounds() {
                return Err(KernelError::InvariantViolation {
                    reason: "carry reduction cycle point has the wrong variable count",
                });
            }
        }

        let opening_tables = BTreeMap::from([(
            carry_reduced(),
            Polynomial::new(dense_view(witness, carry_reduced())?),
        )]);
        let [product_cycle, shift_cycle] = cycle_points;
        // `eq(0, ·)` is the eq table of the all-zeros point.
        let zero_point = vec![F::zero(); relation.rounds()];
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::CarryClaimReduction(CarryClaimReductionPublic::EqCarryProduct),
                Polynomial::new(eq_table(product_cycle)),
            ),
            (
                JoltDerivedId::CarryClaimReduction(CarryClaimReductionPublic::EqCarryShift),
                Polynomial::new(eq_table(shift_cycle)),
            ),
            (
                JoltDerivedId::CarryClaimReduction(CarryClaimReductionPublic::EqZeroSelector),
                Polynomial::new(eq_table(&zero_point)),
            ),
        ]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
