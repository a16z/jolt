//! The registers claim-reduction (stage 3) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is
//! `eq(τ_low, j) · (rd_write_value + γ·rs1_value + γ²·rs2_value)(j)`,
//! degree 2. Its rs1/rs2 openings alias the instruction-input member's
//! (same polynomial, same point) — the generated drivers suppress the
//! duplicate absorbs.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::claim_reductions::registers::{
    rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersClaimReductionPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RegistersClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let product_uniskip_tau_low = relation.product_uniskip_tau_low();
        let ids = [
            rd_write_value_reduced(),
            rs1_value_reduced(),
            rs2_value_reduced(),
        ];
        let opening_tables = ids
            .into_iter()
            .map(|id| Ok((id, Polynomial::new(dense_view(witness, id)?))))
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(product_uniskip_tau_low)),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
