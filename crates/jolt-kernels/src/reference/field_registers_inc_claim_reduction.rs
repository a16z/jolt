use super::views::eq_table;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};
use jolt_claims::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;
use jolt_claims::protocols::field_inline::FieldRegistersIncClaimReductionPublic;
use jolt_claims::protocols::field_inline::{FieldInlineDerivedId, FieldInlineOpeningId};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ConcreteSumcheck as _;
use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
use jolt_witness::{JoltWitnessPlane, WitnessError};
use std::collections::BTreeMap;

impl<F: JoltField> PrepareKernel<F, FieldRegistersIncClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersIncClaimReduction<F>>>,
        KernelError<F>,
    > {
        let oracle =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-register reference oracle",
                }))?;
        let [read_write_cycle, val_evaluation_cycle] = inputs.relation.cycle_points();
        if [read_write_cycle, val_evaluation_cycle]
            .iter()
            .any(|point| point.len() != inputs.relation.rounds())
        {
            return Err(KernelError::InvariantViolation {
                reason:
                    "field-register increment reduction cycle point has the wrong variable count",
            });
        }
        let id = field_rd_inc_reduced();
        let FieldInlineOpeningId::Polynomial { polynomial, .. } = id;
        let opening_tables =
            BTreeMap::from([(id, Polynomial::new(oracle.oracle_table(polynomial)?))]);
        let derived_tables = [
            (
                FieldRegistersIncClaimReductionPublic::EqReadWrite,
                read_write_cycle,
            ),
            (
                FieldRegistersIncClaimReductionPublic::EqValEvaluation,
                val_evaluation_cycle,
            ),
        ]
        .into_iter()
        .map(|(id, point)| {
            (
                FieldInlineDerivedId::from(id),
                Polynomial::new(eq_table(point)),
            )
        })
        .collect();
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
