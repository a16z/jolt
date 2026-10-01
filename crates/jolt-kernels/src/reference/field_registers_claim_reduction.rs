//! Dense witness tables for the symbolic `FieldRegistersClaimReduction` relation.

use super::views::eq_table;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};
use jolt_claims::protocols::field_inline::geometry::claim_reductions::registers::claim_reduction_output_openings;
use jolt_claims::protocols::field_inline::FieldRegistersClaimReductionPublic;
use jolt_claims::protocols::field_inline::{FieldInlineDerivedId, FieldInlineOpeningId};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
use jolt_witness::{JoltWitnessPlane, WitnessError};
use std::collections::BTreeMap;

impl<F: JoltField> PrepareKernel<F, FieldRegistersClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersClaimReduction<F>>>,
        KernelError<F>,
    > {
        let oracle =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-register reference oracle",
                }))?;
        let opening_tables = claim_reduction_output_openings()
            .into_iter()
            .map(|id| {
                let FieldInlineOpeningId::Polynomial { polynomial, .. } = id;
                Ok((id, Polynomial::new(oracle.oracle_table(polynomial)?)))
            })
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            FieldInlineDerivedId::from(FieldRegistersClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(inputs.relation.tau_low())),
        )]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
