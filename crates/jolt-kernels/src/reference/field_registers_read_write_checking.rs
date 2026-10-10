use super::views::{eq_table, tile};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};
use jolt_claims::protocols::field_inline::geometry::registers::read_write_checking_output_openings;
use jolt_claims::protocols::field_inline::{FieldInlineDerivedId, FieldInlineOpeningId};
use jolt_claims::protocols::field_inline::{
    FieldInlinePolynomialId, FieldRegistersReadWritePublic,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_witness::{JoltWitnessPlane, WitnessError};
use std::collections::BTreeMap;

impl<F: JoltField> PrepareKernel<F, FieldRegistersReadWriteChecking<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersReadWriteChecking<F>>>,
        KernelError<F>,
    > {
        let oracle =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-register reference oracle",
                }))?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        // The field-inline phase split is pinned by the compile-time protocol config
        // (phase 1 = log_t, phase 2 = log_k); this kernel's binding order depends on
        // it, so a drifted config is a bug, not a capability gap.
        if dimensions.phase1_num_rounds() != dimensions.log_t()
            || dimensions.phase2_num_rounds() != dimensions.log_k()
        {
            return Err(KernelError::InvariantViolation {
                reason: "field-register read-write dimensions drifted from the config-pinned phase split",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_value;
        if r_cycle.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason:
                    "field-register read-write upstream cycle point has the wrong variable count",
            });
        }

        let copies = 1usize << dimensions.log_k();
        let opening_tables = read_write_checking_output_openings()
            .into_iter()
            .map(|id| {
                let FieldInlineOpeningId::Polynomial { polynomial, .. } = id;
                let table = oracle.oracle_table(polynomial)?;
                let table = match polynomial {
                    FieldInlinePolynomialId::Committed(_) => tile(&table, copies),
                    FieldInlinePolynomialId::Virtual(_) => table,
                };
                Ok((id, Polynomial::new(table)))
            })
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([(
            FieldInlineDerivedId::from(FieldRegistersReadWritePublic::EqCycle),
            Polynomial::new(tile(&eq_table(r_cycle), copies)),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
