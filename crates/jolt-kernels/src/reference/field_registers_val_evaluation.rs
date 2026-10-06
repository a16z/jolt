use super::views::eq_table;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend,
    SumcheckKernel,
};
use jolt_claims::protocols::field_inline::geometry::registers::val_evaluation_output_openings;
use jolt_claims::protocols::field_inline::FieldInlineDerivedId;
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
    FieldRegistersValEvaluationPublic, FIELD_REGISTERS_LOG_K,
};
use jolt_field::JoltField;
use jolt_poly::LtPolynomial;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage5::field_registers_val_evaluation::FieldRegistersValEvaluation;
use jolt_witness::{JoltWitnessPlane, WitnessError};
use std::collections::BTreeMap;

impl<F: JoltField> PrepareKernel<F, FieldRegistersValEvaluation<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = FieldRegistersValEvaluation<F>>>, KernelError<F>>
    {
        let oracle =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-register reference oracle",
                }))?;
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != FIELD_REGISTERS_LOG_K + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "field-register value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(FIELD_REGISTERS_LOG_K);

        let wa_grid = oracle.oracle_table(FieldInlinePolynomialId::Virtual(
            FieldInlineVirtualPolynomial::FieldRdWa,
        ))?;
        let cycles = 1usize << log_t;
        if wa_grid.len() != cycles << FIELD_REGISTERS_LOG_K {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdWa".to_owned(),
                expected: cycles << FIELD_REGISTERS_LOG_K,
                got: wa_grid.len(),
            });
        }
        let eq_address = eq_table(r_address);
        let wa_folded: Vec<F> = (0..cycles)
            .map(|j| {
                eq_address
                    .iter()
                    .enumerate()
                    .map(|(k, eq)| *eq * wa_grid[(k << log_t) | j])
                    .sum()
            })
            .collect();
        let rd_inc = oracle.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;

        let [inc_id, wa_id] = val_evaluation_output_openings();
        let opening_tables = BTreeMap::from([
            (inc_id, Polynomial::new(rd_inc)),
            (wa_id, Polynomial::new(wa_folded)),
        ]);
        let derived_tables = BTreeMap::from([(
            FieldInlineDerivedId::from(FieldRegistersValEvaluationPublic::LtCycle),
            Polynomial::new(LtPolynomial::evaluations(r_cycle)),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
