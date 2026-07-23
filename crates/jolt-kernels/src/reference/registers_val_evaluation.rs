//! The registers value-evaluation (stage 5) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is `LT(j, r_cycle) · rd_inc(j) · rd_wa(r_address, j)` — the
//! "register value at `(r_address, r_cycle)` is the sum of earlier
//! increments" identity. The `rd_wa` opening is the address-bound slice of
//! the `(2^7 × T)` one-hot write-address grid (an opening-side fold), and
//! `LtCycle` is ONE multilinear: `LtPolynomial::evaluations(r_cycle)`.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::registers::{
    rd_inc_val_evaluation, rd_wa_val_evaluation,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersValEvaluationPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, LtPolynomial, Polynomial};
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{address_fold, dense_view};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RegistersValEvaluation<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != REGISTER_ADDRESS_BITS + trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(REGISTER_ADDRESS_BITS);

        // The address-bound `rd_wa` slice, folded from the one-hot grid.
        let wa_folded = address_fold(
            witness,
            rd_wa_val_evaluation(),
            trace_dimensions.log_t(),
            r_address,
        )?;

        let opening_tables = BTreeMap::from([
            (rd_wa_val_evaluation(), Polynomial::new(wa_folded)),
            (
                rd_inc_val_evaluation(),
                Polynomial::new(dense_view(witness, rd_inc_val_evaluation())?),
            ),
        ]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle),
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
