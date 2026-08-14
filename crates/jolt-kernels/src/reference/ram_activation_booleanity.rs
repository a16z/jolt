//! The RAM activation-booleanity (stage 6b, packed path) kernel: a naive
//! member over the cycle domain.
//!
//! The summand is `eq(r_cycle, j) · (B(j)² − B(j))` with `B = Load + Store`
//! the RAM activation sum (`specs/digit-zero-virtualization.md`) — zero on
//! honest traces, so the input claim is zero. `r_cycle` is the stage-1 cycle
//! binding; the verifier's `derive_output_term` pairs it positionally against
//! the raw (un-reversed) sumcheck challenges, so the eq table's big-endian
//! point is the binding reversed.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::ram::{ram_activation_load, ram_activation_store};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamActivationBooleanityPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::ram_activation_booleanity::RamActivationBooleanity;
use jolt_witness::JoltWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamActivationBooleanity<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamActivationBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamActivationBooleanity<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }
        let opening_tables = BTreeMap::from([
            (
                ram_activation_load(),
                Polynomial::new(dense_view(witness, ram_activation_load())?),
            ),
            (
                ram_activation_store(),
                Polynomial::new(dense_view(witness, ram_activation_store())?),
            ),
        ]);
        let eq_point: Vec<F> = stage1_cycle_binding.iter().rev().copied().collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamActivationBooleanityPublic::EqCycle),
            Polynomial::new(eq_table(&eq_point)),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
