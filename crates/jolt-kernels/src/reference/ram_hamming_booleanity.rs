//! The RAM Hamming-weight booleanity (stage 6b) kernel: a naive member over
//! the cycle domain.
//!
//! The summand is `eq(r_cycle, j) · (H(j)² − H(j))` with `H` the RAM
//! Hamming-weight indicator (1 when the cycle touches RAM) — zero on booleans,
//! so the input claim is zero. `r_cycle` is the stage-1 cycle binding; the
//! verifier's `derive_output_term` pairs it positionally against the raw
//! (un-reversed) sumcheck challenges, so the eq table's big-endian point is
//! the binding reversed.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_hamming_weight;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamHammingBooleanity<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamHammingBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }
        let opening_tables = BTreeMap::from([(
            ram_hamming_weight(),
            Polynomial::new(dense_view(witness, ram_hamming_weight())?),
        )]);
        let eq_point: Vec<F> = stage1_cycle_binding.iter().rev().copied().collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
            Polynomial::new(eq_table(&eq_point)),
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
