//! The RAM RA claim-reduction (stage 5) kernel: a naive member over the
//! cycle domain.
//!
//! The summand is
//! `(eq(r_cycle_raf, j) + γ·eq(r_cycle_rw, j) + γ²·eq(r_cycle_val, j)) · ra(r_address, j)`
//! — the three upstream `RamRa` openings share an address prefix (the
//! verifier's `derive_opening_points` hard-checks it), so one address-bound
//! fold of the `(K × T)` RAM `ra` grid serves all three eq-batched terms.
//! Each `EqCycle*` derived leaf is ONE multilinear: the eq table over the
//! corresponding upstream point's cycle suffix.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::ram::ram_ra_claim_reduction;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaClaimReductionPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{address_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamRaClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        inputs: ProverInputs<'_, F, RamRaClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let ram_log_k = relation.ram_log_k();
        let input_points = inputs.points;
        let expected_len = ram_log_k + trace_dimensions.log_t();
        for point in [
            input_points.raf(),
            input_points.read_write(),
            input_points.val_check(),
        ] {
            if point.len() != expected_len {
                return Err(KernelError::InvariantViolation {
                    reason: "RAM RA claim-reduction input point has the wrong variable count",
                });
            }
        }
        // The shared address prefix (the relation's `derive_opening_points`
        // hard-checks that all three inputs agree on it).
        let r_address = &input_points.read_write()[..ram_log_k];

        let ra_folded = address_fold(
            witness,
            ram_ra_claim_reduction(),
            trace_dimensions.log_t(),
            r_address,
        )?;

        let opening_tables =
            BTreeMap::from([(ram_ra_claim_reduction(), Polynomial::new(ra_folded))]);
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleRaf),
                Polynomial::new(eq_table(&input_points.raf()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                Polynomial::new(eq_table(&input_points.read_write()[ram_log_k..])),
            ),
            (
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
                Polynomial::new(eq_table(&input_points.val_check()[ram_log_k..])),
            ),
        ]);

        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            inputs.challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
