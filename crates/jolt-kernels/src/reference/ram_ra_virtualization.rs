//! The RAM RA virtualization (stage 6b) kernel: a naive member over the cycle
//! domain.
//!
//! The summand is `eq(r_cycle, j) · Π_i ra_i(chunk_i, j)`: each committed RAM
//! RA chunk selector is address-folded at its committed-width chunk of the
//! reduced stage-5 address (front-padded with zero coordinates by
//! `committed_address_chunks`).

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::ram::committed_ram_ra;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaVirtualizationPublic};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{address_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, RamRaVirtualization<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaVirtualization<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let ram_reduced_address = relation.ram_reduced_address();
        let ram_reduced_cycle = relation.ram_reduced_cycle();
        let committed_chunk_bits = relation.committed_chunk_bits();
        let chunks = committed_address_chunks(ram_reduced_address, committed_chunk_bits);
        if chunks.len() != dimensions.num_committed_ra_polys() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address chunk count disagrees with the committed RA count",
            });
        }
        let mut opening_tables = BTreeMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let _ = opening_tables.insert(
                committed_ram_ra(index),
                Polynomial::new(address_fold(
                    witness,
                    committed_ram_ra(index),
                    dimensions.log_t(),
                    chunk,
                )?),
            );
        }
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle),
            Polynomial::new(eq_table(ram_reduced_cycle)),
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
