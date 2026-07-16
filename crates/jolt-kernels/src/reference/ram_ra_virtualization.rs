//! The RAM RA virtualization (stage 6b) kernel: a naive member over the cycle
//! domain.
//!
//! The summand is `eq(r_cycle, j) · Π_i ra_i(chunk_i, j)`: each committed RAM
//! RA chunk selector is address-folded at its committed-width chunk of the
//! reduced stage-5 address (front-padded with zero coordinates by
//! `committed_address_chunks`).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::ram::{
    committed_ram_ra, RamRaVirtualizationDimensions,
};
use jolt_claims::protocols::jolt::{JoltDerivedId, RamRaVirtualizationPublic};
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::{address_fold, eq_table};
use crate::ram_ra_virtualization::RamRaVirtualizationProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> RamRaVirtualizationProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: RamRaVirtualizationDimensions,
        ram_reduced_address: &[F],
        ram_reduced_cycle: &[F],
        committed_chunk_bits: usize,
        challenges: &NoChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRaVirtualization<F>>>, KernelError<F>> {
        let relation = RamRaVirtualization::new(
            dimensions,
            ram_reduced_address.to_vec(),
            ram_reduced_cycle.to_vec(),
            committed_chunk_bits,
        );

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
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
