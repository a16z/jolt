//! The instruction RA virtualization (stage 6b) kernel: a naive member over
//! the cycle domain.
//!
//! The summand is `eq(r_cycle, j) · Σ_v γ^v · Π_{i<N} ra_{N·v+i}(chunk, j)`
//! with `N = num_committed_per_virtual` (4 in the default configs): each
//! committed instruction RA chunk selector is address-folded at its own
//! committed-width chunk of the contiguous stage-5 instruction address.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::instruction::committed_instruction_ra;
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::views::{address_fold, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: Field> PrepareKernel<F, InstructionRaVirtualization<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let instruction_r_address = relation.instruction_address();
        let instruction_r_cycle = relation.instruction_read_raf_cycle();
        let committed_chunk_bits = relation.committed_chunk_bits();
        let chunks = committed_address_chunks(instruction_r_address, committed_chunk_bits);
        if chunks.len() != dimensions.num_committed_ra_polys() {
            return Err(KernelError::InvariantViolation {
                reason: "instruction address chunk count disagrees with the committed RA count",
            });
        }
        let mut opening_tables = BTreeMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let _ = opening_tables.insert(
                committed_instruction_ra(index),
                Polynomial::new(address_fold(
                    witness,
                    committed_instruction_ra(index),
                    dimensions.log_t(),
                    chunk,
                )?),
            );
        }
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
            Polynomial::new(eq_table(instruction_r_cycle)),
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
