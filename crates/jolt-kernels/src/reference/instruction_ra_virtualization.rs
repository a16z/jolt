//! The instruction RA virtualization (stage 6b) kernel: a naive member over
//! the cycle domain.
//!
//! The summand is `eq(r_cycle, j) · Σ_v γ^v · Π_{i<N} ra_{N·v+i}(chunk, j)`
//! with `N = num_committed_per_virtual` (4 in the default configs): each
//! committed instruction RA chunk selector is address-folded at its own
//! committed-width chunk of the contiguous stage-5 instruction address.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::geometry::instruction::{
    committed_instruction_ra, InstructionRaVirtualizationDimensions,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationChallenges;
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessOracle;

use super::views::{address_fold, eq_table};
use crate::instruction_ra_virtualization::InstructionRaVirtualizationProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> InstructionRaVirtualizationProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: InstructionRaVirtualizationDimensions,
        instruction_r_address: &[F],
        instruction_r_cycle: &[F],
        committed_chunk_bits: usize,
        challenges: &InstructionRaVirtualizationChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = InstructionRaVirtualization::new(
            dimensions,
            instruction_r_address.to_vec(),
            instruction_r_cycle.to_vec(),
            committed_chunk_bits,
        );

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
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
