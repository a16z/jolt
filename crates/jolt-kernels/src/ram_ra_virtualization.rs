//! The RAM RA virtualization slot: stage 6b.

use jolt_claims::protocols::jolt::geometry::ram::RamRaVirtualizationDimensions;
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-6b RAM RA virtualization slot.
pub trait RamRaVirtualizationProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: RamRaVirtualizationDimensions,
        ram_reduced_address: &[F],
        ram_reduced_cycle: &[F],
        committed_chunk_bits: usize,
        challenges: &NoChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaVirtualization<F>>>, KernelError<F>>;
}
