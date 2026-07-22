//! The reference stage-7 precommitted slot servers: each reclaims (by move)
//! the plain carry its stage-6b cycle kernel parked via `park_residue` and
//! mounts a fresh address-phase kernel over it. The kernels, the carry, and
//! the round compute live in
//! [`precommitted_reduction`](crate::precommitted_reduction); the cycle-phase
//! table builders in the per-kind `reference::*_claim_reduction` modules.

use jolt_field::Field;
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::precommitted_reduction::{AddressReductionKernel, PrecommittedReductionCarry};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The trusted-advice stage-7 slot server: reclaims the carry stage 6b's
/// cycle kernel parked and mounts the final-opening batch member.
pub struct ReferenceTrustedAdviceAddress;

impl<F: Field> PrepareKernel<F, TrustedAdviceAddressPhase<F>> for ReferenceTrustedAdviceAddress {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, TrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        let carry = session
            .take::<PrecommittedReductionCarry<F, TrustedAdviceAddressPhase<F>>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no trusted-advice reduction state for the scheduled address phase",
            })?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}

/// The untrusted-advice stage-7 slot server: reclaims the carry stage 6b's
/// cycle kernel parked and mounts the final-opening batch member.
pub struct ReferenceUntrustedAdviceAddress;

impl<F: Field> PrepareKernel<F, UntrustedAdviceAddressPhase<F>>
    for ReferenceUntrustedAdviceAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, UntrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        let carry = session
            .take::<PrecommittedReductionCarry<F, UntrustedAdviceAddressPhase<F>>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no untrusted-advice reduction state for the scheduled address phase",
            })?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}

/// The committed-bytecode stage-7 slot server: reclaims the carry stage 6b's
/// cycle kernel parked and mounts the final-opening batch member.
pub struct ReferenceBytecodeReductionAddress;

impl<F: Field> PrepareKernel<F, BytecodeReductionAddressPhase<F>>
    for ReferenceBytecodeReductionAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, BytecodeReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = BytecodeReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        let carry = session
            .take::<PrecommittedReductionCarry<F, BytecodeReductionAddressPhase<F>>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no bytecode reduction state for the scheduled address phase",
            })?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}

/// The program-image stage-7 slot server: reclaims the carry stage 6b's
/// cycle kernel parked and mounts the final-opening batch member.
pub struct ReferenceProgramImageReductionAddress;

impl<F: Field> PrepareKernel<F, ProgramImageReductionAddressPhase<F>>
    for ReferenceProgramImageReductionAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, ProgramImageReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        let carry = session
            .take::<PrecommittedReductionCarry<F, ProgramImageReductionAddressPhase<F>>>()
            .ok_or(KernelError::InvariantViolation {
                reason:
                    "stage 6b parked no program-image reduction state for the scheduled address phase",
            })?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}
