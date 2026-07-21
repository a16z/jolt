//! The [`BackendPreparer`]: the one place backend slot names are spelled.
//!
//! The generated `prove_clear` drivers name only the dependency-inverted
//! [`PrepareSumcheck`] bound (jolt-verifier cannot depend on the kernel
//! crate); this context struct implements it for every member relation by
//! forwarding the member's [`ProverInputs`] bundle to the matching
//! [`JoltBackend`] slot. Universal slots forward directly; a slot whose data
//! is not oracle-shaped reads it through the two non-`ProverInputs` channels
//! `PrepareKernel::prepare` receives — the witness plane's typed-row
//! accessors and the `ProofSession` carries — never through a stage-supplied
//! side context.

use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_verifier::stages::relations::{
    PrepareSumcheck, ProverInputs, SumcheckKernel, SumcheckPreparer,
};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::ProverError;

/// The preparer the stage recipes hand to the generated `prove_clear`
/// drivers: the backend value, the proof-scoped session, the witness plane
/// (the oracle provider plus the typed-row accessors kernels fetch rows
/// through), and a stage-specific `context` for the remaining bespoke slots'
/// non-oracle data (`()` for stages whose members are all universally
/// served).
pub struct BackendPreparer<'a, F, PCS, X = ()>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub backend: &'a JoltBackend<F, PCS>,
    pub session: &'a mut ProofSession,
    pub witness: &'a dyn JoltVmWitnessPlane<F>,
    pub context: X,
}

impl<F, PCS, X> SumcheckPreparer<F> for BackendPreparer<'_, F, PCS, X>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    type Error = ProverError<F>;
}

/// Forward a universally served relation to its `PrepareKernel` slot.
macro_rules! forward_prepare {
    ($($relation:ty => $slot:ident),* $(,)?) => {$(
        impl<F, PCS, X> PrepareSumcheck<F, $relation> for BackendPreparer<'_, F, PCS, X>
        where
            F: Field,
            PCS: CommitmentScheme<Field = F>,
        {
            fn prepare(
                &mut self,
                inputs: ProverInputs<'_, F, $relation>,
            ) -> Result<Box<dyn SumcheckKernel<F, Relation = $relation>>, Self::Error> {
                Ok(self.backend.$slot.prepare(self.session, self.witness, inputs)?)
            }
        }
    )*};
}

forward_prepare!(
    jolt_verifier::stages::stage1::outer_remainder::OuterRemainder<F> => outer_remainder,
    jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking<F> => ram_read_write,
    jolt_verifier::stages::stage2::product_remainder::ProductRemainder<F> => product_remainder,
    jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction<F> => instruction_claim_reduction,
    jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation<F> => ram_raf_evaluation,
    jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck<F> => ram_output_check,
    jolt_verifier::stages::stage3::outputs::SpartanShift<F> => spartan_shift,
    jolt_verifier::stages::stage3::outputs::InstructionInput<F> => instruction_input,
    jolt_verifier::stages::stage3::outputs::RegistersClaimReduction<F> => registers_claim_reduction,
    jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking<F> => registers_read_write,
    jolt_verifier::stages::stage4::ram_val_check::RamValCheck<F> => ram_val_check,
    jolt_verifier::stages::stage5::InstructionReadRaf<F> => instruction_read_raf,
    jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction<F> => ram_ra_claim_reduction,
    jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation<F> => registers_val_evaluation,
    jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase<F> => bytecode_read_raf_address,
    jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase<F> => booleanity_address,
    jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle<F> => bytecode_read_raf_cycle,
    jolt_verifier::stages::stage6b::booleanity::Booleanity<F> => booleanity_cycle,
    jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity<F> => ram_hamming_booleanity,
    jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization<F> => ram_ra_virtualization,
    jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization<F> => instruction_ra_virtualization,
    jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction<F> => inc_claim_reduction,
    jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction<F> => hamming_weight_claim_reduction,
);
