//! The [`BackendPreparer`]: the one place backend slot names are spelled.
//!
//! The generated `prove_clear` drivers name only the dependency-inverted
//! [`PrepareSumcheck`] bound (jolt-verifier cannot depend on the kernel
//! crate); this context struct implements it for every member relation by
//! forwarding the member's [`ProverInputs`] bundle to the matching
//! [`JoltBackend`] slot. Universal (naive-served) slots forward directly;
//! bespoke slots with non-oracle witness channels (typed rows,
//! prover-retained program data, carried pre-batch draws) read the extra data
//! from the stage-supplied `context`.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_verifier::stages::relations::{
    PrepareSumcheck, ProverInputs, SumcheckKernel, SumcheckPreparer,
};
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6a::outputs::Stage6aCarriedChallenges;
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, Stage5InstructionReadRafRow};
use jolt_witness::WitnessProvider;

use crate::ProverError;

/// The preparer the stage recipes hand to the generated `prove_clear`
/// drivers: the backend value, the proof-scoped session, the witness plane,
/// and a stage-specific `context` for the bespoke slots' non-oracle data
/// (`()` for stages whose members are all universally served).
pub struct BackendPreparer<'a, F, PCS, X = ()>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub backend: &'a JoltBackend<F, PCS>,
    pub session: &'a mut ProofSession,
    pub witness: &'a dyn WitnessProvider<F, JoltVmNamespace>,
    pub context: X,
}

impl<F, PCS, X> SumcheckPreparer<F> for BackendPreparer<'_, F, PCS, X>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    type Error = ProverError<F>;
}

/// Forward a universal (naive-served) relation to its `PrepareKernel` slot.
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

/// Stage 5's non-oracle witness channel: the typed per-cycle lookup rows the
/// instruction read+RAF slot consumes (index bits, table selection, operand
/// interleaving) — data no field-element oracle view carries losslessly. The
/// recipe fetches them through the witness's typed-rows accessor and stages
/// them here; the slot's `PrepareSumcheck` bridge takes them exactly once.
pub struct Stage5PrepareContext {
    pub instruction_read_raf_rows: Option<Vec<Stage5InstructionReadRafRow>>,
}

impl<F, PCS> PrepareSumcheck<F, InstructionReadRaf<F>>
    for BackendPreparer<'_, F, PCS, Stage5PrepareContext>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &mut self,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, Self::Error> {
        let rows = self.context.instruction_read_raf_rows.take().ok_or(
            ProverError::InvariantViolation {
                reason: "stage-5 instruction rows were staged once but consumed twice",
            },
        )?;
        Ok(self.backend.instruction_read_raf.prepare(
            self.session,
            inputs.relation.dimensions(),
            &inputs.points.lookup_output,
            rows,
            inputs.challenges,
        )?)
    }
}

/// Stage 6a's non-oracle data: both address-phase slots are bespoke. The
/// bytecode member's stage-value table folds the prover-retained program
/// bytecode and its PC pushforward source is the per-cycle bytecode indices
/// (typed stage-6 rows); the booleanity member consumes the hand pre-batch
/// draws carried in `Stage6aCarriedChallenges`, which neither its relation
/// nor its (empty) challenge struct holds.
pub struct Stage6aPrepareContext<'a, F: Field> {
    pub bytecode_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub stage_values: Option<Vec<[F; 5]>>,
    pub stage_cycle_points: &'a [Vec<F>; 5],
    pub bytecode_indices: Option<Vec<usize>>,
    pub entry_bytecode_index: usize,
    pub carried: &'a Stage6aCarriedChallenges<F>,
}

impl<F, PCS> PrepareSumcheck<F, BytecodeReadRafAddressPhase<F>>
    for BackendPreparer<'_, F, PCS, Stage6aPrepareContext<'_, F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &mut self,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, Self::Error>
    {
        let stage_values =
            self.context
                .stage_values
                .take()
                .ok_or(ProverError::InvariantViolation {
                    reason: "stage-6a bytecode stage values were staged once but consumed twice",
                })?;
        let bytecode_indices =
            self.context
                .bytecode_indices
                .take()
                .ok_or(ProverError::InvariantViolation {
                    reason: "stage-6a bytecode indices were staged once but consumed twice",
                })?;
        Ok(self.backend.bytecode_read_raf_address.prepare(
            self.session,
            inputs.relation,
            self.context.bytecode_dimensions,
            stage_values,
            self.context.stage_cycle_points,
            bytecode_indices,
            self.context.entry_bytecode_index,
            inputs.challenges,
        )?)
    }
}

impl<F, PCS> PrepareSumcheck<F, BooleanityAddressPhase<F>>
    for BackendPreparer<'_, F, PCS, Stage6aPrepareContext<'_, F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    fn prepare(
        &mut self,
        inputs: ProverInputs<'_, F, BooleanityAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, Self::Error> {
        Ok(self.backend.booleanity_address.prepare(
            self.session,
            inputs.relation,
            self.context.booleanity_dimensions,
            &self.context.carried.booleanity_reference_address,
            &self.context.carried.booleanity_reference_cycle,
            self.context.carried.booleanity_gamma,
            self.witness,
        )?)
    }
}

forward_prepare!(
    jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking<F> => ram_read_write,
    jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction<F> => instruction_claim_reduction,
    jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation<F> => ram_raf_evaluation,
    jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck<F> => ram_output_check,
    jolt_verifier::stages::stage3::outputs::SpartanShift<F> => spartan_shift,
    jolt_verifier::stages::stage3::outputs::InstructionInput<F> => instruction_input,
    jolt_verifier::stages::stage3::outputs::RegistersClaimReduction<F> => registers_claim_reduction,
    jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking<F> => registers_read_write,
    jolt_verifier::stages::stage4::ram_val_check::RamValCheck<F> => ram_val_check,
    jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction<F> => ram_ra_claim_reduction,
    jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation<F> => registers_val_evaluation,
    jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle<F> => bytecode_read_raf_cycle,
    jolt_verifier::stages::stage6b::booleanity::Booleanity<F> => booleanity_cycle,
    jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity<F> => ram_hamming_booleanity,
    jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization<F> => ram_ra_virtualization,
    jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization<F> => instruction_ra_virtualization,
    jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction<F> => inc_claim_reduction,
    jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction<F> => hamming_weight_claim_reduction,
);
