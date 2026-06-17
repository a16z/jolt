//! CPU sumcheck compute modules.
//!
//! Keep protocol-family optimizations in submodules here, behind
//! hardware-agnostic sumcheck requests.

mod kernels;
pub mod universal;

use std::collections::{HashMap, HashSet};

use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_poly::{boolean_index_msb, EqPolynomial, TensorEqTable};
use jolt_witness::{
    PolynomialChunk, PolynomialView, RaFamilyCycleIndexSource,
    RaFamilyCycleIndices as WitnessRaCycleIndices, WitnessNamespace, WitnessProvider,
};

use crate::{
    Backend, BackendError, BackendValueSlot, RamReadWriteSumcheckBackend, ResolvedSumcheckView,
    Stage3SpartanSumcheckBackend, Stage4ReadWriteSumcheckBackend,
    Stage5ValueEvaluationSumcheckBackend, Stage6RegularBatchSumcheckBackend, SumcheckBackend,
    SumcheckBooleanityOutput, SumcheckBooleanityStateRequest, SumcheckBytecodeReadRafOutput,
    SumcheckBytecodeReadRafStateRequest, SumcheckEvaluationOutput, SumcheckEvaluationRequest,
    SumcheckIncClaimReductionOutput, SumcheckIncClaimReductionStateRequest,
    SumcheckInstructionRaVirtualizationOutput, SumcheckInstructionRaVirtualizationStateRequest,
    SumcheckInstructionReadRafOutput, SumcheckInstructionReadRafStateRequest,
    SumcheckLinearProductOutput, SumcheckLinearProductRequest, SumcheckMaterializationOutput,
    SumcheckMaterializationRequest, SumcheckPrefixProductSumRequest, SumcheckProductUniskipRequest,
    SumcheckRaPushforwardRequest, SumcheckRamHammingBooleanityOutput,
    SumcheckRamHammingBooleanityStateRequest, SumcheckRamOutputCheckStateRequest,
    SumcheckRamRaClaimReductionOutput, SumcheckRamRaClaimReductionStateRequest,
    SumcheckRamRaVirtualizationOutput, SumcheckRamRaVirtualizationStateRequest,
    SumcheckRamRafStateRequest, SumcheckRamReadWriteStateRequest, SumcheckRamValCheckOutput,
    SumcheckRamValCheckStateRequest, SumcheckRegistersReadWriteOutput,
    SumcheckRegistersReadWriteStateRequest, SumcheckRegistersValEvaluationOutput,
    SumcheckRegistersValEvaluationStateRequest, SumcheckRegularBatchRound,
    SumcheckRegularBatchState, SumcheckRequest, SumcheckRowProductRequest, SumcheckSlot,
    SumcheckSpartanOuterRemainderRequest, SumcheckSpartanOuterRemainderRound,
    SumcheckSpartanOuterRemainderRowStateRequest, SumcheckSpartanOuterRemainderState,
    SumcheckSpartanOuterRemainderStateRequest, SumcheckSpartanOuterUniskipRequest,
    SumcheckStage3ShiftStateRequest, SumcheckStage7AdviceAddressState,
    SumcheckStage7AdviceAddressStateRequest, SumcheckStage7HammingState,
    SumcheckStage7HammingStateRequest, SumcheckViewResolution,
};
#[cfg(feature = "field-inline")]
use crate::{
    SumcheckFieldRegistersIncClaimReductionOutput,
    SumcheckFieldRegistersIncClaimReductionStateRequest,
    SumcheckFieldRegistersReadWriteStateRequest, SumcheckFieldRegistersValEvaluationOutput,
    SumcheckFieldRegistersValEvaluationStateRequest,
};

use super::ra::{
    pushforward_indices, RaCycleIndices, RaFamilyLayout, MAX_BYTECODE_CHUNKS,
    MAX_INSTRUCTION_CHUNKS, MAX_RAM_CHUNKS,
};
use super::CpuBackend;

#[cfg(feature = "prover-harness")]
fn record_sumcheck_timing(label: &'static str, start: std::time::Instant) {
    crate::timing::record_backend_timing(label, start.elapsed().as_secs_f64() * 1000.0);
}

#[cfg(not(feature = "prover-harness"))]
#[expect(
    dead_code,
    reason = "fallback timing hook is unused without prover-harness"
)]
const fn record_sumcheck_timing(_label: &'static str, _start: ()) {}

const RESOLUTION_TASK: &str = "sumcheck view resolution";
const EVALUATION_TASK: &str = "sumcheck view evaluation";
const MATERIALIZATION_TASK: &str = "sumcheck view materialization";
const RA_PUSHFORWARD_TASK: &str = "sumcheck RA pushforward materialization";
const LINEAR_PRODUCT_TASK: &str = "sumcheck linear product evaluation";
const PREFIX_PRODUCT_SUM_TASK: &str = "sumcheck prefix product sum evaluation";
const ROW_PRODUCT_TASK: &str = "sumcheck row product evaluation";
const PRODUCT_UNISKIP_ROW_TASK: &str = "sumcheck product uniskip row evaluation";
const SPARTAN_OUTER_UNISKIP_ROW_TASK: &str = "sumcheck Spartan outer uniskip row evaluation";
const SPARTAN_OUTER_REMAINDER_ROW_TASK: &str = "sumcheck Spartan outer remainder row evaluation";
const SPARTAN_OUTER_REMAINDER_STATE_TASK: &str =
    "sumcheck Spartan outer remainder state materialization";
const SPARTAN_OUTER_REMAINDER_ROW_STATE_TASK: &str =
    "sumcheck Spartan outer raw-row remainder state materialization";
const SPARTAN_OUTER_REMAINDER_ROUND_EVALUATION_TASK: &str =
    "sumcheck Spartan outer remainder round evaluation";
const SPARTAN_OUTER_REMAINDER_STATE_BIND_TASK: &str = "sumcheck Spartan outer remainder state bind";
const STAGE3_SHIFT_STATE_TASK: &str = "Stage 3 shift sumcheck state materialization";
const STAGE3_SHIFT_ROUND_TASK: &str = "Stage 3 shift sumcheck round evaluation";
const STAGE3_SHIFT_BIND_TASK: &str = "Stage 3 shift sumcheck state bind";
const REGULAR_BATCH_ROUND_EVALUATION_TASK: &str = "regular batch sumcheck round evaluation";
const REGULAR_BATCH_STATE_BIND_TASK: &str = "regular batch sumcheck state bind";
const STAGE7_HAMMING_STATE_TASK: &str = "Stage 7 hamming sumcheck state materialization";
const STAGE7_HAMMING_ROUND_TASK: &str = "Stage 7 hamming sumcheck round evaluation";
const STAGE7_HAMMING_BIND_TASK: &str = "Stage 7 hamming sumcheck state bind";
const STAGE7_ADVICE_STATE_TASK: &str = "Stage 7 advice address sumcheck state materialization";
const STAGE7_ADVICE_ROUND_TASK: &str = "Stage 7 advice address sumcheck round evaluation";
const STAGE7_ADVICE_BIND_TASK: &str = "Stage 7 advice address sumcheck state bind";
const RAM_READ_WRITE_STATE_TASK: &str = "RAM read-write sumcheck state materialization";
const RAM_READ_WRITE_ROUND_TASK: &str = "RAM read-write sumcheck round evaluation";
const RAM_READ_WRITE_BIND_TASK: &str = "RAM read-write sumcheck state bind";
const RAM_RAF_STATE_TASK: &str = "RAM RAF sumcheck state materialization";
const RAM_RAF_BIND_TASK: &str = "RAM RAF sumcheck state bind";
const RAM_OUTPUT_STATE_TASK: &str = "RAM output-check sumcheck state materialization";
const RAM_OUTPUT_BIND_TASK: &str = "RAM output-check sumcheck state bind";
const REGISTERS_READ_WRITE_STATE_TASK: &str = "registers read-write sumcheck state materialization";
const REGISTERS_READ_WRITE_ROUND_TASK: &str = "registers read-write sumcheck round evaluation";
const REGISTERS_READ_WRITE_BIND_TASK: &str = "registers read-write sumcheck state bind";
const REGISTERS_READ_WRITE_OUTPUT_TASK: &str = "registers read-write sumcheck output claims";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_READ_WRITE_STATE_TASK: &str =
    "field-registers read-write sumcheck state materialization";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_READ_WRITE_ROUND_TASK: &str =
    "field-registers read-write sumcheck round evaluation";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_READ_WRITE_BIND_TASK: &str = "field-registers read-write sumcheck state bind";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_READ_WRITE_OUTPUT_TASK: &str =
    "field-registers read-write sumcheck output claims";
const RAM_VAL_CHECK_STATE_TASK: &str = "RAM value-check sumcheck state materialization";
const RAM_VAL_CHECK_ROUND_TASK: &str = "RAM value-check sumcheck round evaluation";
const RAM_VAL_CHECK_BIND_TASK: &str = "RAM value-check sumcheck state bind";
const RAM_VAL_CHECK_OUTPUT_TASK: &str = "RAM value-check sumcheck output claims";
const RAM_RA_CLAIM_REDUCTION_STATE_TASK: &str =
    "RAM RA claim-reduction sumcheck state materialization";
const RAM_RA_CLAIM_REDUCTION_ROUND_TASK: &str = "RAM RA claim-reduction sumcheck round evaluation";
const RAM_RA_CLAIM_REDUCTION_BIND_TASK: &str = "RAM RA claim-reduction sumcheck state bind";
const RAM_RA_CLAIM_REDUCTION_OUTPUT_TASK: &str = "RAM RA claim-reduction sumcheck output claims";
const REGISTERS_VAL_EVALUATION_STATE_TASK: &str =
    "registers value-evaluation sumcheck state materialization";
const REGISTERS_VAL_EVALUATION_ROUND_TASK: &str =
    "registers value-evaluation sumcheck round evaluation";
const REGISTERS_VAL_EVALUATION_BIND_TASK: &str = "registers value-evaluation sumcheck state bind";
const REGISTERS_VAL_EVALUATION_OUTPUT_TASK: &str =
    "registers value-evaluation sumcheck output claims";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_VAL_EVALUATION_STATE_TASK: &str =
    "field-registers value-evaluation sumcheck state materialization";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_VAL_EVALUATION_ROUND_TASK: &str =
    "field-registers value-evaluation sumcheck round evaluation";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_VAL_EVALUATION_BIND_TASK: &str =
    "field-registers value-evaluation sumcheck state bind";
#[cfg(feature = "field-inline")]
const FIELD_REGISTERS_VAL_EVALUATION_OUTPUT_TASK: &str =
    "field-registers value-evaluation sumcheck output claims";
const INSTRUCTION_READ_RAF_STATE_TASK: &str = "instruction read-RAF sumcheck state materialization";
const INSTRUCTION_READ_RAF_ROUND_TASK: &str = "instruction read-RAF sumcheck round evaluation";
const INSTRUCTION_READ_RAF_BIND_TASK: &str = "instruction read-RAF sumcheck state bind";
const INSTRUCTION_READ_RAF_OUTPUT_TASK: &str = "instruction read-RAF sumcheck output claims";
const STAGE6_BYTECODE_READ_RAF_STATE_TASK: &str =
    "Stage 6 bytecode read-RAF sumcheck state materialization";
const STAGE6_BYTECODE_READ_RAF_ROUND_TASK: &str =
    "Stage 6 bytecode read-RAF sumcheck round evaluation";
const STAGE6_BYTECODE_READ_RAF_BIND_TASK: &str = "Stage 6 bytecode read-RAF sumcheck state bind";
const STAGE6_BYTECODE_READ_RAF_OUTPUT_TASK: &str =
    "Stage 6 bytecode read-RAF sumcheck output claims";
const STAGE6_BOOLEANITY_STATE_TASK: &str = "Stage 6 booleanity sumcheck state materialization";
const STAGE6_BOOLEANITY_ROUND_TASK: &str = "Stage 6 booleanity sumcheck round evaluation";
const STAGE6_BOOLEANITY_BIND_TASK: &str = "Stage 6 booleanity sumcheck state bind";
const STAGE6_BOOLEANITY_OUTPUT_TASK: &str = "Stage 6 booleanity sumcheck output claims";
const STAGE6_RAM_HAMMING_STATE_TASK: &str =
    "Stage 6 RAM hamming booleanity sumcheck state materialization";
const STAGE6_RAM_HAMMING_ROUND_TASK: &str =
    "Stage 6 RAM hamming booleanity sumcheck round evaluation";
const STAGE6_RAM_HAMMING_BIND_TASK: &str = "Stage 6 RAM hamming booleanity sumcheck state bind";
const STAGE6_RAM_HAMMING_OUTPUT_TASK: &str =
    "Stage 6 RAM hamming booleanity sumcheck output claims";
const STAGE6_RAM_RA_VIRTUAL_STATE_TASK: &str =
    "Stage 6 RAM RA virtualization sumcheck state materialization";
const STAGE6_RAM_RA_VIRTUAL_ROUND_TASK: &str =
    "Stage 6 RAM RA virtualization sumcheck round evaluation";
const STAGE6_RAM_RA_VIRTUAL_BIND_TASK: &str = "Stage 6 RAM RA virtualization sumcheck state bind";
const STAGE6_RAM_RA_VIRTUAL_OUTPUT_TASK: &str =
    "Stage 6 RAM RA virtualization sumcheck output claims";
const STAGE6_INSTRUCTION_RA_VIRTUAL_STATE_TASK: &str =
    "Stage 6 instruction RA virtualization sumcheck state materialization";
const STAGE6_INSTRUCTION_RA_VIRTUAL_ROUND_TASK: &str =
    "Stage 6 instruction RA virtualization sumcheck round evaluation";
const STAGE6_INSTRUCTION_RA_VIRTUAL_BIND_TASK: &str =
    "Stage 6 instruction RA virtualization sumcheck state bind";
const STAGE6_INSTRUCTION_RA_VIRTUAL_OUTPUT_TASK: &str =
    "Stage 6 instruction RA virtualization sumcheck output claims";
const STAGE6_INC_STATE_TASK: &str =
    "Stage 6 increment claim-reduction sumcheck state materialization";
const STAGE6_INC_ROUND_TASK: &str = "Stage 6 increment claim-reduction sumcheck round evaluation";
const STAGE6_INC_BIND_TASK: &str = "Stage 6 increment claim-reduction sumcheck state bind";
const STAGE6_INC_OUTPUT_TASK: &str = "Stage 6 increment claim-reduction sumcheck output claims";
#[cfg(feature = "field-inline")]
const STAGE6_FIELD_REGISTERS_INC_STATE_TASK: &str =
    "Stage 6 field-registers increment claim-reduction sumcheck state materialization";
#[cfg(feature = "field-inline")]
const STAGE6_FIELD_REGISTERS_INC_ROUND_TASK: &str =
    "Stage 6 field-registers increment claim-reduction sumcheck round evaluation";
#[cfg(feature = "field-inline")]
const STAGE6_FIELD_REGISTERS_INC_BIND_TASK: &str =
    "Stage 6 field-registers increment claim-reduction sumcheck state bind";
#[cfg(feature = "field-inline")]
const STAGE6_FIELD_REGISTERS_INC_OUTPUT_TASK: &str =
    "Stage 6 field-registers increment claim-reduction sumcheck output claims";

impl<F> Stage3SpartanSumcheckBackend<F> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    type Stage3ShiftState = kernels::stage3_shift::SumcheckStage3ShiftState<F>;

    fn materialize_sumcheck_stage3_shift_state(
        &mut self,
        request: &SumcheckStage3ShiftStateRequest<F>,
    ) -> Result<Self::Stage3ShiftState, BackendError> {
        kernels::stage3_shift::SumcheckStage3ShiftState::new(
            self.name(),
            STAGE3_SHIFT_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_stage3_shift_round(
        &mut self,
        state: &Self::Stage3ShiftState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(self.name(), STAGE3_SHIFT_ROUND_TASK, previous_claim)
    }

    fn bind_sumcheck_stage3_shift_state(
        &mut self,
        state: &mut Self::Stage3ShiftState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), STAGE3_SHIFT_BIND_TASK, challenge)
    }

    fn stage3_shift_output_openings(
        &mut self,
        state: &Self::Stage3ShiftState,
    ) -> Result<[F; 5], BackendError> {
        state.output_openings(self.name(), STAGE3_SHIFT_BIND_TASK)
    }
}

impl<F> RamReadWriteSumcheckBackend<F> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    type RamReadWriteState = super::read_write_matrix::RamReadWriteState<F>;
    type RamRafState = super::read_write_matrix::RamRafState<F>;
    type RamOutputCheckState = super::read_write_matrix::RamOutputCheckState<F>;

    fn materialize_sumcheck_ram_read_write_state(
        &mut self,
        request: &SumcheckRamReadWriteStateRequest<F>,
    ) -> Result<Self::RamReadWriteState, BackendError> {
        super::read_write_matrix::RamReadWriteState::new(
            self.name(),
            RAM_READ_WRITE_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_read_write_round(
        &mut self,
        state: &Self::RamReadWriteState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(self.name(), RAM_READ_WRITE_ROUND_TASK, previous_claim)
    }

    fn bind_sumcheck_ram_read_write_state(
        &mut self,
        state: &mut Self::RamReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), RAM_READ_WRITE_BIND_TASK, challenge)
    }

    fn output_sumcheck_ram_read_write_state(
        &mut self,
        state: &Self::RamReadWriteState,
    ) -> Result<[F; 3], BackendError> {
        state.output_claims(self.name(), RAM_READ_WRITE_BIND_TASK)
    }

    fn materialize_sumcheck_ram_raf_state(
        &mut self,
        request: &SumcheckRamRafStateRequest<F>,
    ) -> Result<Self::RamRafState, BackendError> {
        super::read_write_matrix::RamRafState::new(self.name(), RAM_RAF_STATE_TASK, request)
    }

    fn evaluate_sumcheck_ram_raf_round(
        &mut self,
        state: &Self::RamRafState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(previous_claim)
    }

    fn bind_sumcheck_ram_raf_state(
        &mut self,
        state: &mut Self::RamRafState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_ram_raf_state(
        &mut self,
        state: &Self::RamRafState,
    ) -> Result<F, BackendError> {
        state.output_claim(self.name(), RAM_RAF_BIND_TASK)
    }

    fn materialize_sumcheck_ram_output_check_state(
        &mut self,
        request: &SumcheckRamOutputCheckStateRequest<F>,
    ) -> Result<Self::RamOutputCheckState, BackendError> {
        super::read_write_matrix::RamOutputCheckState::new(
            self.name(),
            RAM_OUTPUT_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_output_check_round(
        &mut self,
        state: &Self::RamOutputCheckState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        Ok(state.evaluate_round(previous_claim))
    }

    fn bind_sumcheck_ram_output_check_state(
        &mut self,
        state: &mut Self::RamOutputCheckState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_ram_output_check_state(
        &mut self,
        state: &Self::RamOutputCheckState,
    ) -> Result<F, BackendError> {
        state.output_claim(self.name(), RAM_OUTPUT_BIND_TASK)
    }
}

impl<F> Stage4ReadWriteSumcheckBackend<F> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    type RegistersReadWriteState = super::read_write_matrix::RegistersReadWriteState<F>;
    #[cfg(feature = "field-inline")]
    type FieldRegistersReadWriteState = super::read_write_matrix::FieldRegistersReadWriteState<F>;
    type RamValCheckState = super::read_write_matrix::RamValCheckState<F>;

    fn materialize_sumcheck_registers_read_write_state(
        &mut self,
        request: &SumcheckRegistersReadWriteStateRequest<F>,
    ) -> Result<Self::RegistersReadWriteState, BackendError> {
        super::read_write_matrix::RegistersReadWriteState::new(
            self.name(),
            REGISTERS_READ_WRITE_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_registers_read_write_round(
        &mut self,
        state: &Self::RegistersReadWriteState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(self.name(), REGISTERS_READ_WRITE_ROUND_TASK, previous_claim)
    }

    fn bind_sumcheck_registers_read_write_state(
        &mut self,
        state: &mut Self::RegistersReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), REGISTERS_READ_WRITE_BIND_TASK, challenge)
    }

    fn output_sumcheck_registers_read_write_state(
        &mut self,
        state: &Self::RegistersReadWriteState,
        opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError> {
        let _ = REGISTERS_READ_WRITE_OUTPUT_TASK;
        state.output_claims(opening_point)
    }

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_read_write_state(
        &mut self,
        request: &SumcheckFieldRegistersReadWriteStateRequest<F>,
    ) -> Result<Self::FieldRegistersReadWriteState, BackendError> {
        super::read_write_matrix::FieldRegistersReadWriteState::new(
            self.name(),
            FIELD_REGISTERS_READ_WRITE_STATE_TASK,
            request,
        )
    }

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_read_write_round(
        &mut self,
        state: &Self::FieldRegistersReadWriteState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            FIELD_REGISTERS_READ_WRITE_ROUND_TASK,
            previous_claim,
        )
    }

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_read_write_state(
        &mut self,
        state: &mut Self::FieldRegistersReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), FIELD_REGISTERS_READ_WRITE_BIND_TASK, challenge)
    }

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_read_write_state(
        &mut self,
        state: &Self::FieldRegistersReadWriteState,
        opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError> {
        let _ = FIELD_REGISTERS_READ_WRITE_OUTPUT_TASK;
        state.output_claims(opening_point)
    }

    fn materialize_sumcheck_ram_val_check_state(
        &mut self,
        request: &SumcheckRamValCheckStateRequest<F>,
    ) -> Result<Self::RamValCheckState, BackendError> {
        super::read_write_matrix::RamValCheckState::new(
            self.name(),
            RAM_VAL_CHECK_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_val_check_round(
        &mut self,
        state: &Self::RamValCheckState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(self.name(), RAM_VAL_CHECK_ROUND_TASK, previous_claim)
    }

    fn bind_sumcheck_ram_val_check_state(
        &mut self,
        state: &mut Self::RamValCheckState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), RAM_VAL_CHECK_BIND_TASK, challenge)
    }

    fn output_sumcheck_ram_val_check_state(
        &mut self,
        state: &Self::RamValCheckState,
    ) -> Result<SumcheckRamValCheckOutput<F>, BackendError> {
        let _ = RAM_VAL_CHECK_OUTPUT_TASK;
        state.output_claims()
    }
}

impl<F> Stage5ValueEvaluationSumcheckBackend<F> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    type InstructionReadRafState = super::read_write_matrix::InstructionReadRafState<F>;
    type RamRaClaimReductionState = super::read_write_matrix::RamRaClaimReductionState<F>;
    type RegistersValEvaluationState = super::read_write_matrix::RegistersValEvaluationState<F>;
    #[cfg(feature = "field-inline")]
    type FieldRegistersValEvaluationState =
        super::read_write_matrix::FieldRegistersValEvaluationState<F>;

    fn materialize_sumcheck_instruction_read_raf_state(
        &mut self,
        request: &SumcheckInstructionReadRafStateRequest<F>,
    ) -> Result<Self::InstructionReadRafState, BackendError> {
        super::read_write_matrix::InstructionReadRafState::new(
            self.name(),
            INSTRUCTION_READ_RAF_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_instruction_read_raf_round(
        &mut self,
        state: &Self::InstructionReadRafState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(self.name(), INSTRUCTION_READ_RAF_ROUND_TASK, previous_claim)
    }

    fn bind_sumcheck_instruction_read_raf_state(
        &mut self,
        state: &mut Self::InstructionReadRafState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), INSTRUCTION_READ_RAF_BIND_TASK, challenge)
    }

    fn output_sumcheck_instruction_read_raf_state(
        &mut self,
        state: &Self::InstructionReadRafState,
    ) -> Result<SumcheckInstructionReadRafOutput<F>, BackendError> {
        let _ = INSTRUCTION_READ_RAF_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        request: &SumcheckRamRaClaimReductionStateRequest<F>,
    ) -> Result<Self::RamRaClaimReductionState, BackendError> {
        super::read_write_matrix::RamRaClaimReductionState::new(
            self.name(),
            RAM_RA_CLAIM_REDUCTION_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_ra_claim_reduction_round(
        &mut self,
        state: &Self::RamRaClaimReductionState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            RAM_RA_CLAIM_REDUCTION_ROUND_TASK,
            previous_claim,
        )
    }

    fn bind_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        state: &mut Self::RamRaClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), RAM_RA_CLAIM_REDUCTION_BIND_TASK, challenge)
    }

    fn output_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        state: &Self::RamRaClaimReductionState,
    ) -> Result<SumcheckRamRaClaimReductionOutput<F>, BackendError> {
        let _ = RAM_RA_CLAIM_REDUCTION_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_registers_val_evaluation_state(
        &mut self,
        request: &SumcheckRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self::RegistersValEvaluationState, BackendError> {
        super::read_write_matrix::RegistersValEvaluationState::new(
            self.name(),
            REGISTERS_VAL_EVALUATION_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_registers_val_evaluation_round(
        &mut self,
        state: &Self::RegistersValEvaluationState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            REGISTERS_VAL_EVALUATION_ROUND_TASK,
            previous_claim,
        )
    }

    fn bind_sumcheck_registers_val_evaluation_state(
        &mut self,
        state: &mut Self::RegistersValEvaluationState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), REGISTERS_VAL_EVALUATION_BIND_TASK, challenge)
    }

    fn output_sumcheck_registers_val_evaluation_state(
        &mut self,
        state: &Self::RegistersValEvaluationState,
    ) -> Result<SumcheckRegistersValEvaluationOutput<F>, BackendError> {
        let _ = REGISTERS_VAL_EVALUATION_OUTPUT_TASK;
        state.output_claims()
    }

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        request: &SumcheckFieldRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self::FieldRegistersValEvaluationState, BackendError> {
        super::read_write_matrix::FieldRegistersValEvaluationState::new(
            self.name(),
            FIELD_REGISTERS_VAL_EVALUATION_STATE_TASK,
            request,
        )
    }

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_val_evaluation_round(
        &mut self,
        state: &Self::FieldRegistersValEvaluationState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            FIELD_REGISTERS_VAL_EVALUATION_ROUND_TASK,
            previous_claim,
        )
    }

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        state: &mut Self::FieldRegistersValEvaluationState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(
            self.name(),
            FIELD_REGISTERS_VAL_EVALUATION_BIND_TASK,
            challenge,
        )
    }

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        state: &Self::FieldRegistersValEvaluationState,
    ) -> Result<SumcheckFieldRegistersValEvaluationOutput<F>, BackendError> {
        let _ = FIELD_REGISTERS_VAL_EVALUATION_OUTPUT_TASK;
        state.output_claims()
    }
}

impl<F> Stage6RegularBatchSumcheckBackend<F> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    type BytecodeReadRafState = super::read_write_matrix::BytecodeReadRafState<F>;
    type BooleanityState = super::read_write_matrix::BooleanityState<F>;
    type RamHammingBooleanityState = super::read_write_matrix::RamHammingBooleanityState<F>;
    type RamRaVirtualizationState = super::read_write_matrix::RamRaVirtualizationState<F>;
    type InstructionRaVirtualizationState =
        super::read_write_matrix::InstructionRaVirtualizationState<F>;
    type IncClaimReductionState = super::read_write_matrix::IncClaimReductionState<F>;
    #[cfg(feature = "field-inline")]
    type FieldRegistersIncClaimReductionState =
        super::read_write_matrix::FieldRegistersIncClaimReductionState<F>;

    fn materialize_sumcheck_bytecode_read_raf_state(
        &mut self,
        request: &SumcheckBytecodeReadRafStateRequest<F>,
    ) -> Result<Self::BytecodeReadRafState, BackendError> {
        super::read_write_matrix::BytecodeReadRafState::new(
            self.name(),
            STAGE6_BYTECODE_READ_RAF_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_bytecode_read_raf_round(
        &mut self,
        state: &Self::BytecodeReadRafState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            STAGE6_BYTECODE_READ_RAF_ROUND_TASK,
            previous_claim,
        )
    }

    fn bind_sumcheck_bytecode_read_raf_state(
        &mut self,
        state: &mut Self::BytecodeReadRafState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), STAGE6_BYTECODE_READ_RAF_BIND_TASK, challenge)
    }

    fn output_sumcheck_bytecode_read_raf_state(
        &mut self,
        state: &Self::BytecodeReadRafState,
    ) -> Result<SumcheckBytecodeReadRafOutput<F>, BackendError> {
        let _ = STAGE6_BYTECODE_READ_RAF_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_booleanity_state(
        &mut self,
        request: &SumcheckBooleanityStateRequest<F>,
    ) -> Result<Self::BooleanityState, BackendError> {
        super::read_write_matrix::BooleanityState::new(
            self.name(),
            STAGE6_BOOLEANITY_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_booleanity_round(
        &mut self,
        state: &Self::BooleanityState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let _ = STAGE6_BOOLEANITY_ROUND_TASK;
        state.evaluate_round(previous_claim)
    }

    fn bind_sumcheck_booleanity_state(
        &mut self,
        state: &mut Self::BooleanityState,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = STAGE6_BOOLEANITY_BIND_TASK;
        state.bind(challenge)
    }

    fn output_sumcheck_booleanity_state(
        &mut self,
        state: &Self::BooleanityState,
    ) -> Result<SumcheckBooleanityOutput<F>, BackendError> {
        let _ = STAGE6_BOOLEANITY_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        request: &SumcheckRamHammingBooleanityStateRequest<F>,
    ) -> Result<Self::RamHammingBooleanityState, BackendError> {
        super::read_write_matrix::RamHammingBooleanityState::new(
            self.name(),
            STAGE6_RAM_HAMMING_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_hamming_booleanity_round(
        &mut self,
        state: &Self::RamHammingBooleanityState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let _ = STAGE6_RAM_HAMMING_ROUND_TASK;
        Ok(state.evaluate_round(previous_claim))
    }

    fn bind_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        state: &mut Self::RamHammingBooleanityState,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = STAGE6_RAM_HAMMING_BIND_TASK;
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        state: &Self::RamHammingBooleanityState,
    ) -> Result<SumcheckRamHammingBooleanityOutput<F>, BackendError> {
        let _ = STAGE6_RAM_HAMMING_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_ram_ra_virtualization_state(
        &mut self,
        request: &SumcheckRamRaVirtualizationStateRequest<F>,
    ) -> Result<Self::RamRaVirtualizationState, BackendError> {
        super::read_write_matrix::RamRaVirtualizationState::new(
            self.name(),
            STAGE6_RAM_RA_VIRTUAL_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_ram_ra_virtualization_round(
        &mut self,
        state: &Self::RamRaVirtualizationState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let _ = STAGE6_RAM_RA_VIRTUAL_ROUND_TASK;
        Ok(state.evaluate_round(previous_claim))
    }

    fn bind_sumcheck_ram_ra_virtualization_state(
        &mut self,
        state: &mut Self::RamRaVirtualizationState,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = STAGE6_RAM_RA_VIRTUAL_BIND_TASK;
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_ram_ra_virtualization_state(
        &mut self,
        state: &Self::RamRaVirtualizationState,
    ) -> Result<SumcheckRamRaVirtualizationOutput<F>, BackendError> {
        let _ = STAGE6_RAM_RA_VIRTUAL_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        request: &SumcheckInstructionRaVirtualizationStateRequest<F>,
    ) -> Result<Self::InstructionRaVirtualizationState, BackendError> {
        super::read_write_matrix::InstructionRaVirtualizationState::new(
            self.name(),
            STAGE6_INSTRUCTION_RA_VIRTUAL_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_instruction_ra_virtualization_round(
        &mut self,
        state: &Self::InstructionRaVirtualizationState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let _ = STAGE6_INSTRUCTION_RA_VIRTUAL_ROUND_TASK;
        Ok(state.evaluate_round(previous_claim))
    }

    fn bind_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        state: &mut Self::InstructionRaVirtualizationState,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = STAGE6_INSTRUCTION_RA_VIRTUAL_BIND_TASK;
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        state: &Self::InstructionRaVirtualizationState,
    ) -> Result<SumcheckInstructionRaVirtualizationOutput<F>, BackendError> {
        let _ = STAGE6_INSTRUCTION_RA_VIRTUAL_OUTPUT_TASK;
        state.output_claims()
    }

    fn materialize_sumcheck_inc_claim_reduction_state(
        &mut self,
        request: &SumcheckIncClaimReductionStateRequest<F>,
    ) -> Result<Self::IncClaimReductionState, BackendError> {
        super::read_write_matrix::IncClaimReductionState::new(
            self.name(),
            STAGE6_INC_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_inc_claim_reduction_round(
        &mut self,
        state: &Self::IncClaimReductionState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let _ = STAGE6_INC_ROUND_TASK;
        state.evaluate_round(previous_claim)
    }

    fn bind_sumcheck_inc_claim_reduction_state(
        &mut self,
        state: &mut Self::IncClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = STAGE6_INC_BIND_TASK;
        state.bind(challenge);
        Ok(())
    }

    fn output_sumcheck_inc_claim_reduction_state(
        &mut self,
        state: &Self::IncClaimReductionState,
    ) -> Result<SumcheckIncClaimReductionOutput<F>, BackendError> {
        let _ = STAGE6_INC_OUTPUT_TASK;
        state.output_claims()
    }

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<F>,
    ) -> Result<Self::FieldRegistersIncClaimReductionState, BackendError> {
        super::read_write_matrix::FieldRegistersIncClaimReductionState::new(
            self.name(),
            STAGE6_FIELD_REGISTERS_INC_STATE_TASK,
            request,
        )
    }

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_inc_claim_reduction_round(
        &mut self,
        state: &Self::FieldRegistersIncClaimReductionState,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        state.evaluate_round(
            self.name(),
            STAGE6_FIELD_REGISTERS_INC_ROUND_TASK,
            previous_claim,
        )
    }

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        state: &mut Self::FieldRegistersIncClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError> {
        state.bind(self.name(), STAGE6_FIELD_REGISTERS_INC_BIND_TASK, challenge)
    }

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        state: &Self::FieldRegistersIncClaimReductionState,
    ) -> Result<SumcheckFieldRegistersIncClaimReductionOutput<F>, BackendError> {
        let _ = STAGE6_FIELD_REGISTERS_INC_OUTPUT_TASK;
        state.output_claims()
    }
}

impl<F, N> SumcheckBackend<F, N> for CpuBackend
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    N: WitnessNamespace,
{
    type Proof = ();

    fn resolve_sumcheck_views<W>(
        &mut self,
        request: &SumcheckRequest<N>,
        witness: &W,
    ) -> Result<SumcheckViewResolution<N>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        resolve_sumcheck_views(self.name(), request, witness)
    }

    fn evaluate_sumcheck_views<W>(
        &mut self,
        request: &SumcheckEvaluationRequest<F, N>,
        witness: &W,
    ) -> Result<Vec<SumcheckEvaluationOutput<F>>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        evaluate_sumcheck_views(self.name(), request, witness)
    }

    fn materialize_sumcheck_views<W>(
        &mut self,
        request: &SumcheckMaterializationRequest<N>,
        witness: &W,
    ) -> Result<Vec<SumcheckMaterializationOutput<F>>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        materialize_sumcheck_views(self.name(), request, witness)
    }

    fn materialize_sumcheck_ra_pushforward<W>(
        &mut self,
        request: &SumcheckRaPushforwardRequest<F, N>,
        witness: &W,
    ) -> Result<Vec<Vec<F>>, BackendError>
    where
        W: WitnessProvider<F, N> + RaFamilyCycleIndexSource<F, N>,
    {
        materialize_sumcheck_ra_pushforward(self.name(), request, witness)
    }

    fn materialize_sumcheck_stage7_hamming_state<W>(
        &mut self,
        request: &SumcheckStage7HammingStateRequest<F, N>,
        witness: &W,
    ) -> Result<SumcheckStage7HammingState<F>, BackendError>
    where
        W: WitnessProvider<F, N> + RaFamilyCycleIndexSource<F, N>,
    {
        materialize_sumcheck_stage7_hamming_state(self.name(), request, witness)
    }

    fn evaluate_sumcheck_stage7_hamming_round(
        &mut self,
        state: &SumcheckStage7HammingState<F>,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        kernels::stage7_hamming::evaluate_round(
            kernels::stage7_hamming::Stage7HammingKernelContext::new(
                self.name(),
                STAGE7_HAMMING_ROUND_TASK,
            ),
            state,
            previous_claim,
        )
    }

    fn bind_sumcheck_stage7_hamming_state(
        &mut self,
        state: &mut SumcheckStage7HammingState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        kernels::stage7_hamming::bind_state(
            kernels::stage7_hamming::Stage7HammingKernelContext::new(
                self.name(),
                STAGE7_HAMMING_BIND_TASK,
            ),
            state,
            challenge,
        )
    }

    fn materialize_sumcheck_stage7_advice_address_state<W>(
        &mut self,
        request: &SumcheckStage7AdviceAddressStateRequest<F, N>,
        witness: &W,
    ) -> Result<SumcheckStage7AdviceAddressState<F>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        materialize_sumcheck_stage7_advice_address_state(self.name(), request, witness)
    }

    fn evaluate_sumcheck_stage7_advice_address_round(
        &mut self,
        state: &SumcheckStage7AdviceAddressState<F>,
        previous_claim: F,
        max_num_rounds: usize,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        kernels::stage7_advice::evaluate_round(
            kernels::stage7_advice::Stage7AdviceKernelContext::new(
                self.name(),
                STAGE7_ADVICE_ROUND_TASK,
            ),
            state,
            previous_claim,
            max_num_rounds,
        )
    }

    fn bind_sumcheck_stage7_advice_address_state(
        &mut self,
        state: &mut SumcheckStage7AdviceAddressState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        kernels::stage7_advice::bind_state(
            kernels::stage7_advice::Stage7AdviceKernelContext::new(
                self.name(),
                STAGE7_ADVICE_BIND_TASK,
            ),
            state,
            challenge,
        )
    }

    fn evaluate_sumcheck_linear_products(
        &mut self,
        request: &SumcheckLinearProductRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        evaluate_sumcheck_linear_products(self.name(), request)
    }

    fn evaluate_sumcheck_prefix_product_sums(
        &mut self,
        request: &SumcheckPrefixProductSumRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        evaluate_sumcheck_prefix_product_sums(self.name(), request)
    }

    fn evaluate_sumcheck_row_products(
        &mut self,
        request: &SumcheckRowProductRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        evaluate_sumcheck_row_products(self.name(), request)
    }

    fn evaluate_sumcheck_product_uniskip_rows(
        &mut self,
        request: &SumcheckProductUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        kernels::spartan_product::evaluate_product_uniskip_rows(
            self.name(),
            PRODUCT_UNISKIP_ROW_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_spartan_outer_uniskip_rows(
        &mut self,
        request: &SumcheckSpartanOuterUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        kernels::spartan_outer::evaluate_spartan_outer_uniskip_rows(
            self.name(),
            SPARTAN_OUTER_UNISKIP_ROW_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_spartan_outer_remainder_rows(
        &mut self,
        request: &SumcheckSpartanOuterRemainderRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        kernels::spartan_outer::evaluate_spartan_outer_remainder_rows(
            self.name(),
            SPARTAN_OUTER_REMAINDER_ROW_TASK,
            request,
        )
    }

    fn materialize_sumcheck_spartan_outer_remainder_state(
        &mut self,
        request: &SumcheckSpartanOuterRemainderStateRequest<F>,
    ) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError> {
        kernels::spartan_outer::materialize_spartan_outer_remainder_state(
            self.name(),
            SPARTAN_OUTER_REMAINDER_STATE_TASK,
            request,
        )
    }

    fn materialize_sumcheck_spartan_outer_remainder_row_state(
        &mut self,
        request: &SumcheckSpartanOuterRemainderRowStateRequest<F>,
    ) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError> {
        kernels::spartan_outer::materialize_spartan_outer_remainder_row_state(
            self.name(),
            SPARTAN_OUTER_REMAINDER_ROW_STATE_TASK,
            request,
        )
    }

    fn evaluate_sumcheck_spartan_outer_remainder_round(
        &mut self,
        state: &SumcheckSpartanOuterRemainderState<F>,
    ) -> Result<SumcheckSpartanOuterRemainderRound<F>, BackendError> {
        kernels::spartan_outer::evaluate_spartan_outer_remainder_round(
            self.name(),
            SPARTAN_OUTER_REMAINDER_ROUND_EVALUATION_TASK,
            state,
        )
    }

    fn bind_sumcheck_spartan_outer_remainder_state(
        &mut self,
        state: &mut SumcheckSpartanOuterRemainderState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        kernels::spartan_outer::bind_spartan_outer_remainder_state(
            self.name(),
            SPARTAN_OUTER_REMAINDER_STATE_BIND_TASK,
            state,
            challenge,
        )
    }

    fn evaluate_sumcheck_regular_batch_round(
        &mut self,
        state: &mut SumcheckRegularBatchState<F>,
        round: usize,
        max_rounds: usize,
        previous_claims: &[F],
    ) -> Result<Vec<SumcheckRegularBatchRound<F>>, BackendError> {
        kernels::regular_batch::evaluate_round(
            kernels::regular_batch::RegularBatchKernelContext::new(
                self.name(),
                REGULAR_BATCH_ROUND_EVALUATION_TASK,
            ),
            state,
            round,
            max_rounds,
            previous_claims,
        )
    }

    fn bind_sumcheck_regular_batch_state(
        &mut self,
        state: &mut SumcheckRegularBatchState<F>,
        round: usize,
        max_rounds: usize,
        challenge: F,
    ) -> Result<(), BackendError> {
        kernels::regular_batch::bind_state(
            kernels::regular_batch::RegularBatchKernelContext::new(
                self.name(),
                REGULAR_BATCH_STATE_BIND_TASK,
            ),
            state,
            round,
            max_rounds,
            challenge,
        )
    }
}

fn resolve_sumcheck_views<F, N, W>(
    backend: &'static str,
    request: &SumcheckRequest<N>,
    witness: &W,
) -> Result<SumcheckViewResolution<N>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let mut slots = HashSet::with_capacity(request.instances.len());
    let mut resolved = Vec::new();

    for instance in &request.instances {
        validate_instance_slot(backend, &mut slots, instance.slot)?;
        validate_statement_shape(backend, instance.slot, instance.rounds, instance.degree)?;
        for (view_index, requirement) in instance.witness_views.iter().copied().enumerate() {
            let descriptor = resolve_requirement(
                backend,
                RESOLUTION_TASK,
                format!("slot {:?} view {view_index}", instance.slot),
                requirement,
                witness,
            )?;
            resolved.push(ResolvedSumcheckView::new(
                instance.slot,
                view_index,
                requirement,
                descriptor,
            ));
        }
    }

    Ok(SumcheckViewResolution::new(resolved))
}

fn evaluate_sumcheck_views<F, N, W>(
    backend: &'static str,
    request: &SumcheckEvaluationRequest<F, N>,
    witness: &W,
) -> Result<Vec<SumcheckEvaluationOutput<F>>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let mut slots = HashSet::with_capacity(request.views.len());
    let mut outputs = Vec::with_capacity(request.views.len());
    outputs.resize_with(request.views.len(), || None);
    let mut pending = Vec::new();

    for (view_index, view_request) in request.views.iter().enumerate() {
        validate_value_slot(backend, &mut slots, view_request.slot, EVALUATION_TASK)?;
        let context = format!("view {view_index}");
        let descriptor = resolve_requirement::<F, N, W>(
            backend,
            EVALUATION_TASK,
            context.clone(),
            view_request.requirement,
            witness,
        )?;
        if view_request.point.len() != descriptor.dimensions.log_rows {
            return Err(BackendError::InvalidRequest {
                backend,
                task: EVALUATION_TASK,
                reason: format!(
                    "{context} point has {} variables, expected {}",
                    view_request.point.len(),
                    descriptor.dimensions.log_rows
                ),
            });
        }

        if let Some(value) = witness
            .try_evaluate_oracle_view(view_request.requirement, &view_request.point)
            .map_err(|error| BackendError::InvalidRequest {
                backend,
                task: EVALUATION_TASK,
                reason: format!(
                    "{context} direct evaluation for {:?} failed: {error}",
                    view_request.requirement.oracle.kind
                ),
            })?
        {
            outputs[view_index] = Some(SumcheckEvaluationOutput::new(view_request.slot, value));
        } else {
            let view = materialize_requirement_view(
                backend,
                EVALUATION_TASK,
                &context,
                view_request.requirement,
                descriptor,
                witness,
            )?;
            pending.push(PendingMaterializedEvaluation {
                output_index: view_index,
                slot: view_request.slot,
                point: view_request.point.clone(),
                view,
            });
        }
    }

    evaluate_pending_materialized_views(backend, &pending, &mut outputs)?;
    outputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| BackendError::InvalidRequest {
            backend,
            task: EVALUATION_TASK,
            reason: "view evaluation did not produce every requested slot".to_owned(),
        })
}

struct PendingMaterializedEvaluation<'a, F: Field, N: WitnessNamespace> {
    output_index: usize,
    slot: BackendValueSlot,
    point: Vec<F>,
    view: PolynomialView<'a, F, N>,
}

fn evaluate_pending_materialized_views<F, N>(
    backend: &'static str,
    pending: &[PendingMaterializedEvaluation<'_, F, N>],
    outputs: &mut [Option<SumcheckEvaluationOutput<F>>],
) -> Result<(), BackendError>
where
    F: Field,
    N: WitnessNamespace,
{
    let mut groups = HashMap::<Vec<F>, Vec<usize>>::with_capacity(pending.len());
    for (pending_index, item) in pending.iter().enumerate() {
        groups
            .entry(item.point.clone())
            .or_default()
            .push(pending_index);
    }

    for (point, pending_indices) in groups {
        let expected_rows = 1usize << point.len();
        let boolean_row = boolean_index_msb(&point);
        let mut value_slices = Vec::with_capacity(pending_indices.len());
        let mut metadata = Vec::with_capacity(pending_indices.len());
        for pending_index in pending_indices {
            let item = &pending[pending_index];
            let Some(values) = item.view.as_slice() else {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task: EVALUATION_TASK,
                    reason: format!(
                        "view {} did not materialize a concrete view",
                        item.output_index
                    ),
                });
            };
            if values.len() != expected_rows {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task: EVALUATION_TASK,
                    reason: format!(
                        "view {} materialized {} rows, expected {expected_rows}",
                        item.output_index,
                        values.len()
                    ),
                });
            }
            value_slices.push(values);
            metadata.push((item.output_index, item.slot));
        }

        if let Some(row) = boolean_row {
            for (values, (output_index, slot)) in value_slices.into_iter().zip(metadata) {
                outputs[output_index] = Some(SumcheckEvaluationOutput::new(slot, values[row]));
            }
        } else {
            let eq_tensor = TensorEqTable::<F>::new(&point);
            if eq_tensor.len() != expected_rows {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task: EVALUATION_TASK,
                    reason: format!(
                        "equality tensor has {} rows, expected {expected_rows}",
                        eq_tensor.len()
                    ),
                });
            }
            let values = eq_tensor.evaluate_slices(&value_slices);
            for (value, (output_index, slot)) in values.into_iter().zip(metadata) {
                outputs[output_index] = Some(SumcheckEvaluationOutput::new(slot, value));
            }
        }
    }

    Ok(())
}

fn materialize_sumcheck_views<F, N, W>(
    backend: &'static str,
    request: &SumcheckMaterializationRequest<N>,
    witness: &W,
) -> Result<Vec<SumcheckMaterializationOutput<F>>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let mut slots = HashSet::with_capacity(request.views.len());
    let mut outputs = Vec::with_capacity(request.views.len());

    for (view_index, view_request) in request.views.iter().enumerate() {
        validate_value_slot(backend, &mut slots, view_request.slot, MATERIALIZATION_TASK)?;
        let context = format!("view {view_index}");
        let descriptor = resolve_requirement(
            backend,
            MATERIALIZATION_TASK,
            context.clone(),
            view_request.requirement,
            witness,
        )?;
        let values = materialize_requirement(
            backend,
            MATERIALIZATION_TASK,
            context,
            view_request.requirement,
            descriptor,
            witness,
        )?;
        outputs.push(SumcheckMaterializationOutput::new(
            view_request.slot,
            values,
        ));
    }

    Ok(outputs)
}

fn ra_chunk_index_u8(
    backend: &'static str,
    task: &'static str,
    value: usize,
) -> Result<u8, BackendError> {
    u8::try_from(value).map_err(|_| BackendError::InvalidRequest {
        backend,
        task,
        reason: format!("RA chunk index {value} exceeds the u8 chunk-index range"),
    })
}

fn materialize_sumcheck_ra_pushforward<F, N, W>(
    backend: &'static str,
    request: &SumcheckRaPushforwardRequest<F, N>,
    witness: &W,
) -> Result<Vec<Vec<F>>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N> + RaFamilyCycleIndexSource<F, N>,
{
    let k_chunk = 1usize << request.log_k_chunk;
    let indices = collect_ra_cycle_indices(
        backend,
        RA_PUSHFORWARD_TASK,
        &request.instruction_ids,
        &request.bytecode_ids,
        &request.ram_ids,
        request.log_k_chunk,
        request.r_cycle.len(),
        request.chunk_size,
        witness,
    )?;
    let layout = RaFamilyLayout::new(
        k_chunk,
        request.instruction_ids.len(),
        request.bytecode_ids.len(),
        request.ram_ids.len(),
    );
    Ok(pushforward_indices(&indices, layout, &request.r_cycle))
}

fn materialize_sumcheck_stage7_hamming_state<F, N, W>(
    backend: &'static str,
    request: &SumcheckStage7HammingStateRequest<F, N>,
    witness: &W,
) -> Result<SumcheckStage7HammingState<F>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N> + RaFamilyCycleIndexSource<F, N>,
{
    let k_chunk = 1usize << request.log_k_chunk;
    let num_polys = request.num_polys();
    if request.r_addr_bool.len() != request.log_k_chunk {
        return Err(BackendError::InvalidRequest {
            backend,
            task: STAGE7_HAMMING_STATE_TASK,
            reason: format!(
                "Stage 7 booleanity address point has {} coordinates, expected {}",
                request.r_addr_bool.len(),
                request.log_k_chunk
            ),
        });
    }
    if request.r_addr_virt.len() != num_polys {
        return Err(BackendError::InvalidRequest {
            backend,
            task: STAGE7_HAMMING_STATE_TASK,
            reason: format!(
                "Stage 7 has {} virtualization address points for {num_polys} RA polynomials",
                request.r_addr_virt.len()
            ),
        });
    }
    for (index, point) in request.r_addr_virt.iter().enumerate() {
        if point.len() != request.log_k_chunk {
            return Err(BackendError::InvalidRequest {
                backend,
                task: STAGE7_HAMMING_STATE_TASK,
                reason: format!(
                    "Stage 7 virtualization address point {index} has {} coordinates, expected {}",
                    point.len(),
                    request.log_k_chunk
                ),
            });
        }
    }

    #[cfg(feature = "prover-harness")]
    let start = std::time::Instant::now();
    let indices = collect_ra_cycle_indices(
        backend,
        STAGE7_HAMMING_STATE_TASK,
        &request.instruction_ids,
        &request.bytecode_ids,
        &request.ram_ids,
        request.log_k_chunk,
        request.r_cycle.len(),
        request.chunk_size,
        witness,
    )?;
    #[cfg(feature = "prover-harness")]
    record_sumcheck_timing("stage7.backend.hamming.collect_ra_indices", start);
    #[cfg(feature = "prover-harness")]
    let start = std::time::Instant::now();
    let layout = RaFamilyLayout::new(
        k_chunk,
        request.instruction_ids.len(),
        request.bytecode_ids.len(),
        request.ram_ids.len(),
    );
    let g_tables = pushforward_indices(&indices, layout, &request.r_cycle);
    #[cfg(feature = "prover-harness")]
    record_sumcheck_timing("stage7.backend.hamming.pushforward_indices", start);
    #[cfg(feature = "prover-harness")]
    let start = std::time::Instant::now();
    let eq_bool_table = EqPolynomial::<F>::evals(&request.r_addr_bool, None);
    #[cfg(feature = "prover-harness")]
    record_sumcheck_timing("stage7.backend.hamming.eq_bool", start);
    #[cfg(feature = "prover-harness")]
    let start = std::time::Instant::now();
    let eq_virt_tables = request
        .r_addr_virt
        .iter()
        .map(|point| EqPolynomial::<F>::evals(point, None))
        .collect::<Vec<_>>();
    #[cfg(feature = "prover-harness")]
    record_sumcheck_timing("stage7.backend.hamming.eq_virt", start);
    #[cfg(feature = "prover-harness")]
    let start = std::time::Instant::now();
    kernels::stage7_hamming::build_state(
        kernels::stage7_hamming::Stage7HammingKernelContext::new(
            backend,
            STAGE7_HAMMING_STATE_TASK,
        ),
        request.label,
        g_tables,
        eq_bool_table,
        eq_virt_tables,
        request.gamma_powers.clone(),
    )
    .inspect(|_| {
        #[cfg(feature = "prover-harness")]
        record_sumcheck_timing("stage7.backend.hamming.build_state", start);
    })
}

fn materialize_sumcheck_stage7_advice_address_state<F, N, W>(
    backend: &'static str,
    request: &SumcheckStage7AdviceAddressStateRequest<F, N>,
    witness: &W,
) -> Result<SumcheckStage7AdviceAddressState<F>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let advice_words = collect_advice_words(
        backend,
        STAGE7_ADVICE_STATE_TASK,
        request.advice_id,
        request.chunk_size,
        request
            .total_rows()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend,
                task: STAGE7_ADVICE_STATE_TASK,
                reason: format!(
                    "Stage 7 advice variable count {} overflows usize",
                    request.total_vars()
                ),
            })?,
        witness,
    )?;
    kernels::stage7_advice::build_state(
        kernels::stage7_advice::Stage7AdviceKernelContext::new(backend, STAGE7_ADVICE_STATE_TASK),
        request,
        advice_words,
    )
}

fn collect_advice_words<F, N, W>(
    backend: &'static str,
    task: &'static str,
    advice_id: N::CommittedId,
    chunk_size: usize,
    expected_len: usize,
    witness: &W,
) -> Result<Vec<u64>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let mut stream = witness
        .committed_stream(advice_id, chunk_size)
        .map_err(BackendError::Witness)?;
    let mut words = Vec::with_capacity(expected_len);
    while let Some(chunk) = stream.next_chunk().map_err(BackendError::Witness)? {
        match chunk {
            PolynomialChunk::U64(values) => words.extend(values),
            other => {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: format!(
                        "Stage 7 advice stream returned {:?} chunks; expected U64",
                        other.kind()
                    ),
                });
            }
        }
    }
    if words.len() != expected_len {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "Stage 7 advice stream returned {} rows, expected {expected_len}",
                words.len()
            ),
        });
    }
    Ok(words)
}

fn convert_witness_ra_cycle_indices(
    backend: &'static str,
    task: &'static str,
    row: WitnessRaCycleIndices,
    instruction_chunks: usize,
    bytecode_chunks: usize,
    ram_chunks: usize,
) -> Result<RaCycleIndices, BackendError> {
    if instruction_chunks > row.instruction.len()
        || bytecode_chunks > row.bytecode.len()
        || ram_chunks > row.ram.len()
    {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "RA fast path shape ({instruction_chunks}, {bytecode_chunks}, {ram_chunks}) exceeds witness bounds ({}, {}, {})",
                row.instruction.len(),
                row.bytecode.len(),
                row.ram.len()
            ),
        });
    }

    let mut converted = RaCycleIndices::default();
    converted.instruction[..instruction_chunks]
        .copy_from_slice(&row.instruction[..instruction_chunks]);
    converted.bytecode[..bytecode_chunks].copy_from_slice(&row.bytecode[..bytecode_chunks]);
    converted.ram[..ram_chunks].copy_from_slice(&row.ram[..ram_chunks]);
    Ok(converted)
}

#[expect(clippy::too_many_arguments)]
fn collect_ra_cycle_indices<F, N, W>(
    backend: &'static str,
    task: &'static str,
    instruction_ids: &[N::CommittedId],
    bytecode_ids: &[N::CommittedId],
    ram_ids: &[N::CommittedId],
    log_k_chunk: usize,
    log_t: usize,
    chunk_size: usize,
    witness: &W,
) -> Result<Vec<RaCycleIndices>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N> + RaFamilyCycleIndexSource<F, N>,
{
    let instruction_chunks = instruction_ids.len();
    let bytecode_chunks = bytecode_ids.len();
    let ram_chunks = ram_ids.len();
    if instruction_chunks > MAX_INSTRUCTION_CHUNKS
        || bytecode_chunks > MAX_BYTECODE_CHUNKS
        || ram_chunks > MAX_RAM_CHUNKS
    {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "RA family ({instruction_chunks}, {bytecode_chunks}, {ram_chunks}) exceeds CPU bounds ({MAX_INSTRUCTION_CHUNKS}, {MAX_BYTECODE_CHUNKS}, {MAX_RAM_CHUNKS})"
            ),
        });
    }
    if log_k_chunk > 8 {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("log_k_chunk {log_k_chunk} exceeds the u8 chunk-index range (max 8)"),
        });
    }

    let rows = 1usize << log_t;
    if let Some(fast_indices) = witness.try_collect_ra_family_cycle_indices(
        instruction_ids,
        bytecode_ids,
        ram_ids,
        log_k_chunk,
        log_t,
    )? {
        if fast_indices.len() != rows {
            return Err(BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "RA fast path returned {} rows, expected {rows}",
                    fast_indices.len()
                ),
            });
        }
        return fast_indices
            .into_iter()
            .map(|row| {
                convert_witness_ra_cycle_indices(
                    backend,
                    task,
                    row,
                    instruction_chunks,
                    bytecode_chunks,
                    ram_chunks,
                )
            })
            .collect();
    }

    let ids = instruction_ids
        .iter()
        .chain(bytecode_ids)
        .chain(ram_ids)
        .copied()
        .collect::<Vec<_>>();
    let mut indices = vec![RaCycleIndices::default(); rows];
    let mut stream = witness.committed_batch_stream(&ids, chunk_size.max(1))?;
    let mut cycle = 0usize;
    while let Some(batch) = stream.next_batch()? {
        if batch.chunks.len() != ids.len() {
            return Err(BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "RA batch stream returned {} chunks, expected {}",
                    batch.chunks.len(),
                    ids.len()
                ),
            });
        }
        let batch_len = batch.len();
        if batch_len == 0 {
            return Err(BackendError::InvalidRequest {
                backend,
                task,
                reason: "RA batch stream returned an empty batch".to_owned(),
            });
        }
        if cycle + batch_len > rows {
            return Err(BackendError::InvalidRequest {
                backend,
                task,
                reason: format!(
                    "RA batch stream produced more than {rows} rows for r_cycle length {log_t}"
                ),
            });
        }

        for (position, (id, chunk)) in batch.chunks.into_iter().enumerate() {
            if id != ids[position] {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: "RA batch stream returned committed IDs out of order".to_owned(),
                });
            }
            let PolynomialChunk::OneHot(values) = chunk else {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: format!("expected a one-hot RA stream, got {:?}", chunk.kind()),
                });
            };
            if values.len() != batch_len {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task,
                    reason: format!(
                        "RA batch chunk {position} has {} rows, expected {batch_len}",
                        values.len()
                    ),
                });
            }

            if position < instruction_chunks {
                for (offset, value) in values.into_iter().enumerate() {
                    let index = value.ok_or_else(|| BackendError::InvalidRequest {
                        backend,
                        task,
                        reason: "instruction RA stream produced a padding (None) chunk index"
                            .to_owned(),
                    })?;
                    indices[cycle + offset].instruction[position] =
                        ra_chunk_index_u8(backend, task, index)?;
                }
            } else if position < instruction_chunks + bytecode_chunks {
                let chunk = position - instruction_chunks;
                for (offset, value) in values.into_iter().enumerate() {
                    let index = value.ok_or_else(|| BackendError::InvalidRequest {
                        backend,
                        task,
                        reason: "bytecode RA stream produced a padding (None) chunk index"
                            .to_owned(),
                    })?;
                    indices[cycle + offset].bytecode[chunk] =
                        ra_chunk_index_u8(backend, task, index)?;
                }
            } else {
                let chunk = position - instruction_chunks - bytecode_chunks;
                for (offset, value) in values.into_iter().enumerate() {
                    indices[cycle + offset].ram[chunk] = match value {
                        Some(index) => Some(ra_chunk_index_u8(backend, task, index)?),
                        None => None,
                    };
                }
            }
        }
        cycle += batch_len;
    }
    if cycle != rows {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "RA batch stream produced {cycle} rows, expected {rows} (1 << r_cycle.len())"
            ),
        });
    }
    Ok(indices)
}

fn evaluate_sumcheck_linear_products<F>(
    backend: &'static str,
    request: &SumcheckLinearProductRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
{
    if kernels::spartan_outer::matches_linear_product(request) {
        return kernels::spartan_outer::evaluate_linear_products(
            backend,
            LINEAR_PRODUCT_TASK,
            request,
        );
    }

    kernels::evaluate_linear_product_queries(backend, LINEAR_PRODUCT_TASK, request)
}

fn evaluate_sumcheck_prefix_product_sums<F>(
    backend: &'static str,
    request: &SumcheckPrefixProductSumRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
{
    if kernels::spartan_outer::matches_prefix_product_sum(request) {
        return kernels::spartan_outer::evaluate_prefix_product_sums(
            backend,
            PREFIX_PRODUCT_SUM_TASK,
            request,
        );
    }

    kernels::spartan_outer::evaluate_prefix_product_sums(backend, PREFIX_PRODUCT_SUM_TASK, request)
}

fn evaluate_sumcheck_row_products<F>(
    backend: &'static str,
    request: &SumcheckRowProductRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    if kernels::spartan_product::matches_row_product(request) {
        return kernels::spartan_product::evaluate_row_products(backend, ROW_PRODUCT_TASK, request);
    }

    kernels::evaluate_row_product_queries(backend, ROW_PRODUCT_TASK, request)
}

fn resolve_requirement<F, N, W>(
    backend: &'static str,
    task: &'static str,
    context: String,
    requirement: jolt_witness::ViewRequirement<N>,
    witness: &W,
) -> Result<jolt_witness::OracleDescriptor<N>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let descriptor = witness
        .describe_oracle(requirement.oracle)
        .map_err(|error| BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "{context} describe {:?} failed: {error}",
                requirement.oracle.kind
            ),
        })?;
    if descriptor.reference.kind != requirement.oracle.kind {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("{context} resolved the wrong oracle"),
        });
    }
    if descriptor.encoding != requirement.encoding {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "{context} resolved encoding {:?}, expected {:?}",
                descriptor.encoding, requirement.encoding
            ),
        });
    }
    if descriptor.dimensions.rows() == 0 {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("{context} has zero rows"),
        });
    }
    Ok(descriptor)
}

fn materialize_requirement<F, N, W>(
    backend: &'static str,
    task: &'static str,
    context: String,
    requirement: jolt_witness::ViewRequirement<N>,
    descriptor: jolt_witness::OracleDescriptor<N>,
    witness: &W,
) -> Result<Vec<F>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let view =
        materialize_requirement_view(backend, task, &context, requirement, descriptor, witness)?;
    let Some(values) = view.as_slice() else {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("{context} did not materialize a concrete view"),
        });
    };
    if values.len() != descriptor.dimensions.rows() {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "{context} materialized {} rows, expected {}",
                values.len(),
                descriptor.dimensions.rows()
            ),
        });
    }
    Ok(values.to_vec())
}

fn materialize_requirement_view<'a, F, N, W>(
    backend: &'static str,
    task: &'static str,
    context: &str,
    requirement: jolt_witness::ViewRequirement<N>,
    descriptor: jolt_witness::OracleDescriptor<N>,
    witness: &'a W,
) -> Result<PolynomialView<'a, F, N>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let view = witness
        .oracle_view(requirement)
        .map_err(|error| BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "{context} materialize {:?} failed: {error}",
                requirement.oracle.kind
            ),
        })?;
    if view.descriptor().reference.kind != descriptor.reference.kind {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("{context} materialized the wrong oracle"),
        });
    }
    if view.encoding() != descriptor.encoding {
        return Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!(
                "{context} materialized encoding {:?}, expected {:?}",
                view.encoding(),
                descriptor.encoding
            ),
        });
    }
    Ok(view)
}

fn validate_instance_slot(
    backend: &'static str,
    slots: &mut HashSet<SumcheckSlot>,
    slot: SumcheckSlot,
) -> Result<(), BackendError> {
    if slots.insert(slot) {
        Ok(())
    } else {
        Err(BackendError::InvalidRequest {
            backend,
            task: RESOLUTION_TASK,
            reason: format!("duplicate sumcheck slot {slot:?}"),
        })
    }
}

fn validate_value_slot(
    backend: &'static str,
    slots: &mut HashSet<BackendValueSlot>,
    slot: BackendValueSlot,
    task: &'static str,
) -> Result<(), BackendError> {
    if slots.insert(slot) {
        Ok(())
    } else {
        Err(BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("duplicate value slot {slot:?}"),
        })
    }
}

fn validate_statement_shape(
    backend: &'static str,
    slot: SumcheckSlot,
    rounds: usize,
    degree: usize,
) -> Result<(), BackendError> {
    if degree == 0 {
        return Err(BackendError::InvalidRequest {
            backend,
            task: RESOLUTION_TASK,
            reason: format!("slot {slot:?} has zero degree"),
        });
    }
    if rounds == 0 {
        return Err(BackendError::InvalidRequest {
            backend,
            task: RESOLUTION_TASK,
            reason: format!("slot {slot:?} has zero rounds"),
        });
    }
    Ok(())
}
