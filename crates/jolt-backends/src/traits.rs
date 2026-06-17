use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::UnivariatePoly;
use jolt_witness::{WitnessNamespace, WitnessProvider};

use crate::{
    BackendError, CommitmentRequest, CommitmentResult, OpeningRequest, OpeningResult,
    OpeningRlcMaterializationRequest, OpeningRlcMaterializationResult, SumcheckBooleanityOutput,
    SumcheckBooleanityStateRequest, SumcheckBytecodeReadRafOutput,
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
    SumcheckRegularBatchState, SumcheckRequest, SumcheckResult, SumcheckRowProductRequest,
    SumcheckSpartanOuterRemainderRequest, SumcheckSpartanOuterRemainderRound,
    SumcheckSpartanOuterRemainderRowStateRequest, SumcheckSpartanOuterRemainderState,
    SumcheckSpartanOuterRemainderStateRequest, SumcheckSpartanOuterUniskipRequest,
    SumcheckStage3ShiftStateRequest, SumcheckStage7AdviceAddressState,
    SumcheckStage7AdviceAddressStateRequest, SumcheckStage7HammingState,
    SumcheckStage7HammingStateRequest, SumcheckViewResolution,
};

#[cfg(feature = "zk")]
use crate::{
    BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest, BlindFoldErrorRowsResult,
    BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest,
    BlindFoldFoldRowsResult, BlindFoldFoldScalarsRequest, BlindFoldFoldScalarsResult,
    BlindFoldRequest, BlindFoldResult, BlindFoldRowCommitmentRequest, BlindFoldRowCommitmentResult,
    BlindFoldRowOpeningRequest, BlindFoldRowOpeningResult,
};
#[cfg(feature = "field-inline")]
use crate::{
    SumcheckFieldRegistersIncClaimReductionOutput,
    SumcheckFieldRegistersIncClaimReductionStateRequest,
    SumcheckFieldRegistersReadWriteStateRequest, SumcheckFieldRegistersValEvaluationOutput,
    SumcheckFieldRegistersValEvaluationStateRequest,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;

pub trait Backend {
    fn name(&self) -> &'static str;
}

pub trait RamReadWriteSumcheckBackend<F: Field>: Backend {
    type RamReadWriteState;
    type RamRafState;
    type RamOutputCheckState;

    fn materialize_sumcheck_ram_read_write_state(
        &mut self,
        request: &SumcheckRamReadWriteStateRequest<F>,
    ) -> Result<Self::RamReadWriteState, BackendError>;

    fn evaluate_sumcheck_ram_read_write_round(
        &mut self,
        state: &Self::RamReadWriteState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_read_write_state(
        &mut self,
        state: &mut Self::RamReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_read_write_state(
        &mut self,
        state: &Self::RamReadWriteState,
    ) -> Result<[F; 3], BackendError>;

    fn materialize_sumcheck_ram_raf_state(
        &mut self,
        request: &SumcheckRamRafStateRequest<F>,
    ) -> Result<Self::RamRafState, BackendError>;

    fn evaluate_sumcheck_ram_raf_round(
        &mut self,
        state: &Self::RamRafState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_raf_state(
        &mut self,
        state: &mut Self::RamRafState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_raf_state(
        &mut self,
        state: &Self::RamRafState,
    ) -> Result<F, BackendError>;

    fn materialize_sumcheck_ram_output_check_state(
        &mut self,
        request: &SumcheckRamOutputCheckStateRequest<F>,
    ) -> Result<Self::RamOutputCheckState, BackendError>;

    fn evaluate_sumcheck_ram_output_check_round(
        &mut self,
        state: &Self::RamOutputCheckState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_output_check_state(
        &mut self,
        state: &mut Self::RamOutputCheckState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_output_check_state(
        &mut self,
        state: &Self::RamOutputCheckState,
    ) -> Result<F, BackendError>;
}

pub trait Stage3SpartanSumcheckBackend<F: Field>: Backend {
    type Stage3ShiftState;

    fn materialize_sumcheck_stage3_shift_state(
        &mut self,
        request: &SumcheckStage3ShiftStateRequest<F>,
    ) -> Result<Self::Stage3ShiftState, BackendError>;

    fn evaluate_sumcheck_stage3_shift_round(
        &mut self,
        state: &Self::Stage3ShiftState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_stage3_shift_state(
        &mut self,
        state: &mut Self::Stage3ShiftState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn stage3_shift_output_openings(
        &mut self,
        state: &Self::Stage3ShiftState,
    ) -> Result<[F; 5], BackendError>;
}

pub trait Stage4ReadWriteSumcheckBackend<F: Field>: Backend {
    type RegistersReadWriteState;
    #[cfg(feature = "field-inline")]
    type FieldRegistersReadWriteState;
    type RamValCheckState;

    fn materialize_sumcheck_registers_read_write_state(
        &mut self,
        request: &SumcheckRegistersReadWriteStateRequest<F>,
    ) -> Result<Self::RegistersReadWriteState, BackendError>;

    fn evaluate_sumcheck_registers_read_write_round(
        &mut self,
        state: &Self::RegistersReadWriteState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_registers_read_write_state(
        &mut self,
        state: &mut Self::RegistersReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_registers_read_write_state(
        &mut self,
        state: &Self::RegistersReadWriteState,
        opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_read_write_state(
        &mut self,
        request: &SumcheckFieldRegistersReadWriteStateRequest<F>,
    ) -> Result<Self::FieldRegistersReadWriteState, BackendError>;

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_read_write_round(
        &mut self,
        state: &Self::FieldRegistersReadWriteState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_read_write_state(
        &mut self,
        state: &mut Self::FieldRegistersReadWriteState,
        challenge: F,
    ) -> Result<(), BackendError>;

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_read_write_state(
        &mut self,
        state: &Self::FieldRegistersReadWriteState,
        opening_point: &[F],
    ) -> Result<SumcheckRegistersReadWriteOutput<F>, BackendError>;

    fn materialize_sumcheck_ram_val_check_state(
        &mut self,
        request: &SumcheckRamValCheckStateRequest<F>,
    ) -> Result<Self::RamValCheckState, BackendError>;

    fn evaluate_sumcheck_ram_val_check_round(
        &mut self,
        state: &Self::RamValCheckState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_val_check_state(
        &mut self,
        state: &mut Self::RamValCheckState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_val_check_state(
        &mut self,
        state: &Self::RamValCheckState,
    ) -> Result<SumcheckRamValCheckOutput<F>, BackendError>;
}

pub trait Stage5ValueEvaluationSumcheckBackend<F: Field>: Backend {
    type InstructionReadRafState;
    type RamRaClaimReductionState;
    type RegistersValEvaluationState;
    #[cfg(feature = "field-inline")]
    type FieldRegistersValEvaluationState;

    fn materialize_sumcheck_instruction_read_raf_state(
        &mut self,
        request: &SumcheckInstructionReadRafStateRequest<F>,
    ) -> Result<Self::InstructionReadRafState, BackendError>;

    fn evaluate_sumcheck_instruction_read_raf_round(
        &mut self,
        state: &Self::InstructionReadRafState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_instruction_read_raf_state(
        &mut self,
        state: &mut Self::InstructionReadRafState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_instruction_read_raf_state(
        &mut self,
        state: &Self::InstructionReadRafState,
    ) -> Result<SumcheckInstructionReadRafOutput<F>, BackendError>;

    fn materialize_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        request: &SumcheckRamRaClaimReductionStateRequest<F>,
    ) -> Result<Self::RamRaClaimReductionState, BackendError>;

    fn evaluate_sumcheck_ram_ra_claim_reduction_round(
        &mut self,
        state: &Self::RamRaClaimReductionState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        state: &mut Self::RamRaClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_ra_claim_reduction_state(
        &mut self,
        state: &Self::RamRaClaimReductionState,
    ) -> Result<SumcheckRamRaClaimReductionOutput<F>, BackendError>;

    fn materialize_sumcheck_registers_val_evaluation_state(
        &mut self,
        request: &SumcheckRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self::RegistersValEvaluationState, BackendError>;

    fn evaluate_sumcheck_registers_val_evaluation_round(
        &mut self,
        state: &Self::RegistersValEvaluationState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_registers_val_evaluation_state(
        &mut self,
        state: &mut Self::RegistersValEvaluationState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_registers_val_evaluation_state(
        &mut self,
        state: &Self::RegistersValEvaluationState,
    ) -> Result<SumcheckRegistersValEvaluationOutput<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        request: &SumcheckFieldRegistersValEvaluationStateRequest<F>,
    ) -> Result<Self::FieldRegistersValEvaluationState, BackendError>;

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_val_evaluation_round(
        &mut self,
        state: &Self::FieldRegistersValEvaluationState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        state: &mut Self::FieldRegistersValEvaluationState,
        challenge: F,
    ) -> Result<(), BackendError>;

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_val_evaluation_state(
        &mut self,
        state: &Self::FieldRegistersValEvaluationState,
    ) -> Result<SumcheckFieldRegistersValEvaluationOutput<F>, BackendError>;
}

pub trait Stage6RegularBatchSumcheckBackend<F: Field>: Backend {
    type BytecodeReadRafState;
    type BooleanityState;
    type RamHammingBooleanityState;
    type RamRaVirtualizationState;
    type InstructionRaVirtualizationState;
    type IncClaimReductionState;
    #[cfg(feature = "field-inline")]
    type FieldRegistersIncClaimReductionState;

    fn materialize_sumcheck_bytecode_read_raf_state(
        &mut self,
        request: &SumcheckBytecodeReadRafStateRequest<F>,
    ) -> Result<Self::BytecodeReadRafState, BackendError>;

    fn evaluate_sumcheck_bytecode_read_raf_round(
        &mut self,
        state: &Self::BytecodeReadRafState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_bytecode_read_raf_state(
        &mut self,
        state: &mut Self::BytecodeReadRafState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_bytecode_read_raf_state(
        &mut self,
        state: &Self::BytecodeReadRafState,
    ) -> Result<SumcheckBytecodeReadRafOutput<F>, BackendError>;

    fn materialize_sumcheck_booleanity_state(
        &mut self,
        request: &SumcheckBooleanityStateRequest<F>,
    ) -> Result<Self::BooleanityState, BackendError>;

    fn evaluate_sumcheck_booleanity_round(
        &mut self,
        state: &Self::BooleanityState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_booleanity_state(
        &mut self,
        state: &mut Self::BooleanityState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_booleanity_state(
        &mut self,
        state: &Self::BooleanityState,
    ) -> Result<SumcheckBooleanityOutput<F>, BackendError>;

    fn materialize_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        request: &SumcheckRamHammingBooleanityStateRequest<F>,
    ) -> Result<Self::RamHammingBooleanityState, BackendError>;

    fn evaluate_sumcheck_ram_hamming_booleanity_round(
        &mut self,
        state: &Self::RamHammingBooleanityState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        state: &mut Self::RamHammingBooleanityState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_hamming_booleanity_state(
        &mut self,
        state: &Self::RamHammingBooleanityState,
    ) -> Result<SumcheckRamHammingBooleanityOutput<F>, BackendError>;

    fn materialize_sumcheck_ram_ra_virtualization_state(
        &mut self,
        request: &SumcheckRamRaVirtualizationStateRequest<F>,
    ) -> Result<Self::RamRaVirtualizationState, BackendError>;

    fn evaluate_sumcheck_ram_ra_virtualization_round(
        &mut self,
        state: &Self::RamRaVirtualizationState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_ram_ra_virtualization_state(
        &mut self,
        state: &mut Self::RamRaVirtualizationState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_ram_ra_virtualization_state(
        &mut self,
        state: &Self::RamRaVirtualizationState,
    ) -> Result<SumcheckRamRaVirtualizationOutput<F>, BackendError>;

    fn materialize_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        request: &SumcheckInstructionRaVirtualizationStateRequest<F>,
    ) -> Result<Self::InstructionRaVirtualizationState, BackendError>;

    fn evaluate_sumcheck_instruction_ra_virtualization_round(
        &mut self,
        state: &Self::InstructionRaVirtualizationState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        state: &mut Self::InstructionRaVirtualizationState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_instruction_ra_virtualization_state(
        &mut self,
        state: &Self::InstructionRaVirtualizationState,
    ) -> Result<SumcheckInstructionRaVirtualizationOutput<F>, BackendError>;

    fn materialize_sumcheck_inc_claim_reduction_state(
        &mut self,
        request: &SumcheckIncClaimReductionStateRequest<F>,
    ) -> Result<Self::IncClaimReductionState, BackendError>;

    fn evaluate_sumcheck_inc_claim_reduction_round(
        &mut self,
        state: &Self::IncClaimReductionState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    fn bind_sumcheck_inc_claim_reduction_state(
        &mut self,
        state: &mut Self::IncClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn output_sumcheck_inc_claim_reduction_state(
        &mut self,
        state: &Self::IncClaimReductionState,
    ) -> Result<SumcheckIncClaimReductionOutput<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn materialize_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        request: &SumcheckFieldRegistersIncClaimReductionStateRequest<F>,
    ) -> Result<Self::FieldRegistersIncClaimReductionState, BackendError>;

    #[cfg(feature = "field-inline")]
    fn evaluate_sumcheck_field_registers_inc_claim_reduction_round(
        &mut self,
        state: &Self::FieldRegistersIncClaimReductionState,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError>;

    #[cfg(feature = "field-inline")]
    fn bind_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        state: &mut Self::FieldRegistersIncClaimReductionState,
        challenge: F,
    ) -> Result<(), BackendError>;

    #[cfg(feature = "field-inline")]
    fn output_sumcheck_field_registers_inc_claim_reduction_state(
        &mut self,
        state: &Self::FieldRegistersIncClaimReductionState,
    ) -> Result<SumcheckFieldRegistersIncClaimReductionOutput<F>, BackendError>;
}

pub trait CommitmentBackend<F, N, PCS>: Backend
where
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
{
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<CommitmentResult<N, PCS>, BackendError>
    where
        W: WitnessProvider<F, N> + Sync + ?Sized;
}

pub trait SumcheckBackend<F, N>: Backend
where
    F: Field,
    N: WitnessNamespace,
{
    type Proof;

    fn resolve_sumcheck_views<W>(
        &mut self,
        request: &SumcheckRequest<N>,
        witness: &W,
    ) -> Result<SumcheckViewResolution<N>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck view resolution",
        })
    }

    fn evaluate_sumcheck_views<W>(
        &mut self,
        request: &SumcheckEvaluationRequest<F, N>,
        witness: &W,
    ) -> Result<Vec<SumcheckEvaluationOutput<F>>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck view evaluation",
        })
    }

    fn materialize_sumcheck_views<W>(
        &mut self,
        request: &SumcheckMaterializationRequest<N>,
        witness: &W,
    ) -> Result<Vec<SumcheckMaterializationOutput<F>>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck view materialization",
        })
    }

    /// Materialize the Stage 7 hamming-weight RA-family pushforward tables
    /// `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)`, one dense table of length
    /// `2^k_chunk` per committed RA polynomial (canonical instruction → bytecode
    /// → RAM order). Heavy compute: streams per-cycle one-hot chunk indices from
    /// the witness and reduces over the cycle hypercube.
    fn materialize_sumcheck_ra_pushforward<W>(
        &mut self,
        request: &SumcheckRaPushforwardRequest<F, N>,
        witness: &W,
    ) -> Result<Vec<Vec<F>>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck RA pushforward materialization",
        })
    }

    fn materialize_sumcheck_stage7_hamming_state<W>(
        &mut self,
        request: &SumcheckStage7HammingStateRequest<F, N>,
        witness: &W,
    ) -> Result<SumcheckStage7HammingState<F>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 hamming sumcheck state materialization",
        })
    }

    fn evaluate_sumcheck_stage7_hamming_round(
        &mut self,
        state: &SumcheckStage7HammingState<F>,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        let _ = state;
        let _ = previous_claim;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 hamming sumcheck round evaluation",
        })
    }

    fn bind_sumcheck_stage7_hamming_state(
        &mut self,
        state: &mut SumcheckStage7HammingState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = state;
        let _ = challenge;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 hamming sumcheck state bind",
        })
    }

    fn materialize_sumcheck_stage7_advice_address_state<W>(
        &mut self,
        request: &SumcheckStage7AdviceAddressStateRequest<F, N>,
        witness: &W,
    ) -> Result<SumcheckStage7AdviceAddressState<F>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 advice address sumcheck state materialization",
        })
    }

    fn evaluate_sumcheck_stage7_advice_address_round(
        &mut self,
        state: &SumcheckStage7AdviceAddressState<F>,
        previous_claim: F,
        max_num_rounds: usize,
    ) -> Result<UnivariatePoly<F>, BackendError> {
        let _ = state;
        let _ = previous_claim;
        let _ = max_num_rounds;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 advice address sumcheck round evaluation",
        })
    }

    fn bind_sumcheck_stage7_advice_address_state(
        &mut self,
        state: &mut SumcheckStage7AdviceAddressState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = state;
        let _ = challenge;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "Stage 7 advice address sumcheck state bind",
        })
    }

    fn evaluate_sumcheck_linear_products(
        &mut self,
        request: &SumcheckLinearProductRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck linear product evaluation",
        })
    }

    fn evaluate_sumcheck_prefix_product_sums(
        &mut self,
        request: &SumcheckPrefixProductSumRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck prefix product sum evaluation",
        })
    }

    fn evaluate_sumcheck_row_products(
        &mut self,
        request: &SumcheckRowProductRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck row product evaluation",
        })
    }

    fn evaluate_sumcheck_product_uniskip_rows(
        &mut self,
        request: &SumcheckProductUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck product uniskip row evaluation",
        })
    }

    fn evaluate_sumcheck_spartan_outer_uniskip_rows(
        &mut self,
        request: &SumcheckSpartanOuterUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer uniskip row evaluation",
        })
    }

    fn evaluate_sumcheck_spartan_outer_remainder_rows(
        &mut self,
        request: &SumcheckSpartanOuterRemainderRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer remainder row evaluation",
        })
    }

    fn materialize_sumcheck_spartan_outer_remainder_state(
        &mut self,
        request: &SumcheckSpartanOuterRemainderStateRequest<F>,
    ) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer remainder state materialization",
        })
    }

    fn materialize_sumcheck_spartan_outer_remainder_row_state(
        &mut self,
        request: &SumcheckSpartanOuterRemainderRowStateRequest<F>,
    ) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError> {
        let _ = request;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer raw-row remainder state materialization",
        })
    }

    fn evaluate_sumcheck_spartan_outer_remainder_round(
        &mut self,
        state: &SumcheckSpartanOuterRemainderState<F>,
    ) -> Result<SumcheckSpartanOuterRemainderRound<F>, BackendError> {
        let _ = state;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer remainder round evaluation",
        })
    }

    fn bind_sumcheck_spartan_outer_remainder_state(
        &mut self,
        state: &mut SumcheckSpartanOuterRemainderState<F>,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = state;
        let _ = challenge;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck Spartan outer remainder state bind",
        })
    }

    fn evaluate_sumcheck_regular_batch_round(
        &mut self,
        state: &mut SumcheckRegularBatchState<F>,
        round: usize,
        max_rounds: usize,
        previous_claims: &[F],
    ) -> Result<Vec<SumcheckRegularBatchRound<F>>, BackendError> {
        let _ = (state, round, max_rounds, previous_claims);
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "regular batch sumcheck round evaluation",
        })
    }

    fn bind_sumcheck_regular_batch_state(
        &mut self,
        state: &mut SumcheckRegularBatchState<F>,
        round: usize,
        max_rounds: usize,
        challenge: F,
    ) -> Result<(), BackendError> {
        let _ = (state, round, max_rounds, challenge);
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "regular batch sumcheck state bind",
        })
    }

    fn prove_sumcheck<W>(
        &mut self,
        request: &SumcheckRequest<N>,
        witness: &W,
    ) -> Result<SumcheckResult<F, Self::Proof>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "sumcheck",
        })
    }
}

pub trait OpeningBackend<F, N, PCS>: Backend
where
    F: Field,
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
{
    fn materialize_opening_rlc<W>(
        &mut self,
        request: &OpeningRlcMaterializationRequest<F, N>,
        witness: &W,
    ) -> Result<OpeningRlcMaterializationResult<F>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "opening RLC materialization",
        })
    }

    fn open<W>(
        &mut self,
        request: &OpeningRequest<F, N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<OpeningResult<F, PCS::Proof>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let _ = request;
        let _ = witness;
        let _ = setup;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "opening",
        })
    }
}

#[cfg(feature = "zk")]
pub trait BlindFoldBackend<F>: Backend
where
    F: Field,
{
    type Proof;

    fn commit_blindfold_rows<VC>(
        &mut self,
        request: BlindFoldRowCommitmentRequest<'_, F>,
        setup: &VC::Setup,
    ) -> Result<BlindFoldRowCommitmentResult<VC::Output>, BackendError>
    where
        VC: VectorCommitment<Field = F>,
    {
        request.validate(self.name(), VC::capacity(setup))?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold row commitments",
        })
    }

    fn compute_blindfold_error_rows(
        &mut self,
        request: BlindFoldErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldErrorRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold error rows",
        })
    }

    fn compute_blindfold_cross_term_error_rows(
        &mut self,
        request: BlindFoldCrossTermErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldErrorRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold cross-term error rows",
        })
    }

    fn open_blindfold_rows<VC>(
        &mut self,
        request: BlindFoldRowOpeningRequest<'_, F>,
        setup: &VC::Setup,
    ) -> Result<BlindFoldRowOpeningResult<F>, BackendError>
    where
        VC: VectorCommitment<Field = F>,
    {
        request.validate(self.name(), VC::capacity(setup))?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold row openings",
        })
    }

    fn fold_blindfold_rows(
        &mut self,
        request: BlindFoldFoldRowsRequest<'_, F>,
    ) -> Result<BlindFoldFoldRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold row folding",
        })
    }

    fn fold_blindfold_scalars(
        &mut self,
        request: BlindFoldFoldScalarsRequest<'_, F>,
    ) -> Result<BlindFoldFoldScalarsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold scalar folding",
        })
    }

    fn fold_blindfold_error_rows(
        &mut self,
        request: BlindFoldFoldErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldFoldRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold error row folding",
        })
    }

    fn fold_blindfold_error_scalars(
        &mut self,
        request: BlindFoldFoldErrorScalarsRequest<'_, F>,
    ) -> Result<BlindFoldFoldScalarsResult<F>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold error scalar folding",
        })
    }

    fn prove_blindfold(
        &mut self,
        request: &BlindFoldRequest<F>,
    ) -> Result<BlindFoldResult<F, Self::Proof>, BackendError> {
        request.validate(self.name())?;
        Err(BackendError::UnsupportedTask {
            backend: self.name(),
            task: "blindfold",
        })
    }
}
