//! Backend compute implementations for modular Jolt proving.
//!
//! Backends own compute traits and request/result types. Protocol crates decide
//! what work to schedule.

#[cfg(feature = "zk")]
mod blindfold;
mod commitments;
mod error;
mod ids;
mod openings;
mod sumcheck;
#[cfg(feature = "frontier-harness")]
mod timing;
mod traits;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cpu")]
pub mod poly {
    pub use crate::cpu::poly::{
        stage8_streaming_rlc_vector_matrix_product, Stage8StreamingRlcVectorMatrixProductInput,
    };
}

#[cfg(feature = "zk")]
pub use blindfold::{
    BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest, BlindFoldErrorRowsResult,
    BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest,
    BlindFoldFoldRowsResult, BlindFoldFoldScalarsRequest, BlindFoldFoldScalarsResult,
    BlindFoldPrivateOpening, BlindFoldRequest, BlindFoldResult, BlindFoldRoundRequest,
    BlindFoldRowCommitmentRequest, BlindFoldRowCommitmentResult, BlindFoldRowOpeningRequest,
    BlindFoldRowOpeningResult, BlindFoldSlot,
};
pub use commitments::{
    CommitmentMode, CommitmentRequest, CommitmentRequestItem, CommitmentResult, CommitmentSlot,
    CommittedPolynomialOutput, ResolvedWitnessRequirement, StreamedWitnessChunk,
    StreamedWitnessOutput, TracePolynomialEmbedding,
};
pub use error::BackendError;
pub use ids::{BackendKernelMetadata, BackendRelationId, BackendValueSlot};
pub use openings::{
    OpeningEvaluationOutput, OpeningProofOutput, OpeningQueryRequest, OpeningRequest,
    OpeningResult, OpeningRlcComponent, OpeningRlcMaterializationRequest,
    OpeningRlcMaterializationResult, OpeningSlot,
};
pub use sumcheck::{
    ResolvedSumcheckView, SumcheckAdviceTraceOrder, SumcheckBooleanityOutput,
    SumcheckBooleanityStateRequest, SumcheckBytecodeReadRafExtraStageValues,
    SumcheckBytecodeReadRafOutput, SumcheckBytecodeReadRafStateRequest, SumcheckEvaluationOutput,
    SumcheckEvaluationRequest, SumcheckFieldRegisterRead, SumcheckFieldRegisterWrite,
    SumcheckFieldRegistersIncClaimReductionOutput,
    SumcheckFieldRegistersIncClaimReductionStateRequest, SumcheckFieldRegistersReadWriteRow,
    SumcheckFieldRegistersReadWriteStateRequest, SumcheckFieldRegistersValEvaluationOutput,
    SumcheckFieldRegistersValEvaluationStateRequest, SumcheckIncClaimReductionOutput,
    SumcheckIncClaimReductionStateRequest, SumcheckInstanceRequest,
    SumcheckInstructionRaVirtualizationOutput, SumcheckInstructionRaVirtualizationStateRequest,
    SumcheckInstructionReadRafOutput, SumcheckInstructionReadRafRow,
    SumcheckInstructionReadRafStateRequest, SumcheckLinearProductOutput,
    SumcheckLinearProductQuery, SumcheckLinearProductRequest, SumcheckMaterializationOutput,
    SumcheckMaterializationRequest, SumcheckPrefixProductSumQuery, SumcheckPrefixProductSumRequest,
    SumcheckProductUniskipRequest, SumcheckProductUniskipRow, SumcheckProofOutput,
    SumcheckRaPushforwardRequest, SumcheckRamHammingBooleanityOutput,
    SumcheckRamHammingBooleanityStateRequest, SumcheckRamOutputCheckStateRequest,
    SumcheckRamRaClaimReductionOutput, SumcheckRamRaClaimReductionStateRequest,
    SumcheckRamRaVirtualizationOutput, SumcheckRamRaVirtualizationStateRequest,
    SumcheckRamRafStateRequest, SumcheckRamReadWriteRow, SumcheckRamReadWriteStateRequest,
    SumcheckRamValCheckOutput, SumcheckRamValCheckStateRequest, SumcheckRegisterRead,
    SumcheckRegisterWrite, SumcheckRegistersReadWriteOutput, SumcheckRegistersReadWriteRow,
    SumcheckRegistersReadWriteStateRequest, SumcheckRegistersValEvaluationOutput,
    SumcheckRegistersValEvaluationStateRequest, SumcheckRegularBatchInstance,
    SumcheckRegularBatchLinearFactor, SumcheckRegularBatchLinearTerm, SumcheckRegularBatchProduct,
    SumcheckRegularBatchRound, SumcheckRegularBatchState, SumcheckRequest, SumcheckResult,
    SumcheckRowProductQuery, SumcheckRowProductRequest, SumcheckSlot,
    SumcheckSpartanOuterRemainderQuery, SumcheckSpartanOuterRemainderRequest,
    SumcheckSpartanOuterRemainderRound, SumcheckSpartanOuterRemainderRowStateRequest,
    SumcheckSpartanOuterRemainderState, SumcheckSpartanOuterRemainderStateRequest,
    SumcheckSpartanOuterRow, SumcheckSpartanOuterUniskipQuery, SumcheckSpartanOuterUniskipRequest,
    SumcheckStage3ShiftRow, SumcheckStage3ShiftStateRequest, SumcheckStage6IncRow,
    SumcheckStage6RaRow, SumcheckStage7AdviceAddressState, SumcheckStage7AdviceAddressStateRequest,
    SumcheckStage7HammingState, SumcheckStage7HammingStateRequest, SumcheckViewEvaluationRequest,
    SumcheckViewMaterializationRequest, SumcheckViewResolution,
};
#[cfg(feature = "frontier-harness")]
pub use timing::{reset_backend_timings, take_backend_timings, BackendTiming};
#[cfg(feature = "zk")]
pub use traits::BlindFoldBackend;
pub use traits::{
    Backend, CommitmentBackend, OpeningBackend, RamReadWriteSumcheckBackend,
    Stage3SpartanSumcheckBackend, Stage4ReadWriteSumcheckBackend,
    Stage5ValueEvaluationSumcheckBackend, Stage6RegularBatchSumcheckBackend, SumcheckBackend,
};
