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
#[cfg(feature = "prover-harness")]
mod timing;
mod traits;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cpu")]
pub mod poly;

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
#[cfg(not(feature = "field-inline"))]
pub use sumcheck::stage2_regular_batch_instances;
#[cfg(feature = "field-inline")]
pub use sumcheck::{
    field_register_read_write_row, field_register_read_write_rows,
    stage1_field_inline_r1cs_input_slot, stage2_field_inline_factor_openings,
    stage2_field_inline_materialize_product_factors,
    stage2_field_inline_product_uniskip_extended_evals,
    stage2_field_inline_regular_batch_instances, Stage2FieldInlineFactorOpenings,
    Stage2FieldInlineMaterializedFactors, Stage2FieldInlineProductUniskipEvalRequest,
    Stage2FieldInlineRegularBatchInstanceRequest, STAGE1_FIELD_INLINE_R1CS_INPUT_SLOT_START,
};
pub use sumcheck::{
    instruction_read_raf_row, instruction_read_raf_rows, product_uniskip_row_from_stage2_trace,
    product_uniskip_rows_from_stage2_trace, ram_read_write_row, ram_read_write_rows,
    ram_read_write_rows_from_trace, register_read_write_row, register_read_write_rows,
    spartan_outer_row, spartan_outer_rows, stage3_shift_row, stage3_shift_rows,
    stage6_bytecode_pc_indices, stage6_hamming_weight, stage6_inc_rows, stage6_ra_rows,
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
pub use sumcheck::{
    stage1_r1cs_input_slot, stage2_product_instruction_openings_from_rows,
    stage2_product_uniskip_extended_eval_outputs, stage2_product_uniskip_extended_eval_request,
    stage2_product_uniskip_extended_eval_slot, stage2_product_uniskip_extended_eval_targets,
    stage2_product_uniskip_first_round, stage2_product_uniskip_rows_from_stage2_trace,
    stage2_ram_state_requests, stage3_instruction_input_opening_slot,
    stage3_registers_claim_reduction_opening_slot, stage3_shift_opening_slot,
    stage4_ram_val_check_opening_slot, stage4_registers_read_write_opening_slot,
    stage5_instruction_lookup_table_flag_opening_slot, stage5_instruction_ra_opening_slot,
    stage5_registers_val_evaluation_opening_slot, stage6_booleanity_bytecode_ra_opening_slot,
    stage6_booleanity_instruction_ra_opening_slot, stage6_booleanity_ram_ra_opening_slot,
    stage6_bytecode_ra_opening_slot, stage6_instruction_ra_virtualization_opening_slot,
    stage6_ram_ra_virtualization_opening_slot, Stage2InstructionClaimReductionOpenings,
    Stage2ProductInstructionOpenings, Stage2ProductRemainderOpenings,
    Stage2ProductUniskipFirstRound, Stage2ProductUniskipFirstRoundRequest, Stage2RamStateRequests,
    Stage2RamStateRequestsRequest, Stage2RegularBatchInstanceRequest,
    SPARTAN_OUTER_REMAINDER_RELATION, SPARTAN_OUTER_UNISKIP_RELATION,
    SPARTAN_PRODUCT_UNISKIP_RELATION, STAGE1_R1CS_INPUT_SLOT_START, STAGE1_REMAINDER_OUTPUT_SLOT,
    STAGE1_REMAINDER_SLOT, STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS, STAGE1_UNISKIP_INPUT_SLOT,
    STAGE1_UNISKIP_OUTPUT_SLOT, STAGE1_UNISKIP_SLOT, STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_SLOT_START, STAGE2_PRODUCT_UNISKIP_INPUT_SLOT,
    STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS, STAGE2_PRODUCT_UNISKIP_OUTPUT_SLOT,
    STAGE2_PRODUCT_UNISKIP_SLOT, STAGE2_RAM_OUTPUT_CHECK_RELATION, STAGE2_RAM_RAF_RELATION,
    STAGE2_RAM_READ_WRITE_RELATION, STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS,
    STAGE3_INSTRUCTION_INPUT_OPENING_SLOT_START,
    STAGE3_REGISTERS_CLAIM_REDUCTION_OPENING_SLOT_START, STAGE3_SHIFT_OPENING_SLOT_START,
    STAGE4_RAM_VAL_CHECK_OPENING_SLOT_START, STAGE4_REGISTERS_READ_WRITE_OPENING_SLOT_START,
    STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT, STAGE5_INSTRUCTION_RA_OPENING_SLOT_START,
    STAGE5_LOOKUP_TABLE_FLAG_OPENING_SLOT_START, STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
    STAGE5_REGISTERS_VAL_EVALUATION_OPENING_SLOT_START,
    STAGE6_BOOLEANITY_BYTECODE_RA_OPENING_SLOT_START,
    STAGE6_BOOLEANITY_INSTRUCTION_RA_OPENING_SLOT_START,
    STAGE6_BOOLEANITY_RAM_RA_OPENING_SLOT_START, STAGE6_BYTECODE_RA_OPENING_SLOT_START,
    STAGE6_INC_RAM_OPENING_SLOT, STAGE6_INC_RD_OPENING_SLOT,
    STAGE6_INSTRUCTION_RA_VIRTUALIZATION_OPENING_SLOT_START,
    STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT, STAGE6_RAM_RA_VIRTUALIZATION_OPENING_SLOT_START,
};
#[cfg(feature = "prover-harness")]
pub use timing::{reset_backend_timings, take_backend_timings, BackendTiming};
#[cfg(feature = "zk")]
pub use traits::BlindFoldBackend;
pub use traits::{
    Backend, CommitmentBackend, OpeningBackend, RamReadWriteSumcheckBackend,
    Stage3SpartanSumcheckBackend, Stage4ReadWriteSumcheckBackend,
    Stage5ValueEvaluationSumcheckBackend, Stage6RegularBatchSumcheckBackend, SumcheckBackend,
};
