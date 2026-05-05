#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::protocols::jolt::emit::rust) struct StageProverImportShape {
    has_opening_equalities: bool,
    has_transcript_absorb_bytes: bool,
    has_point_zeros: bool,
}

impl StageProverImportShape {
    pub(in crate::protocols::jolt::emit::rust) const STAGE2: Self = Self {
        has_opening_equalities: false,
        has_transcript_absorb_bytes: false,
        has_point_zeros: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const STAGE3: Self = Self {
        has_opening_equalities: true,
        has_transcript_absorb_bytes: false,
        has_point_zeros: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const STAGE4_OR_5: Self = Self {
        has_opening_equalities: true,
        has_transcript_absorb_bytes: true,
        has_point_zeros: false,
    };
    pub(in crate::protocols::jolt::emit::rust) const STAGE6_OR_7: Self = Self {
        has_opening_equalities: true,
        has_transcript_absorb_bytes: true,
        has_point_zeros: true,
    };
}

pub(in crate::protocols::jolt::emit::rust) fn stage_prover_imports(
    stage: usize,
    shape: StageProverImportShape,
) -> String {
    let opening_equality_import = if shape.has_opening_equalities {
        format!(", Stage{stage}OpeningClaimEqualityPlan")
    } else {
        String::new()
    };
    let point_zero_import = if shape.has_point_zeros {
        format!(", Stage{stage}PointZeroPlan")
    } else {
        String::new()
    };
    let transcript_absorb_import = if shape.has_transcript_absorb_bytes {
        format!(", Stage{stage}TranscriptAbsorbBytesPlan")
    } else {
        String::new()
    };
    format!(
        "use jolt_field::Fr;\nuse jolt_kernels::stage{stage}::{{execute_stage{stage}_program, Stage{stage}CpuProgramPlan, Stage{stage}ExecutionArtifacts, Stage{stage}ExecutionMode, Stage{stage}FieldConstantPlan, Stage{stage}FieldExprPlan, Stage{stage}KernelError, Stage{stage}KernelExecutor, Stage{stage}KernelPlan, Stage{stage}OpeningBatchPlan{opening_equality_import}, Stage{stage}OpeningClaimPlan, Stage{stage}OpeningInputPlan, Stage{stage}Params, Stage{stage}PointConcatPlan, Stage{stage}PointSlicePlan{point_zero_import}, Stage{stage}ProgramStepPlan, Stage{stage}SumcheckBatchPlan, Stage{stage}SumcheckClaimPlan, Stage{stage}SumcheckDriverPlan, Stage{stage}SumcheckEvalPlan, Stage{stage}SumcheckInstanceResultPlan{transcript_absorb_import}, Stage{stage}TranscriptSqueezePlan}};\nuse jolt_transcript::{{Blake2bTranscript, Transcript}};"
    )
}
