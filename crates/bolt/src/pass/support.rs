mod attr_sources;
mod attrs;
mod diagnostic;
mod functions;
mod lowering;
mod module_role;
mod opening_lowering;
mod opening_proofs;
mod params;
mod pcs_lowering;
mod relations;
mod result_count;
mod source;
mod sumcheck_lowering;
mod sumcheck_proofs;
mod transcript;
mod value_lowering;
mod values;

pub(super) use attr_sources::{lower_attr_sources, LoweredAttr};
pub(crate) use attrs::string_attr;
pub(super) use attrs::{copy_attrs, symbol_attr};
pub(super) use functions::{
    compute_function_attrs, cpu_function_symbol_ref_attrs, COMPUTE_FUNCTION_ATTRS,
};
pub(super) use lowering::{append_copied_named_op, append_lowered_result_count};
pub(super) use module_role::require_module_role;
pub(super) use opening_lowering::{
    classify_compute_opening_op, lower_opening_op, OpeningDialect, OpeningOpFamily,
};
pub(super) use opening_proofs::{
    COMPUTE_OPENING_BATCH_OPENING_RESULT_TYPES, COMPUTE_OPENING_BATCH_RESULT_TYPES,
    COMPUTE_OPENING_CLAIM_RESULT_TYPES, CPU_OPENING_BATCH_OPENING_RESULT_TYPES,
    CPU_OPENING_BATCH_RESULT_TYPES, CPU_OPENING_CLAIM_RESULT_TYPES,
};
pub(super) use params::{
    compute_params_symbol_ref_attrs, protocol_params_attrs, PROTOCOL_PARAM_ATTRS,
};
pub(super) use pcs_lowering::{
    classify_compute_pcs_op, lower_pcs_op, PcsDialect, PcsLoweringRole, PcsOpFamily,
};
pub(super) use relations::{COMPUTE_RELATION_ATTRS, CPU_KERNEL_ATTRS};
pub(super) use result_count::LoweredResultCount;
pub(super) use source::{compute_to_cpu_op_name, string_attr_source, symbol_ref};
pub(super) use sumcheck_lowering::{
    classify_compute_sumcheck_value_op, lower_sumcheck_value_op, SumcheckValueDialect,
    SumcheckValueFamily, COMPUTE_SUMCHECK_EVAL_RESULT_TYPES,
    COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES, CPU_SUMCHECK_EVAL_RESULT_TYPES,
    CPU_SUMCHECK_INSTANCE_RESULT_TYPES,
};
pub(super) use sumcheck_proofs::{
    lower_kernel_sumcheck_claim_op, lower_kernel_sumcheck_driver_op, lower_sumcheck_batch_op,
    lower_sumcheck_claim_op, lower_sumcheck_driver_op, COMPUTE_SUMCHECK_BATCH_RESULT_TYPES,
    COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES, COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
    CPU_SUMCHECK_BATCH_RESULT_TYPES, CPU_SUMCHECK_CLAIM_RESULT_TYPES,
    CPU_SUMCHECK_DRIVER_RESULT_TYPES, SUMCHECK_KERNEL_CLAIM_SOURCE_ATTRS,
    SUMCHECK_KERNEL_DRIVER_SOURCE_ATTRS,
};
pub(crate) use transcript::transcript_squeeze_protocol_result_type;
pub(super) use transcript::{
    classify_compute_transcript_op, lower_transcript_op, transcript_squeeze_compute_result_types,
    transcript_squeeze_cpu_result_types, TranscriptDialect, TranscriptOpFamily,
    COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES, CPU_TRANSCRIPT_STATE_RESULT_TYPES,
};
pub(super) use value_lowering::{
    classify_compute_value_op, lower_value_op, ValueDialect, ValueOpFamily,
    COMPUTE_FIELD_RESULT_TYPES, COMPUTE_OPENING_INPUT_RESULT_TYPES, COMPUTE_POINT_RESULT_TYPES,
    CPU_FIELD_RESULT_TYPES, CPU_OPENING_INPUT_RESULT_TYPES, CPU_POINT_RESULT_TYPES,
};
pub(super) use values::{append_and_map_result_count, lowered_operands, required_lowered_operand};
