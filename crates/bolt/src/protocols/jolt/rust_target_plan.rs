use std::fmt::{self, Display, Formatter};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RustTargetPlanError {
    kind: &'static str,
    value: String,
}

impl RustTargetPlanError {
    fn unsupported(kind: &'static str, value: &str) -> Self {
        Self {
            kind,
            value: value.to_owned(),
        }
    }
}

impl Display for RustTargetPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "unsupported {} `{}`", self.kind, self.value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgramStepKind {
    TranscriptSqueeze,
    TranscriptAbsorbBytes,
    SumcheckDriver,
}

impl ProgramStepKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "transcript_squeeze" => Ok(Self::TranscriptSqueeze),
            "transcript_absorb_bytes" => Ok(Self::TranscriptAbsorbBytes),
            "sumcheck_driver" => Ok(Self::SumcheckDriver),
            _ => Err(RustTargetPlanError::unsupported("program step kind", value)),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::TranscriptSqueeze => "TranscriptSqueeze",
            Self::TranscriptAbsorbBytes => "TranscriptAbsorbBytes",
            Self::SumcheckDriver => "SumcheckDriver",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TranscriptSqueezeKind {
    ChallengeScalar,
    ChallengeVector,
    Scalar,
}

impl TranscriptSqueezeKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "challenge_scalar" => Ok(Self::ChallengeScalar),
            "challenge_vector" => Ok(Self::ChallengeVector),
            "scalar" => Ok(Self::Scalar),
            _ => Err(RustTargetPlanError::unsupported(
                "transcript squeeze kind",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::ChallengeScalar => "ChallengeScalar",
            Self::ChallengeVector => "ChallengeVector",
            Self::Scalar => "Scalar",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClaimKind {
    Committed,
    Virtual,
}

impl ClaimKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "committed" => Ok(Self::Committed),
            "virtual" => Ok(Self::Virtual),
            _ => Err(RustTargetPlanError::unsupported(
                "opening claim kind",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::Committed => "Committed",
            Self::Virtual => "Virtual",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SumcheckPointOrder {
    AsIs,
    Reverse,
    Stage4RegistersReadWrite,
    InstructionReadRaf,
    BytecodeReadRaf,
    Stage6Booleanity,
}

impl SumcheckPointOrder {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "as_is" => Ok(Self::AsIs),
            "reverse" => Ok(Self::Reverse),
            "stage4_registers_rw" => Ok(Self::Stage4RegistersReadWrite),
            "instruction_read_raf" => Ok(Self::InstructionReadRaf),
            "bytecode_read_raf" => Ok(Self::BytecodeReadRaf),
            "stage6_booleanity" => Ok(Self::Stage6Booleanity),
            _ => Err(RustTargetPlanError::unsupported(
                "sumcheck point order",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::AsIs => "AsIs",
            Self::Reverse => "Reverse",
            Self::Stage4RegistersReadWrite => "Stage4RegistersReadWrite",
            Self::InstructionReadRaf => "InstructionReadRaf",
            Self::BytecodeReadRaf => "BytecodeReadRaf",
            Self::Stage6Booleanity => "Stage6Booleanity",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoltVerifierRelationKind {
    Stage1OuterUniskip,
    Stage1OuterRemaining,
    Stage2ProductVirtualUniskip,
    Stage2RamReadWrite,
    Stage2ProductVirtualRemainder,
    Stage2InstructionLookupClaimReduction,
    Stage2RamRafEvaluation,
    Stage2RamOutputCheck,
    Stage2Batched,
    Stage3SpartanShift,
    Stage3InstructionInput,
    Stage3RegistersClaimReduction,
    Stage3Batched,
    Stage4RegistersReadWrite,
    Stage4RamValCheck,
    Stage4Batched,
    Stage5InstructionReadRaf,
    Stage5RamRaClaimReduction,
    Stage5RegistersValEvaluation,
    Stage5Batched,
    Stage6BytecodeReadRaf,
    Stage6Booleanity,
    Stage6HammingBooleanity,
    Stage6RamRaVirtual,
    Stage6InstructionRaVirtual,
    Stage6IncClaimReduction,
    Stage6Batched,
    Stage7HammingWeightClaimReduction,
    Stage7Batched,
}

impl JoltVerifierRelationKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "jolt.stage1.outer.uniskip" => Ok(Self::Stage1OuterUniskip),
            "jolt.stage1.outer.remaining" => Ok(Self::Stage1OuterRemaining),
            "jolt.stage2.product_virtual.uniskip" => Ok(Self::Stage2ProductVirtualUniskip),
            "jolt.stage2.ram.read_write" => Ok(Self::Stage2RamReadWrite),
            "jolt.stage2.product_virtual.remainder" => Ok(Self::Stage2ProductVirtualRemainder),
            "jolt.stage2.instruction_lookup.claim_reduction" => {
                Ok(Self::Stage2InstructionLookupClaimReduction)
            }
            "jolt.stage2.ram.raf_evaluation" => Ok(Self::Stage2RamRafEvaluation),
            "jolt.stage2.ram.output_check" => Ok(Self::Stage2RamOutputCheck),
            "jolt.stage2.batched" => Ok(Self::Stage2Batched),
            "jolt.stage3.spartan_shift" => Ok(Self::Stage3SpartanShift),
            "jolt.stage3.instruction_input" => Ok(Self::Stage3InstructionInput),
            "jolt.stage3.registers_claim_reduction" => Ok(Self::Stage3RegistersClaimReduction),
            "jolt.stage3.batched" => Ok(Self::Stage3Batched),
            "jolt.stage4.registers_read_write" => Ok(Self::Stage4RegistersReadWrite),
            "jolt.stage4.ram_val_check" => Ok(Self::Stage4RamValCheck),
            "jolt.stage4.batched" => Ok(Self::Stage4Batched),
            "jolt.stage5.instruction_read_raf" => Ok(Self::Stage5InstructionReadRaf),
            "jolt.stage5.ram_ra_claim_reduction" => Ok(Self::Stage5RamRaClaimReduction),
            "jolt.stage5.registers_val_evaluation" => Ok(Self::Stage5RegistersValEvaluation),
            "jolt.stage5.batched" => Ok(Self::Stage5Batched),
            "jolt.stage6.bytecode_read_raf" => Ok(Self::Stage6BytecodeReadRaf),
            "jolt.stage6.booleanity" => Ok(Self::Stage6Booleanity),
            "jolt.stage6.hamming_booleanity" => Ok(Self::Stage6HammingBooleanity),
            "jolt.stage6.ram_ra_virtual" => Ok(Self::Stage6RamRaVirtual),
            "jolt.stage6.instruction_ra_virtual" => Ok(Self::Stage6InstructionRaVirtual),
            "jolt.stage6.inc_claim_reduction" => Ok(Self::Stage6IncClaimReduction),
            "jolt.stage6.batched" => Ok(Self::Stage6Batched),
            "jolt.stage7.hamming_weight_claim_reduction" => {
                Ok(Self::Stage7HammingWeightClaimReduction)
            }
            "jolt.stage7.batched" => Ok(Self::Stage7Batched),
            _ => Err(RustTargetPlanError::unsupported("relation", value)),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::Stage1OuterUniskip => "Stage1OuterUniskip",
            Self::Stage1OuterRemaining => "Stage1OuterRemaining",
            Self::Stage2ProductVirtualUniskip => "Stage2ProductVirtualUniskip",
            Self::Stage2RamReadWrite => "Stage2RamReadWrite",
            Self::Stage2ProductVirtualRemainder => "Stage2ProductVirtualRemainder",
            Self::Stage2InstructionLookupClaimReduction => "Stage2InstructionLookupClaimReduction",
            Self::Stage2RamRafEvaluation => "Stage2RamRafEvaluation",
            Self::Stage2RamOutputCheck => "Stage2RamOutputCheck",
            Self::Stage2Batched => "Stage2Batched",
            Self::Stage3SpartanShift => "Stage3SpartanShift",
            Self::Stage3InstructionInput => "Stage3InstructionInput",
            Self::Stage3RegistersClaimReduction => "Stage3RegistersClaimReduction",
            Self::Stage3Batched => "Stage3Batched",
            Self::Stage4RegistersReadWrite => "Stage4RegistersReadWrite",
            Self::Stage4RamValCheck => "Stage4RamValCheck",
            Self::Stage4Batched => "Stage4Batched",
            Self::Stage5InstructionReadRaf => "Stage5InstructionReadRaf",
            Self::Stage5RamRaClaimReduction => "Stage5RamRaClaimReduction",
            Self::Stage5RegistersValEvaluation => "Stage5RegistersValEvaluation",
            Self::Stage5Batched => "Stage5Batched",
            Self::Stage6BytecodeReadRaf => "Stage6BytecodeReadRaf",
            Self::Stage6Booleanity => "Stage6Booleanity",
            Self::Stage6HammingBooleanity => "Stage6HammingBooleanity",
            Self::Stage6RamRaVirtual => "Stage6RamRaVirtual",
            Self::Stage6InstructionRaVirtual => "Stage6InstructionRaVirtual",
            Self::Stage6IncClaimReduction => "Stage6IncClaimReduction",
            Self::Stage6Batched => "Stage6Batched",
            Self::Stage7HammingWeightClaimReduction => "Stage7HammingWeightClaimReduction",
            Self::Stage7Batched => "Stage7Batched",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FieldExprKind {
    OpeningEval,
    Add,
    Sub,
    Mul,
    Sum,
    Product,
    Neg,
    Pow(usize),
    LagrangeBasisEval {
        domain_start: i64,
        domain_size: usize,
        index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ValueExprKind {
    FieldVectorSum,
    FieldVectorProduct,
    StructuredPolynomial {
        polynomial: StructuredPolynomialKind,
        x_point: StructuredPolynomialPointTransform,
        y_point: StructuredPolynomialPointTransform,
    },
    PowerStridedWeightedSum {
        row_count: usize,
        power_stride: usize,
        value_term_offsets: Vec<usize>,
        shared_term_offsets: Vec<usize>,
        row_term_offsets: Vec<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuredPolynomialKind {
    Eq,
    EqPlusOne,
    Lt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuredPolynomialPointSegment {
    Full,
    Prefix,
    Suffix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuredPolynomialPointLength {
    Full,
    XPoint,
    YPoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuredPolynomialPointOrder {
    AsIs,
    Reverse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StructuredPolynomialPointTransform {
    segment: StructuredPolynomialPointSegment,
    length: StructuredPolynomialPointLength,
    order: StructuredPolynomialPointOrder,
}

impl FieldExprKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "opening_eval" => Ok(Self::OpeningEval),
            "field.add" => Ok(Self::Add),
            "field.sub" => Ok(Self::Sub),
            "field.mul" => Ok(Self::Mul),
            "field.sum" => Ok(Self::Sum),
            "field.product" => Ok(Self::Product),
            "field.neg" => Ok(Self::Neg),
            value if value.starts_with("field.pow:") => parse_pow(value),
            value if value.starts_with("poly.lagrange_basis_eval:") => parse_lagrange(value),
            _ => Err(RustTargetPlanError::unsupported(
                "field expression formula",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant_expr(&self) -> String {
        match self {
            Self::OpeningEval => "OpeningEval".to_owned(),
            Self::Add => "Add".to_owned(),
            Self::Sub => "Sub".to_owned(),
            Self::Mul => "Mul".to_owned(),
            Self::Sum => "Sum".to_owned(),
            Self::Product => "Product".to_owned(),
            Self::Neg => "Neg".to_owned(),
            Self::Pow(exponent) => format!("Pow({exponent})"),
            Self::LagrangeBasisEval {
                domain_start,
                domain_size,
                index,
            } => format!("LagrangeBasisEval({domain_start}, {domain_size}, {index})"),
        }
    }
}

impl ValueExprKind {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "field_vector.sum" => Ok(Self::FieldVectorSum),
            "field_vector.product" => Ok(Self::FieldVectorProduct),
            value if value.starts_with("poly.structured_eval:") => {
                parse_structured_polynomial_value(value)
            }
            value if value.starts_with("field.power_strided_weighted_sum:") => {
                parse_power_strided_weighted_sum(value)
            }
            _ => Err(RustTargetPlanError::unsupported(
                "value expression formula",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant_expr(&self) -> String {
        match self {
            Self::FieldVectorSum => "FieldVectorSum".to_owned(),
            Self::FieldVectorProduct => "FieldVectorProduct".to_owned(),
            Self::StructuredPolynomial {
                polynomial,
                x_point,
                y_point,
            } => format!(
                "StructuredPolynomial {{ polynomial: {}, x_point: {}, y_point: {} }}",
                polynomial.rust_variant_expr(),
                x_point.rust_expr(),
                y_point.rust_expr(),
            ),
            Self::PowerStridedWeightedSum {
                row_count,
                power_stride,
                value_term_offsets,
                shared_term_offsets,
                row_term_offsets,
            } => format!(
                "PowerStridedWeightedSum {{ row_count: {row_count}, power_stride: {power_stride}, value_term_offsets: {}, shared_term_offsets: {}, row_term_offsets: {} }}",
                usize_slice_expr(value_term_offsets),
                usize_slice_expr(shared_term_offsets),
                usize_slice_expr(row_term_offsets),
            ),
        }
    }
}

impl StructuredPolynomialKind {
    fn from_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "eq" => Ok(Self::Eq),
            "eq_plus_one" => Ok(Self::EqPlusOne),
            "lt" => Ok(Self::Lt),
            _ => Err(RustTargetPlanError::unsupported(
                "structured polynomial kind",
                value,
            )),
        }
    }

    fn rust_variant_expr(&self) -> &'static str {
        match self {
            Self::Eq => "bolt_verifier_runtime::StructuredPolynomialKind::Eq",
            Self::EqPlusOne => "bolt_verifier_runtime::StructuredPolynomialKind::EqPlusOne",
            Self::Lt => "bolt_verifier_runtime::StructuredPolynomialKind::Lt",
        }
    }
}

impl StructuredPolynomialPointSegment {
    fn from_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "full" => Ok(Self::Full),
            "prefix" => Ok(Self::Prefix),
            "suffix" => Ok(Self::Suffix),
            _ => Err(RustTargetPlanError::unsupported(
                "structured polynomial point segment",
                value,
            )),
        }
    }

    fn rust_variant_expr(&self) -> &'static str {
        match self {
            Self::Full => "bolt_verifier_runtime::StructuredPolynomialPointSegment::Full",
            Self::Prefix => "bolt_verifier_runtime::StructuredPolynomialPointSegment::Prefix",
            Self::Suffix => "bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix",
        }
    }
}

impl StructuredPolynomialPointLength {
    fn from_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "full" => Ok(Self::Full),
            "x_point" => Ok(Self::XPoint),
            "y_point" => Ok(Self::YPoint),
            _ => Err(RustTargetPlanError::unsupported(
                "structured polynomial point length",
                value,
            )),
        }
    }

    fn rust_variant_expr(&self) -> &'static str {
        match self {
            Self::Full => "bolt_verifier_runtime::StructuredPolynomialPointLength::Full",
            Self::XPoint => "bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint",
            Self::YPoint => "bolt_verifier_runtime::StructuredPolynomialPointLength::YPoint",
        }
    }
}

impl StructuredPolynomialPointOrder {
    fn from_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "as_is" => Ok(Self::AsIs),
            "reverse" => Ok(Self::Reverse),
            _ => Err(RustTargetPlanError::unsupported(
                "structured polynomial point order",
                value,
            )),
        }
    }

    fn rust_variant_expr(&self) -> &'static str {
        match self {
            Self::AsIs => "bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs",
            Self::Reverse => "bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse",
        }
    }
}

impl StructuredPolynomialPointTransform {
    fn new(segment: &str, length: &str, order: &str) -> Result<Self, RustTargetPlanError> {
        Ok(Self {
            segment: StructuredPolynomialPointSegment::from_attr(segment)?,
            length: StructuredPolynomialPointLength::from_attr(length)?,
            order: StructuredPolynomialPointOrder::from_attr(order)?,
        })
    }

    fn rust_expr(&self) -> String {
        format!(
            "bolt_verifier_runtime::StructuredPolynomialPointTransform {{ segment: {}, length: {}, order: {} }}",
            self.segment.rust_variant_expr(),
            self.length.rust_variant_expr(),
            self.order.rust_variant_expr(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PcsProofMode {
    Open,
    Verify,
}

impl PcsProofMode {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "open" => Ok(Self::Open),
            "verify" => Ok(Self::Verify),
            _ => Err(RustTargetPlanError::unsupported("PCS proof mode", value)),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Verify => "Verify",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpeningEqualityMode {
    PointAndEval,
}

impl OpeningEqualityMode {
    pub(crate) fn from_cpu_attr(value: &str) -> Result<Self, RustTargetPlanError> {
        match value {
            "point_and_eval" => Ok(Self::PointAndEval),
            _ => Err(RustTargetPlanError::unsupported(
                "opening equality mode",
                value,
            )),
        }
    }

    pub(crate) fn rust_variant(self) -> &'static str {
        match self {
            Self::PointAndEval => "PointAndEval",
        }
    }
}

fn parse_pow(value: &str) -> Result<FieldExprKind, RustTargetPlanError> {
    let exponent = value
        .strip_prefix("field.pow:")
        .and_then(|exponent| exponent.parse::<usize>().ok())
        .ok_or_else(|| RustTargetPlanError::unsupported("field expression formula", value))?;
    Ok(FieldExprKind::Pow(exponent))
}

fn parse_lagrange(value: &str) -> Result<FieldExprKind, RustTargetPlanError> {
    let spec = value
        .strip_prefix("poly.lagrange_basis_eval:")
        .ok_or_else(|| RustTargetPlanError::unsupported("field expression formula", value))?;
    let parts = spec.split(':').collect::<Vec<_>>();
    let [domain_start, domain_size, index] = parts.as_slice() else {
        return Err(RustTargetPlanError::unsupported(
            "field expression formula",
            value,
        ));
    };
    let domain_start = domain_start
        .parse::<i64>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    let domain_size = domain_size
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    let index = index
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    Ok(FieldExprKind::LagrangeBasisEval {
        domain_start,
        domain_size,
        index,
    })
}

fn parse_power_strided_weighted_sum(value: &str) -> Result<ValueExprKind, RustTargetPlanError> {
    let spec = value
        .strip_prefix("field.power_strided_weighted_sum:")
        .ok_or_else(|| RustTargetPlanError::unsupported("field expression formula", value))?;
    let parts = spec.split(':').collect::<Vec<_>>();
    let [row_count, power_stride, value_offsets, shared_offsets, row_offsets] = parts.as_slice()
    else {
        return Err(RustTargetPlanError::unsupported(
            "field expression formula",
            value,
        ));
    };
    let row_count = row_count
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    let power_stride = power_stride
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    Ok(ValueExprKind::PowerStridedWeightedSum {
        row_count,
        power_stride,
        value_term_offsets: parse_usize_list(value_offsets, value)?,
        shared_term_offsets: parse_usize_list(shared_offsets, value)?,
        row_term_offsets: parse_usize_list(row_offsets, value)?,
    })
}

fn parse_structured_polynomial_value(value: &str) -> Result<ValueExprKind, RustTargetPlanError> {
    let spec = value
        .strip_prefix("poly.structured_eval:")
        .ok_or_else(|| RustTargetPlanError::unsupported("value expression formula", value))?;
    let [polynomial, x_segment, x_length, x_order, y_segment, y_length, y_order] =
        spec.split(':').collect::<Vec<_>>()[..]
    else {
        return Err(RustTargetPlanError::unsupported(
            "value expression formula",
            value,
        ));
    };
    Ok(ValueExprKind::StructuredPolynomial {
        polynomial: StructuredPolynomialKind::from_attr(polynomial)?,
        x_point: StructuredPolynomialPointTransform::new(x_segment, x_length, x_order)?,
        y_point: StructuredPolynomialPointTransform::new(y_segment, y_length, y_order)?,
    })
}

pub(crate) fn power_strided_weighted_sum_formula(
    row_count: usize,
    power_stride: usize,
    value_term_offsets: &[usize],
    shared_term_offsets: &[usize],
    row_term_offsets: &[usize],
) -> String {
    format!(
        "field.power_strided_weighted_sum:{row_count}:{power_stride}:{}:{}:{}",
        join_usize_list(value_term_offsets),
        join_usize_list(shared_term_offsets),
        join_usize_list(row_term_offsets),
    )
}

pub(crate) fn structured_polynomial_value_formula(
    polynomial: &str,
    x_segment: &str,
    x_length: &str,
    x_order: &str,
    y_segment: &str,
    y_length: &str,
    y_order: &str,
) -> String {
    format!(
        "poly.structured_eval:{polynomial}:{x_segment}:{x_length}:{x_order}:{y_segment}:{y_length}:{y_order}"
    )
}

fn parse_usize_list(value: &str, full_formula: &str) -> Result<Vec<usize>, RustTargetPlanError> {
    if value == "_" {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                RustTargetPlanError::unsupported("field expression formula", full_formula)
            })
        })
        .collect()
}

fn join_usize_list(values: &[usize]) -> String {
    if values.is_empty() {
        return "_".to_owned();
    }
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn usize_slice_expr(values: &[usize]) -> String {
    if values.is_empty() {
        return "&[]".to_owned();
    }
    let values = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{values}]")
}

#[cfg(test)]
mod tests {
    use super::{
        ClaimKind, FieldExprKind, JoltVerifierRelationKind, OpeningEqualityMode, PcsProofMode,
        ProgramStepKind, StructuredPolynomialKind, StructuredPolynomialPointLength,
        StructuredPolynomialPointOrder, StructuredPolynomialPointSegment,
        StructuredPolynomialPointTransform, TranscriptSqueezeKind, ValueExprKind,
    };

    #[test]
    fn parses_typed_rust_target_ids_from_cpu_attrs() {
        assert_eq!(
            ProgramStepKind::from_cpu_attr("sumcheck_driver").ok(),
            Some(ProgramStepKind::SumcheckDriver)
        );
        assert_eq!(
            TranscriptSqueezeKind::from_cpu_attr("challenge_vector").ok(),
            Some(TranscriptSqueezeKind::ChallengeVector)
        );
        assert_eq!(
            ClaimKind::from_cpu_attr("virtual").ok(),
            Some(ClaimKind::Virtual)
        );
        assert_eq!(
            JoltVerifierRelationKind::from_cpu_attr("jolt.stage6.booleanity").ok(),
            Some(JoltVerifierRelationKind::Stage6Booleanity)
        );
        assert_eq!(
            PcsProofMode::from_cpu_attr("verify").ok(),
            Some(PcsProofMode::Verify)
        );
        assert_eq!(
            OpeningEqualityMode::from_cpu_attr("point_and_eval").ok(),
            Some(OpeningEqualityMode::PointAndEval)
        );
    }

    #[test]
    fn parses_compound_field_expr_kinds() {
        assert_eq!(
            FieldExprKind::from_cpu_attr("field.pow:32").ok(),
            Some(FieldExprKind::Pow(32))
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("field.sum").ok(),
            Some(FieldExprKind::Sum)
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("field.product").ok(),
            Some(FieldExprKind::Product)
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("poly.lagrange_basis_eval:-1:3:2").ok(),
            Some(FieldExprKind::LagrangeBasisEval {
                domain_start: -1,
                domain_size: 3,
                index: 2,
            })
        );
        assert!(FieldExprKind::from_cpu_attr("field_vector.sum").is_err());
        assert!(FieldExprKind::from_cpu_attr("field.pow:nope").is_err());
        assert!(FieldExprKind::from_cpu_attr("poly.lagrange_basis_eval:1:2").is_err());
    }

    #[test]
    fn parses_value_expr_kinds() {
        assert_eq!(
            ValueExprKind::from_cpu_attr("field_vector.sum").ok(),
            Some(ValueExprKind::FieldVectorSum)
        );
        assert_eq!(
            ValueExprKind::from_cpu_attr("field_vector.product").ok(),
            Some(ValueExprKind::FieldVectorProduct)
        );
        assert_eq!(
            ValueExprKind::from_cpu_attr("field.power_strided_weighted_sum:39:3:0:1:2").ok(),
            Some(ValueExprKind::PowerStridedWeightedSum {
                row_count: 39,
                power_stride: 3,
                value_term_offsets: vec![0],
                shared_term_offsets: vec![1],
                row_term_offsets: vec![2],
            })
        );
        assert_eq!(
            ValueExprKind::from_cpu_attr(
                "poly.structured_eval:lt:suffix:y_point:reverse:full:full:as_is"
            )
            .ok(),
            Some(ValueExprKind::StructuredPolynomial {
                polynomial: StructuredPolynomialKind::Lt,
                x_point: StructuredPolynomialPointTransform {
                    segment: StructuredPolynomialPointSegment::Suffix,
                    length: StructuredPolynomialPointLength::YPoint,
                    order: StructuredPolynomialPointOrder::Reverse,
                },
                y_point: StructuredPolynomialPointTransform {
                    segment: StructuredPolynomialPointSegment::Full,
                    length: StructuredPolynomialPointLength::Full,
                    order: StructuredPolynomialPointOrder::AsIs,
                },
            })
        );
        assert!(ValueExprKind::from_cpu_attr("field.power_strided_weighted_sum:1:2:0").is_err());
        assert!(ValueExprKind::from_cpu_attr("poly.structured_eval:eq:full").is_err());
    }

    #[test]
    fn emits_generated_rust_variant_names_without_cpu_attr_spelling() {
        assert_eq!(
            JoltVerifierRelationKind::Stage5RegistersValEvaluation.rust_variant(),
            "Stage5RegistersValEvaluation"
        );
        assert_eq!(
            FieldExprKind::LagrangeBasisEval {
                domain_start: -1,
                domain_size: 3,
                index: 0,
            }
            .rust_variant_expr(),
            "LagrangeBasisEval(-1, 3, 0)"
        );
        assert_eq!(
            ValueExprKind::PowerStridedWeightedSum {
                row_count: 39,
                power_stride: 3,
                value_term_offsets: vec![0],
                shared_term_offsets: vec![1],
                row_term_offsets: vec![2],
            }
            .rust_variant_expr(),
            "PowerStridedWeightedSum { row_count: 39, power_stride: 3, value_term_offsets: &[0], shared_term_offsets: &[1], row_term_offsets: &[2] }"
        );
        assert_eq!(
            ValueExprKind::StructuredPolynomial {
                polynomial: StructuredPolynomialKind::EqPlusOne,
                x_point: StructuredPolynomialPointTransform {
                    segment: StructuredPolynomialPointSegment::Prefix,
                    length: StructuredPolynomialPointLength::YPoint,
                    order: StructuredPolynomialPointOrder::Reverse,
                },
                y_point: StructuredPolynomialPointTransform {
                    segment: StructuredPolynomialPointSegment::Full,
                    length: StructuredPolynomialPointLength::Full,
                    order: StructuredPolynomialPointOrder::AsIs,
                },
            }
            .rust_variant_expr(),
            "StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::EqPlusOne, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Prefix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::YPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }"
        );
    }
}
