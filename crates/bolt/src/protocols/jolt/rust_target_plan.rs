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
    FieldVectorSum,
    FieldVectorProduct,
    Neg,
    Pow(usize),
    LagrangeBasisEval {
        domain_start: i64,
        domain_size: usize,
        index: usize,
    },
    EvalFamilyWeightedSum {
        eval_count: usize,
        power_stride: usize,
        value_term_offsets: Vec<usize>,
        shared_term_offsets: Vec<usize>,
        item_term_offsets: Vec<usize>,
    },
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
            "field_vector.sum" => Ok(Self::FieldVectorSum),
            "field_vector.product" => Ok(Self::FieldVectorProduct),
            "field.neg" => Ok(Self::Neg),
            value if value.starts_with("field.pow:") => parse_pow(value),
            value if value.starts_with("poly.lagrange_basis_eval:") => parse_lagrange(value),
            value if value.starts_with("eval_family.weighted_sum:") => {
                parse_eval_family_weighted_sum(value)
            }
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
            Self::FieldVectorSum => "FieldVectorSum".to_owned(),
            Self::FieldVectorProduct => "FieldVectorProduct".to_owned(),
            Self::Neg => "Neg".to_owned(),
            Self::Pow(exponent) => format!("Pow({exponent})"),
            Self::LagrangeBasisEval {
                domain_start,
                domain_size,
                index,
            } => format!("LagrangeBasisEval({domain_start}, {domain_size}, {index})"),
            Self::EvalFamilyWeightedSum {
                eval_count,
                power_stride,
                value_term_offsets,
                shared_term_offsets,
                item_term_offsets,
            } => format!(
                "EvalFamilyWeightedSum {{ eval_count: {eval_count}, power_stride: {power_stride}, value_term_offsets: {}, shared_term_offsets: {}, item_term_offsets: {} }}",
                usize_slice_expr(value_term_offsets),
                usize_slice_expr(shared_term_offsets),
                usize_slice_expr(item_term_offsets),
            ),
        }
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

fn parse_eval_family_weighted_sum(value: &str) -> Result<FieldExprKind, RustTargetPlanError> {
    let spec = value
        .strip_prefix("eval_family.weighted_sum:")
        .ok_or_else(|| RustTargetPlanError::unsupported("field expression formula", value))?;
    let parts = spec.split(':').collect::<Vec<_>>();
    let [eval_count, power_stride, value_offsets, shared_offsets, item_offsets] = parts.as_slice()
    else {
        return Err(RustTargetPlanError::unsupported(
            "field expression formula",
            value,
        ));
    };
    let eval_count = eval_count
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    let power_stride = power_stride
        .parse::<usize>()
        .map_err(|_| RustTargetPlanError::unsupported("field expression formula", value))?;
    Ok(FieldExprKind::EvalFamilyWeightedSum {
        eval_count,
        power_stride,
        value_term_offsets: parse_usize_list(value_offsets, value)?,
        shared_term_offsets: parse_usize_list(shared_offsets, value)?,
        item_term_offsets: parse_usize_list(item_offsets, value)?,
    })
}

pub(crate) fn eval_family_weighted_sum_formula(
    eval_count: usize,
    power_stride: usize,
    value_term_offsets: &[usize],
    shared_term_offsets: &[usize],
    item_term_offsets: &[usize],
) -> String {
    format!(
        "eval_family.weighted_sum:{eval_count}:{power_stride}:{}:{}:{}",
        join_usize_list(value_term_offsets),
        join_usize_list(shared_term_offsets),
        join_usize_list(item_term_offsets),
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
        ProgramStepKind, TranscriptSqueezeKind,
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
            FieldExprKind::from_cpu_attr("field_vector.sum").ok(),
            Some(FieldExprKind::FieldVectorSum)
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("field_vector.product").ok(),
            Some(FieldExprKind::FieldVectorProduct)
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("poly.lagrange_basis_eval:-1:3:2").ok(),
            Some(FieldExprKind::LagrangeBasisEval {
                domain_start: -1,
                domain_size: 3,
                index: 2,
            })
        );
        assert_eq!(
            FieldExprKind::from_cpu_attr("eval_family.weighted_sum:39:3:0:1:2").ok(),
            Some(FieldExprKind::EvalFamilyWeightedSum {
                eval_count: 39,
                power_stride: 3,
                value_term_offsets: vec![0],
                shared_term_offsets: vec![1],
                item_term_offsets: vec![2],
            })
        );
        assert!(FieldExprKind::from_cpu_attr("field.pow:nope").is_err());
        assert!(FieldExprKind::from_cpu_attr("poly.lagrange_basis_eval:1:2").is_err());
        assert!(FieldExprKind::from_cpu_attr("eval_family.weighted_sum:1:2:0").is_err());
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
            FieldExprKind::EvalFamilyWeightedSum {
                eval_count: 39,
                power_stride: 3,
                value_term_offsets: vec![0],
                shared_term_offsets: vec![1],
                item_term_offsets: vec![2],
            }
            .rust_variant_expr(),
            "EvalFamilyWeightedSum { eval_count: 39, power_stride: 3, value_term_offsets: &[0], shared_term_offsets: &[1], item_term_offsets: &[2] }"
        );
    }
}
