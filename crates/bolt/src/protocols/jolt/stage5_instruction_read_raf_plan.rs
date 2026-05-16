use crate::emit::rust::{push_format, EmitError};
use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;
use crate::protocols::jolt::verifier_relation_outputs::{
    RelationOutputPlan, RelationOutputProductFamilyPlan, RelationOutputProductFamilyTermPlan,
    StructuredPolynomialEvalPlan, StructuredPolynomialEvalRefPlan, StructuredPolynomialKind,
    StructuredPolynomialPointLength, StructuredPolynomialPointOrder, StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment,
};

pub(crate) const STAGE5_TABLE_FLAG_EVAL_FAMILY: &str =
    "stage5.instruction_read_raf.eval.LookupTableFlag";
pub(crate) const STAGE5_INSTRUCTION_RA_EVAL_FAMILY: &str =
    "stage5.instruction_read_raf.eval.InstructionRa";
const STAGE5_INDEXED_EVAL_FAMILIES_CONST: &str = "STAGE5_INDEXED_EVAL_FAMILIES";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafEmitPlan {
    pub(crate) point: String,
    pub(crate) lookup_output_point: String,
    pub(crate) table_flag_evals: IndexedEvalFamilyPlan,
    pub(crate) table_flag_evals_ref: String,
    pub(crate) instruction_ra_evals: IndexedEvalFamilyPlan,
    pub(crate) instruction_ra_evals_ref: String,
    pub(crate) raf_flag_eval: String,
    pub(crate) gamma: String,
    pub(crate) point_values: Vec<Stage5InstructionReadRafPointValueEmitPlan>,
    pub(crate) log_k: usize,
}

impl Stage5InstructionReadRafEmitPlan {
    pub(crate) fn from_eval_families(
        eval_families: &[IndexedEvalFamilyPlan],
    ) -> Result<Self, EmitError> {
        let (table_flag_evals_index, table_flag_evals) =
            IndexedEvalFamilyPlan::find_with_index(eval_families, STAGE5_TABLE_FLAG_EVAL_FAMILY)?;
        let (instruction_ra_evals_index, instruction_ra_evals) =
            IndexedEvalFamilyPlan::find_with_index(
                eval_families,
                STAGE5_INSTRUCTION_RA_EVAL_FAMILY,
            )?;
        Ok(Self {
            point: "stage5.instruction_read_raf.instance".to_owned(),
            lookup_output_point: "stage5.input.stage2.instruction.LookupOutput".to_owned(),
            point_values: point_value_plans(table_flag_evals.evals.len()),
            table_flag_evals: table_flag_evals.clone(),
            table_flag_evals_ref: indexed_eval_family_ref(table_flag_evals_index),
            instruction_ra_evals: instruction_ra_evals.clone(),
            instruction_ra_evals_ref: indexed_eval_family_ref(instruction_ra_evals_index),
            raf_flag_eval: "stage5.instruction_read_raf.eval.InstructionRafFlag".to_owned(),
            gamma: "stage5.instruction_read_raf.gamma".to_owned(),
            log_k: 128,
        })
    }

    pub(crate) fn emit_runtime_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&emit_point_value_constants(&self.point_values));
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_INSTRUCTION_READ_RAF_PLAN: Stage5InstructionReadRafPlan = Stage5InstructionReadRafPlan {{\n\
                 \x20   point: {},\n\
                 \x20   lookup_output_point: {},\n\
                 \x20   table_flag_evals: {},\n\
                 \x20   instruction_ra_evals: {},\n\
                 \x20   raf_flag_eval: {},\n\
                 \x20   gamma: {},\n\
                 \x20   point_values: STAGE5_INSTRUCTION_READ_RAF_POINT_VALUES,\n\
                 \x20   log_k: {},\n\
                 }};\n\n",
                rust_str(&self.point),
                rust_str(&self.lookup_output_point),
                self.table_flag_evals_ref,
                self.instruction_ra_evals_ref,
                rust_str(&self.raf_flag_eval),
                rust_str(&self.gamma),
                self.log_k,
            ),
        );
        source
    }

    pub(crate) fn relation_output_plan(&self) -> Stage5InstructionReadRafOutputPlan {
        const PREFIX: &str = "stage5.instruction_read_raf.output";

        let table_value_family = RelationOutputProductFamilyPlan {
            symbol: format!("{PREFIX}.product.LookupTableValues"),
            gamma: None,
            terms: self
                .table_flag_evals
                .evals
                .iter()
                .zip(
                    self.point_values
                        .iter()
                        .filter(|value| value.is_lookup_table()),
                )
                .map(
                    |(flag_eval, table_value)| RelationOutputProductFamilyTermPlan {
                        gamma_power_offset: 0,
                        evals: vec![table_value.symbol.clone(), flag_eval.clone()],
                        eval_families: Vec::new(),
                        factors: Vec::new(),
                    },
                )
                .collect(),
        };
        let ra_product_family = RelationOutputProductFamilyPlan {
            symbol: format!("{PREFIX}.product.InstructionRa"),
            gamma: None,
            terms: vec![RelationOutputProductFamilyTermPlan {
                gamma_power_offset: 0,
                evals: Vec::new(),
                eval_families: vec![self.instruction_ra_evals.symbol.clone()],
                factors: Vec::new(),
            }],
        };
        let eq = StructuredPolynomialEvalPlan {
            symbol: format!("{PREFIX}.eq.LookupOutputCycle"),
            polynomial: StructuredPolynomialKind::Eq,
            x_point: StructuredPolynomialPointPlan {
                source: self.point.clone(),
                segment: StructuredPolynomialPointSegment::Suffix,
                length: StructuredPolynomialPointLength::YPoint,
                order: StructuredPolynomialPointOrder::Reverse,
            },
            y_point: StructuredPolynomialPointPlan {
                source: self.lookup_output_point.clone(),
                segment: StructuredPolynomialPointSegment::Full,
                length: StructuredPolynomialPointLength::Full,
                order: StructuredPolynomialPointOrder::AsIs,
            },
        };

        let left = "stage5.instruction_read_raf.point_value.LeftLookupOperand".to_owned();
        let right = "stage5.instruction_read_raf.point_value.RightLookupOperand".to_owned();
        let identity = "stage5.instruction_read_raf.point_value.Identity".to_owned();
        let gamma_right = format!("{PREFIX}.term.GammaRightLookupOperand");
        let left_plus_gamma_right = format!("{PREFIX}.partial.LeftPlusGammaRight");
        let raf_flag_left_plus_gamma_right = format!("{PREFIX}.term.RafFlagLeftPlusGammaRight");
        let non_raf_lookup_operands = format!("{PREFIX}.partial.NonRafLookupOperands");
        let gamma_identity = format!("{PREFIX}.term.GammaIdentity");
        let raf_flag_gamma_identity = format!("{PREFIX}.term.RafFlagGammaIdentity");
        let raf_claim = format!("{PREFIX}.partial.RafClaim");
        let gamma_raf_claim = format!("{PREFIX}.term.GammaRafClaim");
        let lookup_or_raf = format!("{PREFIX}.partial.LookupOrRaf");
        let eq_ra = format!("{PREFIX}.partial.EqRa");
        let claim_expr = format!("{PREFIX}.claim_expr");

        let field_exprs = vec![
            output_field_expr(
                gamma_right.clone(),
                "field.mul",
                vec![self.gamma.clone(), right],
            ),
            output_field_expr(
                left_plus_gamma_right.clone(),
                "field.add",
                vec![left, gamma_right],
            ),
            output_field_expr(
                raf_flag_left_plus_gamma_right.clone(),
                "field.mul",
                vec![self.raf_flag_eval.clone(), left_plus_gamma_right.clone()],
            ),
            output_field_expr(
                non_raf_lookup_operands.clone(),
                "field.sub",
                vec![left_plus_gamma_right, raf_flag_left_plus_gamma_right],
            ),
            output_field_expr(
                gamma_identity.clone(),
                "field.mul",
                vec![self.gamma.clone(), identity],
            ),
            output_field_expr(
                raf_flag_gamma_identity.clone(),
                "field.mul",
                vec![self.raf_flag_eval.clone(), gamma_identity],
            ),
            output_field_expr(
                raf_claim.clone(),
                "field.add",
                vec![non_raf_lookup_operands, raf_flag_gamma_identity],
            ),
            output_field_expr(
                gamma_raf_claim.clone(),
                "field.mul",
                vec![self.gamma.clone(), raf_claim],
            ),
            output_field_expr(
                lookup_or_raf.clone(),
                "field.add",
                vec![table_value_family.symbol.clone(), gamma_raf_claim],
            ),
            output_field_expr(
                eq_ra.clone(),
                "field.mul",
                vec![eq.symbol.clone(), ra_product_family.symbol.clone()],
            ),
            output_field_expr(claim_expr.clone(), "field.mul", vec![eq_ra, lookup_or_raf]),
        ];

        Stage5InstructionReadRafOutputPlan {
            relation_output_values: vec![eq.clone()],
            field_exprs,
            claim: RelationOutputPlan {
                relation: "jolt.stage5.instruction_read_raf".to_owned(),
                structured_polynomial_evals: vec![StructuredPolynomialEvalRefPlan {
                    symbol: eq.symbol,
                    index: 0,
                }],
                eval_families: Vec::new(),
                product_families: vec![table_value_family, ra_product_family],
                function_families: Vec::new(),
                local_scalars: self
                    .point_values
                    .iter()
                    .map(|value| value.symbol.clone())
                    .collect(),
                expected_output: claim_expr,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafPointValueEmitPlan {
    pub(crate) symbol: String,
    pub(crate) kind: Stage5InstructionReadRafPointValueKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafOutputPlan {
    pub(crate) relation_output_values: Vec<StructuredPolynomialEvalPlan>,
    pub(crate) field_exprs: Vec<Stage5InstructionReadRafOutputFieldExprPlan>,
    pub(crate) claim: RelationOutputPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafOutputFieldExprPlan {
    pub(crate) symbol: String,
    pub(crate) formula: String,
    pub(crate) operands: Vec<String>,
}

impl Stage5InstructionReadRafPointValueEmitPlan {
    pub(crate) fn is_lookup_table(&self) -> bool {
        matches!(
            self.kind,
            Stage5InstructionReadRafPointValueKind::LookupTable { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stage5InstructionReadRafPointValueKind {
    LookupTable { index: usize },
    LeftOperand,
    RightOperand,
    Identity,
}

fn point_value_plans(table_count: usize) -> Vec<Stage5InstructionReadRafPointValueEmitPlan> {
    let mut values = (0..table_count)
        .map(|index| Stage5InstructionReadRafPointValueEmitPlan {
            symbol: format!("stage5.instruction_read_raf.point_value.LookupTable_{index}"),
            kind: Stage5InstructionReadRafPointValueKind::LookupTable { index },
        })
        .collect::<Vec<_>>();
    values.extend([
        Stage5InstructionReadRafPointValueEmitPlan {
            symbol: "stage5.instruction_read_raf.point_value.LeftLookupOperand".to_owned(),
            kind: Stage5InstructionReadRafPointValueKind::LeftOperand,
        },
        Stage5InstructionReadRafPointValueEmitPlan {
            symbol: "stage5.instruction_read_raf.point_value.RightLookupOperand".to_owned(),
            kind: Stage5InstructionReadRafPointValueKind::RightOperand,
        },
        Stage5InstructionReadRafPointValueEmitPlan {
            symbol: "stage5.instruction_read_raf.point_value.Identity".to_owned(),
            kind: Stage5InstructionReadRafPointValueKind::Identity,
        },
    ]);
    values
}

fn output_field_expr(
    symbol: String,
    formula: &str,
    operands: Vec<String>,
) -> Stage5InstructionReadRafOutputFieldExprPlan {
    Stage5InstructionReadRafOutputFieldExprPlan {
        symbol,
        formula: formula.to_owned(),
        operands,
    }
}

fn emit_point_value_constants(values: &[Stage5InstructionReadRafPointValueEmitPlan]) -> String {
    let values = values
        .iter()
        .map(|value| {
            format!(
                "    Stage5InstructionReadRafPointValuePlan {{ symbol: {}, kind: {} }},",
                rust_str(&value.symbol),
                point_value_kind_expr(&value.kind),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const STAGE5_INSTRUCTION_READ_RAF_POINT_VALUES: &[Stage5InstructionReadRafPointValuePlan] = &[\n{values}\n];\n\n"
    )
}

fn point_value_kind_expr(kind: &Stage5InstructionReadRafPointValueKind) -> String {
    match kind {
        Stage5InstructionReadRafPointValueKind::LookupTable { index } => {
            format!("Stage5InstructionReadRafPointValueKind::LookupTable {{ index: {index} }}")
        }
        Stage5InstructionReadRafPointValueKind::LeftOperand => {
            "Stage5InstructionReadRafPointValueKind::LeftOperand".to_owned()
        }
        Stage5InstructionReadRafPointValueKind::RightOperand => {
            "Stage5InstructionReadRafPointValueKind::RightOperand".to_owned()
        }
        Stage5InstructionReadRafPointValueKind::Identity => {
            "Stage5InstructionReadRafPointValueKind::Identity".to_owned()
        }
    }
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn indexed_eval_family_ref(index: usize) -> String {
    format!("&{STAGE5_INDEXED_EVAL_FAMILIES_CONST}[{index}]")
}

#[cfg(test)]
mod tests {
    use crate::emit::rust::EmitError;
    use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;

    use super::{
        Stage5InstructionReadRafEmitPlan, STAGE5_INSTRUCTION_RA_EVAL_FAMILY,
        STAGE5_TABLE_FLAG_EVAL_FAMILY,
    };

    #[test]
    fn instruction_read_raf_plan_groups_indexed_eval_families() -> Result<(), EmitError> {
        let families = instruction_read_raf_families([
            (
                "LookupTableFlag_1",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1",
            ),
            (
                "InstructionRafFlag",
                "stage5.instruction_read_raf.eval.InstructionRafFlag",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
            (
                "LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
            ),
        ]);
        let plan = Stage5InstructionReadRafEmitPlan::from_eval_families(&families)?;

        assert_eq!(
            plan.table_flag_evals.evals,
            vec![
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1"
            ]
        );
        assert_eq!(
            plan.instruction_ra_evals.evals,
            vec!["stage5.instruction_read_raf.eval.InstructionRa_0"]
        );
        assert_eq!(
            plan.table_flag_evals_ref,
            "&STAGE5_INDEXED_EVAL_FAMILIES[0]"
        );
        assert_eq!(
            plan.instruction_ra_evals_ref,
            "&STAGE5_INDEXED_EVAL_FAMILIES[1]"
        );
        Ok(())
    }

    #[test]
    fn instruction_read_raf_plan_references_indexed_eval_family_rows() -> Result<(), EmitError> {
        let families = instruction_read_raf_families([
            (
                "LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
        ]);
        let plan = Stage5InstructionReadRafEmitPlan::from_eval_families(&families)?;
        let source = plan.emit_runtime_constants();

        assert!(!source.contains("STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS"));
        assert!(!source.contains("STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS"));
        assert!(source.contains("table_flag_evals: &STAGE5_INDEXED_EVAL_FAMILIES[0]"));
        assert!(source.contains("instruction_ra_evals: &STAGE5_INDEXED_EVAL_FAMILIES[1]"));
        Ok(())
    }

    #[test]
    fn instruction_read_raf_relation_output_plan_is_typed() -> Result<(), EmitError> {
        let families = instruction_read_raf_families([
            (
                "LookupTableFlag_0",
                "stage5.instruction_read_raf.eval.LookupTableFlag_0",
            ),
            (
                "LookupTableFlag_1",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
        ]);
        let plan = Stage5InstructionReadRafEmitPlan::from_eval_families(&families)?;
        let output_plan = plan.relation_output_plan();

        assert_eq!(
            output_plan.claim.relation,
            "jolt.stage5.instruction_read_raf"
        );
        assert_eq!(
            output_plan.claim.expected_output,
            "stage5.instruction_read_raf.output.claim_expr"
        );
        assert_eq!(output_plan.claim.structured_polynomial_evals.len(), 1);
        assert_eq!(
            output_plan.claim.structured_polynomial_evals[0].symbol,
            "stage5.instruction_read_raf.output.eq.LookupOutputCycle"
        );
        assert_eq!(output_plan.claim.structured_polynomial_evals[0].index, 0);
        assert_eq!(output_plan.claim.product_families.len(), 2);
        assert_eq!(
            output_plan.claim.product_families[0].symbol,
            "stage5.instruction_read_raf.output.product.LookupTableValues"
        );
        assert_eq!(
            output_plan.claim.product_families[0].terms[0].evals,
            vec![
                "stage5.instruction_read_raf.point_value.LookupTable_0".to_owned(),
                "stage5.instruction_read_raf.eval.LookupTableFlag_0".to_owned()
            ]
        );
        assert_eq!(
            output_plan.claim.product_families[1].terms[0].evals,
            Vec::<String>::new()
        );
        assert_eq!(
            output_plan.claim.product_families[1].terms[0].eval_families,
            vec![STAGE5_INSTRUCTION_RA_EVAL_FAMILY.to_owned()]
        );
        assert_eq!(
            output_plan.claim.local_scalars,
            vec![
                "stage5.instruction_read_raf.point_value.LookupTable_0".to_owned(),
                "stage5.instruction_read_raf.point_value.LookupTable_1".to_owned(),
                "stage5.instruction_read_raf.point_value.LeftLookupOperand".to_owned(),
                "stage5.instruction_read_raf.point_value.RightLookupOperand".to_owned(),
                "stage5.instruction_read_raf.point_value.Identity".to_owned(),
            ]
        );
        assert!(output_plan.field_exprs.iter().any(|expr| {
            expr.symbol == "stage5.instruction_read_raf.output.claim_expr"
                && expr.formula == "field.mul"
                && expr.operands
                    == vec![
                        "stage5.instruction_read_raf.output.partial.EqRa".to_owned(),
                        "stage5.instruction_read_raf.output.partial.LookupOrRaf".to_owned(),
                    ]
        }));
        Ok(())
    }

    #[test]
    fn instruction_read_raf_plan_requires_explicit_eval_family_rows() {
        let families = [IndexedEvalFamilyPlan {
            symbol: STAGE5_TABLE_FLAG_EVAL_FAMILY.to_owned(),
            evals: vec!["stage5.instruction_read_raf.eval.LookupTableFlag_0".to_owned()],
        }];

        let error = Stage5InstructionReadRafEmitPlan::from_eval_families(&families)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(
            error.contains("missing eval family `stage5.instruction_read_raf.eval.InstructionRa`")
        );
    }

    fn instruction_read_raf_families<'a>(
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Vec<IndexedEvalFamilyPlan> {
        let evals = evals.into_iter().collect::<Vec<_>>();
        vec![
            IndexedEvalFamilyPlan {
                symbol: STAGE5_TABLE_FLAG_EVAL_FAMILY.to_owned(),
                evals: indexed_names("LookupTableFlag_", &evals),
            },
            IndexedEvalFamilyPlan {
                symbol: STAGE5_INSTRUCTION_RA_EVAL_FAMILY.to_owned(),
                evals: indexed_names("InstructionRa_", &evals),
            },
        ]
    }

    fn indexed_names(prefix: &str, evals: &[(&str, &str)]) -> Vec<String> {
        let mut names = evals
            .iter()
            .filter_map(|(oracle, name)| {
                oracle
                    .strip_prefix(prefix)
                    .and_then(|index| index.parse::<usize>().ok())
                    .map(|index| (index, (*name).to_owned()))
            })
            .collect::<Vec<_>>();
        names.sort_by_key(|(index, _)| *index);
        names.into_iter().map(|(_, name)| name).collect()
    }
}
