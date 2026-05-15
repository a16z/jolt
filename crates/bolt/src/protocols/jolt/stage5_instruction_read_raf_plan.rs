use crate::emit::rust::{push_format, EmitError};
use crate::protocols::jolt::verifier_output_claims::{
    StructuredPolynomialEvalPlan, StructuredPolynomialKind, StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder, StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment, SumcheckOutputClaimPlan, SumcheckOutputProductFamilyPlan,
    SumcheckOutputProductFamilyTermPlan,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5InstructionReadRafEmitPlan {
    pub(crate) point: String,
    pub(crate) lookup_output_point: String,
    pub(crate) table_flag_evals: Stage5NamedEvalFamilyEmitPlan,
    pub(crate) instruction_ra_evals: Stage5NamedEvalFamilyEmitPlan,
    pub(crate) raf_flag_eval: String,
    pub(crate) gamma: String,
    pub(crate) point_values: Vec<Stage5InstructionReadRafPointValueEmitPlan>,
    pub(crate) log_k: usize,
}

impl Stage5InstructionReadRafEmitPlan {
    pub(crate) fn from_evals<'a>(
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, EmitError> {
        let evals = evals.into_iter().collect::<Vec<_>>();
        let table_flag_evals = Stage5NamedEvalFamilyEmitPlan::from_indexed_oracles(
            "stage5.instruction_read_raf.eval.LookupTableFlag",
            "LookupTableFlag_",
            evals.iter().copied(),
        )?;
        let instruction_ra_evals = Stage5NamedEvalFamilyEmitPlan::from_indexed_oracles(
            "stage5.instruction_read_raf.eval.InstructionRa",
            "InstructionRa_",
            evals.iter().copied(),
        )?;
        Ok(Self {
            point: "stage5.instruction_read_raf.point".to_owned(),
            lookup_output_point: "stage5.input.stage2.instruction.LookupOutput".to_owned(),
            point_values: point_value_plans(table_flag_evals.evals.len()),
            table_flag_evals,
            instruction_ra_evals,
            raf_flag_eval: "stage5.instruction_read_raf.eval.InstructionRafFlag".to_owned(),
            gamma: "stage5.instruction_read_raf.gamma".to_owned(),
            log_k: 128,
        })
    }

    pub(crate) fn emit_runtime_constants(&self) -> String {
        let families = [
            (
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS",
                &self.table_flag_evals,
            ),
            (
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS",
                &self.instruction_ra_evals,
            ),
        ];

        let mut source = String::new();
        for (names_const, family_const, family) in families {
            let names_source = family
                .evals
                .iter()
                .map(|name| rust_str(name))
                .collect::<Vec<_>>()
                .join(", ");
            push_format(
                &mut source,
                format_args!(
                    "#[rustfmt::skip]\npub const {names_const}: &[&str] = &[{names_source}];\n"
                ),
            );
            push_format(
                &mut source,
                format_args!(
                    "pub const {family_const}: NamedEvalFamilyPlan = NamedEvalFamilyPlan {{ symbol: {}, evals: {names_const} }};\n\n",
                    rust_str(&family.symbol),
                ),
            );
        }
        source.push_str(&emit_point_value_constants(&self.point_values));
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_INSTRUCTION_READ_RAF_PLAN: Stage5InstructionReadRafPlan = Stage5InstructionReadRafPlan {{\n\
                 \x20   point: {},\n\
                 \x20   lookup_output_point: {},\n\
                 \x20   table_flag_evals: &STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS,\n\
                 \x20   instruction_ra_evals: &STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS,\n\
                 \x20   raf_flag_eval: {},\n\
                 \x20   gamma: {},\n\
                 \x20   point_values: STAGE5_INSTRUCTION_READ_RAF_POINT_VALUES,\n\
                 \x20   log_k: {},\n\
                 }};\n\n",
                rust_str(&self.point),
                rust_str(&self.lookup_output_point),
                rust_str(&self.raf_flag_eval),
                rust_str(&self.gamma),
                self.log_k,
            ),
        );
        source
    }

    pub(crate) fn output_claim_plan(&self) -> Stage5InstructionReadRafOutputPlan {
        const PREFIX: &str = "stage5.instruction_read_raf.output";

        let table_value_family = SumcheckOutputProductFamilyPlan {
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
                    |(flag_eval, table_value)| SumcheckOutputProductFamilyTermPlan {
                        gamma_power_offset: 0,
                        evals: vec![table_value.symbol.clone(), flag_eval.clone()],
                        factors: Vec::new(),
                    },
                )
                .collect(),
        };
        let ra_product_family = SumcheckOutputProductFamilyPlan {
            symbol: format!("{PREFIX}.product.InstructionRa"),
            gamma: None,
            terms: vec![SumcheckOutputProductFamilyTermPlan {
                gamma_power_offset: 0,
                evals: self.instruction_ra_evals.evals.clone(),
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
            field_exprs,
            claim: SumcheckOutputClaimPlan {
                relation: "jolt.stage5.instruction_read_raf".to_owned(),
                polynomial_evals: vec![eq],
                eval_families: Vec::new(),
                product_families: vec![table_value_family, ra_product_family],
                function_families: Vec::new(),
                local_scalars: self
                    .point_values
                    .iter()
                    .map(|value| value.symbol.clone())
                    .collect(),
                claim_value: claim_expr,
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
    pub(crate) field_exprs: Vec<Stage5InstructionReadRafOutputFieldExprPlan>,
    pub(crate) claim: SumcheckOutputClaimPlan,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage5NamedEvalFamilyEmitPlan {
    pub(crate) symbol: String,
    pub(crate) evals: Vec<String>,
}

impl Stage5NamedEvalFamilyEmitPlan {
    fn from_indexed_oracles<'a>(
        symbol: &str,
        oracle_prefix: &str,
        evals: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, EmitError> {
        let mut indexed_names = Vec::new();
        for (oracle, name) in evals {
            let Some(suffix) = oracle.strip_prefix(oracle_prefix) else {
                continue;
            };
            let index = suffix.parse::<usize>().map_err(|_| {
                EmitError::new(format!(
                    "invalid indexed eval oracle `{oracle}` for family `{symbol}`"
                ))
            })?;
            indexed_names.push((index, name.to_owned()));
        }
        if indexed_names.is_empty() {
            return Err(EmitError::new(format!("missing eval family `{symbol}`")));
        }
        indexed_names.sort_by_key(|(index, _)| *index);
        for (expected, (actual, _)) in indexed_names.iter().enumerate() {
            if expected != *actual {
                return Err(EmitError::new(format!(
                    "non-contiguous eval family `{symbol}` at index {actual}"
                )));
            }
        }
        Ok(Self {
            symbol: symbol.to_owned(),
            evals: indexed_names
                .into_iter()
                .map(|(_, name)| name)
                .collect::<Vec<_>>(),
        })
    }
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

#[cfg(test)]
mod tests {
    use crate::emit::rust::EmitError;

    use super::Stage5InstructionReadRafEmitPlan;

    #[test]
    fn instruction_read_raf_plan_groups_indexed_eval_families() -> Result<(), EmitError> {
        let plan = Stage5InstructionReadRafEmitPlan::from_evals([
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
        ])?;

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
        Ok(())
    }

    #[test]
    fn instruction_read_raf_output_claim_plan_is_typed() -> Result<(), EmitError> {
        let plan = Stage5InstructionReadRafEmitPlan::from_evals([
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
        ])?;
        let output_plan = plan.output_claim_plan();

        assert_eq!(
            output_plan.claim.relation,
            "jolt.stage5.instruction_read_raf"
        );
        assert_eq!(
            output_plan.claim.claim_value,
            "stage5.instruction_read_raf.output.claim_expr"
        );
        assert_eq!(output_plan.claim.polynomial_evals.len(), 1);
        assert_eq!(
            output_plan.claim.polynomial_evals[0].symbol,
            "stage5.instruction_read_raf.output.eq.LookupOutputCycle"
        );
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
            vec!["stage5.instruction_read_raf.eval.InstructionRa_0".to_owned()]
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
    fn instruction_read_raf_plan_rejects_non_contiguous_eval_families() -> Result<(), EmitError> {
        let error = match Stage5InstructionReadRafEmitPlan::from_evals([
            (
                "LookupTableFlag_1",
                "stage5.instruction_read_raf.eval.LookupTableFlag_1",
            ),
            (
                "InstructionRa_0",
                "stage5.instruction_read_raf.eval.InstructionRa_0",
            ),
        ]) {
            Ok(_) => {
                return Err(EmitError::new(
                    "non-contiguous table flag family should fail planning",
                ));
            }
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("non-contiguous eval family `stage5.instruction_read_raf.eval.LookupTableFlag` at index 1")
        );
        Ok(())
    }
}
