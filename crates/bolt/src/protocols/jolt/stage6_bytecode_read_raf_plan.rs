use crate::emit::rust::push_format;
use crate::protocols::jolt::verifier_output_claims::{
    SumcheckOutputClaimPlan, SumcheckOutputProductFamilyPlan, SumcheckOutputProductFamilyTermPlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeReadRafPlan {
    pub(crate) const_name: &'static str,
    pub(crate) point: &'static str,
    pub(crate) gamma: &'static str,
    pub(crate) bytecode_ra_eval_names_const: &'static str,
    pub(crate) bytecode_ra_eval_family_const: &'static str,
    pub(crate) bytecode_ra_eval_family_symbol: &'static str,
    pub(crate) bytecode_ra_evals: &'static [&'static str],
    pub(crate) entries: &'static str,
    pub(crate) entry_bytecode_index: &'static str,
    pub(crate) stages_const: &'static str,
    pub(crate) stages: &'static [BytecodeReadRafStagePlan],
    pub(crate) output_terms_const: &'static str,
    pub(crate) output_terms: &'static [BytecodeOutputTermPlan],
    pub(crate) output_contribution: &'static str,
    pub(crate) registers: BytecodeRegisterSymbols,
    pub(crate) entry_lookup_table: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeReadRafStagePlan {
    pub(crate) terms_const: &'static str,
    pub(crate) gamma: &'static str,
    pub(crate) cycle_point: &'static str,
    pub(crate) register_point: Option<&'static str>,
    pub(crate) terms: &'static [BytecodeReadRafTermPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BytecodeOutputTermPlan {
    StageValue {
        stage_index: usize,
        gamma_power: usize,
        identity_gamma_power: Option<usize>,
    },
    Entry {
        gamma_power: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeRegisterSymbols {
    pub(crate) rd: &'static str,
    pub(crate) rs1: &'static str,
    pub(crate) rs2: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BytecodeReadRafTermPlan {
    Address {
        gamma_power: usize,
    },
    Imm {
        gamma_power: usize,
    },
    CircuitFlag {
        index: usize,
        gamma_power: usize,
    },
    EntryFlag {
        flag: BytecodeFlag,
        expected: bool,
        gamma_power: usize,
    },
    RegisterEq {
        register: BytecodeRegister,
        gamma_power: usize,
    },
    LookupTable {
        gamma_base: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BytecodeFlag {
    IsInterleaved,
    IsBranch,
    LeftIsRs1,
    LeftIsPc,
    RightIsRs2,
    RightIsImm,
    IsNoop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BytecodeRegister {
    Rd,
    Rs1,
    Rs2,
}

const STAGE6_BYTECODE_STAGE1_TERMS: &[BytecodeReadRafTermPlan] = &[
    BytecodeReadRafTermPlan::Address { gamma_power: 0 },
    BytecodeReadRafTermPlan::Imm { gamma_power: 1 },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 0,
        gamma_power: 2,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 1,
        gamma_power: 3,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 2,
        gamma_power: 4,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 3,
        gamma_power: 5,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 4,
        gamma_power: 6,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 5,
        gamma_power: 7,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 6,
        gamma_power: 8,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 7,
        gamma_power: 9,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 8,
        gamma_power: 10,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 9,
        gamma_power: 11,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 10,
        gamma_power: 12,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 11,
        gamma_power: 13,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 12,
        gamma_power: 14,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 13,
        gamma_power: 15,
    },
];

const STAGE6_BYTECODE_STAGE2_TERMS: &[BytecodeReadRafTermPlan] = &[
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 5,
        gamma_power: 0,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::IsBranch,
        expected: true,
        gamma_power: 1,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 6,
        gamma_power: 2,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 7,
        gamma_power: 3,
    },
];

const STAGE6_BYTECODE_STAGE3_TERMS: &[BytecodeReadRafTermPlan] = &[
    BytecodeReadRafTermPlan::Imm { gamma_power: 0 },
    BytecodeReadRafTermPlan::Address { gamma_power: 1 },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::LeftIsRs1,
        expected: true,
        gamma_power: 2,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::LeftIsPc,
        expected: true,
        gamma_power: 3,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::RightIsRs2,
        expected: true,
        gamma_power: 4,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::RightIsImm,
        expected: true,
        gamma_power: 5,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::IsNoop,
        expected: true,
        gamma_power: 6,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 7,
        gamma_power: 7,
    },
    BytecodeReadRafTermPlan::CircuitFlag {
        index: 12,
        gamma_power: 8,
    },
];

const STAGE6_BYTECODE_STAGE4_TERMS: &[BytecodeReadRafTermPlan] = &[
    BytecodeReadRafTermPlan::RegisterEq {
        register: BytecodeRegister::Rd,
        gamma_power: 0,
    },
    BytecodeReadRafTermPlan::RegisterEq {
        register: BytecodeRegister::Rs1,
        gamma_power: 1,
    },
    BytecodeReadRafTermPlan::RegisterEq {
        register: BytecodeRegister::Rs2,
        gamma_power: 2,
    },
];

const STAGE6_BYTECODE_STAGE5_TERMS: &[BytecodeReadRafTermPlan] = &[
    BytecodeReadRafTermPlan::RegisterEq {
        register: BytecodeRegister::Rd,
        gamma_power: 0,
    },
    BytecodeReadRafTermPlan::EntryFlag {
        flag: BytecodeFlag::IsInterleaved,
        expected: false,
        gamma_power: 1,
    },
    BytecodeReadRafTermPlan::LookupTable { gamma_base: 2 },
];

const STAGE6_BYTECODE_STAGES: &[BytecodeReadRafStagePlan] = &[
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE1_TERMS",
        gamma: "stage6.bytecode_read_raf.stage1_gamma",
        cycle_point: "stage6.input.stage1.Imm",
        register_point: None,
        terms: STAGE6_BYTECODE_STAGE1_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE2_TERMS",
        gamma: "stage6.bytecode_read_raf.stage2_gamma",
        cycle_point: "stage6.input.stage2.OpFlagJump",
        register_point: None,
        terms: STAGE6_BYTECODE_STAGE2_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE3_TERMS",
        gamma: "stage6.bytecode_read_raf.stage3_gamma",
        cycle_point: "stage6.input.stage3.spartan_shift.UnexpandedPC",
        register_point: None,
        terms: STAGE6_BYTECODE_STAGE3_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE4_TERMS",
        gamma: "stage6.bytecode_read_raf.stage4_gamma",
        cycle_point: "stage6.input.stage4.Rs1Ra",
        register_point: Some("stage6.input.stage4.Rs1Ra"),
        terms: STAGE6_BYTECODE_STAGE4_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE5_TERMS",
        gamma: "stage6.bytecode_read_raf.stage5_gamma",
        cycle_point: "stage6.input.stage5.registers_val_evaluation.RdWa",
        register_point: Some("stage6.input.stage5.registers_val_evaluation.RdWa"),
        terms: STAGE6_BYTECODE_STAGE5_TERMS,
    },
];

const STAGE6_BYTECODE_OUTPUT_TERMS: &[BytecodeOutputTermPlan] = &[
    BytecodeOutputTermPlan::StageValue {
        stage_index: 0,
        gamma_power: 0,
        identity_gamma_power: Some(5),
    },
    BytecodeOutputTermPlan::StageValue {
        stage_index: 1,
        gamma_power: 1,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::StageValue {
        stage_index: 2,
        gamma_power: 2,
        identity_gamma_power: Some(4),
    },
    BytecodeOutputTermPlan::StageValue {
        stage_index: 3,
        gamma_power: 3,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::StageValue {
        stage_index: 4,
        gamma_power: 4,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::Entry { gamma_power: 7 },
];

const STAGE6_BYTECODE_READ_RAF_PLAN: BytecodeReadRafPlan = BytecodeReadRafPlan {
    const_name: "STAGE6_BYTECODE_PLAN",
    point: "stage6.bytecode_read_raf.point",
    gamma: "stage6.bytecode_read_raf.gamma",
    bytecode_ra_eval_names_const: "STAGE6_BYTECODE_RA_EVAL_NAMES",
    bytecode_ra_eval_family_const: "STAGE6_BYTECODE_RA_EVALS",
    bytecode_ra_eval_family_symbol: "stage6.bytecode_read_raf.eval.BytecodeRa",
    bytecode_ra_evals: &[
        "stage6.bytecode_read_raf.eval.BytecodeRa_0",
        "stage6.bytecode_read_raf.eval.BytecodeRa_1",
        "stage6.bytecode_read_raf.eval.BytecodeRa_2",
    ],
    entries: "stage6.bytecode_read_raf.entries",
    entry_bytecode_index: "stage6.bytecode_read_raf.entry_bytecode_index",
    stages_const: "STAGE6_BYTECODE_STAGES",
    stages: STAGE6_BYTECODE_STAGES,
    output_terms_const: "STAGE6_BYTECODE_OUTPUT_TERMS",
    output_terms: STAGE6_BYTECODE_OUTPUT_TERMS,
    output_contribution: "stage6.bytecode_read_raf.output.contribution",
    registers: BytecodeRegisterSymbols {
        rd: "stage6.bytecode.entry.rd",
        rs1: "stage6.bytecode.entry.rs1",
        rs2: "stage6.bytecode.entry.rs2",
    },
    entry_lookup_table: "stage6.bytecode.entry.lookup_table",
};

pub(crate) fn emit_stage6_bytecode_read_raf_plan_constants() -> String {
    emit_bytecode_read_raf_plan(&STAGE6_BYTECODE_READ_RAF_PLAN)
}

#[cfg(test)]
pub(crate) fn stage6_bytecode_read_raf_output_contribution_symbol() -> &'static str {
    STAGE6_BYTECODE_READ_RAF_PLAN.output_contribution
}

pub(crate) fn stage6_bytecode_read_raf_output_claim_plan() -> SumcheckOutputClaimPlan {
    STAGE6_BYTECODE_READ_RAF_PLAN.output_claim_plan()
}

impl BytecodeReadRafPlan {
    fn output_claim_plan(&self) -> SumcheckOutputClaimPlan {
        let product_family = SumcheckOutputProductFamilyPlan {
            symbol: "stage6.bytecode_read_raf.output.product.BytecodeReadRaf".to_owned(),
            gamma: None,
            terms: vec![SumcheckOutputProductFamilyTermPlan {
                gamma_power_offset: 0,
                evals: std::iter::once(self.output_contribution.to_owned())
                    .chain(self.bytecode_ra_evals.iter().map(|eval| (*eval).to_owned()))
                    .collect(),
                factors: Vec::new(),
            }],
        };
        SumcheckOutputClaimPlan {
            relation: "jolt.stage6.bytecode_read_raf".to_owned(),
            polynomial_evals: Vec::new(),
            eval_families: Vec::new(),
            product_families: vec![product_family.clone()],
            function_families: Vec::new(),
            local_scalars: vec![self.output_contribution.to_owned()],
            claim_value: product_family.symbol,
        }
    }
}

fn emit_bytecode_read_raf_plan(plan: &BytecodeReadRafPlan) -> String {
    let mut source = "\n".to_owned();

    push_format(
        &mut source,
        format_args!(
            "#[rustfmt::skip]\nconst {}: &[&str] = &[{}];\n",
            plan.bytecode_ra_eval_names_const,
            plan.bytecode_ra_evals
                .iter()
                .map(|eval| rust_str(eval))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    );
    push_format(
        &mut source,
        format_args!(
            "const {}: bolt_verifier_runtime::NamedEvalFamilyPlan = bolt_verifier_runtime::NamedEvalFamilyPlan {{ symbol: {}, evals: {} }};\n\n",
            plan.bytecode_ra_eval_family_const,
            rust_str(plan.bytecode_ra_eval_family_symbol),
            plan.bytecode_ra_eval_names_const,
        ),
    );

    for stage in plan.stages {
        push_format(
            &mut source,
            format_args!(
                "const {}: &[Stage67BytecodeTermPlan] = &[\n",
                stage.terms_const
            ),
        );
        for term in stage.terms {
            push_format(
                &mut source,
                format_args!("    {},\n", emit_bytecode_read_raf_term_plan(term)),
            );
        }
        source.push_str("];\n");
    }

    push_format(
        &mut source,
        format_args!(
            "const {}: &[Stage67BytecodeStagePlan] = &[\n",
            plan.stages_const
        ),
    );
    for stage in plan.stages {
        push_format(
            &mut source,
            format_args!(
                "    Stage67BytecodeStagePlan {{ gamma: {}, cycle_point: {}, register_point: {}, terms: {} }},\n",
                rust_str(stage.gamma),
                rust_str(stage.cycle_point),
                rust_option_str(stage.register_point),
                stage.terms_const,
            ),
        );
    }
    source.push_str("];\n\n");

    push_format(
        &mut source,
        format_args!(
            "const {}: &[Stage67BytecodeOutputTermPlan] = &[\n",
            plan.output_terms_const
        ),
    );
    for term in plan.output_terms {
        push_format(
            &mut source,
            format_args!("    {},\n", emit_bytecode_output_term_plan(term)),
        );
    }
    source.push_str("];\n\n");

    push_format(
        &mut source,
        format_args!(
            "const {}: Stage67BytecodeReadRafPlan = Stage67BytecodeReadRafPlan {{\n",
            plan.const_name
        ),
    );
    push_format(
        &mut source,
        format_args!(
            "    point: {},\n    gamma: {},\n    bytecode_ra_evals: &{},\n    entries: {},\n    entry_bytecode_index: {},\n    stages: {},\n    output_terms: {},\n    output_contribution: {},\n",
            rust_str(plan.point),
            rust_str(plan.gamma),
            plan.bytecode_ra_eval_family_const,
            rust_str(plan.entries),
            rust_str(plan.entry_bytecode_index),
            plan.stages_const,
            plan.output_terms_const,
            rust_str(plan.output_contribution),
        ),
    );
    source.push_str("    registers: Stage67BytecodeRegisterSymbols {\n");
    push_format(
        &mut source,
        format_args!(
            "        rd: {},\n        rs1: {},\n        rs2: {},\n",
            rust_str(plan.registers.rd),
            rust_str(plan.registers.rs1),
            rust_str(plan.registers.rs2),
        ),
    );
    source.push_str("    },\n");
    push_format(
        &mut source,
        format_args!(
            "    entry_lookup_table: {},\n",
            rust_str(plan.entry_lookup_table)
        ),
    );
    source.push_str("};\n");
    source
}

fn emit_bytecode_read_raf_term_plan(term: &BytecodeReadRafTermPlan) -> String {
    match term {
        BytecodeReadRafTermPlan::Address { gamma_power } => {
            format!("Stage67BytecodeTermPlan::Address {{ gamma_power: {gamma_power} }}")
        }
        BytecodeReadRafTermPlan::Imm { gamma_power } => {
            format!("Stage67BytecodeTermPlan::Imm {{ gamma_power: {gamma_power} }}")
        }
        BytecodeReadRafTermPlan::CircuitFlag { index, gamma_power } => format!(
            "Stage67BytecodeTermPlan::CircuitFlag {{ index: {index}, gamma_power: {gamma_power} }}"
        ),
        BytecodeReadRafTermPlan::EntryFlag {
            flag,
            expected,
            gamma_power,
        } => format!(
            "Stage67BytecodeTermPlan::EntryFlag {{ flag: {}, expected: {expected}, gamma_power: {gamma_power} }}",
            emit_bytecode_flag(*flag)
        ),
        BytecodeReadRafTermPlan::RegisterEq {
            register,
            gamma_power,
        } => format!(
            "Stage67BytecodeTermPlan::RegisterEq {{ register: {}, gamma_power: {gamma_power} }}",
            emit_bytecode_register(*register)
        ),
        BytecodeReadRafTermPlan::LookupTable { gamma_base } => {
            format!("Stage67BytecodeTermPlan::LookupTable {{ gamma_base: {gamma_base} }}")
        }
    }
}

fn emit_bytecode_output_term_plan(term: &BytecodeOutputTermPlan) -> String {
    match *term {
        BytecodeOutputTermPlan::StageValue {
            stage_index,
            gamma_power,
            identity_gamma_power,
        } => format!(
            "Stage67BytecodeOutputTermPlan::StageValue {{ stage_index: {stage_index}, gamma_power: {gamma_power}, identity_gamma_power: {} }}",
            emit_option_usize(identity_gamma_power),
        ),
        BytecodeOutputTermPlan::Entry { gamma_power } => {
            format!("Stage67BytecodeOutputTermPlan::Entry {{ gamma_power: {gamma_power} }}")
        }
    }
}

fn emit_bytecode_flag(flag: BytecodeFlag) -> &'static str {
    match flag {
        BytecodeFlag::IsInterleaved => "Stage67BytecodeFlag::IsInterleaved",
        BytecodeFlag::IsBranch => "Stage67BytecodeFlag::IsBranch",
        BytecodeFlag::LeftIsRs1 => "Stage67BytecodeFlag::LeftIsRs1",
        BytecodeFlag::LeftIsPc => "Stage67BytecodeFlag::LeftIsPc",
        BytecodeFlag::RightIsRs2 => "Stage67BytecodeFlag::RightIsRs2",
        BytecodeFlag::RightIsImm => "Stage67BytecodeFlag::RightIsImm",
        BytecodeFlag::IsNoop => "Stage67BytecodeFlag::IsNoop",
    }
}

fn emit_bytecode_register(register: BytecodeRegister) -> &'static str {
    match register {
        BytecodeRegister::Rd => "Stage67BytecodeRegister::Rd",
        BytecodeRegister::Rs1 => "Stage67BytecodeRegister::Rs1",
        BytecodeRegister::Rs2 => "Stage67BytecodeRegister::Rs2",
    }
}

fn emit_option_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| format!("Some({value})"))
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        emit_stage6_bytecode_read_raf_plan_constants, stage6_bytecode_read_raf_output_claim_plan,
        stage6_bytecode_read_raf_output_contribution_symbol, BytecodeFlag, BytecodeOutputTermPlan,
        BytecodeReadRafTermPlan, BytecodeRegister, STAGE6_BYTECODE_READ_RAF_PLAN,
    };

    #[test]
    fn stage6_bytecode_plan_rows_encode_the_read_raf_reduction() {
        let plan = &STAGE6_BYTECODE_READ_RAF_PLAN;

        assert_eq!(plan.stages.len(), 5);
        assert_eq!(plan.output_terms.len(), 6);
        assert_eq!(plan.registers.rd, "stage6.bytecode.entry.rd");
        assert_eq!(
            plan.entry_lookup_table,
            "stage6.bytecode.entry.lookup_table"
        );
        assert_eq!(
            plan.output_contribution,
            "stage6.bytecode_read_raf.output.contribution"
        );

        let stage1 = &plan.stages[0];
        assert_eq!(stage1.terms.len(), 16);
        assert_eq!(
            plan.output_terms[0],
            BytecodeOutputTermPlan::StageValue {
                stage_index: 0,
                gamma_power: 0,
                identity_gamma_power: Some(5),
            }
        );
        assert_eq!(
            plan.output_terms[5],
            BytecodeOutputTermPlan::Entry { gamma_power: 7 }
        );
        assert_eq!(
            stage1.terms[0],
            BytecodeReadRafTermPlan::Address { gamma_power: 0 }
        );
        assert_eq!(
            stage1.terms[1],
            BytecodeReadRafTermPlan::Imm { gamma_power: 1 }
        );
        for (index, term) in stage1.terms[2..].iter().enumerate() {
            assert_eq!(
                *term,
                BytecodeReadRafTermPlan::CircuitFlag {
                    index,
                    gamma_power: index + 2,
                }
            );
        }

        assert_eq!(
            plan.stages[1].terms[1],
            BytecodeReadRafTermPlan::EntryFlag {
                flag: BytecodeFlag::IsBranch,
                expected: true,
                gamma_power: 1,
            }
        );
        assert_eq!(
            plan.stages[3].terms,
            &[
                BytecodeReadRafTermPlan::RegisterEq {
                    register: BytecodeRegister::Rd,
                    gamma_power: 0,
                },
                BytecodeReadRafTermPlan::RegisterEq {
                    register: BytecodeRegister::Rs1,
                    gamma_power: 1,
                },
                BytecodeReadRafTermPlan::RegisterEq {
                    register: BytecodeRegister::Rs2,
                    gamma_power: 2,
                },
            ]
        );
        assert_eq!(
            plan.stages[4].terms,
            &[
                BytecodeReadRafTermPlan::RegisterEq {
                    register: BytecodeRegister::Rd,
                    gamma_power: 0,
                },
                BytecodeReadRafTermPlan::EntryFlag {
                    flag: BytecodeFlag::IsInterleaved,
                    expected: false,
                    gamma_power: 1,
                },
                BytecodeReadRafTermPlan::LookupTable { gamma_base: 2 },
            ]
        );
    }

    #[test]
    fn stage6_bytecode_output_claim_plan_uses_point_derived_contribution() {
        let claim = stage6_bytecode_read_raf_output_claim_plan();

        assert_eq!(claim.relation, "jolt.stage6.bytecode_read_raf");
        assert_eq!(
            claim.claim_value,
            "stage6.bytecode_read_raf.output.product.BytecodeReadRaf"
        );
        assert!(claim.polynomial_evals.is_empty());
        assert!(claim.eval_families.is_empty());
        assert_eq!(claim.product_families.len(), 1);
        assert_eq!(claim.product_families[0].terms.len(), 1);
        assert_eq!(
            claim.local_scalars,
            vec![stage6_bytecode_read_raf_output_contribution_symbol().to_owned()]
        );
        assert_eq!(
            claim.product_families[0].terms[0].evals,
            vec![
                stage6_bytecode_read_raf_output_contribution_symbol().to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_0".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_1".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_2".to_owned(),
            ]
        );
    }

    #[test]
    fn stage6_bytecode_plan_renderer_emits_stage67_constants() {
        let source = emit_stage6_bytecode_read_raf_plan_constants();

        assert!(source.contains("const STAGE6_BYTECODE_STAGE1_TERMS"));
        assert!(source.contains("Stage67BytecodeTermPlan::LookupTable { gamma_base: 2 }"));
        assert!(source.contains("Stage67BytecodeFlag::IsInterleaved"));
        assert!(source.contains("Stage67BytecodeRegister::Rs2"));
        assert!(source.contains("const STAGE6_BYTECODE_RA_EVAL_NAMES"));
        assert!(source.contains(
            "const STAGE6_BYTECODE_RA_EVALS: bolt_verifier_runtime::NamedEvalFamilyPlan"
        ));
        assert!(source.contains("const STAGE6_BYTECODE_OUTPUT_TERMS"));
        assert!(source.contains("Stage67BytecodeOutputTermPlan::Entry { gamma_power: 7 }"));
        assert!(source
            .contains("output_contribution: \"stage6.bytecode_read_raf.output.contribution\""));
        assert!(source.contains("const STAGE6_BYTECODE_PLAN: Stage67BytecodeReadRafPlan"));
    }
}
