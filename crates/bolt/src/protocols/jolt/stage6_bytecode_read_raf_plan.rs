use crate::emit::rust::{push_format, EmitError};
use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;
use crate::protocols::jolt::verifier_relation_outputs::{
    RelationOutputFieldExprPlan, RelationOutputPlan,
};

pub(crate) const STAGE6_BYTECODE_RA_EVAL_FAMILY: &str = "stage6.bytecode_read_raf.eval.BytecodeRa";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeReadRafPlan {
    pub(crate) const_name: &'static str,
    pub(crate) point: &'static str,
    pub(crate) gamma: &'static str,
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
        symbol: &'static str,
        stage_index: usize,
        gamma_power: usize,
        identity_gamma_power: Option<usize>,
    },
    Entry {
        symbol: &'static str,
        gamma_power: usize,
    },
}

impl BytecodeOutputTermPlan {
    fn symbol(self) -> &'static str {
        match self {
            Self::StageValue { symbol, .. } | Self::Entry { symbol, .. } => symbol,
        }
    }
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
        symbol: "stage6.bytecode_read_raf.output.term.Stage1",
        stage_index: 0,
        gamma_power: 0,
        identity_gamma_power: Some(5),
    },
    BytecodeOutputTermPlan::StageValue {
        symbol: "stage6.bytecode_read_raf.output.term.Stage2",
        stage_index: 1,
        gamma_power: 1,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::StageValue {
        symbol: "stage6.bytecode_read_raf.output.term.Stage3",
        stage_index: 2,
        gamma_power: 2,
        identity_gamma_power: Some(4),
    },
    BytecodeOutputTermPlan::StageValue {
        symbol: "stage6.bytecode_read_raf.output.term.Stage4",
        stage_index: 3,
        gamma_power: 3,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::StageValue {
        symbol: "stage6.bytecode_read_raf.output.term.Stage5",
        stage_index: 4,
        gamma_power: 4,
        identity_gamma_power: None,
    },
    BytecodeOutputTermPlan::Entry {
        symbol: "stage6.bytecode_read_raf.output.term.Entry",
        gamma_power: 7,
    },
];

const STAGE6_BYTECODE_READ_RAF_PLAN: BytecodeReadRafPlan = BytecodeReadRafPlan {
    const_name: "STAGE6_BYTECODE_PLAN",
    point: "stage6.bytecode_read_raf.point",
    gamma: "stage6.bytecode_read_raf.gamma",
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage6BytecodeReadRafEmitPlan {
    pub(crate) bytecode_ra_evals: IndexedEvalFamilyPlan,
    pub(crate) bytecode_ra_evals_ref: String,
}

impl Stage6BytecodeReadRafEmitPlan {
    pub(crate) fn from_eval_families(
        eval_families: &[IndexedEvalFamilyPlan],
    ) -> Result<Self, EmitError> {
        let (bytecode_ra_evals_ref, bytecode_ra_evals) = stage6_bytecode_read_raf_eval_family_ref(
            eval_families,
            "STAGE6_INDEXED_EVAL_FAMILIES",
        )?;
        Ok(Self {
            bytecode_ra_evals: bytecode_ra_evals.clone(),
            bytecode_ra_evals_ref,
        })
    }

    pub(crate) fn emit_runtime_constants(&self) -> String {
        emit_bytecode_read_raf_plan(&STAGE6_BYTECODE_READ_RAF_PLAN, &self.bytecode_ra_evals_ref)
    }

    pub(crate) fn relation_output_plan(&self) -> Stage6BytecodeReadRafRelationOutputPlan {
        STAGE6_BYTECODE_READ_RAF_PLAN.relation_output_plan(&self.bytecode_ra_evals)
    }

    pub(crate) fn local_scalar_symbols() -> impl Iterator<Item = &'static str> {
        STAGE6_BYTECODE_READ_RAF_PLAN
            .output_terms
            .iter()
            .copied()
            .map(BytecodeOutputTermPlan::symbol)
    }
}

#[cfg(test)]
pub(crate) fn stage6_bytecode_read_raf_output_contribution_symbol() -> &'static str {
    STAGE6_BYTECODE_READ_RAF_PLAN.output_contribution
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage6BytecodeReadRafRelationOutputPlan {
    pub(crate) field_exprs: Vec<RelationOutputFieldExprPlan>,
    pub(crate) claim: RelationOutputPlan,
}

impl BytecodeReadRafPlan {
    fn relation_output_plan(
        &self,
        bytecode_ra_evals: &IndexedEvalFamilyPlan,
    ) -> Stage6BytecodeReadRafRelationOutputPlan {
        let bytecode_ra_product = "stage6.bytecode_read_raf.output.product.BytecodeRa".to_owned();
        let claim_expr = "stage6.bytecode_read_raf.output.claim_expr".to_owned();
        let output_term_symbols = self
            .output_terms
            .iter()
            .copied()
            .map(BytecodeOutputTermPlan::symbol)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        Stage6BytecodeReadRafRelationOutputPlan {
            field_exprs: vec![
                RelationOutputFieldExprPlan {
                    symbol: bytecode_ra_product.clone(),
                    formula: "field_vector.product".to_owned(),
                    operands: vec![bytecode_ra_evals.symbol.clone()],
                },
                RelationOutputFieldExprPlan {
                    symbol: self.output_contribution.to_owned(),
                    formula: "field.sum".to_owned(),
                    operands: output_term_symbols.clone(),
                },
                RelationOutputFieldExprPlan {
                    symbol: claim_expr.clone(),
                    formula: "field.product".to_owned(),
                    operands: vec![self.output_contribution.to_owned(), bytecode_ra_product],
                },
            ],
            claim: RelationOutputPlan::with_local_scalars(
                "jolt.stage6.bytecode_read_raf",
                output_term_symbols,
                claim_expr,
            ),
        }
    }
}

fn emit_bytecode_read_raf_plan(plan: &BytecodeReadRafPlan, bytecode_ra_evals_ref: &str) -> String {
    let mut source = "\n".to_owned();

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
            "pub const {}: Stage67BytecodeReadRafPlan = Stage67BytecodeReadRafPlan {{\n",
            plan.const_name
        ),
    );
    push_format(
        &mut source,
        format_args!(
            "    point: {},\n    gamma: {},\n    bytecode_ra_evals: &{},\n    entries: {},\n    entry_bytecode_index: {},\n    stages: {},\n    output_terms: {},\n    output_contribution: {},\n",
            rust_str(plan.point),
            rust_str(plan.gamma),
            bytecode_ra_evals_ref,
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

fn stage6_bytecode_read_raf_eval_family_ref<'a>(
    eval_families: &'a [IndexedEvalFamilyPlan],
    families_const: &str,
) -> Result<(String, &'a IndexedEvalFamilyPlan), EmitError> {
    let (index, family) =
        IndexedEvalFamilyPlan::find_with_index(eval_families, STAGE6_BYTECODE_RA_EVAL_FAMILY)?;
    Ok((format!("{families_const}[{index}]"), family))
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
            symbol,
            stage_index,
            gamma_power,
            identity_gamma_power,
        } => format!(
            "Stage67BytecodeOutputTermPlan::StageValue {{ symbol: {}, stage_index: {stage_index}, gamma_power: {gamma_power}, identity_gamma_power: {} }}",
            rust_str(symbol),
            emit_option_usize(identity_gamma_power),
        ),
        BytecodeOutputTermPlan::Entry {
            symbol,
            gamma_power,
        } => {
            format!(
                "Stage67BytecodeOutputTermPlan::Entry {{ symbol: {}, gamma_power: {gamma_power} }}",
                rust_str(symbol)
            )
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
    use crate::emit::rust::EmitError;
    use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;

    use super::{
        stage6_bytecode_read_raf_output_contribution_symbol, BytecodeFlag, BytecodeOutputTermPlan,
        BytecodeReadRafTermPlan, BytecodeRegister, Stage6BytecodeReadRafEmitPlan,
        STAGE6_BYTECODE_RA_EVAL_FAMILY, STAGE6_BYTECODE_READ_RAF_PLAN,
    };

    fn bytecode_ra_evals() -> IndexedEvalFamilyPlan {
        IndexedEvalFamilyPlan {
            symbol: "stage6.bytecode_read_raf.eval.BytecodeRa".to_owned(),
            evals: (0..4)
                .map(|index| format!("stage6.bytecode_read_raf.eval.BytecodeRa_{index}"))
                .collect(),
        }
    }

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
                symbol: "stage6.bytecode_read_raf.output.term.Stage1",
                stage_index: 0,
                gamma_power: 0,
                identity_gamma_power: Some(5),
            }
        );
        assert_eq!(
            plan.output_terms[5],
            BytecodeOutputTermPlan::Entry {
                symbol: "stage6.bytecode_read_raf.output.term.Entry",
                gamma_power: 7
            }
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
    fn stage6_bytecode_relation_output_plan_uses_point_derived_contribution(
    ) -> Result<(), EmitError> {
        let output_plan =
            Stage6BytecodeReadRafEmitPlan::from_eval_families(&[bytecode_ra_evals()])?
                .relation_output_plan();
        let claim = output_plan.claim;

        assert_eq!(claim.relation, "jolt.stage6.bytecode_read_raf");
        assert_eq!(
            claim.expected_output,
            "stage6.bytecode_read_raf.output.claim_expr"
        );
        assert_eq!(
            claim.local_scalar_symbols().cloned().collect::<Vec<_>>(),
            vec![
                "stage6.bytecode_read_raf.output.term.Stage1".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage2".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage3".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage4".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage5".to_owned(),
                "stage6.bytecode_read_raf.output.term.Entry".to_owned(),
            ]
        );
        assert_eq!(output_plan.field_exprs[0].formula, "field_vector.product");
        assert_eq!(
            output_plan.field_exprs[0].operands,
            vec![STAGE6_BYTECODE_RA_EVAL_FAMILY.to_owned()]
        );
        assert_eq!(output_plan.field_exprs[1].formula, "field.sum");
        assert_eq!(
            output_plan.field_exprs[1].operands,
            vec![
                "stage6.bytecode_read_raf.output.term.Stage1".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage2".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage3".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage4".to_owned(),
                "stage6.bytecode_read_raf.output.term.Stage5".to_owned(),
                "stage6.bytecode_read_raf.output.term.Entry".to_owned(),
            ]
        );
        assert_eq!(output_plan.field_exprs[2].formula, "field.product");
        assert_eq!(
            output_plan.field_exprs[2].operands,
            vec![
                stage6_bytecode_read_raf_output_contribution_symbol().to_owned(),
                "stage6.bytecode_read_raf.output.product.BytecodeRa".to_owned()
            ]
        );
        Ok(())
    }

    #[test]
    fn stage6_bytecode_plan_renderer_emits_stage67_constants() -> Result<(), EmitError> {
        let source = Stage6BytecodeReadRafEmitPlan::from_eval_families(&[bytecode_ra_evals()])?
            .emit_runtime_constants();

        assert!(source.contains("const STAGE6_BYTECODE_STAGE1_TERMS"));
        assert!(source.contains("Stage67BytecodeTermPlan::LookupTable { gamma_base: 2 }"));
        assert!(source.contains("Stage67BytecodeFlag::IsInterleaved"));
        assert!(source.contains("Stage67BytecodeRegister::Rs2"));
        assert!(!source.contains("STAGE6_BYTECODE_RA_EVAL_NAMES"));
        assert!(!source.contains("STAGE6_BYTECODE_RA_EVALS"));
        assert!(source.contains("const STAGE6_BYTECODE_OUTPUT_TERMS"));
        assert!(source.contains(
            "Stage67BytecodeOutputTermPlan::Entry { symbol: \"stage6.bytecode_read_raf.output.term.Entry\", gamma_power: 7 }"
        ));
        assert!(source
            .contains("output_contribution: \"stage6.bytecode_read_raf.output.contribution\""));
        assert!(source.contains("bytecode_ra_evals: &STAGE6_INDEXED_EVAL_FAMILIES[0]"));
        assert!(source.contains("pub const STAGE6_BYTECODE_PLAN: Stage67BytecodeReadRafPlan"));
        Ok(())
    }

    #[test]
    fn stage6_bytecode_plan_references_indexed_eval_family_row() -> Result<(), EmitError> {
        let families = [bytecode_ra_evals()];
        let plan = Stage6BytecodeReadRafEmitPlan::from_eval_families(&families)?;

        assert_eq!(
            plan.bytecode_ra_evals_ref,
            "STAGE6_INDEXED_EVAL_FAMILIES[0]"
        );
        assert_eq!(
            plan.bytecode_ra_evals.symbol,
            STAGE6_BYTECODE_RA_EVAL_FAMILY
        );
        Ok(())
    }
}
