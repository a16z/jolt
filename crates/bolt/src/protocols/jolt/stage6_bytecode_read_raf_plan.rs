#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeReadRafPlan {
    pub(crate) const_name: &'static str,
    pub(crate) point: &'static str,
    pub(crate) gamma: &'static str,
    pub(crate) bytecode_ra_eval_prefix: &'static str,
    pub(crate) entries: &'static str,
    pub(crate) entry_bytecode_index: &'static str,
    pub(crate) stages_const: &'static str,
    pub(crate) stages: &'static [BytecodeReadRafStagePlan],
    pub(crate) entry_contribution: BytecodeEntryContributionPlan,
    pub(crate) registers: BytecodeRegisterSymbols,
    pub(crate) entry_lookup_table: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeReadRafStagePlan {
    pub(crate) terms_const: &'static str,
    pub(crate) gamma: &'static str,
    pub(crate) cycle_point: &'static str,
    pub(crate) register_point: Option<&'static str>,
    pub(crate) output_gamma_power: usize,
    pub(crate) identity_gamma_power: Option<usize>,
    pub(crate) terms: &'static [BytecodeReadRafTermPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BytecodeEntryContributionPlan {
    pub(crate) gamma_power: usize,
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
        output_gamma_power: 0,
        identity_gamma_power: Some(5),
        terms: STAGE6_BYTECODE_STAGE1_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE2_TERMS",
        gamma: "stage6.bytecode_read_raf.stage2_gamma",
        cycle_point: "stage6.input.stage2.OpFlagJump",
        register_point: None,
        output_gamma_power: 1,
        identity_gamma_power: None,
        terms: STAGE6_BYTECODE_STAGE2_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE3_TERMS",
        gamma: "stage6.bytecode_read_raf.stage3_gamma",
        cycle_point: "stage6.input.stage3.spartan_shift.UnexpandedPC",
        register_point: None,
        output_gamma_power: 2,
        identity_gamma_power: Some(4),
        terms: STAGE6_BYTECODE_STAGE3_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE4_TERMS",
        gamma: "stage6.bytecode_read_raf.stage4_gamma",
        cycle_point: "stage6.input.stage4.Rs1Ra",
        register_point: Some("stage6.input.stage4.Rs1Ra"),
        output_gamma_power: 3,
        identity_gamma_power: None,
        terms: STAGE6_BYTECODE_STAGE4_TERMS,
    },
    BytecodeReadRafStagePlan {
        terms_const: "STAGE6_BYTECODE_STAGE5_TERMS",
        gamma: "stage6.bytecode_read_raf.stage5_gamma",
        cycle_point: "stage6.input.stage5.registers_val_evaluation.RdWa",
        register_point: Some("stage6.input.stage5.registers_val_evaluation.RdWa"),
        output_gamma_power: 4,
        identity_gamma_power: None,
        terms: STAGE6_BYTECODE_STAGE5_TERMS,
    },
];

pub(crate) const STAGE6_BYTECODE_READ_RAF_PLAN: BytecodeReadRafPlan = BytecodeReadRafPlan {
    const_name: "STAGE6_BYTECODE_PLAN",
    point: "stage6.bytecode_read_raf.point",
    gamma: "stage6.bytecode_read_raf.gamma",
    bytecode_ra_eval_prefix: "stage6.bytecode_read_raf.eval.BytecodeRa_",
    entries: "stage6.bytecode_read_raf.entries",
    entry_bytecode_index: "stage6.bytecode_read_raf.entry_bytecode_index",
    stages_const: "STAGE6_BYTECODE_STAGES",
    stages: STAGE6_BYTECODE_STAGES,
    entry_contribution: BytecodeEntryContributionPlan { gamma_power: 7 },
    registers: BytecodeRegisterSymbols {
        rd: "stage6.bytecode.entry.rd",
        rs1: "stage6.bytecode.entry.rs1",
        rs2: "stage6.bytecode.entry.rs2",
    },
    entry_lookup_table: "stage6.bytecode.entry.lookup_table",
};

pub(crate) fn stage6_bytecode_read_raf_plan() -> &'static BytecodeReadRafPlan {
    &STAGE6_BYTECODE_READ_RAF_PLAN
}

#[cfg(test)]
mod tests {
    use super::{
        stage6_bytecode_read_raf_plan, BytecodeFlag, BytecodeReadRafTermPlan, BytecodeRegister,
    };

    #[test]
    fn stage6_bytecode_plan_rows_encode_the_read_raf_reduction() {
        let plan = stage6_bytecode_read_raf_plan();

        assert_eq!(plan.stages.len(), 5);
        assert_eq!(plan.entry_contribution.gamma_power, 7);
        assert_eq!(plan.registers.rd, "stage6.bytecode.entry.rd");
        assert_eq!(
            plan.entry_lookup_table,
            "stage6.bytecode.entry.lookup_table"
        );

        let stage1 = &plan.stages[0];
        assert_eq!(stage1.output_gamma_power, 0);
        assert_eq!(stage1.identity_gamma_power, Some(5));
        assert_eq!(stage1.terms.len(), 16);
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
}
