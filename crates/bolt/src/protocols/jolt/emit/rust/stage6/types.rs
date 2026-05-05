use super::super::stage_common::{
    stage_runtime_verifier_program_aliases, stage_verifier_error_enum, stage_verifier_type_aliases,
    StageRuntimeVerifierTypeShape, StageVerifierErrorShape,
};
use super::Stage6CpuProgram;

impl Stage6CpuProgram {
    pub(super) fn emit_verifier_types() -> String {
        let mut source = stage_verifier_type_aliases(6, StageRuntimeVerifierTypeShape::STAGE6_OR_7);
        source.push_str(&stage_runtime_verifier_program_aliases(6));
        source.push_str(
            r#"
#[derive(Clone, Debug)]
pub struct Stage6BytecodeEntry {
    pub address: Fr,
    pub imm: Fr,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

impl Stage67BytecodeEntry for Stage6BytecodeEntry {
    fn address(&self) -> Fr { self.address }
    fn imm(&self) -> Fr { self.imm }
    fn circuit_flags(&self) -> &[bool; 14] { &self.circuit_flags }
    fn rd(&self) -> Option<usize> { self.rd }
    fn rs1(&self) -> Option<usize> { self.rs1 }
    fn rs2(&self) -> Option<usize> { self.rs2 }
    fn lookup_table(&self) -> Option<usize> { self.lookup_table }
    fn is_interleaved(&self) -> bool { self.is_interleaved }
    fn is_branch(&self) -> bool { self.is_branch }
    fn left_is_rs1(&self) -> bool { self.left_is_rs1 }
    fn left_is_pc(&self) -> bool { self.left_is_pc }
    fn right_is_rs2(&self) -> bool { self.right_is_rs2 }
    fn right_is_imm(&self) -> bool { self.right_is_imm }
    fn is_noop(&self) -> bool { self.is_noop }
}


#[derive(Clone, Debug)]
pub struct Stage6BytecodeReadRafData {
    pub entries: Vec<Stage6BytecodeEntry>,
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Debug)]
pub struct Stage6VerifierData {
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafData>,
}

const STAGE6_RELATION_SYMBOLS: Stage67RelationSymbols = Stage67RelationSymbols {
    hamming_booleanity_relation: "jolt.stage6.hamming_booleanity",
    hamming_booleanity_instance: "stage6.hamming_booleanity.instance",
    booleanity_point: "stage6.booleanity.point",
    stage5_instruction_ra0: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    booleanity_combined_point: "stage6.booleanity.combined_point",
    booleanity_gamma: "stage6.booleanity.gamma",
    booleanity_instruction_ra_prefix: "stage6.booleanity.eval.InstructionRa_",
    booleanity_bytecode_ra_prefix: "stage6.booleanity.eval.BytecodeRa_",
    booleanity_ram_ra_prefix: "stage6.booleanity.eval.RamRa_",
    hamming_weight_eval: "stage6.hamming_booleanity.eval.HammingWeight",
    hamming_lookup_output: "stage6.input.stage1.LookupOutput",
    ram_ra_virtual_cycle: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    ram_ra_virtual_eval_prefix: "stage6.ram_ra_virtual.eval.RamRa_",
    instruction_ra_virtual_cycle: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    instruction_ra_virtual_eval_prefix: "stage6.instruction_ra_virtual.eval.InstructionRa_",
    instruction_ra_virtual_input_prefix: "stage6.input.stage5.instruction_read_raf.InstructionRa_",
    instruction_ra_virtual_gamma: "stage6.instruction_ra_virtual.gamma",
    inc_ram_stage2: "stage6.input.stage2.ram_read_write.RamInc",
    inc_ram_stage4: "stage6.input.stage4.ram_val_check.RamInc",
    inc_rd_stage4: "stage6.input.stage4.registers_read_write.RdInc",
    inc_rd_stage5: "stage6.input.stage5.registers_val_evaluation.RdInc",
    inc_gamma: "stage6.inc_claim_reduction.gamma",
    inc_ram_eval: "stage6.inc_claim_reduction.eval.RamInc",
    inc_rd_eval: "stage6.inc_claim_reduction.eval.RdInc",
};

const STAGE6_BYTECODE_SYMBOLS: Stage67BytecodeSymbols = Stage67BytecodeSymbols {
    point: "stage6.bytecode_read_raf.point",
    gamma: "stage6.bytecode_read_raf.gamma",
    bytecode_ra_eval_prefix: "stage6.bytecode_read_raf.eval.BytecodeRa_",
    entries: "stage6.bytecode_read_raf.entries",
    entry_bytecode_index: "stage6.bytecode_read_raf.entry_bytecode_index",
    stage_gammas: [
        "stage6.bytecode_read_raf.stage1_gamma",
        "stage6.bytecode_read_raf.stage2_gamma",
        "stage6.bytecode_read_raf.stage3_gamma",
        "stage6.bytecode_read_raf.stage4_gamma",
        "stage6.bytecode_read_raf.stage5_gamma",
    ],
    stage_cycle_points: [
        "stage6.input.stage1.Imm",
        "stage6.input.stage2.OpFlagJump",
        "stage6.input.stage3.spartan_shift.UnexpandedPC",
        "stage6.input.stage4.Rs1Ra",
        "stage6.input.stage5.registers_val_evaluation.RdWa",
    ],
    stage4_register_point: "stage6.input.stage4.Rs1Ra",
    stage5_register_point: "stage6.input.stage5.registers_val_evaluation.RdWa",
    entry_rd: "stage6.bytecode.entry.rd",
    entry_rs1: "stage6.bytecode.entry.rs1",
    entry_rs2: "stage6.bytecode.entry.rs2",
    entry_lookup_table: "stage6.bytecode.entry.lookup_table",
};
"#,
        );
        source.push_str(&stage_verifier_error_enum(
            6,
            StageVerifierErrorShape::STANDARD,
        ));
        source
    }
}
