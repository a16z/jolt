use jolt_riscv::NormalizedInstruction;

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BytecodePreprocessing {
    pub bytecode: Vec<NormalizedInstruction>,
    pub entry_address: u64,
}
