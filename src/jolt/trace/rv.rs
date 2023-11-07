use crate::jolt::vm::{test_vm::TestInstructionSet, pc::ELFRow};

use super::{JoltProvableTrace, MemoryOp};

struct RVTraceRow {
  pc: u64,
  opcode: u64,

  rd: Option<u64>,
  rs1: Option<u64>,
  rs2: Option<u64>,
  imm: Option<u64>,

  rd_pre_val: Option<u64>,
  rd_post_val: Option<u64>,

  rs1_val: Option<u64>,
  rs2_val: Option<u64>,

  memory_bytes_before: Option<Vec<u8>>,
  memory_bytes_after: Option<Vec<u8>>,
}

impl JoltProvableTrace for RVTraceRow {
  type JoltInstructionEnum = TestInstructionSet;
  fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum> {
    // Handle fan-out 1-to-many
    todo!("massive match");
    // vec![TestInstructionSet::from(self.opcode)]
  }

  fn to_ram_ops(&self) -> Vec<MemoryOp> {
    todo!("massive match")
  }

  fn to_pc_trace(&self) -> ELFRow {
    // TODO(sragss): Is 0 padding correct?
    ELFRow::new(
      self.pc.try_into().unwrap(),
      self.opcode,
      self.rd.unwrap_or(0),
      self.rs1.unwrap_or(0),
      self.rs2.unwrap_or(0),
      self.imm.unwrap_or(0),
    )
  }
}
