use crate::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    instruction::{ELFInstruction, RV32IM},
    memory::{MemoryOp, MemoryState},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RVTraceRow {
    pub instruction: ELFInstruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
    pub advice_value: Option<u64>,
    pub precompile_input: Option<[u32; 16]>,
    pub precompile_output_address: Option<u64>,
}

fn sum_u64_i32(a: u64, b: i32) -> u64 {
    if b.is_negative() {
        let abs_b = b.unsigned_abs() as u64;
        if a < abs_b {
            panic!("overflow")
        }
        a - abs_b
    } else {
        let b_u64: u64 = b.try_into().expect("failed u64 conversion");
        a + b_u64
    }
}

impl RVTraceRow {
    pub fn imm_u64(&self) -> u64 {
        self.instruction.imm.unwrap() as u64
    }

    pub fn imm_u32(&self) -> u32 {
        self.instruction.imm.unwrap() as u64 as u32
    }

    pub fn to_memory_ops(&self) -> [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] {
        let rs1_read = || MemoryOp::Read(self.instruction.rs1.unwrap());
        let rs2_read = || MemoryOp::Read(self.instruction.rs2.unwrap());
        let rd_write = || {
            MemoryOp::Write(
                self.instruction.rd.unwrap(),
                self.register_state.rd_post_val.unwrap(),
            )
        };

        let ram_write_value = || match self.memory_state {
            Some(MemoryState::Read {
                address: _,
                value: _,
            }) => panic!("Unexpected MemoryState::Read"),
            Some(MemoryState::Write {
                address: _,
                pre_value: _,
                post_value,
            }) => post_value,
            None => panic!("Memory state not found"),
        };

        let rs1_offset = || -> u64 {
            let rs1_val = self.register_state.rs1_val.unwrap();
            let imm = self.instruction.imm.unwrap();
            sum_u64_i32(rs1_val, imm as i32)
        };

        // Canonical ordering for memory instructions
        // 0: rs1
        // 1: rs2
        // 2: rd
        // 3: byte_0
        // 4: byte_1
        // 5: byte_2
        // 6: byte_3
        // If any are empty a no_op is inserted.

        match self.instruction.opcode {
            RV32IM::ADD
            | RV32IM::SUB
            | RV32IM::XOR
            | RV32IM::OR
            | RV32IM::AND
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::MUL
            | RV32IM::MULH
            | RV32IM::MULHU
            | RV32IM::MULHSU
            | RV32IM::MULU
            | RV32IM::DIV
            | RV32IM::DIVU
            | RV32IM::REM
            | RV32IM::REMU => [rs1_read(), rs2_read(), rd_write(), MemoryOp::noop_read()],

            RV32IM::LUI | RV32IM::AUIPC | RV32IM::VIRTUAL_ADVICE => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT => [
                rs1_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::ADDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::ANDI
            | RV32IM::ORI
            | RV32IM::XORI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::JALR
            | RV32IM::VIRTUAL_POW2
            | RV32IM::VIRTUAL_SHIFT_RIGHT_BITMASK
            | RV32IM::VIRTUAL_MOVE
            | RV32IM::VIRTUAL_MOVSIGN => [
                rs1_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::LW => [
                rs1_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::Read(rs1_offset()),
            ],
            RV32IM::FENCE => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::SB | RV32IM::SH | RV32IM::SW => [
                rs1_read(),
                rs2_read(),
                MemoryOp::noop_write(),
                MemoryOp::Write(rs1_offset(), ram_write_value()),
            ],

            // RV32IM::LB | RV32IM::LH | RV32IM::LBU | RV32IM::LHU => [
            RV32IM::JAL | RV32IM::VIRTUAL_POW2I | RV32IM::VIRTUAL_SHIFT_RIGHT_BITMASKI => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                rd_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER => [
                rs1_read(),
                rs2_read(),
                MemoryOp::noop_write(),
                MemoryOp::noop_read(),
            ],

            RV32IM::ECALL => [
                MemoryOp::noop_read(),
                MemoryOp::noop_read(),
                MemoryOp::noop_write(),
                MemoryOp::Write(rs1_offset(), ram_write_value()),
            ],

            _ => unreachable!("{self:?}"),
        }
    }
}
