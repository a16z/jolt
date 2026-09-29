use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth},
};

use super::format::format_r::FormatR;
use super::{Cycle, Instruction, RAMRead, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRW,
    mask   = 0xf9f0707f,
    match  = 0x1000202f,
    format = FormatR,
    ram    = RAMRead
);

impl LRW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LRW as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRW: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1 as usize] as u64;

        // Load the word from memory
        let value = cpu.mmu.load_word(address);

        let write_value = match value {
            Ok((word, _memory_read)) => {
                cpu.set_reservation(address, ReservationWidth::Word);
                // Sign extend the 32-bit value
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, write_value);
    }
}

impl RISCVTrace for LRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address, ReservationWidth::Word);

        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

impl LRW {}

#[cfg(test)]
mod tests {
    use crate::instruction::Instruction;

    fn encode_lr(funct3: u32, rs2: u32) -> u32 {
        (0b00010 << 27) | (rs2 << 20) | (1 << 15) | (funct3 << 12) | (2 << 7) | 0x2f
    }

    /// LR.W and LR.D have no rs2 operand and their encodings require rs2 = 0,
    /// so `decode` must return an error for other values instead of passing
    /// the word to `LRW::new`/`LRD::new`.
    #[test]
    fn decode_rejects_lr_with_nonzero_rs2() {
        for funct3 in [0b010, 0b011] {
            for rs2 in 1..32 {
                let error = Instruction::decode(encode_lr(funct3, rs2), 0x1000, false)
                    .expect_err("decode must reject LR with rs2 != 0 with an Err, not panic");
                assert!(
                    error.contains("LR rs2"),
                    "funct3={funct3:03b} rs2={rs2}: {error}"
                );
            }
        }
    }

    #[test]
    fn decode_accepts_lr_with_zero_rs2() {
        let lrw = Instruction::decode(encode_lr(0b010, 0), 0x1000, false).expect("lr.w");
        assert!(matches!(lrw, Instruction::LRW(_)), "{lrw:?}");
        let lrd = Instruction::decode(encode_lr(0b011, 0), 0x1000, false).expect("lr.d");
        assert!(matches!(lrd, Instruction::LRD(_)), "{lrd:?}");
    }
}
