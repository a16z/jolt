use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = CSRRW,
    mask   = 0x0000707f,
    match  = 0x00001073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl CSRRW {
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRW as RISCVInstruction>::RAMAccess) {
        let csr_addr = self.csr_address();
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

        let old_val = cpu.read_csr_raw(csr_addr);

        cpu.write_csr_raw(csr_addr, rs1_val);

        if self.operands.rd != 0 {
            cpu.write_register(self.operands.rd as usize, cpu.sign_extend(old_val as i64));
        }
    }
}

impl RISCVTrace for CSRRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::Cycle;
    use crate::instruction::Instruction;
    use crate::instruction::RISCVTrace;

    #[test]
    fn test_csrrw_mtvec_decode() {
        let instr: u32 = 0x30529073;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRW");

        match decoded {
            Instruction::CSRRW(csrrw) => {
                assert_eq!(csrrw.operands.rd, 0, "rd should be x0");
                assert_eq!(csrrw.operands.rs1, 5, "rs1 should be t0 (x5)");
                assert_eq!(csrrw.csr_address(), 0x305, "CSR should be mtvec (0x305)");
            }
            _ => panic!("Expected CSRRW instruction, got {decoded:?}"),
        }
    }

    #[test]
    fn test_csrrw_unsupported_csr_rejected_at_decode() {
        let instr: u32 = (0x180 << 20) | (5 << 15) | (1 << 12) | 0x73;
        let err = Instruction::decode(instr, 0x1000, false)
            .expect_err("decode must reject unsupported CSR (satp) with an Err, not panic");
        assert!(err.contains("CSR"), "error should mention CSR: {err}");
    }

    #[test]
    fn test_csrrw_with_rd() {
        let instr: u32 = 0x30529573;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRW");

        match decoded {
            Instruction::CSRRW(csrrw) => {
                assert_eq!(csrrw.operands.rd, 10, "rd should be a0 (x10)");
                assert_eq!(csrrw.operands.rs1, 5, "rs1 should be t0 (x5)");
                assert_eq!(csrrw.csr_address(), 0x305, "CSR should be mtvec (0x305)");
            }
            _ => panic!("Expected CSRRW instruction, got {decoded:?}"),
        }
    }

    #[test]
    fn test_csrrw_trace_rd_eq_rs1_preserves_rs1_for_assert() {
        // csrrw t0, mtvec, t0 (rd == rs1 == x5)
        let instr: u32 = (0x305 << 20) | (5 << 15) | (1 << 12) | (5 << 7) | 0x73;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRW");
        let Instruction::CSRRW(csrrw) = decoded else {
            panic!("Expected CSRRW instruction");
        };

        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        let old_vr_val: u64 = 0x2222_3333;
        let write_val: u64 = 0x1111_0000;

        cpu.x[34] = old_vr_val as i64;
        cpu.x[5] = write_val as i64;

        let mut trace: Vec<Cycle> = Vec::new();
        csrrw.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[5] as u64, old_vr_val);

        assert_eq!(cpu.x[34] as u64, write_val);
    }
}
