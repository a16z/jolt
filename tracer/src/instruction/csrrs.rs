use crate::instruction::registers::i::RegisterStateI;

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = CSRRS,
    mask   = 0x0000707f,
    match  = 0x00002073,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl CSRRS {
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRS as RISCVInstruction>::RAMAccess) {
        let csr_addr = self.csr_address();
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

        let old_val = cpu.read_csr_raw(csr_addr);

        if self.operands.rs1 != 0 {
            cpu.write_csr_raw(csr_addr, old_val | rs1_val);
        }

        if self.operands.rd != 0 {
            cpu.write_register(self.operands.rd as usize, cpu.sign_extend(old_val as i64));
        }
    }
}

impl RISCVTrace for CSRRS {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::{Cycle, Instruction, RISCVTrace};

    #[test]
    fn test_csrr_mtvec_decode() {
        let instr: u32 = 0x305022f3;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");

        match decoded {
            Instruction::CSRRS(csrrs) => {
                assert_eq!(csrrs.operands.rd, 5, "rd should be t0 (x5)");
                assert_eq!(csrrs.operands.rs1, 0, "rs1 should be x0");
                assert_eq!(csrrs.csr_address(), 0x305, "CSR should be mtvec (0x305)");
            }
            _ => panic!("Expected CSRRS instruction, got {decoded:?}"),
        }
    }

    #[test]
    fn test_csrrs_unsupported_csr_rejected_at_decode() {
        let instr: u32 = (0x180 << 20) | (2 << 12) | (5 << 7) | 0x73;
        let err = Instruction::decode(instr, 0x1000, false)
            .expect_err("decode must reject unsupported CSR (satp) with an Err, not panic");
        assert!(err.contains("CSR"), "error should mention CSR: {err}");
    }

    #[test]
    fn test_csrrs_with_rs1() {
        let instr: u32 = 0x3052a573;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");

        match decoded {
            Instruction::CSRRS(csrrs) => {
                assert_eq!(csrrs.operands.rd, 10, "rd should be a0 (x10)");
                assert_eq!(csrrs.operands.rs1, 5, "rs1 should be t0 (x5)");
                assert_eq!(csrrs.csr_address(), 0x305, "CSR should be mtvec (0x305)");
            }
            _ => panic!("Expected CSRRS instruction, got {decoded:?}"),
        }
    }

    #[test]
    fn test_csrrs_trace_full() {
        let instr: u32 = 0x3052a573;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");
        let Instruction::CSRRS(csrrs) = decoded else {
            panic!("Expected CSRRS instruction");
        };

        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        let old_csr: u64 = 0x00FF;
        let rs1_val: u64 = 0xFF00;

        cpu.x[34] = old_csr as i64;
        cpu.x[5] = rs1_val as i64;

        let mut trace: Vec<Cycle> = Vec::new();
        csrrs.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[10] as u64, old_csr, "rd should get old CSR value");
        assert_eq!(
            cpu.x[34] as u64,
            old_csr | rs1_val,
            "CSR should have bits set from rs1"
        );
    }

    #[test]
    fn test_csrrs_trace_rd_eq_rs1() {
        let instr: u32 = (0x305 << 20) | (5 << 15) | (2 << 12) | (5 << 7) | 0x73;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");
        let Instruction::CSRRS(csrrs) = decoded else {
            panic!("Expected CSRRS instruction");
        };

        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        let old_csr: u64 = 0x00FF;
        let rs1_val: u64 = 0xFF00;

        cpu.x[34] = old_csr as i64;
        cpu.x[5] = rs1_val as i64;

        let mut trace: Vec<Cycle> = Vec::new();
        csrrs.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[5] as u64, old_csr, "rd should get old CSR value");
        assert_eq!(
            cpu.x[34] as u64,
            old_csr | rs1_val,
            "CSR should have bits set from original rs1"
        );
    }
}
