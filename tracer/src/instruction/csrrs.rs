//! CSRRS (CSR Read-Set) — Read CSR to rd, set bits from rs1.
//!
//! Encoding: `csr[31:20] | rs1[19:15] | funct3=010[14:12] | rd[11:7] | opcode=1110011[6:0]`
//!
//! The `csrr rd, csr` pseudo-instruction is `csrrs rd, csr, x0` (read only, no bits set).
//! The `csrs csr, rs` pseudo-instruction is `csrrs x0, csr, rs` (set only, discard old value).

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
        // Don't call self.execute() - the inline sequence handles all register writes.
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::{Cycle, Instruction, RISCVTrace};

    #[test]
    fn test_csrrs_unsupported_csr_rejected_at_decode() {
        // satp = 0x180 — valid RISC-V supervisor CSR but not modelled by Jolt.
        // Encoding: 0x180 << 20 | 0 << 15 | 2 << 12 | 5 << 7 | 0x73
        let instr: u32 = (0x180 << 20) | (2 << 12) | (5 << 7) | 0x73;
        let err = Instruction::decode(instr, 0x1000, false)
            .expect_err("decode must reject unsupported CSR (satp) with an Err, not panic");
        assert!(err.contains("CSR"), "error should mention CSR: {err}");
    }

    #[test]
    fn test_csrrs_trace_rd_eq_rs1() {
        // csrrs t0, mtvec, t0 (rd == rs1 == x5)
        let instr: u32 = (0x305 << 20) | (5 << 15) | (2 << 12) | (5 << 7) | 0x73;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");
        let Instruction::CSRRS(csrrs) = decoded else {
            panic!("Expected CSRRS instruction");
        };

        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        let old_csr: u64 = 0x00FF;
        let rs1_val: u64 = 0xFF00;

        cpu.x[34] = old_csr as i64; // vr34 = mtvec
        cpu.x[5] = rs1_val as i64;

        let mut trace: Vec<Cycle> = Vec::new();
        csrrs.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[5] as u64, old_csr, "rd should get old CSR value");
        // vr should have old | rs1 (using preserved rs1, not clobbered value)
        assert_eq!(
            cpu.x[34] as u64,
            old_csr | rs1_val,
            "CSR should have bits set from original rs1"
        );
    }
}
