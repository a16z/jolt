//! CSRRS (CSR Read-Set) — Read CSR to rd, set bits from rs1.
//!
//! Encoding: csr[31:20] | rs1[19:15] | funct3=010[14:12] | rd[11:7] | opcode=1110011[6:0]
//!
//! The `csrr rd, csr` pseudo-instruction is `csrrs rd, csr, x0` (read only, no bits set).
//! The `csrs csr, rs` pseudo-instruction is `csrrs x0, csr, rs` (set only, discard old value).

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    addi::ADDI, format::format_i::FormatI, or::OR, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = CSRRS,
    mask   = 0x0000707f,  // Match opcode (7 bits) + funct3 (3 bits)
    match  = 0x00002073,  // opcode=1110011, funct3=010
    format = FormatI,
    ram    = ()
);

impl CSRRS {
    /// Extract CSR address from the immediate field (bits [31:20] of instruction)
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRS as RISCVInstruction>::RAMAccess) {
        let csr_addr = self.csr_address();
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

        // Read old CSR value from CSR state
        let old_val = cpu.read_csr_raw(csr_addr);

        // If rs1 != 0, set bits (OR with rs1)
        if self.operands.rs1 != 0 {
            cpu.write_csr_raw(csr_addr, old_val | rs1_val);
        }

        // Write old value to rd (if rd != x0)
        if self.operands.rd != 0 {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(old_val as i64);
        }
    }
}

impl RISCVTrace for CSRRS {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles all register writes.
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generate inline sequence for CSRRS.
    ///
    /// Semantics: `rd = CSR; CSR |= rs1`
    ///
    /// For rs1 = 0, rd != 0 (csrr — read only):
    ///   ADDI(rd, vr, 0)
    ///
    /// For rs1 != 0, rd = 0 (csrs — set only):
    ///   OR(vr, vr, rs1)
    ///
    /// For rs1 != 0, rd != 0, rd != rs1 (full read-set):
    ///   ADDI(rd, vr, 0)     — read old value to rd
    ///   OR(vr, vr, rs1)     — set bits
    ///
    /// For rs1 != 0, rd != 0, rd == rs1 (read-set, dest clobbers source):
    ///   ADDI(temp, rs1, 0)  — preserve rs1
    ///   ADDI(rd, vr, 0)     — read old value to rd (clobbers rs1)
    ///   OR(vr, vr, temp)    — set bits from preserved value
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let csr_addr = self.csr_address();

        // CSR 0 is never valid - return no-op for default-constructed instructions
        if csr_addr == 0 {
            warn!("CSRRS with CSR address 0 is invalid, returning NoOp");
            return vec![Instruction::NoOp];
        }

        let virtual_reg = allocator
            .csr_to_virtual_register(csr_addr)
            .unwrap_or_else(|| panic!("CSRRS: Unsupported CSR 0x{csr_addr:03x}"));

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        if self.operands.rs1 == 0 {
            // csrr: read only
            asm.emit_i::<ADDI>(self.operands.rd, virtual_reg, 0);
        } else if self.operands.rd == 0 {
            // csrs: set only
            asm.emit_r::<OR>(virtual_reg, virtual_reg, self.operands.rs1);
        } else if self.operands.rd == self.operands.rs1 {
            // rd == rs1: preserve rs1 before clobbering
            let temp = allocator.allocate();
            asm.emit_i::<ADDI>(*temp, self.operands.rs1, 0);
            asm.emit_i::<ADDI>(self.operands.rd, virtual_reg, 0);
            asm.emit_r::<OR>(virtual_reg, virtual_reg, *temp);
        } else {
            // rd != rs1: read old, then set
            asm.emit_i::<ADDI>(self.operands.rd, virtual_reg, 0);
            asm.emit_r::<OR>(virtual_reg, virtual_reg, self.operands.rs1);
        }

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::{Cycle, Instruction, RISCVTrace};

    /// Test decoding of `csrr t0, mtvec` (csrrs t0, mtvec, x0)
    #[test]
    fn test_csrr_mtvec_decode() {
        // csrr t0, mtvec = csrrs t0, 0x305, x0
        // Encoding: 0x305 << 20 | 0 << 15 | 2 << 12 | 5 << 7 | 0x73
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

    /// Test decoding with rs1 != 0 (full csrrs)
    #[test]
    fn test_csrrs_with_rs1() {
        // csrrs a0, mtvec, t0 (read mtvec to a0, set bits from t0)
        // Encoding: 0x305 << 20 | 5 << 15 | 2 << 12 | 10 << 7 | 0x73
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

    /// Test trace of full csrrs: rd = old CSR, CSR |= rs1
    #[test]
    fn test_csrrs_trace_full() {
        // csrrs a0, mtvec, t0 (rd=a0(10), rs1=t0(5), csr=mtvec)
        let instr: u32 = 0x3052a573;
        let address: u64 = 0x1000;

        let decoded = Instruction::decode(instr, address, false).expect("Failed to decode CSRRS");
        let Instruction::CSRRS(csrrs) = decoded else {
            panic!("Expected CSRRS instruction");
        };

        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        let old_csr: u64 = 0x00FF;
        let rs1_val: u64 = 0xFF00;

        cpu.x[34] = old_csr as i64; // vr34 = mtvec
        cpu.x[5] = rs1_val as i64; // t0

        let mut trace: Vec<Cycle> = Vec::new();
        csrrs.trace(&mut cpu, Some(&mut trace));

        // rd (a0) should have old CSR value
        assert_eq!(cpu.x[10] as u64, old_csr, "rd should get old CSR value");
        // vr (mtvec) should have old | rs1
        assert_eq!(
            cpu.x[34] as u64,
            old_csr | rs1_val,
            "CSR should have bits set from rs1"
        );
    }

    /// Test trace of csrrs with rd == rs1 (clobber case)
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
        cpu.x[5] = rs1_val as i64; // t0

        let mut trace: Vec<Cycle> = Vec::new();
        csrrs.trace(&mut cpu, Some(&mut trace));

        // rd (t0) should have old CSR value
        assert_eq!(cpu.x[5] as u64, old_csr, "rd should get old CSR value");
        // vr should have old | rs1 (using preserved rs1, not clobbered value)
        assert_eq!(
            cpu.x[34] as u64,
            old_csr | rs1_val,
            "CSR should have bits set from original rs1"
        );
    }
}
