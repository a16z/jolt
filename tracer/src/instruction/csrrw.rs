//! CSRRW (CSR Read-Write) — Write rs1 to CSR, read old value to rd.
//!
//! Encoding: csr[31:20] | rs1[19:15] | funct3=001[14:12] | rd[11:7] | opcode=1110011[6:0]
//!
//! For ZeroOS: Single-core, no-interrupts, M-mode-only. Supports the following CSRs
//! mapped to virtual registers for proof verification:
//!   - mtvec (0x305) → vr34
//!   - mscratch (0x340) → vr35
//!   - mepc (0x341) → vr36
//!   - mcause (0x342) → vr37
//!   - mtval (0x343) → vr38
//!   - mstatus (0x300) → vr39
//!
//! The `csrw csr, rs` pseudo-instruction is `csrrw x0, csr, rs` (rd=0, discard old value).
//! The full `csrrw rd, csr, rs` swaps rd ← old_CSR, CSR ← rs.

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = CSRRW,
    mask   = 0x0000707f,  // Match opcode (7 bits) + funct3 (3 bits)
    match  = 0x00001073,  // opcode=1110011, funct3=001
    format = FormatI,
    ram    = (),
    side_effects = true
);

impl CSRRW {
    /// Extract CSR address from the immediate field (bits [31:20] of instruction)
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRW as RISCVInstruction>::RAMAccess) {
        let csr_addr = self.csr_address();
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

        // Read old CSR value from CSR state
        let old_val = cpu.read_csr_raw(csr_addr);

        // Write new value to CSR state
        cpu.write_csr_raw(csr_addr, rs1_val);

        // Write old value to rd (if rd != x0)
        if self.operands.rd != 0 {
            cpu.write_register(self.operands.rd as usize, cpu.sign_extend(old_val as i64));
        }
    }
}

impl RISCVTrace for CSRRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles everything.
        // Virtual registers are the single source of truth; we don't use cpu.csr[].

        // Generate and execute inline sequence
        // The inline sequence reads from virtual register and writes to rd,
        // then writes rs1 to virtual register.
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::Cycle;
    use crate::instruction::Instruction;
    use crate::instruction::RISCVTrace;

    /// Test decoding of `csrw mtvec, t0` (csrrw x0, mtvec, t0)
    /// Encoding: csr=0x305, rs1=t0(5), funct3=001, rd=x0(0), opcode=1110011
    #[test]
    fn test_csrrw_mtvec_decode() {
        // csrw mtvec, t0 = csrrw x0, 0x305, t0
        // Encoding: 0x305 << 20 | 5 << 15 | 1 << 12 | 0 << 7 | 0x73
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

    /// Test decoding with rd != 0 (full csrrw, not just csrw pseudo-instruction)
    #[test]
    fn test_csrrw_with_rd() {
        // csrrw a0, mtvec, t0 (read old mtvec to a0, write t0 to mtvec)
        // Encoding: 0x305 << 20 | 5 << 15 | 1 << 12 | 10 << 7 | 0x73
        let instr: u32 = 0x30529573; // rd=a0(10)
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

        // Choose distinct values so that if the inline sequence accidentally uses
        // the post-clobber rs1 value, the test will fail.
        let old_vr_val: u64 = 0x2222_3333;
        let write_val: u64 = 0x1111_0000;

        // Set up the virtual register (single source of truth)
        cpu.x[34] = old_vr_val as i64; // vr34 = mtvec
        cpu.x[5] = write_val as i64; // rs1 = t0 = write_val

        let mut trace: Vec<Cycle> = Vec::new();
        csrrw.trace(&mut cpu, Some(&mut trace));

        // Architectural rd (t0) gets old value from virtual register.
        assert_eq!(cpu.x[5] as u64, old_vr_val);

        // Virtual register (vr34 = mtvec) should have the new value (from rs1).
        assert_eq!(cpu.x[34] as u64, write_val);
    }
}
