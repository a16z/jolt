//! CSRRW (CSR Read-Write) — Write rs1 to CSR, read old value to rd.
//!
//! Encoding: `csr[31:20] | rs1[19:15] | funct3=001[14:12] | rd[11:7] | opcode=1110011[6:0]`
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
        // Don't call self.execute() - the inline sequence handles everything.
        // Virtual registers are the single source of truth; we don't use cpu.csr[].

        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::Cycle;
    use crate::instruction::Instruction;
    use crate::instruction::RISCVTrace;

    /// `decode` must reject unsupported CSRs with a typed error instead of
    /// letting them reach the inline-sequence path, which would previously
    /// panic the prover process.
    #[test]
    fn test_csrrw_unsupported_csr_rejected_at_decode() {
        // satp = 0x180 — valid RISC-V supervisor CSR but not modelled by Jolt.
        // Encoding: 0x180 << 20 | 5 << 15 | 1 << 12 | 0 << 7 | 0x73
        let instr: u32 = (0x180 << 20) | (5 << 15) | (1 << 12) | 0x73;
        let err = Instruction::decode(instr, 0x1000, false)
            .expect_err("decode must reject unsupported CSR (satp) with an Err, not panic");
        assert!(err.contains("CSR"), "error should mention CSR: {err}");
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

        assert_eq!(cpu.x[5] as u64, old_vr_val);

        assert_eq!(cpu.x[34] as u64, write_val);
    }
}
