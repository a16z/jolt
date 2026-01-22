//! CSRRS (CSR Read-Set) — Read CSR to rd, set bits from rs1.
//!
//! Encoding: csr[31:20] | rs1[19:15] | funct3=010[14:12] | rd[11:7] | opcode=1110011[6:0]
//!
//! The `csrr rd, csr` pseudo-instruction is `csrrs rd, csr, x0` (read only, no bits set).
//!
//! ## Limitation: Only `csrr` pseudo-instruction supported
//!
//! The full CSRRS instruction atomically reads and sets bits: `rd = CSR; CSR |= rs1`.
//! We only support the read-only `csrr` pseudo-instruction (rs1 = x0).
//!
//! **Why?** Implementing the bit-set operation would require an OR instruction in the
//! inline sequence, which we don't have. We'd need to either:
//! - Add a virtual OR instruction, or
//! - Synthesize OR from AND/XOR (more complex)
//!
//! **Impact:** If ZeroOS or guest code uses `csrrs rd, csr, rs1` with rs1 != 0,
//! tracing will panic. This is acceptable for ZeroOS M-mode which only uses `csrr`.

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    addi::ADDI,
    format::format_i::FormatI,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

/// CSR addresses for M-mode CSRs
const CSR_MSTATUS: u16 = 0x300;
const CSR_MTVEC: u16 = 0x305;
const CSR_MSCRATCH: u16 = 0x340;
const CSR_MEPC: u16 = 0x341;
const CSR_MCAUSE: u16 = 0x342;
const CSR_MTVAL: u16 = 0x343;

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
        // Only support csrr pseudo-instruction (rs1=0)
        if self.operands.rs1 != 0 {
            panic!(
                "CSRRS: rs1 != 0 not supported (only csrr pseudo-instruction is used). \
                 Use csrr rd, csr instead of csrrs rd, csr, rs1."
            );
        }

        // Don't call self.execute() - the inline sequence handles all register writes.
        // For csrr (rs1=0), there's no CSR state modification needed.

        // Generate and execute inline sequence
        // The inline sequence reads from the virtual register (source of truth for proofs)
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generate inline sequence for CSRRS (csrr pseudo-instruction only).
    ///
    /// Reads CSR value from the virtual register and copies to rd.
    /// Virtual registers are updated by:
    /// - CSRRW for mtvec, mscratch
    /// - ECALL inline sequence for mepc, mcause, mtval, mstatus (when trap is taken)
    ///
    /// For rs1 = 0 (csrr, read only) with rd != 0:
    ///   0: ADDI(rd, vr, 0) - Copy from virtual register to rd
    ///
    /// For rs1 = 0, rd = 0: No-op (empty sequence)
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // If rd == 0, this is a no-op
        if self.operands.rd == 0 {
            return Vec::new();
        }

        let csr_addr = self.csr_address();

        // Validate CSR address is supported
        match csr_addr {
            CSR_MSTATUS | CSR_MTVEC | CSR_MSCRATCH | CSR_MEPC | CSR_MCAUSE | CSR_MTVAL => {}
            _ => panic!("CSRRS: Unsupported CSR 0x{:03x}", csr_addr),
        };

        // Map CSR address to virtual register
        let virtual_reg = match csr_addr {
            CSR_MSTATUS => allocator.mstatus_register(),
            CSR_MTVEC => allocator.trap_handler_register(),
            CSR_MSCRATCH => allocator.mscratch_register(),
            CSR_MEPC => allocator.mepc_register(),
            CSR_MCAUSE => allocator.mcause_register(),
            CSR_MTVAL => allocator.mtval_register(),
            _ => unreachable!(), // Already validated above
        };

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Read from virtual register to rd
        asm.emit_i::<ADDI>(self.operands.rd, virtual_reg, 0);

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use crate::instruction::Instruction;

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
            _ => panic!("Expected CSRRS instruction, got {:?}", decoded),
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
            _ => panic!("Expected CSRRS instruction, got {:?}", decoded),
        }
    }
}
