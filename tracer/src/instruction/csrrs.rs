//! CSRRS (CSR Read-Set) — Read CSR to rd, set bits from rs1.
//!
//! Encoding: csr[31:20] | rs1[19:15] | funct3=010[14:12] | rd[11:7] | opcode=1110011[6:0]
//!
//! The `csrr rd, csr` pseudo-instruction is `csrrs rd, csr, x0` (read only, no bits set).
//!
//! For ZeroOS M-mode: Used primarily for reading CSRs (csrr pseudo-instruction).

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
    virtual_advice::VirtualAdvice,
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

        // Read old CSR value
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
        // CSRRS (including csrr pseudo) causes Sumcheck verification failure
        // when used after csrrw. See context.md for details.
        // For now, panic to prevent silent failures.
        panic!(
            "CSRRS: Not supported (causes Sumcheck failure). \
             CSR reads are not yet working correctly in Jolt."
        );

        #[allow(unreachable_code)]
        {
            let csr_addr = self.csr_address();
            let virtual_reg = match csr_addr {
                CSR_MSTATUS => cpu.vr_allocator.mstatus_register(),
                CSR_MTVEC => cpu.vr_allocator.trap_handler_register(),
                CSR_MSCRATCH => cpu.vr_allocator.mscratch_register(),
                CSR_MEPC => cpu.vr_allocator.mepc_register(),
                CSR_MCAUSE => cpu.vr_allocator.mcause_register(),
                CSR_MTVAL => cpu.vr_allocator.mtval_register(),
                _ => panic!("CSRRS: Unsupported CSR 0x{:03x}", csr_addr),
            };

            // Get values BEFORE execute (in case rd overlaps virtual_reg)
            let old_vr_val = cpu.x[virtual_reg as usize] as u64;
            let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

            // Execute the CSR operation (updates emulation state)
            let mut ram_access = ();
            self.execute(cpu, &mut ram_access);

            // Determine the value to write back:
            // - For rs1=0 (csrr): same value (no change)
            // - For rs1!=0 (csrrs): old | rs1
            let write_val = if self.operands.rs1 != 0 {
                old_vr_val | rs1_val
            } else {
                old_vr_val
            };

            // Generate inline sequence and fill in advice
            let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

            // VirtualAdvice is always at index 0 now (write-then-read pattern)
            if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
                instr.advice = write_val;
            } else {
                panic!(
                    "CSRRS: Expected VirtualAdvice at index 0, got {:?}",
                    inline_sequence[0]
                );
            }

            // Execute inline sequence to record in trace
            let mut trace = trace;
            for instr in inline_sequence {
                instr.trace(cpu, trace.as_deref_mut());
            }
        }
    }

    /// Generate inline sequence for CSRRS.
    ///
    /// IMPORTANT: We use write-then-read pattern to match ECALL's approach.
    /// Virtual registers must be "initialized" within the inline sequence before
    /// being read, otherwise R1CS constraints fail.
    ///
    /// For rs1 = 0 (csrr, read only) with rd != 0:
    ///   0: VirtualAdvice(temp)       - Get advice (current vr value)
    ///   1: ADDI(vr, temp, 0)         - Write advice to vr (makes it "defined")
    ///   2: ADDI(rd, vr, 0)           - Read vr to rd
    ///
    /// For rs1 = 0 (csrr, read only) with rd == 0 (no-op, just preserve vr):
    ///   0: VirtualAdvice(temp)       - Get advice (current vr value)
    ///   1: ADDI(vr, temp, 0)         - Write advice back to vr
    ///
    /// For rs1 != 0 (csrrs, read and set bits):
    ///   0: VirtualAdvice(temp)       - Get advice (new value = old | rs1)
    ///   1: ADDI(vr, temp, 0)         - Write new value to vr
    ///   2: ADDI(rd, vr, 0)           - Read vr to rd (gets old value via separate advice)
    ///   Note: For rs1!=0, we need a different approach since rd should get OLD value
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let csr_addr = self.csr_address();

        // Map CSR address to virtual register
        let virtual_reg = match csr_addr {
            CSR_MSTATUS => allocator.mstatus_register(),
            CSR_MTVEC => allocator.trap_handler_register(),
            CSR_MSCRATCH => allocator.mscratch_register(),
            CSR_MEPC => allocator.mepc_register(),
            CSR_MCAUSE => allocator.mcause_register(),
            CSR_MTVAL => allocator.mtval_register(),
            _ => panic!(
                "CSRRS: Unsupported CSR 0x{:03x}",
                csr_addr
            ),
        };

        let value_reg = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        if self.operands.rs1 == 0 && self.operands.rd != 0 {
            // csrr with rd != 0: Write-then-read pattern (matches ECALL)
            // Index 0: VirtualAdvice provides the current vr value
            asm.emit_j::<VirtualAdvice>(*value_reg, 0);

            // Index 1: Write advice to vr (makes it "defined" in this sequence)
            asm.emit_i::<ADDI>(virtual_reg, *value_reg, 0);

            // Index 2: Read vr to rd
            asm.emit_i::<ADDI>(self.operands.rd, virtual_reg, 0);
        } else if self.operands.rs1 != 0 {
            // csrrs with rs1 != 0: Need both old value (for rd) and new value (for vr)
            // This case is complex - for now panic as we don't use csrrs with rs1!=0
            panic!("CSRRS: rs1 != 0 not yet supported (only csrr pseudo-instruction is used)");
        } else {
            // rd == 0 and rs1 == 0: No-op, just preserve vr
            // Index 0: VirtualAdvice provides current vr value
            asm.emit_j::<VirtualAdvice>(*value_reg, 0);

            // Index 1: Write advice back to vr
            asm.emit_i::<ADDI>(virtual_reg, *value_reg, 0);
        }

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::CSRRS;
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
