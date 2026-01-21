//! CSRRW (CSR Read-Write) — Write rs1 to CSR, read old value to rd.
//!
//! Encoding: csr[31:20] | rs1[19:15] | funct3=001[14:12] | rd[11:7] | opcode=1110011[6:0]
//!
//! For ZeroOS: Single-core, no-interrupts, M-mode-only. Supports the following CSRs
//! mapped to virtual registers for proof verification:
//!   - mtvec (0x305) → vr33
//!   - mscratch (0x340) → vr34
//!   - mepc (0x341) → vr35
//!   - mcause (0x342) → vr36
//!   - mtval (0x343) → vr37
//!   - mstatus (0x300) → vr38
//!
//! The `csrw csr, rs` pseudo-instruction is `csrrw x0, csr, rs` (rd=0, discard old value).
//! The full `csrrw rd, csr, rs` swaps rd ← old_CSR, CSR ← rs.

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
    virtual_assert_eq::VirtualAssertEQ,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

/// CSR addresses for M-mode CSRs
const CSR_MSTATUS: u16 = 0x300;  // Machine Status
const CSR_MTVEC: u16 = 0x305;    // Machine Trap-Vector Base Address
const CSR_MSCRATCH: u16 = 0x340; // Machine Scratch Register
const CSR_MEPC: u16 = 0x341;     // Machine Exception Program Counter
const CSR_MCAUSE: u16 = 0x342;   // Machine Trap Cause
const CSR_MTVAL: u16 = 0x343;    // Machine Trap Value

declare_riscv_instr!(
    name   = CSRRW,
    mask   = 0x0000707f,  // Match opcode (7 bits) + funct3 (3 bits)
    match  = 0x00001073,  // opcode=1110011, funct3=001
    format = FormatI,
    ram    = ()
);

impl CSRRW {
    /// Extract CSR address from the immediate field (bits [31:20] of instruction)
    fn csr_address(&self) -> u16 {
        (self.operands.imm & 0xfff) as u16
    }

    fn exec(&self, cpu: &mut Cpu, _: &mut <CSRRW as RISCVInstruction>::RAMAccess) {
        let csr_addr = self.csr_address();
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;

        // Read old CSR value (for rd, if rd != 0)
        let old_val = cpu.read_csr_raw(csr_addr);

        // Write new value to CSR (emulation state - NOT proven)
        cpu.write_csr_raw(csr_addr, rs1_val);

        // Write old value to rd (if rd != x0)
        if self.operands.rd != 0 {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(old_val as i64);
        }
    }
}

impl RISCVTrace for CSRRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let csr_addr = self.csr_address();

        // Get the OLD CSR value before executing (for rd when rd != 0)
        let old_csr_val = cpu.read_csr_raw(csr_addr);

        // Get the value being written (from rs1)
        let write_val = cpu.x[self.operands.rs1 as usize] as u64;

        // Execute the CSR operation (updates emulation state)
        let mut ram_access = ();
        self.execute(cpu, &mut ram_access);

        // Generate inline sequence for proof verification
        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        if self.operands.rd == 0 {
            // csrw pseudo-instruction: just one advice (new CSR value)
            if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
                instr.advice = write_val;
            } else {
                panic!("Expected VirtualAdvice at index 0, got {:?}", inline_sequence[0]);
            }
        } else {
            // Full csrrw: need two advice values
            // When rd == rs1, there's an extra ADDI at index 0 to preserve rs1,
            // so the advice indices shift by 1.
            let (old_advice_idx, new_advice_idx) = if self.operands.rd == self.operands.rs1 {
                (1, 3)  // Extra ADDI at index 0
            } else {
                (0, 2)
            };

            // Set old CSR value advice
            if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[old_advice_idx] {
                instr.advice = old_csr_val;
            } else {
                panic!("Expected VirtualAdvice at index {}, got {:?}", old_advice_idx, inline_sequence[old_advice_idx]);
            }

            // Set new CSR value advice (from rs1)
            if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[new_advice_idx] {
                instr.advice = write_val;
            } else {
                panic!("Expected VirtualAdvice at index {}, got {:?}", new_advice_idx, inline_sequence[new_advice_idx]);
            }
        }

        // Execute inline sequence to record in trace
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generate inline sequence for CSRRW.
    ///
    /// For rd = 0 (csrw pseudo-instruction):
    ///   0: VirtualAdvice(temp)     - Get write value as advice
    ///   1: ADDI(vr, temp, 0)       - Write to virtual register
    ///
    /// For rd != 0, rd != rs1 (full csrrw, atomic swap):
    ///   0: VirtualAdvice(temp_old) - Get OLD CSR value as advice
    ///   1: ADDI(rd, temp_old, 0)   - Write old value to rd
    ///   2: VirtualAdvice(temp_new) - Get NEW value (from rs1) as advice
    ///   3: ADDI(vr, temp_new, 0)   - Write new value to virtual register
    ///   4: VirtualAssertEQ(temp_new, rs1) - Assert advice matches rs1
    ///
    /// For rd != 0, rd == rs1 (swap where dest equals source):
    ///   0: ADDI(rs1_copy, rs1, 0)  - Preserve rs1 before it gets clobbered
    ///   1: VirtualAdvice(temp_old) - Get OLD CSR value as advice
    ///   2: ADDI(rd, temp_old, 0)   - Write old value to rd (clobbers rs1!)
    ///   3: VirtualAdvice(temp_new) - Get NEW value as advice
    ///   4: ADDI(vr, temp_new, 0)   - Write new value to virtual register
    ///   5: VirtualAssertEQ(temp_new, rs1_copy) - Assert advice matches preserved rs1
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let csr_addr = self.csr_address();

        // Map CSR address to virtual register
        let virtual_reg = match csr_addr {
            CSR_MSTATUS => allocator.mstatus_register(),         // mstatus → vr38
            CSR_MTVEC => allocator.trap_handler_register(),      // mtvec → vr33
            CSR_MSCRATCH => allocator.mscratch_register(),       // mscratch → vr34
            CSR_MEPC => allocator.mepc_register(),               // mepc → vr35
            CSR_MCAUSE => allocator.mcause_register(),           // mcause → vr36
            CSR_MTVAL => allocator.mtval_register(),             // mtval → vr37
            _ => panic!(
                "CSRRW: Unsupported CSR 0x{:03x}",
                csr_addr
            ),
        };

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        if self.operands.rd == 0 {
            // csrw pseudo-instruction: just write new value
            let value_reg = allocator.allocate();

            // Index 0: Get write value as advice
            asm.emit_j::<VirtualAdvice>(*value_reg, 0);

            // Index 1: Write value to virtual register
            asm.emit_i::<ADDI>(virtual_reg, *value_reg, 0);
        } else {
            // Full csrrw: rd ← old_CSR, CSR ← rs1
            let temp_old = allocator.allocate();
            let temp_new = allocator.allocate();

            // When rd == rs1, we need to preserve rs1's value before clobbering it
            // because the ADDI that writes to rd will destroy the original rs1 value.
            let rs1_copy = if self.operands.rd == self.operands.rs1 {
                let copy_reg = allocator.allocate();
                // Index 0 (when rd == rs1): Copy rs1 to temp before clobbering
                asm.emit_i::<ADDI>(*copy_reg, self.operands.rs1, 0);
                Some(copy_reg)
            } else {
                None
            };

            // Get old CSR value as advice
            asm.emit_j::<VirtualAdvice>(*temp_old, 0);

            // Write old value to rd (this clobbers rs1 if rd == rs1)
            asm.emit_i::<ADDI>(self.operands.rd, *temp_old, 0);

            // Get new value (rs1) as advice
            asm.emit_j::<VirtualAdvice>(*temp_new, 0);

            // Write new value to virtual register
            asm.emit_i::<ADDI>(virtual_reg, *temp_new, 0);

            // Assert that advice equals original rs1 value (verifies prover honesty)
            // This prevents the prover from lying about what value is being written.
            let verify_reg = rs1_copy.map(|r| *r).unwrap_or(self.operands.rs1);
            asm.emit_b::<VirtualAssertEQ>(*temp_new, verify_reg, 0);
        }

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::CSRRW;
    use crate::instruction::Instruction;

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
            _ => panic!("Expected CSRRW instruction, got {:?}", decoded),
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
            _ => panic!("Expected CSRRW instruction, got {:?}", decoded),
        }
    }
}
