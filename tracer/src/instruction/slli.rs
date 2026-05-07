use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_muli::VirtualMULI,
};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SLLI,
    mask   = 0xfc00707f,
    match  = 0x00001013,
    format = FormatI,
    ram    = ()
);

impl SLLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize].wrapping_shl(self.operands.imm as u32 & mask),
            ),
        );
    }
}

impl RISCVTrace for SLLI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for shift left logical immediate.
    ///
    /// SLLI (Shift Left Logical Immediate) shifts rs1 left by a constant amount.
    /// The implementation uses multiplication by 2^shift_amount, leveraging the
    /// mathematical equivalence: x << n = x * 2^n
    ///
    /// This approach is zkVM-friendly as multiplication is a native operation
    /// that can be efficiently verified in the constraint system.
    ///
    /// The shift amount is masked to 5 bits on RV32 (0-31) or 6 bits on RV64 (0-63),
    /// ensuring the shift stays within valid bounds for the architecture.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // Determine word size based on immediate value and instruction encoding
        // For SLLI: RV32 uses 5-bit immediates (0-31), RV64 uses 6-bit immediates (0-63)
        let mask = match xlen {
            Xlen::Bit32 => 0x1f, //low 5bits
            Xlen::Bit64 => 0x3f, //low 6bits
        };
        let shift = self.operands.imm & mask;

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualMULI>(self.operands.rd, self.operands.rs1, 1 << shift);
        asm.finalize()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use crate::emulator::cpu::{Cpu, Xlen};
    use crate::emulator::default_terminal::DefaultTerminal;
    use crate::instruction::{uncompress_instruction, Instruction};

    fn setup_rv64_cpu() -> Cpu {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.update_xlen(Xlen::Bit64);
        cpu
    }

    /// c.slli RV64 encoding roundtrip: assembles the compressed halfword, runs it
    /// through uncompress_instruction, decodes as SLLI, and executes. Mirrors the
    /// per-shift test cases in ACT4's Zca-c.slli-00.S.
    fn run_c_slli(cpu: &mut Cpu, rd: u8, shamt: u8) {
        assert!(rd != 0, "C.SLLI with rd=0 is reserved");
        assert!(shamt < 64, "shamt must fit in 6 bits");
        let shamt5 = shamt & 0x1f;
        let shamt_hi = (shamt >> 5) & 0x1;
        // c.slli: funct3=000 at bits [15:13] (implicit 0), bit 12 = imm[5],
        // bits [11:7] = rd, bits [6:2] = imm[4:0], op = 0b10 at bits [1:0].
        let halfword: u32 =
            ((shamt_hi as u32) << 12) | ((rd as u32) << 7) | ((shamt5 as u32) << 2) | 0b10;
        let word = uncompress_instruction(halfword, Xlen::Bit64);
        let decoded = Instruction::decode(word, 0x80000000, true).unwrap();
        let Instruction::SLLI(ref slli) = decoded else {
            panic!("expected SLLI after uncompress; got {decoded:?}");
        };
        let mut ram_access = ();
        slli.exec(cpu, &mut ram_access);
    }

    #[test]
    fn test_c_slli_matches_spec_for_all_shifts() {
        let src: u64 = 0x98795a79bef0db26;
        for shamt in 1u8..=63 {
            let mut cpu = setup_rv64_cpu();
            cpu.x[1] = src as i64;
            run_c_slli(&mut cpu, 1, shamt);
            let expected = src.wrapping_shl(shamt as u32) as i64;
            assert_eq!(
                cpu.x[1], expected,
                "c.slli x1, {shamt} with x1=0x{src:016x}: got 0x{:016x}, expected 0x{:016x}",
                cpu.x[1] as u64, expected as u64,
            );
        }
    }

    /// Reproduces the first test case from ACT4's Zca-c.slli-00.S:
    ///   initial x1 = 0x98795a79bef0db26
    ///   c.slli x1, 33
    ///   expected x1 = 0x7de1b64c00000000
    #[test]
    fn test_c_slli_act4_first_case() {
        let mut cpu = setup_rv64_cpu();
        cpu.x[1] = 0x98795a79bef0db26_u64 as i64;
        run_c_slli(&mut cpu, 1, 33);
        assert_eq!(
            cpu.x[1] as u64, 0x7de1b64c00000000,
            "RV64 c.slli x1, 33: got 0x{:016x}",
            cpu.x[1] as u64,
        );
    }

    /// c.slli with rd=x0 is a HINT (no-op) per the RISC-V C standard, not a
    /// reserved encoding. ACT4's Zca-c.slli-00.S includes a case
    /// `c.slli x0, 52`. The prior uncompressor rejected r=0 and returned
    /// 0xffffffff, which caused a decode panic at runtime. Verify we now
    /// decode and execute it as a no-op without panicking.
    #[test]
    fn test_c_slli_rd_x0_is_hint_noop() {
        let mut cpu = setup_rv64_cpu();
        // c.slli x0, 52: funct3=000 (implicit 0), bit12=shamt[5]=1,
        // rd=x0 (bits[11:7]=0), shamt[4:0]=20 at bits[6:2], op=10.
        let halfword: u32 = (1u32 << 12) | ((52u32 & 0x1f) << 2) | 0b10;
        let word = uncompress_instruction(halfword, Xlen::Bit64);
        assert_ne!(
            word, 0xffffffff,
            "c.slli x0, 52 should decode (HINT), not return the sentinel"
        );
        let decoded = Instruction::decode(word, 0x80000000, true).unwrap();
        let Instruction::SLLI(ref slli) = decoded else {
            panic!("expected SLLI, got {decoded:?}");
        };
        let mut ram_access = ();
        slli.exec(&mut cpu, &mut ram_access);
        assert_eq!(cpu.x[0], 0, "x0 must remain 0 after any write");
    }
}
