use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_move::VirtualMove, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = REMU,
    mask   = 0xfe00707f,
    match  = 0x02007033,
    format = FormatR,
    ram    = ()
);

impl REMU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        cpu.x[self.operands.rd as usize] = match divisor {
            0 => cpu.sign_extend(dividend as i64),
            _ => cpu.sign_extend(dividend.wrapping_rem(divisor) as i64),
        };
    }
}

impl RISCVTrace for REMU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = if cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) == 0 {
                match cpu.xlen {
                    Xlen::Bit32 => u32::MAX as u64,
                    Xlen::Bit64 => u64::MAX,
                }
            } else {
                cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    / cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
            };
        } else {
            panic!("Expected Advice instruction");
        }
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = match cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) {
                0 => cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]),
                divisor => {
                    let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
                    let quotient = dividend / divisor;
                    dividend - quotient * divisor
                }
            };
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify unsigned remainder operation.
    ///
    /// This function implements a verifiable unsigned remainder (modulo) operation by
    /// decomposing it into simpler operations that can be proven correct. Unlike signed
    /// remainder, unsigned remainder is simpler as all values are non-negative.
    ///
    /// The approach:
    /// 1. Receives untrusted quotient and remainder advice from an oracle
    /// 2. Verifies the fundamental division property: dividend = quotient × divisor + remainder
    /// 3. Ensures remainder < divisor (modulo property for unsigned)
    /// 4. Handles division by zero per RISC-V spec (returns dividend)
    ///
    /// Note: When divisor is 0, the quotient advice is ignored (can be any value) since
    /// multiplication by 0 yields 0, and the check becomes: 0 + remainder = dividend.
    /// The VirtualAssertValidUnsignedRemainder handles the special case where divisor is 0.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend (unsigned input)
        let a1 = self.operands.rs2; // divisor (unsigned input)
        let a2 = allocator.allocate(); // quotient from oracle (ignored when divisor==0)
        let a3 = allocator.allocate(); // remainder from oracle (untrusted)
        let t0 = allocator.allocate(); // temporary for multiplication result
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        // Both quotient and remainder are unsigned values
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get remainder advice

        // Step 2: Compute quotient × divisor
        // When divisor is 0, this multiplication yields 0, which is correct
        // for the special case handling (remainder should equal dividend)
        asm.emit_r::<MUL>(*t0, *a2, a1); // t0 = quotient × divisor

        // Step 3: Verify fundamental division property
        // dividend = quotient × divisor + remainder (all operations mod 2^n)
        // When divisor is 0: dividend = 0 + remainder, so remainder must equal dividend
        asm.emit_r::<ADD>(*t0, *t0, *a3); // t0 = (quotient × divisor) + remainder
        asm.emit_b::<VirtualAssertEQ>(*t0, a0, 0); // assert t0 == dividend

        // Step 4: Verify remainder constraint
        // For valid unsigned division: remainder < divisor
        // Special case: when divisor is 0, this check ensures remainder == dividend
        // (the assertion handles this by checking remainder < MAX when divisor is 0)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, a1, 0);

        // Step 5: Move verified remainder to destination register
        // The remainder is the final result for REMU instruction
        asm.emit_i::<VirtualMove>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
