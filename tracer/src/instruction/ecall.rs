//! ECALL (SYSTEM 0x0000_0073) — Environment call for syscalls and special operations.

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, PrivilegeMode, Trap, TrapType, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    addi::ADDI,
    format::format_i::FormatI,
    jalr::JALR,
    mul::MUL,
    sub::SUB,
    virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ECALL,
    mask   = 0xffff_ffff,
    match  = 0x0000_0073,
    format = FormatI,
    ram    = ()
);

impl ECALL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ECALL as RISCVInstruction>::RAMAccess) {
        let trap_type = match cpu.privilege_mode {
            PrivilegeMode::User => TrapType::EnvironmentCallFromUMode,
            PrivilegeMode::Supervisor => TrapType::EnvironmentCallFromSMode,
            PrivilegeMode::Machine | PrivilegeMode::Reserved => TrapType::EnvironmentCallFromMMode,
        };

        cpu.raise_trap(
            Trap {
                trap_type,
                value: 0,
            },
            self.address,
        );
    }

}

impl RISCVTrace for ECALL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // First, execute the ECALL to trigger trap handling.
        // This calls cpu.raise_trap() -> handle_trap() -> handle_syscall()
        // which sets cpu.pending_csr_result with the syscall return value.
        let mut ram_access = ();
        self.execute(cpu, &mut ram_access);

        // Check if this was a CSR ECALL (for setting trap handler address)
        let is_csr_ecall = cpu.vr_allocator.is_csr_ecall();
        cpu.vr_allocator.set_is_csr_ecall(false); // reset flag

        // Get trap_handler_advice:
        // - For CSR ECALL: use a3 (the trap handler address we want to store)
        // - For regular ECALL: use current reg33 value (preserves it)
        let trap_handler_advice = if is_csr_ecall {
            cpu.x[13] as u64 // a3 contains trap handler for CSR ECALL
        } else {
            cpu.x[33] as u64 // Current reg33 value for regular ECALLs
        };

        let ecall_result = cpu.pending_csr_result.take().unwrap_or(0) as u64;

        // After trap handling, CPU's PC is either:
        // - trap handler address (if trap was taken, e.g., for Linux syscalls with ZeroOS)
        // - self.address + 4 (if trap was not taken, e.g., for Jolt-specific ECALLs)
        let target_pc = cpu.read_pc();

        // Return address for trap handler: ECALL_addr + 4 (ECALL is always 4 bytes)
        // This is passed in t1 so the trap handler can return without using mepc.
        let return_addr = self.address + 4;

        // Determine if trap was taken (target != return address)
        let trap_taken = target_pc != return_addr;
        cpu.vr_allocator.set_last_ecall_trap_taken(trap_taken);

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // Fill in the advice values:
        // - Index 0: ecall result (for a0)
        // - Index 2: return address (for t1, used by trap handler to return)
        // - Index 4: target PC (for JALR)
        // - Index 5: trap handler address (for writing to register 33)
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = ecall_result;
        } else {
            panic!("Expected VirtualAdvice instruction at index 0, got {:?}", inline_sequence[0]);
        }

        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[2] {
            instr.advice = return_addr;
        } else {
            panic!("Expected VirtualAdvice instruction at index 2, got {:?}", inline_sequence[2]);
        }

        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[4] {
            instr.advice = target_pc;
        } else {
            panic!("Expected VirtualAdvice instruction at index 4, got {:?}", inline_sequence[4]);
        }

        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[5] {
            instr.advice = trap_handler_advice;
        } else {
            panic!("Expected VirtualAdvice instruction at index 5, got {:?}", inline_sequence[5]);
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// ECALL inline sequence: use VirtualAdvice to write ecall return value to a0,
    /// return address to t1, and target PC, then verify and JALR to the target.
    ///
    /// Syscalls return their result in a0 (register 10), but ECALL's encoding has rd=0.
    /// We use VirtualAdvice to provide the ecall result as untrusted advice,
    /// which gets written to a0.
    ///
    /// The return address (ECALL_addr+4) is passed in t1 (register 6). For trap-taking
    /// ECALLs, the trap handler saves t1 and uses it to return, eliminating the need
    /// for mepc CSR operations.
    ///
    /// The target PC is provided via VirtualAdvice and then VERIFIED against proven state.
    /// We assert: (target_pc == return_addr) OR (target_pc == trap_handler)
    /// This is computed as: (target_pc - return_addr) * (target_pc - trap_handler) == 0
    /// - For non-trap ECALLs: target_pc == return_addr, so first diff is 0
    /// - For trap-taking ECALLs: target_pc == trap_handler (from register 33), so second diff is 0
    /// - If prover lies: neither diff is 0, product != 0, assertion fails
    ///
    /// The sequence ends with JALR to avoid the NextUnexpPCUpdateOtherwise constraint
    /// failing when the inline sequence is followed by NoOp padding. JALR has Jump=true,
    /// which makes the constraint guard `!(ShouldBranch || Jump)` false.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_trap_handler_reg = allocator.trap_handler_register();

        let ecall_result = allocator.allocate(); // temporary for ecall result
        let return_addr = allocator.allocate(); // temporary for return address
        let next_pc = allocator.allocate(); // temporary for target PC
        let trap_handler_advice = allocator.allocate(); // advice for trap handler to write to reg33
        let trap_handler = allocator.allocate(); // copy of register 33 (after write)
        let diff1 = allocator.allocate(); // target_pc - return_addr
        let diff2 = allocator.allocate(); // target_pc - trap_handler
        // Note: product reuses diff1's register after SUB is done

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Index 0: Get ecall result as advice and write to temp register
        asm.emit_j::<VirtualAdvice>(*ecall_result, 0);

        // Index 1: Move result to a0 (register 10)
        asm.emit_i::<ADDI>(10, *ecall_result, 0);

        // Index 2: Get return address (ECALL_addr+4) as advice and write to temp register
        asm.emit_j::<VirtualAdvice>(*return_addr, 0);

        // Index 3: Move return address to t1 (register 6) for trap handler to use
        asm.emit_i::<ADDI>(6, *return_addr, 0);

        // Index 4: Get target PC as advice
        asm.emit_j::<VirtualAdvice>(*next_pc, 0);

        // Index 5: Get trap handler address as advice
        // - For CSR ECALL: this is a3 (the trap handler address being set)
        // - For regular ECALL: this is current reg33 value (preserves it)
        asm.emit_j::<VirtualAdvice>(*trap_handler_advice, 0);

        // Index 6: Write trap handler advice to register 33
        // This sets reg33 for CSR ECALL, or preserves it for regular ECALL
        asm.emit_i::<ADDI>(v_trap_handler_reg, *trap_handler_advice, 0);

        // Index 7: Read trap handler from register 33
        asm.emit_i::<ADDI>(*trap_handler, v_trap_handler_reg, 0);

        // Verify: (target_pc == return_addr) OR (target_pc == trap_handler)
        // Compute: (target_pc - return_addr) * (target_pc - trap_handler) == 0
        // Index 8: diff1 = target_pc - return_addr
        asm.emit_r::<SUB>(*diff1, *next_pc, *return_addr);
        // Index 9: diff2 = target_pc - trap_handler
        asm.emit_r::<SUB>(*diff2, *next_pc, *trap_handler);
        // Index 10: product = diff1 * diff2 (reuse diff1 register)
        asm.emit_r::<MUL>(*diff1, *diff1, *diff2);
        // Index 11: Assert product == 0
        asm.emit_b::<VirtualAssertEQ>(*diff1, 0, 0);

        // Index 12: Jump to target PC. JALR has Jump=true, so the NextUnexpPCUpdateOtherwise
        // constraint won't fire (guard is !(ShouldBranch || Jump) which is false).
        // Using rd=0 means we don't save the return address.
        asm.emit_i::<JALR>(0, *next_pc, 0);

        asm.finalize()
    }
}
