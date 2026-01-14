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
    virtual_advice::VirtualAdvice,
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

        // Get syscall result that was stored by cpu.handle_syscall()
        let syscall_result = cpu.pending_csr_result.take().unwrap_or(0) as u64;

        // After trap handling, CPU's PC is either:
        // - mtvec (if trap was taken, e.g., for Linux syscalls with ZeroOS)
        // - self.address + 4 (if trap was not taken, e.g., for Jolt-specific ECALLs)
        // We need the inline sequence to jump to wherever CPU's PC is now.
        let target_pc = cpu.read_pc();

        // Return address for trap handler: ECALL_addr + 4 (ECALL is always 4 bytes)
        // This is passed in t1 so the trap handler can return without using mepc.
        let return_addr = self.address + 4;

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // Fill in the advice values:
        // - Index 0: syscall result (for a0)
        // - Index 2: return address (for t1, used by trap handler to return)
        // - Index 4: target PC (for JALR)
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = syscall_result;
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

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// ECALL inline sequence: use VirtualAdvice to write syscall return value to a0,
    /// return address to t1, and target PC for JALR, then JALR to the target.
    ///
    /// Syscalls return their result in a0 (register 10), but ECALL's encoding has rd=0.
    /// We use VirtualAdvice to provide the syscall result as untrusted advice,
    /// which gets written to a0.
    ///
    /// The return address (ECALL_addr+4) is passed in t1 (register 6). For trap-taking
    /// ECALLs, the trap handler saves t1 and uses it to return, eliminating the need
    /// for mepc CSR operations.
    ///
    /// The target PC is also provided via VirtualAdvice. This allows the inline sequence
    /// to jump to either ECALL_addr+4 (for non-trap-taking ECALLs) or mtvec (for trap-taking
    /// ECALLs handled by ZeroOS).
    ///
    /// The sequence ends with JALR to avoid the NextUnexpPCUpdateOtherwise constraint
    /// failing when the inline sequence is followed by NoOp padding. JALR has Jump=true,
    /// which makes the constraint guard `!(ShouldBranch || Jump)` false.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let syscall_result = allocator.allocate(); // temporary for syscall result
        let return_addr = allocator.allocate(); // temporary for return address
        let next_pc = allocator.allocate(); // temporary for target PC

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get syscall result as advice and write to temp register
        asm.emit_j::<VirtualAdvice>(*syscall_result, 0);

        // Move result to a0 (register 10)
        asm.emit_i::<ADDI>(10, *syscall_result, 0);

        // Get return address (ECALL_addr+4) as advice and write to temp register
        asm.emit_j::<VirtualAdvice>(*return_addr, 0);

        // Move return address to t1 (register 6) for trap handler to use
        asm.emit_i::<ADDI>(6, *return_addr, 0);

        // Get target PC as advice and write to temp register
        // This handles both trap-taking (PC = mtvec) and non-trap-taking (PC = ECALL_addr + 4) cases
        asm.emit_j::<VirtualAdvice>(*next_pc, 0);

        // Jump to target PC. JALR has Jump=true, so the NextUnexpPCUpdateOtherwise
        // constraint won't fire (guard is !(ShouldBranch || Jump) which is false).
        // Using rd=0 means we don't save the return address.
        asm.emit_i::<JALR>(0, *next_pc, 0);

        asm.finalize()
    }
}
