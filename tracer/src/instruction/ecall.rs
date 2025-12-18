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
        // which sets cpu.pending_syscall_result with the syscall return value.
        let mut ram_access = ();
        self.execute(cpu, &mut ram_access);

        // Get syscall result that was stored by cpu.handle_syscall()
        let syscall_result = cpu.pending_syscall_result.take().unwrap_or(0) as u64;

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // Fill in the advice value for VirtualAdvice instruction
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = syscall_result;
        } else {
            panic!("Expected VirtualAdvice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// ECALL inline sequence: use VirtualAdvice to write syscall return value to a0.
    ///
    /// Syscalls return their result in a0 (register 10), but ECALL's encoding has rd=0.
    /// We use VirtualAdvice to provide the syscall result as untrusted advice,
    /// which gets written to a0.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = allocator.allocate(); // temporary for syscall result

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get syscall result as advice and write to temp register
        asm.emit_j::<VirtualAdvice>(*a0, 0);

        // Move result to a0 (register 10)
        asm.emit_i::<ADDI>(10, *a0, 0);

        asm.finalize()
    }
}
