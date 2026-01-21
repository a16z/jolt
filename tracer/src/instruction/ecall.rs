//! ECALL (SYSTEM 0x0000_0073) — Environment call for syscalls and special operations.
//!
//! When ECALL triggers a trap (falls through to ZeroOS trap handler), the inline
//! sequence writes mepc, mcause, mtval, and mstatus to their virtual registers.
//! This keeps virtual registers as the single source of truth for proofs.
//!
//! Key insight: For M-mode only operation, we can compute all trap values:
//! - mepc = ECALL instruction address
//! - mcause = 11 (ECALL from M-mode)
//! - mtval = 0 (always)
//! - mstatus = computed from current mstatus

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

/// mcause value for ECALL from M-mode
const MCAUSE_ECALL_FROM_MMODE: u64 = 11;

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

impl ECALL {
    /// Generate inline sequence for ECALL.
    ///
    /// Always includes trap CSR writes to maintain consistent sequence length.
    /// When trap is not taken, the CSR writes preserve current values (no-op).
    /// TODO(sagar) we should just move cycle_tracking to a custom instruction
    ///
    /// Key insight: We compute trap values directly without reading from cpu.rs CSR state:
    /// - mepc = self.address (ECALL instruction address)
    /// - mcause = 11 (constant for ECALL from M-mode)
    /// - mtval = 0 (always)
    /// - mstatus = computed from current mstatus
    fn generate_inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_trap_handler_reg = allocator.trap_handler_register();
        let vr_mepc = allocator.mepc_register();
        let vr_mcause = allocator.mcause_register();
        let vr_mtval = allocator.mtval_register();
        let vr_mstatus = allocator.mstatus_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Always write trap CSRs (sequence length must be consistent)
        // When trap is not taken, advice values preserve current vr values
        // Note: We explicitly drop temp registers after use to stay within the 7-register limit

        // Write mepc (need VirtualAdvice for address or preserved value)
        let temp_mepc = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*temp_mepc, 0);
        asm.emit_i::<ADDI>(vr_mepc, *temp_mepc, 0);
        drop(temp_mepc); // Free register for reuse

        // Write mcause (need VirtualAdvice - 11 when trap taken, preserve when not)
        let temp_mcause = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*temp_mcause, 0);
        asm.emit_i::<ADDI>(vr_mcause, *temp_mcause, 0);
        drop(temp_mcause); // Free register for reuse

        // Write mtval (need VirtualAdvice - 0 when trap taken, preserve when not)
        let temp_mtval = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*temp_mtval, 0);
        asm.emit_i::<ADDI>(vr_mtval, *temp_mtval, 0);
        drop(temp_mtval); // Free register for reuse

        // Write mstatus (need VirtualAdvice for computed value or preserved)
        let temp_status = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*temp_status, 0);
        asm.emit_i::<ADDI>(vr_mstatus, *temp_status, 0);
        drop(temp_status); // Free register for reuse

        // Set up return address and jump
        let return_addr = allocator.allocate();
        let next_pc = allocator.allocate();
        let diff1 = allocator.allocate();
        let diff2 = allocator.allocate();

        // Get return address as advice (used for verification)
        asm.emit_j::<VirtualAdvice>(*return_addr, 0);

        // Get target PC as advice
        asm.emit_j::<VirtualAdvice>(*next_pc, 0);

        // Verify: (target_pc == return_addr) OR (target_pc == trap_handler)
        asm.emit_r::<SUB>(*diff1, *next_pc, *return_addr);
        asm.emit_r::<SUB>(*diff2, *next_pc, v_trap_handler_reg);
        asm.emit_r::<MUL>(*diff1, *diff1, *diff2);
        asm.emit_b::<VirtualAssertEQ>(*diff1, 0, 0);

        // Jump to target PC
        asm.emit_i::<JALR>(0, *next_pc, 0);

        asm.finalize()
    }
}

impl RISCVTrace for ECALL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Execute the ECALL to trigger trap handling.
        let mut ram_access = ();
        self.execute(cpu, &mut ram_access);

        // After trap handling, CPU's PC is either:
        // - trap handler address (if trap was taken, e.g., for Linux syscalls with ZeroOS)
        // - self.address + 4 (if trap was not taken, e.g., for Jolt-specific ECALLs)
        let target_pc = cpu.read_pc();
        let return_addr = self.address + 4;
        let trap_taken = target_pc != return_addr;

        let mut inline_sequence = self.generate_inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // Fill in the advice values (sequence always has same structure)
        // Index 0: mepc VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = if trap_taken {
                self.address // ECALL instruction address
            } else {
                // Preserve current value when trap not taken
                cpu.x[cpu.vr_allocator.mepc_register() as usize] as u64
            };
        }
        // Index 1: ADDI to vr_mepc (skip)

        // Index 2: mcause VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[2] {
            instr.advice = if trap_taken {
                MCAUSE_ECALL_FROM_MMODE // 11 for ECALL from M-mode
            } else {
                // Preserve current value when trap not taken
                cpu.x[cpu.vr_allocator.mcause_register() as usize] as u64
            };
        }
        // Index 3: ADDI to vr_mcause (skip)

        // Index 4: mtval VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[4] {
            instr.advice = if trap_taken {
                0 // mtval is always 0 for ECALL
            } else {
                // Preserve current value when trap not taken
                cpu.x[cpu.vr_allocator.mtval_register() as usize] as u64
            };
        }
        // Index 5: ADDI to vr_mtval (skip)

        // Index 6: mstatus VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[6] {
            if trap_taken {
                // Compute new mstatus per RISC-V spec:
                // new_status = (old_status & !0x1888) | (mie << 7) | (3 << 11)
                let old_status = cpu.x[cpu.vr_allocator.mstatus_register() as usize] as u64;
                let mie = (old_status >> 3) & 1;
                instr.advice = (old_status & !0x1888) | (mie << 7) | (3 << 11);
            } else {
                // Preserve current value when trap not taken
                instr.advice = cpu.x[cpu.vr_allocator.mstatus_register() as usize] as u64;
            }
        }
        // Index 7: ADDI to vr_mstatus (skip)

        // Index 8: return address VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[8] {
            instr.advice = return_addr;
        }

        // Index 9: target PC VirtualAdvice
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[9] {
            instr.advice = target_pc;
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        self.generate_inline_sequence(allocator, xlen)
    }
}
