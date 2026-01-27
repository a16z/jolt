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

use jolt_platform::{JOLT_CYCLE_TRACK_ECALL_NUM, JOLT_PRINT_ECALL_NUM};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, PrivilegeMode, Trap, TrapType, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    addi::ADDI, auipc::AUIPC, format::format_i::FormatI, jalr::JALR, mul::MUL, slli::SLLI,
    sub::SUB, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
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
    /// All trap CSR writes use constrained instructions (no VirtualAdvice):
    /// - mepc = ecall_addr (computed via AUIPC, circuit-constrained)
    /// - mcause = 11 (constant, ECALL from M-mode)
    /// - mtval = 0 (constant)
    /// - mstatus = 0x1800 (M-mode, MPP=3, MIE=0)
    ///
    /// The only advice value is target_pc, which is constrained to be either:
    /// - return_addr (ecall_addr + 4) for Jolt-specific ECALLs, or
    /// - trap_handler (from virtual register) for real traps
    ///
    /// This makes ECALL fully sound: a malicious prover cannot corrupt CSR values
    /// or jump to arbitrary addresses.
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

        // === Constrained CSR writes ===
        // All values computed without advice - circuit-enforced correctness

        // Compute ecall_addr via AUIPC (circuit constrains rd = PC + imm)
        // Since all inline instructions use self.address as PC, AUIPC(t, 0) = ecall_addr
        let ecall_addr = allocator.allocate();
        asm.emit_u::<AUIPC>(*ecall_addr, 0);

        // mepc = ecall_addr (ADDI copies the value)
        asm.emit_i::<ADDI>(vr_mepc, *ecall_addr, 0);

        // return_addr = ecall_addr + 4 (next instruction after ECALL)
        let return_addr = allocator.allocate();
        asm.emit_i::<ADDI>(*return_addr, *ecall_addr, 4);
        drop(ecall_addr); // Free for reuse

        // mcause = 11 (ECALL from M-mode, constant)
        asm.emit_i::<ADDI>(vr_mcause, 0, MCAUSE_ECALL_FROM_MMODE);

        // mtval = 0 (always 0 for ECALL, constant)
        asm.emit_i::<ADDI>(vr_mtval, 0, 0);

        // mstatus = 0x1800 (MPP=3 at bits 12:11, M-mode only)
        // Computed as: 3 << 11 = 0x1800
        let three = allocator.allocate();
        asm.emit_i::<ADDI>(*three, 0, 3);
        asm.emit_i::<SLLI>(vr_mstatus, *three, 11);
        drop(three); // Free for reuse

        // === Target PC with constraint ===
        // Only VirtualAdvice is target_pc, constrained to {return_addr, trap_handler}

        let target_pc = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*target_pc, 0);

        // Verify: (target_pc == return_addr) OR (target_pc == trap_handler)
        // Using: (target_pc - return_addr) * (target_pc - trap_handler) == 0
        let diff1 = allocator.allocate();
        let diff2 = allocator.allocate();
        asm.emit_r::<SUB>(*diff1, *target_pc, *return_addr);
        asm.emit_r::<SUB>(*diff2, *target_pc, v_trap_handler_reg);
        asm.emit_r::<MUL>(*diff1, *diff1, *diff2);
        asm.emit_b::<VirtualAssertEQ>(*diff1, 0, 0);

        // Jump to target PC
        asm.emit_i::<JALR>(0, *target_pc, 0);

        asm.finalize()
    }
}

impl RISCVTrace for ECALL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Don't call self.execute() - the inline sequence handles all register/PC updates.
        // CSR values are computed with constrained instructions (no advice needed).

        let return_addr = self.address + 4;
        let call_id = cpu.x[10] as u32; // a0

        // Handle Jolt-specific ECALLs (these don't take trap)
        let trap_taken = if call_id == JOLT_CYCLE_TRACK_ECALL_NUM {
            let marker_ptr = cpu.x[11] as u32; // a1
            let marker_len = cpu.x[12] as u32; // a2
            let event_type = cpu.x[13] as u32; // a3
            let _ = cpu.handle_jolt_cycle_marker(marker_ptr, marker_len, event_type);
            false
        } else if call_id == JOLT_PRINT_ECALL_NUM {
            let string_ptr = cpu.x[11] as u32; // a1
            let string_len = cpu.x[12] as u32; // a2
            let event_type = cpu.x[13] as u32; // a3
            let _ = cpu.handle_jolt_print(string_ptr, string_len, event_type as u8);
            false
        } else {
            true
        };

        // Compute target PC:
        // - Jolt ECALLs: return to next instruction (no trap)
        // - Other ECALLs: jump to trap handler (from virtual register)
        let target_pc = if trap_taken {
            cpu.x[cpu.vr_allocator.trap_handler_register() as usize] as u64
        } else {
            return_addr
        };

        let mut inline_sequence = self.generate_inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // The inline sequence structure (all constrained except target_pc):
        // 0: AUIPC(ecall_addr, 0)
        // 1: ADDI(vr_mepc, ecall_addr, 0)
        // 2: ADDI(return_addr, ecall_addr, 4)
        // 3: ADDI(vr_mcause, x0, 11)
        // 4: ADDI(vr_mtval, x0, 0)
        // 5: ADDI(three, x0, 3)
        // 6: VirtualMULI(vr_mstatus, three, 2048)  <- SLLI expands to this
        // 7: VirtualAdvice(target_pc)              <- Only advice, constrained below
        // 8: SUB(diff1, target_pc, return_addr)
        // 9: SUB(diff2, target_pc, v_trap_handler)
        // 10: MUL(diff1, diff1, diff2)
        // 11: VirtualAssertEQ(diff1, 0, 0)
        // 12: JALR(x0, target_pc, 0)

        // Fill in the only advice value: target_pc at index 7
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[7] {
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
