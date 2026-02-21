//! ECALL (SYSTEM 0x0000_0073) — Environment call, always traps to mtvec.
//!
//! The inline sequence writes mepc, mcause, mtval, and mstatus to their
//! virtual registers, then jumps unconditionally to the trap handler.

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, PrivilegeMode, Trap, TrapType, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    addi::ADDI, auipc::AUIPC, format::format_i::FormatI, jalr::JALR, slli::SLLI, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
};

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

impl RISCVTrace for ECALL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
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
        let v_trap_handler_reg = allocator.trap_handler_register();
        let vr_mepc = allocator.mepc_register();
        let vr_mcause = allocator.mcause_register();
        let vr_mtval = allocator.mtval_register();
        let vr_mstatus = allocator.mstatus_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        let ecall_addr = allocator.allocate();
        asm.emit_u::<AUIPC>(*ecall_addr, 0);
        asm.emit_i::<ADDI>(vr_mepc, *ecall_addr, 0);
        drop(ecall_addr);

        asm.emit_i::<ADDI>(vr_mcause, 0, MCAUSE_ECALL_FROM_MMODE);
        asm.emit_i::<ADDI>(vr_mtval, 0, 0);

        // mstatus = 0x1800 (MPP=3 at bits 12:11)
        let three = allocator.allocate();
        asm.emit_i::<ADDI>(*three, 0, 3);
        asm.emit_i::<SLLI>(vr_mstatus, *three, 11);
        drop(three);

        // Use virtual register for rd to discard write value
        let jalr_rd = allocator.allocate();
        asm.emit_i::<JALR>(*jalr_rd, v_trap_handler_reg, 0);

        asm.finalize()
    }
}
