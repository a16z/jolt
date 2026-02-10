use serde::{Deserialize, Serialize};

use jolt_platform::{JOLT_ADVICE_WRITE_CALL_ID, JOLT_CYCLE_TRACK_CALL_ID, JOLT_PRINT_CALL_ID};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = VirtualHostIO,
    mask   = 0,
    match  = 0,
    format = FormatI,
    ram    = ()
);

impl VirtualHostIO {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualHostIO as RISCVInstruction>::RAMAccess) {
        let call_id = cpu.x[10] as u32;

        if call_id == JOLT_ADVICE_WRITE_CALL_ID {
            let ptr = cpu.x[11] as u64;
            let len = cpu.x[12] as u64;
            let _ = cpu.handle_advice_write(ptr, len);
        } else {
            let ptr = cpu.x[11] as u32;
            let len = cpu.x[12] as u32;
            let event = cpu.x[13] as u32;

            if call_id == JOLT_CYCLE_TRACK_CALL_ID {
                let _ = cpu.handle_jolt_cycle_marker(ptr, len, event);
            } else if call_id == JOLT_PRINT_CALL_ID {
                let _ = cpu.handle_jolt_print(ptr, len, event);
            }
        }
    }
}

impl RISCVTrace for VirtualHostIO {}
