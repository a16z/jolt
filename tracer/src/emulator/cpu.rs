#![allow(clippy::useless_format, clippy::type_complexity, dead_code)]

#[cfg(feature = "std")]
extern crate fnv;

#[cfg(feature = "std")]
use self::fnv::FnvHashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::btree_map::BTreeMap as FnvHashMap;
use core::convert::TryInto;

use crate::instruction::{RV32IMCycle, RV32IMInstruction};

use super::mmu::{AddressingMode, Mmu};
use super::terminal::Terminal;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};

const CSR_CAPACITY: usize = 4096;

const CSR_USTATUS_ADDRESS: u16 = 0x000;
const CSR_FFLAGS_ADDRESS: u16 = 0x001;
const CSR_FRM_ADDRESS: u16 = 0x002;
const CSR_FCSR_ADDRESS: u16 = 0x003;
const CSR_UIE_ADDRESS: u16 = 0x004;
const CSR_UTVEC_ADDRESS: u16 = 0x005;
const _CSR_USCRATCH_ADDRESS: u16 = 0x040;
const CSR_UEPC_ADDRESS: u16 = 0x041;
const CSR_UCAUSE_ADDRESS: u16 = 0x042;
const CSR_UTVAL_ADDRESS: u16 = 0x043;
const _CSR_UIP_ADDRESS: u16 = 0x044;
const CSR_SSTATUS_ADDRESS: u16 = 0x100;
const CSR_SEDELEG_ADDRESS: u16 = 0x102;
const CSR_SIDELEG_ADDRESS: u16 = 0x103;
const CSR_SIE_ADDRESS: u16 = 0x104;
const CSR_STVEC_ADDRESS: u16 = 0x105;
const _CSR_SSCRATCH_ADDRESS: u16 = 0x140;
const CSR_SEPC_ADDRESS: u16 = 0x141;
const CSR_SCAUSE_ADDRESS: u16 = 0x142;
const CSR_STVAL_ADDRESS: u16 = 0x143;
const CSR_SIP_ADDRESS: u16 = 0x144;
const CSR_SATP_ADDRESS: u16 = 0x180;
const CSR_MSTATUS_ADDRESS: u16 = 0x300;
const CSR_MISA_ADDRESS: u16 = 0x301;
const CSR_MEDELEG_ADDRESS: u16 = 0x302;
const CSR_MIDELEG_ADDRESS: u16 = 0x303;
const CSR_MIE_ADDRESS: u16 = 0x304;

const CSR_MTVEC_ADDRESS: u16 = 0x305;
const _CSR_MSCRATCH_ADDRESS: u16 = 0x340;
const CSR_MEPC_ADDRESS: u16 = 0x341;
const CSR_MCAUSE_ADDRESS: u16 = 0x342;
const CSR_MTVAL_ADDRESS: u16 = 0x343;
const CSR_MIP_ADDRESS: u16 = 0x344;
const _CSR_PMPCFG0_ADDRESS: u16 = 0x3a0;
const _CSR_PMPADDR0_ADDRESS: u16 = 0x3b0;
const _CSR_MCYCLE_ADDRESS: u16 = 0xb00;
const CSR_CYCLE_ADDRESS: u16 = 0xc00;
const CSR_TIME_ADDRESS: u16 = 0xc01;
const _CSR_INSERT_ADDRESS: u16 = 0xc02;
const _CSR_MHARTID_ADDRESS: u16 = 0xf14;

const MIP_MEIP: u64 = 0x800;
pub const MIP_MTIP: u64 = 0x080;
pub const MIP_MSIP: u64 = 0x008;
pub const MIP_SEIP: u64 = 0x200;
const MIP_STIP: u64 = 0x020;
const MIP_SSIP: u64 = 0x002;

// Must be a power of 2.
pub const TOTAL_REGISTERS: usize = 128;

pub const JOLT_CYCLE_TRACK_ECALL_NUM: u32 = 0xC7C1E;
pub const JOLT_CYCLE_MARKER_START: u32 = 1;
pub const JOLT_CYCLE_MARKER_END: u32 = 2;
#[derive(Clone)]
struct ActiveMarker {
    label: String,
    start_instrs: u64,      // executed_instrs  at ‘start’
    start_trace_len: usize, // trace.len()      at ‘start’
}

/// Emulates a RISC-V CPU core
#[derive(Clone)]
pub struct Cpu {
    clock: u64,
    pub(crate) xlen: Xlen,
    pub(crate) privilege_mode: PrivilegeMode,
    wfi: bool,
    // using only lower 32bits of x, pc, and csr registers
    // for 32-bit mode
    pub x: [i64; TOTAL_REGISTERS],
    f: [f64; 32],
    pub(crate) pc: u64,
    csr: [u64; CSR_CAPACITY],
    pub(crate) mmu: Mmu,
    reservation: u64, // @TODO: Should support multiple address reservations
    is_reservation_set: bool,
    _dump_flag: bool,
    unsigned_data_mask: u64,
    // pub trace: Vec<RV32IMCycle>,
    pub trace_len: usize,
    executed_instrs: u64, // “real” RV32IM cycles
    active_markers: FnvHashMap<u32, ActiveMarker>,
}

#[derive(Clone)]
pub enum Xlen {
    Bit32,
    Bit64, // @TODO: Support Bit128
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum PrivilegeMode {
    User,
    Supervisor,
    Reserved,
    Machine,
}

#[derive(Debug)]
pub struct Trap {
    pub trap_type: TrapType,
    pub value: u64, // Trap type specific value
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum TrapType {
    InstructionAddressMisaligned,
    InstructionAccessFault,
    IllegalInstruction,
    Breakpoint,
    LoadAddressMisaligned,
    LoadAccessFault,
    StoreAddressMisaligned,
    StoreAccessFault,
    EnvironmentCallFromUMode,
    EnvironmentCallFromSMode,
    EnvironmentCallFromMMode,
    InstructionPageFault,
    LoadPageFault,
    StorePageFault,
    UserSoftwareInterrupt,
    SupervisorSoftwareInterrupt,
    MachineSoftwareInterrupt,
    UserTimerInterrupt,
    SupervisorTimerInterrupt,
    MachineTimerInterrupt,
    UserExternalInterrupt,
    SupervisorExternalInterrupt,
    MachineExternalInterrupt,
}

fn _get_privilege_mode_name(mode: &PrivilegeMode) -> &'static str {
    match mode {
        PrivilegeMode::User => "User",
        PrivilegeMode::Supervisor => "Supervisor",
        PrivilegeMode::Reserved => "Reserved",
        PrivilegeMode::Machine => "Machine",
    }
}

// bigger number is higher privilege level
fn get_privilege_encoding(mode: &PrivilegeMode) -> u8 {
    match mode {
        PrivilegeMode::User => 0,
        PrivilegeMode::Supervisor => 1,
        PrivilegeMode::Reserved => panic!(),
        PrivilegeMode::Machine => 3,
    }
}

/// Returns `PrivilegeMode` from encoded privilege mode bits
pub fn get_privilege_mode(encoding: u64) -> PrivilegeMode {
    match encoding {
        0 => PrivilegeMode::User,
        1 => PrivilegeMode::Supervisor,
        3 => PrivilegeMode::Machine,
        _ => panic!("Unknown privilege encoding"),
    }
}

fn _get_trap_type_name(trap_type: &TrapType) -> &'static str {
    match trap_type {
        TrapType::InstructionAddressMisaligned => "InstructionAddressMisaligned",
        TrapType::InstructionAccessFault => "InstructionAccessFault",
        TrapType::IllegalInstruction => "IllegalInstruction",
        TrapType::Breakpoint => "Breakpoint",
        TrapType::LoadAddressMisaligned => "LoadAddressMisaligned",
        TrapType::LoadAccessFault => "LoadAccessFault",
        TrapType::StoreAddressMisaligned => "StoreAddressMisaligned",
        TrapType::StoreAccessFault => "StoreAccessFault",
        TrapType::EnvironmentCallFromUMode => "EnvironmentCallFromUMode",
        TrapType::EnvironmentCallFromSMode => "EnvironmentCallFromSMode",
        TrapType::EnvironmentCallFromMMode => "EnvironmentCallFromMMode",
        TrapType::InstructionPageFault => "InstructionPageFault",
        TrapType::LoadPageFault => "LoadPageFault",
        TrapType::StorePageFault => "StorePageFault",
        TrapType::UserSoftwareInterrupt => "UserSoftwareInterrupt",
        TrapType::SupervisorSoftwareInterrupt => "SupervisorSoftwareInterrupt",
        TrapType::MachineSoftwareInterrupt => "MachineSoftwareInterrupt",
        TrapType::UserTimerInterrupt => "UserTimerInterrupt",
        TrapType::SupervisorTimerInterrupt => "SupervisorTimerInterrupt",
        TrapType::MachineTimerInterrupt => "MachineTimerInterrupt",
        TrapType::UserExternalInterrupt => "UserExternalInterrupt",
        TrapType::SupervisorExternalInterrupt => "SupervisorExternalInterrupt",
        TrapType::MachineExternalInterrupt => "MachineExternalInterrupt",
    }
}

fn get_trap_cause(trap: &Trap, xlen: &Xlen) -> u64 {
    let interrupt_bit = match xlen {
        Xlen::Bit32 => 0x80000000_u64,
        Xlen::Bit64 => 0x8000000000000000_u64,
    };
    match trap.trap_type {
        TrapType::InstructionAddressMisaligned => 0,
        TrapType::InstructionAccessFault => 1,
        TrapType::IllegalInstruction => 2,
        TrapType::Breakpoint => 3,
        TrapType::LoadAddressMisaligned => 4,
        TrapType::LoadAccessFault => 5,
        TrapType::StoreAddressMisaligned => 6,
        TrapType::StoreAccessFault => 7,
        TrapType::EnvironmentCallFromUMode => 8,
        TrapType::EnvironmentCallFromSMode => 9,
        TrapType::EnvironmentCallFromMMode => 11,
        TrapType::InstructionPageFault => 12,
        TrapType::LoadPageFault => 13,
        TrapType::StorePageFault => 15,
        TrapType::UserSoftwareInterrupt => interrupt_bit,
        TrapType::SupervisorSoftwareInterrupt => interrupt_bit + 1,
        TrapType::MachineSoftwareInterrupt => interrupt_bit + 3,
        TrapType::UserTimerInterrupt => interrupt_bit + 4,
        TrapType::SupervisorTimerInterrupt => interrupt_bit + 5,
        TrapType::MachineTimerInterrupt => interrupt_bit + 7,
        TrapType::UserExternalInterrupt => interrupt_bit + 8,
        TrapType::SupervisorExternalInterrupt => interrupt_bit + 9,
        TrapType::MachineExternalInterrupt => interrupt_bit + 11,
    }
}

impl Cpu {
    /// Creates a new `Cpu`.
    ///
    /// # Arguments
    /// * `Terminal`
    pub fn new(terminal: Box<dyn Terminal>) -> Self {
        let mut cpu = Cpu {
            clock: 0,
            xlen: Xlen::Bit64,
            privilege_mode: PrivilegeMode::Machine,
            wfi: false,
            x: [0; TOTAL_REGISTERS],
            f: [0.0; 32],
            pc: 0,
            csr: [0; CSR_CAPACITY],
            mmu: Mmu::new(Xlen::Bit64, terminal),
            reservation: 0,
            is_reservation_set: false,
            _dump_flag: false,
            unsigned_data_mask: 0xffffffffffffffff,
            // trace: Vec::with_capacity(1 << 24), // TODO(moodlezoup): make configurable
            trace_len: 0,
            executed_instrs: 0,
            active_markers: FnvHashMap::default(),
        };
        // cpu.x[0xb] = 0x1020; // I don't know why but Linux boot seems to require this initialization
        cpu.write_csr_raw(CSR_MISA_ADDRESS, 0x800000008014312f);
        cpu
    }
    /// trap wrapper for cycle tracking tool
    #[inline(always)]
    pub fn raise_trap(&mut self, trap: Trap, faulting_pc: u64) {
        let _ = self.handle_trap(trap, faulting_pc, false);
    }

    /// Updates Program Counter content
    ///
    /// # Arguments
    /// * `value`
    pub fn update_pc(&mut self, value: u64) {
        self.pc = value;
    }

    /// Updates XLEN, 32-bit or 64-bit
    ///
    /// # Arguments
    /// * `xlen`
    pub fn update_xlen(&mut self, xlen: Xlen) {
        self.xlen = xlen.clone();
        self.unsigned_data_mask = match xlen {
            Xlen::Bit32 => 0xffffffff,
            Xlen::Bit64 => 0xffffffffffffffff,
        };
        self.mmu.update_xlen(xlen.clone());
    }

    /// Reads integer register content
    ///
    /// # Arguments
    /// * `reg` Register number. Must be 0-31
    pub fn read_register(&self, reg: u8) -> i64 {
        debug_assert!(reg <= 31, "reg must be 0-31. {reg}");
        match reg {
            0 => 0, // 0th register is hardwired zero
            _ => self.x[reg as usize],
        }
    }

    /// Reads Program counter content
    pub fn read_pc(&self) -> u64 {
        self.pc
    }

    /// Sets the reservation address for atomic memory operations
    pub fn set_reservation(&mut self, address: u64) {
        self.reservation = address;
        self.is_reservation_set = true;
    }

    /// Clears the reservation for atomic memory operations
    pub fn clear_reservation(&mut self) {
        self.is_reservation_set = false;
    }

    /// Checks if a reservation is set for the given address
    pub fn has_reservation(&self, address: u64) -> bool {
        self.is_reservation_set && self.reservation == address
    }

    pub fn is_reservation_set(&self) -> bool {
        self.is_reservation_set
    }

    /// Runs program one cycle. Fetch, decode, and execution are completed in a cycle so far.
    pub fn tick(&mut self, trace: Option<&mut Vec<RV32IMCycle>>) {
        let instruction_address = self.pc;
        match self.tick_operate(trace) {
            Ok(()) => {}
            Err(e) => self.handle_exception(e, instruction_address),
        }
        self.mmu.tick();
        self.handle_interrupt(self.pc);
        self.clock = self.clock.wrapping_add(1);

        // cpu core clock : mtime clock in clint = 8 : 1 is
        // just an arbitrary ratio.
        // @TODO: Implement more properly
        self.write_csr_raw(CSR_CYCLE_ADDRESS, self.clock * 8);
    }

    // @TODO: Rename?
    fn tick_operate(&mut self, trace: Option<&mut Vec<RV32IMCycle>>) -> Result<(), Trap> {
        if self.wfi {
            if (self.read_csr_raw(CSR_MIE_ADDRESS) & self.read_csr_raw(CSR_MIP_ADDRESS)) != 0 {
                self.wfi = false;
            }
            return Ok(());
        }

        let original_word = self.fetch()?;
        let instruction_address = normalize_u64(self.pc, &self.xlen);
        let word = match (original_word & 0x3) == 0x3 {
            true => {
                self.pc = self.pc.wrapping_add(4); // 32-bit length non-compressed instruction
                original_word
            }
            false => {
                self.pc = self.pc.wrapping_add(2); // 16-bit length compressed instruction
                self.uncompress(original_word & 0xffff)
            }
        };

        let instr = RV32IMInstruction::decode(word, instruction_address)
            .ok()
            .unwrap();

        if trace.is_none() {
            instr.execute(self);
        } else {
            instr.trace(self, trace);
        }

        // check if current instruction is real or not for cycle profiling
        if instr.is_real() {
            self.executed_instrs += 1;
        }
        self.x[0] = 0; // hardwired zero

        Ok(())
    }

    fn handle_interrupt(&mut self, instruction_address: u64) {
        // @TODO: Optimize
        let minterrupt = self.read_csr_raw(CSR_MIP_ADDRESS) & self.read_csr_raw(CSR_MIE_ADDRESS);

        if (minterrupt & MIP_MEIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::MachineExternalInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            // Who should clear mip bit?
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_MEIP,
            );
            self.wfi = false;
            return;
        }
        if (minterrupt & MIP_MSIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::MachineSoftwareInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_MSIP,
            );
            self.wfi = false;
            return;
        }
        if (minterrupt & MIP_MTIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::MachineTimerInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_MTIP,
            );
            self.wfi = false;
            return;
        }
        if (minterrupt & MIP_SEIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::SupervisorExternalInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_SEIP,
            );
            self.wfi = false;
            return;
        }
        if (minterrupt & MIP_SSIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::SupervisorSoftwareInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_SSIP,
            );
            self.wfi = false;
            return;
        }
        if (minterrupt & MIP_STIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::SupervisorTimerInterrupt,
                    value: self.pc, // dummy
                },
                instruction_address,
                true,
            )
        {
            self.write_csr_raw(
                CSR_MIP_ADDRESS,
                self.read_csr_raw(CSR_MIP_ADDRESS) & !MIP_STIP,
            );
            self.wfi = false;
        }
    }

    fn handle_exception(&mut self, exception: Trap, instruction_address: u64) {
        self.handle_trap(exception, instruction_address, false);
    }

    fn handle_trap(&mut self, trap: Trap, instruction_address: u64, is_interrupt: bool) -> bool {
        // non-interrupt case is an ECALL
        if !is_interrupt
            && matches!(
                trap.trap_type,
                TrapType::EnvironmentCallFromUMode
                    | TrapType::EnvironmentCallFromSMode
                    | TrapType::EnvironmentCallFromMMode
            )
        {
            let call_id = self.x[10] as u32; // a0
            if call_id == JOLT_CYCLE_TRACK_ECALL_NUM {
                let marker_ptr = self.x[11] as u32; // a1
                let event_type = self.x[12] as u32; // a2

                // Read / update the per-label counters.
                //
                // Any fault raised while touching guest memory (e.g. a bad
                // string pointer) is swallowed here and will manifest as the
                // usual access-fault on the *next* instruction fetch.
                let _ = self.handle_jolt_cycle_marker(marker_ptr, event_type);

                return false; // we don't take the trap
            }
        }

        let current_privilege_encoding = get_privilege_encoding(&self.privilege_mode) as u64;
        let cause = get_trap_cause(&trap, &self.xlen);

        // First, determine which privilege mode should handle the trap.
        // @TODO: Check if this logic is correct
        let mdeleg = match is_interrupt {
            true => self.read_csr_raw(CSR_MIDELEG_ADDRESS),
            false => self.read_csr_raw(CSR_MEDELEG_ADDRESS),
        };
        let sdeleg = match is_interrupt {
            true => self.read_csr_raw(CSR_SIDELEG_ADDRESS),
            false => self.read_csr_raw(CSR_SEDELEG_ADDRESS),
        };
        let pos = cause & 0xffff;

        let new_privilege_mode = match ((mdeleg >> pos) & 1) == 0 {
            true => PrivilegeMode::Machine,
            false => match ((sdeleg >> pos) & 1) == 0 {
                true => PrivilegeMode::Supervisor,
                false => PrivilegeMode::User,
            },
        };
        let new_privilege_encoding = get_privilege_encoding(&new_privilege_mode) as u64;

        let current_status = match self.privilege_mode {
            PrivilegeMode::Machine => self.read_csr_raw(CSR_MSTATUS_ADDRESS),
            PrivilegeMode::Supervisor => self.read_csr_raw(CSR_SSTATUS_ADDRESS),
            PrivilegeMode::User => self.read_csr_raw(CSR_USTATUS_ADDRESS),
            PrivilegeMode::Reserved => panic!(),
        };

        // Second, ignore the interrupt if it's disabled by some conditions

        if is_interrupt {
            let ie = match new_privilege_mode {
                PrivilegeMode::Machine => self.read_csr_raw(CSR_MIE_ADDRESS),
                PrivilegeMode::Supervisor => self.read_csr_raw(CSR_SIE_ADDRESS),
                PrivilegeMode::User => self.read_csr_raw(CSR_UIE_ADDRESS),
                PrivilegeMode::Reserved => panic!(),
            };

            let current_mie = (current_status >> 3) & 1;
            let current_sie = (current_status >> 1) & 1;
            let current_uie = current_status & 1;

            let msie = (ie >> 3) & 1;
            let ssie = (ie >> 1) & 1;
            let usie = ie & 1;

            let mtie = (ie >> 7) & 1;
            let stie = (ie >> 5) & 1;
            let utie = (ie >> 4) & 1;

            let meie = (ie >> 11) & 1;
            let seie = (ie >> 9) & 1;
            let ueie = (ie >> 8) & 1;

            // 1. Interrupt is always enabled if new privilege level is higher
            // than current privilege level
            // 2. Interrupt is always disabled if new privilege level is lower
            // than current privilege level
            // 3. Interrupt is enabled if xIE in xstatus is 1 where x is privilege level
            // and new privilege level equals to current privilege level

            #[allow(clippy::comparison_chain)]
            if new_privilege_encoding < current_privilege_encoding {
                return false;
            } else if current_privilege_encoding == new_privilege_encoding {
                match self.privilege_mode {
                    PrivilegeMode::Machine => {
                        if current_mie == 0 {
                            return false;
                        }
                    }
                    PrivilegeMode::Supervisor => {
                        if current_sie == 0 {
                            return false;
                        }
                    }
                    PrivilegeMode::User => {
                        if current_uie == 0 {
                            return false;
                        }
                    }
                    PrivilegeMode::Reserved => panic!(),
                };
            }

            // Interrupt can be maskable by xie csr register
            // where x is a new privilege mode.

            match trap.trap_type {
                TrapType::UserSoftwareInterrupt => {
                    if usie == 0 {
                        return false;
                    }
                }
                TrapType::SupervisorSoftwareInterrupt => {
                    if ssie == 0 {
                        return false;
                    }
                }
                TrapType::MachineSoftwareInterrupt => {
                    if msie == 0 {
                        return false;
                    }
                }
                TrapType::UserTimerInterrupt => {
                    if utie == 0 {
                        return false;
                    }
                }
                TrapType::SupervisorTimerInterrupt => {
                    if stie == 0 {
                        return false;
                    }
                }
                TrapType::MachineTimerInterrupt => {
                    if mtie == 0 {
                        return false;
                    }
                }
                TrapType::UserExternalInterrupt => {
                    if ueie == 0 {
                        return false;
                    }
                }
                TrapType::SupervisorExternalInterrupt => {
                    if seie == 0 {
                        return false;
                    }
                }
                TrapType::MachineExternalInterrupt => {
                    if meie == 0 {
                        return false;
                    }
                }
                _ => {}
            };
        }

        // So, this trap should be taken

        self.privilege_mode = new_privilege_mode;
        self.mmu.update_privilege_mode(self.privilege_mode.clone());
        let csr_epc_address = match self.privilege_mode {
            PrivilegeMode::Machine => CSR_MEPC_ADDRESS,
            PrivilegeMode::Supervisor => CSR_SEPC_ADDRESS,
            PrivilegeMode::User => CSR_UEPC_ADDRESS,
            PrivilegeMode::Reserved => panic!(),
        };
        let csr_cause_address = match self.privilege_mode {
            PrivilegeMode::Machine => CSR_MCAUSE_ADDRESS,
            PrivilegeMode::Supervisor => CSR_SCAUSE_ADDRESS,
            PrivilegeMode::User => CSR_UCAUSE_ADDRESS,
            PrivilegeMode::Reserved => panic!(),
        };
        let csr_tval_address = match self.privilege_mode {
            PrivilegeMode::Machine => CSR_MTVAL_ADDRESS,
            PrivilegeMode::Supervisor => CSR_STVAL_ADDRESS,
            PrivilegeMode::User => CSR_UTVAL_ADDRESS,
            PrivilegeMode::Reserved => panic!(),
        };
        let csr_tvec_address = match self.privilege_mode {
            PrivilegeMode::Machine => CSR_MTVEC_ADDRESS,
            PrivilegeMode::Supervisor => CSR_STVEC_ADDRESS,
            PrivilegeMode::User => CSR_UTVEC_ADDRESS,
            PrivilegeMode::Reserved => panic!(),
        };

        self.write_csr_raw(csr_epc_address, instruction_address);
        self.write_csr_raw(csr_cause_address, cause);
        self.write_csr_raw(csr_tval_address, trap.value);
        self.pc = self.read_csr_raw(csr_tvec_address);

        // Add 4 * cause if tvec has vector type address
        if (self.pc & 0x3) != 0 {
            self.pc = (self.pc & !0x3) + 4 * (cause & 0xffff);
        }

        match self.privilege_mode {
            PrivilegeMode::Machine => {
                let status = self.read_csr_raw(CSR_MSTATUS_ADDRESS);
                let mie = (status >> 3) & 1;
                // clear MIE[3], override MPIE[7] with MIE[3], override MPP[12:11] with current privilege encoding
                let new_status =
                    (status & !0x1888) | (mie << 7) | (current_privilege_encoding << 11);
                self.write_csr_raw(CSR_MSTATUS_ADDRESS, new_status);
            }
            PrivilegeMode::Supervisor => {
                let status = self.read_csr_raw(CSR_SSTATUS_ADDRESS);
                let sie = (status >> 1) & 1;
                // clear SIE[1], override SPIE[5] with SIE[1], override SPP[8] with current privilege encoding
                let new_status =
                    (status & !0x122) | (sie << 5) | ((current_privilege_encoding & 1) << 8);
                self.write_csr_raw(CSR_SSTATUS_ADDRESS, new_status);
            }
            PrivilegeMode::User => {
                panic!("Not implemented yet");
            }
            PrivilegeMode::Reserved => panic!(), // shouldn't happen
        };
        //println!("Trap! {:x} Clock:{:x}", cause, self.clock);
        true
    }

    fn fetch(&mut self) -> Result<u32, Trap> {
        let word = match self.mmu.fetch_word(self.pc) {
            Ok(word) => word,
            Err(e) => {
                self.pc = self.pc.wrapping_add(4); // @TODO: What if instruction is compressed?
                return Err(e);
            }
        };
        Ok(word)
    }

    fn has_csr_access_privilege(&self, address: u16) -> bool {
        let privilege = (address >> 8) & 0x3; // the lowest privilege level that can access the CSR
        privilege as u8 <= get_privilege_encoding(&self.privilege_mode)
    }

    fn read_csr(&mut self, address: u16) -> Result<u64, Trap> {
        match self.has_csr_access_privilege(address) {
            true => Ok(self.read_csr_raw(address)),
            false => Err(Trap {
                trap_type: TrapType::IllegalInstruction,
                value: self.pc.wrapping_sub(4), // @TODO: Is this always correct?
            }),
        }
    }

    fn write_csr(&mut self, address: u16, value: u64) -> Result<(), Trap> {
        match self.has_csr_access_privilege(address) {
            true => {
                /*
                // Checking writability fails some tests so disabling so far
                let read_only = ((address >> 10) & 0x3) == 0x3;
                if read_only {
                    return Err(Exception::IllegalInstruction);
                }
                */
                self.write_csr_raw(address, value);
                if address == CSR_SATP_ADDRESS {
                    self.update_addressing_mode(value);
                }
                Ok(())
            }
            false => Err(Trap {
                trap_type: TrapType::IllegalInstruction,
                value: self.pc.wrapping_sub(4), // @TODO: Is this always correct?
            }),
        }
    }

    // SSTATUS, SIE, and SIP are subsets of MSTATUS, MIE, and MIP
    fn read_csr_raw(&self, address: u16) -> u64 {
        match address {
            // @TODO: Mask should consider of 32-bit mode
            CSR_FFLAGS_ADDRESS => self.csr[CSR_FCSR_ADDRESS as usize] & 0x1f,
            CSR_FRM_ADDRESS => (self.csr[CSR_FCSR_ADDRESS as usize] >> 5) & 0x7,
            CSR_SSTATUS_ADDRESS => self.csr[CSR_MSTATUS_ADDRESS as usize] & 0x80000003000de162,
            CSR_SIE_ADDRESS => self.csr[CSR_MIE_ADDRESS as usize] & 0x222,
            CSR_SIP_ADDRESS => self.csr[CSR_MIP_ADDRESS as usize] & 0x222,
            CSR_TIME_ADDRESS => panic!("CLINT is unsupported."),
            _ => self.csr[address as usize],
        }
    }

    fn write_csr_raw(&mut self, address: u16, value: u64) {
        match address {
            CSR_FFLAGS_ADDRESS => {
                self.csr[CSR_FCSR_ADDRESS as usize] &= !0x1f;
                self.csr[CSR_FCSR_ADDRESS as usize] |= value & 0x1f;
            }
            CSR_FRM_ADDRESS => {
                self.csr[CSR_FCSR_ADDRESS as usize] &= !0xe0;
                self.csr[CSR_FCSR_ADDRESS as usize] |= (value << 5) & 0xe0;
            }
            CSR_SSTATUS_ADDRESS => {
                self.csr[CSR_MSTATUS_ADDRESS as usize] &= !0x80000003000de162;
                self.csr[CSR_MSTATUS_ADDRESS as usize] |= value & 0x80000003000de162;
                self.mmu
                    .update_mstatus(self.read_csr_raw(CSR_MSTATUS_ADDRESS));
            }
            CSR_SIE_ADDRESS => {
                self.csr[CSR_MIE_ADDRESS as usize] &= !0x222;
                self.csr[CSR_MIE_ADDRESS as usize] |= value & 0x222;
            }
            CSR_SIP_ADDRESS => {
                self.csr[CSR_MIP_ADDRESS as usize] &= !0x222;
                self.csr[CSR_MIP_ADDRESS as usize] |= value & 0x222;
            }
            CSR_MIDELEG_ADDRESS => {
                self.csr[address as usize] = value & 0x666; // from qemu
            }
            CSR_MSTATUS_ADDRESS => {
                self.csr[address as usize] = value;
                self.mmu
                    .update_mstatus(self.read_csr_raw(CSR_MSTATUS_ADDRESS));
            }
            CSR_TIME_ADDRESS => {
                panic!("CLINT is unsupported.")
            }
            _ => {
                self.csr[address as usize] = value;
            }
        };
    }

    fn _set_fcsr_nv(&mut self) {
        self.csr[CSR_FCSR_ADDRESS as usize] |= 0x10;
    }

    fn set_fcsr_dz(&mut self) {
        self.csr[CSR_FCSR_ADDRESS as usize] |= 0x8;
    }

    fn _set_fcsr_of(&mut self) {
        self.csr[CSR_FCSR_ADDRESS as usize] |= 0x4;
    }

    fn _set_fcsr_uf(&mut self) {
        self.csr[CSR_FCSR_ADDRESS as usize] |= 0x2;
    }

    fn _set_fcsr_nx(&mut self) {
        self.csr[CSR_FCSR_ADDRESS as usize] |= 0x1;
    }

    fn update_addressing_mode(&mut self, value: u64) {
        let addressing_mode = match self.xlen {
            Xlen::Bit32 => match value & 0x80000000 {
                0 => AddressingMode::None,
                _ => AddressingMode::SV32,
            },
            Xlen::Bit64 => match value >> 60 {
                0 => AddressingMode::None,
                8 => AddressingMode::SV39,
                9 => AddressingMode::SV48,
                _ => {
                    #[cfg(feature = "std")]
                    println!("Unknown addressing_mode {:x}", value >> 60);
                    panic!();
                }
            },
        };
        let ppn = match self.xlen {
            Xlen::Bit32 => value & 0x3fffff,
            Xlen::Bit64 => value & 0xfffffffffff,
        };
        self.mmu.update_addressing_mode(addressing_mode);
        self.mmu.update_ppn(ppn);
    }

    // @TODO: Rename to better name?
    pub(crate) fn sign_extend(&self, value: i64) -> i64 {
        match self.xlen {
            Xlen::Bit32 => value as i32 as i64,
            Xlen::Bit64 => value,
        }
    }

    // @TODO: Rename to better name?
    pub(crate) fn unsigned_data(&self, value: i64) -> u64 {
        (value as u64) & self.unsigned_data_mask
    }

    // @TODO: Rename to better name?
    pub(crate) fn most_negative(&self) -> i64 {
        match self.xlen {
            Xlen::Bit32 => i32::MIN as i64,
            Xlen::Bit64 => i64::MIN,
        }
    }

    // @TODO: Optimize
    fn uncompress(&self, halfword: u32) -> u32 {
        let op = halfword & 0x3; // [1:0]
        let funct3 = (halfword >> 13) & 0x7; // [15:13]

        match op {
            0 => match funct3 {
                0 => {
                    // C.ADDI4SPN
                    // addi rd+8, x2, nzuimm
                    let rd = (halfword >> 2) & 0x7; // [4:2]
                    let nzuimm = ((halfword >> 7) & 0x30) | // nzuimm[5:4] <= [12:11]
                        ((halfword >> 1) & 0x3c0) | // nzuimm{9:6] <= [10:7]
                        ((halfword >> 4) & 0x4) | // nzuimm[2] <= [6]
                        ((halfword >> 2) & 0x8); // nzuimm[3] <= [5]
                                                 // nzuimm == 0 is reserved instruction
                    if nzuimm != 0 {
                        return (nzuimm << 20) | (2 << 15) | ((rd + 8) << 7) | 0x13;
                    }
                }
                1 => {
                    // @TODO: Support C.LQ for 128-bit
                    // C.FLD for 32, 64-bit
                    // fld rd+8, offset(rs1+8)
                    let rd = (halfword >> 2) & 0x7; // [4:2]
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
                    return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x7;
                }
                2 => {
                    // C.LW
                    // lw rd+8, offset(rs1+8)
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let rd = (halfword >> 2) & 0x7; // [4:2]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword >> 4) & 0x4) | // offset[2] <= [6]
                        ((halfword << 1) & 0x40); // offset[6] <= [5]
                    return (offset << 20) | ((rs1 + 8) << 15) | (2 << 12) | ((rd + 8) << 7) | 0x3;
                }
                3 => {
                    // @TODO: Support C.FLW in 32-bit mode
                    // C.LD in 64-bit mode
                    // ld rd+8, offset(rs1+8)
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let rd = (halfword >> 2) & 0x7; // [4:2]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
                    return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x3;
                }
                4 => {
                    // Reserved
                }
                5 => {
                    // C.FSD
                    // fsd rs2+8, offset(rs1+8)
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let rs2 = (halfword >> 2) & 0x7; // [4:2]
                    let offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
                    let imm11_5 = (offset >> 5) & 0x7f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | ((rs2 + 8) << 20)
                        | ((rs1 + 8) << 15)
                        | (3 << 12)
                        | (imm4_0 << 7)
                        | 0x27;
                }
                6 => {
                    // C.SW
                    // sw rs2+8, offset(rs1+8)
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let rs2 = (halfword >> 2) & 0x7; // [4:2]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0x40) | // offset[6] <= [5]
                        ((halfword >> 4) & 0x4); // offset[2] <= [6]
                    let imm11_5 = (offset >> 5) & 0x7f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | ((rs2 + 8) << 20)
                        | ((rs1 + 8) << 15)
                        | (2 << 12)
                        | (imm4_0 << 7)
                        | 0x23;
                }
                7 => {
                    // @TODO: Support C.FSW in 32-bit mode
                    // C.SD
                    // sd rs2+8, offset(rs1+8)
                    let rs1 = (halfword >> 7) & 0x7; // [9:7]
                    let rs2 = (halfword >> 2) & 0x7; // [4:2]
                    let offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
                    let imm11_5 = (offset >> 5) & 0x7f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | ((rs2 + 8) << 20)
                        | ((rs1 + 8) << 15)
                        | (3 << 12)
                        | (imm4_0 << 7)
                        | 0x23;
                }
                _ => {} // Not happens
            },
            1 => {
                match funct3 {
                    0 => {
                        // C.ADDI
                        let r = (halfword >> 7) & 0x1f; // [11:7]
                        let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]

                        match (r, imm) {
                            (0, 0) => {
                                // NOP
                                return 0x13;
                            }
                            (0, _) => {
                                // HINT
                                return 0x13;
                            }
                            (_, 0) => {
                                // HINT
                                return 0x13;
                            }
                            (_, _) => {
                                return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                            }
                        }
                    }
                    1 => {
                        match self.xlen {
                            Xlen::Bit32 => {
                                // C.JAL (RV32C only)
                                // jal x1, offset
                                let offset = match halfword & 0x1000 {
                                    0x1000 => 0xfffff000,
                                    _ => 0
                                } | // offset[31:12] <= [12]
                                ((halfword >> 1) & 0x800) | // offset[11] <= [12]
                                ((halfword >> 7) & 0x10) | // offset[4] <= [11]
                                ((halfword >> 1) & 0x300) | // offset[9:8] <= [10:9]
                                ((halfword << 2) & 0x400) | // offset[10] <= [8]
                                ((halfword >> 1) & 0x40) | // offset[6] <= [7]
                                ((halfword << 1) & 0x80) | // offset[7] <= [6]
                                ((halfword >> 2) & 0xe) | // offset[3:1] <= [5:3]
                                ((halfword << 3) & 0x20); // offset[5] <= [2]
                                let imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                                    ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                                    ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                                    ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
                                return (imm << 12) | (1 << 7) | 0x6f;
                            }
                            Xlen::Bit64 => {
                                // C.ADDIW (RV64C only)
                                let r = (halfword >> 7) & 0x1f;
                                let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                                if r == 0 {
                                    // Reserved
                                } else if imm == 0 {
                                    // sext.w rd
                                    return (r << 15) | (r << 7) | 0x1b;
                                } else {
                                    // addiw r, r, imm
                                    return (imm << 20) | (r << 15) | (r << 7) | 0x1b;
                                }
                            }
                        }
                    }
                    2 => {
                        // C.LI
                        let r = (halfword >> 7) & 0x1f;
                        let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                        if r != 0 {
                            // addi rd, x0, imm
                            return (imm << 20) | (r << 7) | 0x13;
                        } else {
                            // HINT
                            return 0x13;
                        }
                    }
                    3 => {
                        let r = (halfword >> 7) & 0x1f; // [11:7]
                        if r == 2 {
                            // C.ADDI16SP
                            // addi r, r, nzimm
                            let imm = match halfword & 0x1000 {
                                0x1000 => 0xfffffc00,
                                _ => 0
                            } | // imm[31:10] <= [12]
                            ((halfword >> 3) & 0x200) | // imm[9] <= [12]
                            ((halfword >> 2) & 0x10) | // imm[4] <= [6]
                            ((halfword << 1) & 0x40) | // imm[6] <= [5]
                            ((halfword << 4) & 0x180) | // imm[8:7] <= [4:3]
                            ((halfword << 3) & 0x20); // imm[5] <= [2]
                            if imm != 0 {
                                return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                            }
                            // imm == 0 is for reserved instruction
                        }
                        if r != 0 && r != 2 {
                            // C.LUI
                            // lui r, nzimm
                            let nzimm = match halfword & 0x1000 {
                                0x1000 => 0xfffc0000,
                                _ => 0
                            } | // nzimm[31:18] <= [12]
                            ((halfword << 5) & 0x20000) | // nzimm[17] <= [12]
                            ((halfword << 10) & 0x1f000); // nzimm[16:12] <= [6:2]
                            if nzimm != 0 {
                                return nzimm | (r << 7) | 0x37;
                            }
                            // nzimm == 0 is for reserved instruction
                        }
                        if r == 0 {
                            // NOP
                            return 0x13;
                        }
                    }
                    4 => {
                        let funct2 = (halfword >> 10) & 0x3; // [11:10]
                        match funct2 {
                            0 => {
                                // C.SRLI
                                // c.srli rs1+8, rs1+8, shamt
                                let shamt = ((halfword >> 7) & 0x20) | // shamt[5] <= [12]
                                    ((halfword >> 2) & 0x1f); // shamt[4:0] <= [6:2]
                                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                                return (shamt << 20)
                                    | ((rs1 + 8) << 15)
                                    | (5 << 12)
                                    | ((rs1 + 8) << 7)
                                    | 0x13;
                            }
                            1 => {
                                // C.SRAI
                                // srai rs1+8, rs1+8, shamt
                                let shamt = ((halfword >> 7) & 0x20) | // shamt[5] <= [12]
                                    ((halfword >> 2) & 0x1f); // shamt[4:0] <= [6:2]
                                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                                return (0x20 << 25)
                                    | (shamt << 20)
                                    | ((rs1 + 8) << 15)
                                    | (5 << 12)
                                    | ((rs1 + 8) << 7)
                                    | 0x13;
                            }
                            2 => {
                                // C.ANDI
                                // andi, r+8, r+8, imm
                                let r = (halfword >> 7) & 0x7; // [9:7]
                                let imm = match halfword & 0x1000 {
                                    0x1000 => 0xffffffc0,
                                    _ => 0
                                } | // imm[31:6] <= [12]
                                ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                                ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                                return (imm << 20)
                                    | ((r + 8) << 15)
                                    | (7 << 12)
                                    | ((r + 8) << 7)
                                    | 0x13;
                            }
                            3 => {
                                let funct1 = (halfword >> 12) & 1; // [12]
                                let funct2_2 = (halfword >> 5) & 0x3; // [6:5]
                                let rs1 = (halfword >> 7) & 0x7;
                                let rs2 = (halfword >> 2) & 0x7;
                                match funct1 {
                                    0 => match funct2_2 {
                                        0 => {
                                            // C.SUB
                                            // sub rs1+8, rs1+8, rs2+8
                                            return (0x20 << 25)
                                                | ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | ((rs1 + 8) << 7)
                                                | 0x33;
                                        }
                                        1 => {
                                            // C.XOR
                                            // xor rs1+8, rs1+8, rs2+8
                                            return ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | (4 << 12)
                                                | ((rs1 + 8) << 7)
                                                | 0x33;
                                        }
                                        2 => {
                                            // C.OR
                                            // or rs1+8, rs1+8, rs2+8
                                            return ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | (6 << 12)
                                                | ((rs1 + 8) << 7)
                                                | 0x33;
                                        }
                                        3 => {
                                            // C.AND
                                            // and rs1+8, rs1+8, rs2+8
                                            return ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | (7 << 12)
                                                | ((rs1 + 8) << 7)
                                                | 0x33;
                                        }
                                        _ => {} // Not happens
                                    },
                                    1 => match funct2_2 {
                                        0 => {
                                            // C.SUBW
                                            // subw r1+8, r1+8, r2+8
                                            return (0x20 << 25)
                                                | ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | ((rs1 + 8) << 7)
                                                | 0x3b;
                                        }
                                        1 => {
                                            // C.ADDW
                                            // addw r1+8, r1+8, r2+8
                                            return ((rs2 + 8) << 20)
                                                | ((rs1 + 8) << 15)
                                                | ((rs1 + 8) << 7)
                                                | 0x3b;
                                        }
                                        2 => {
                                            // Reserved
                                        }
                                        3 => {
                                            // Reserved
                                        }
                                        _ => {} // Not happens
                                    },
                                    _ => {} // No happens
                                };
                            }
                            _ => {} // not happens
                        };
                    }
                    5 => {
                        // C.J
                        // jal x0, imm
                        let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffff000,
                                _ => 0
                            } | // offset[31:12] <= [12]
                            ((halfword >> 1) & 0x800) | // offset[11] <= [12]
                            ((halfword >> 7) & 0x10) | // offset[4] <= [11]
                            ((halfword >> 1) & 0x300) | // offset[9:8] <= [10:9]
                            ((halfword << 2) & 0x400) | // offset[10] <= [8]
                            ((halfword >> 1) & 0x40) | // offset[6] <= [7]
                            ((halfword << 1) & 0x80) | // offset[7] <= [6]
                            ((halfword >> 2) & 0xe) | // offset[3:1] <= [5:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                        let imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                            ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                            ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                            ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
                        return (imm << 12) | 0x6f;
                    }
                    6 => {
                        // C.BEQZ
                        // beq r+8, x0, offset
                        let r = (halfword >> 7) & 0x7;
                        let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffffe00,
                                _ => 0
                            } | // offset[31:9] <= [12]
                            ((halfword >> 4) & 0x100) | // offset[8] <= [12]
                            ((halfword >> 7) & 0x18) | // offset[4:3] <= [11:10]
                            ((halfword << 1) & 0xc0) | // offset[7:6] <= [6:5]
                            ((halfword >> 2) & 0x6) | // offset[2:1] <= [4:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                        let imm2 = ((offset >> 6) & 0x40) | // imm2[6] <= [12]
                            ((offset >> 5) & 0x3f); // imm2[5:0] <= [10:5]
                        let imm1 = (offset & 0x1e) | // imm1[4:1] <= [4:1]
                            ((offset >> 11) & 0x1); // imm1[0] <= [11]
                        return (imm2 << 25) | ((r + 8) << 20) | (imm1 << 7) | 0x63;
                    }
                    7 => {
                        // C.BNEZ
                        // bne r+8, x0, offset
                        let r = (halfword >> 7) & 0x7;
                        let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffffe00,
                                _ => 0
                            } | // offset[31:9] <= [12]
                            ((halfword >> 4) & 0x100) | // offset[8] <= [12]
                            ((halfword >> 7) & 0x18) | // offset[4:3] <= [11:10]
                            ((halfword << 1) & 0xc0) | // offset[7:6] <= [6:5]
                            ((halfword >> 2) & 0x6) | // offset[2:1] <= [4:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                        let imm2 = ((offset >> 6) & 0x40) | // imm2[6] <= [12]
                            ((offset >> 5) & 0x3f); // imm2[5:0] <= [10:5]
                        let imm1 = (offset & 0x1e) | // imm1[4:1] <= [4:1]
                            ((offset >> 11) & 0x1); // imm1[0] <= [11]
                        return (imm2 << 25) | ((r + 8) << 20) | (1 << 12) | (imm1 << 7) | 0x63;
                    }
                    _ => {} // No happens
                };
            }
            2 => {
                match funct3 {
                    0 => {
                        // C.SLLI
                        // slli r, r, shamt
                        let r = (halfword >> 7) & 0x1f;
                        let shamt = ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                            ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                        if r != 0 {
                            return (shamt << 20) | (r << 15) | (1 << 12) | (r << 7) | 0x13;
                        }
                        // r == 0 is reserved instruction?
                    }
                    1 => {
                        // C.FLDSP
                        // fld rd, offset(x2)
                        let rd = (halfword >> 7) & 0x1f;
                        let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                            ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
                        if rd != 0 {
                            return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x7;
                        }
                        // rd == 0 is reserved instruction
                    }
                    2 => {
                        // C.LWSP
                        // lw r, offset(x2)
                        let r = (halfword >> 7) & 0x1f;
                        let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x1c) | // offset[4:2] <= [6:4]
                            ((halfword << 4) & 0xc0); // offset[7:6] <= [3:2]
                        if r != 0 {
                            return (offset << 20) | (2 << 15) | (2 << 12) | (r << 7) | 0x3;
                        }
                        // r == 0 is reserved instruction
                    }
                    3 => {
                        // @TODO: Support C.FLWSP in 32-bit mode
                        // C.LDSP
                        // ld rd, offset(x2)
                        let rd = (halfword >> 7) & 0x1f;
                        let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                            ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
                        if rd != 0 {
                            return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x3;
                        }
                        // rd == 0 is reserved instruction
                    }
                    4 => {
                        let funct1 = (halfword >> 12) & 1; // [12]
                        let rs1 = (halfword >> 7) & 0x1f; // [11:7]
                        let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                        match funct1 {
                            0 => {
                                // C.MV
                                match (rs1, rs2) {
                                    (0, 0) => {
                                        // Reserved
                                    }
                                    (r, 0) if r != 0 => {
                                        // C.JR: jalr x0, 0(rs1)
                                        return (rs1 << 15) | 0x67;
                                    }
                                    (0, r2) if r2 != 0 => {
                                        // HINT
                                        return 0x13;
                                    }
                                    (rd, rs2) => {
                                        // add rd, x0, rs2
                                        return (rs2 << 20) | (rd << 7) | 0x33;
                                    }
                                }
                            }
                            1 => {
                                // C.ADD
                                match (rs1, rs2) {
                                    (0, 0) => {
                                        // C.EBREAK
                                        // ebreak
                                        return 0x00100073;
                                    }
                                    (rs1, 0) if rs1 != 0 => {
                                        // C.JALR
                                        // jalr x1, 0(rs1)
                                        return (rs1 << 15) | (1 << 7) | 0x67;
                                    }
                                    (0, rs2) if rs2 != 0 => {
                                        // HINT
                                        return 0x13;
                                    }
                                    (rs1, rs2) => {
                                        // C.ADD
                                        // add rs1, rs1, rs2
                                        return (rs2 << 20) | (rs1 << 15) | (rs1 << 7) | 0x33;
                                    }
                                }
                            }
                            _ => {} // Not happens
                        };
                    }
                    5 => {
                        // @TODO: Implement
                        // C.FSDSP
                        // fsd rs2, offset(x2)
                        let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                        let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                            ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
                        let imm11_5 = (offset >> 5) & 0x3f;
                        let imm4_0 = offset & 0x1f;
                        return (imm11_5 << 25)
                            | (rs2 << 20)
                            | (2 << 15)
                            | (3 << 12)
                            | (imm4_0 << 7)
                            | 0x27;
                    }
                    6 => {
                        // C.SWSP
                        // sw rs2, offset(x2)
                        let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                        let offset = ((halfword >> 7) & 0x3c) | // offset[5:2] <= [12:9]
                            ((halfword >> 1) & 0xc0); // offset[7:6] <= [8:7]
                        let imm11_5 = (offset >> 5) & 0x3f;
                        let imm4_0 = offset & 0x1f;
                        return (imm11_5 << 25)
                            | (rs2 << 20)
                            | (2 << 15)
                            | (2 << 12)
                            | (imm4_0 << 7)
                            | 0x23;
                    }
                    7 => {
                        // @TODO: Support C.FSWSP in 32-bit mode
                        // C.SDSP
                        // sd rs, offset(x2)
                        let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                        let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                            ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
                        let imm11_5 = (offset >> 5) & 0x3f;
                        let imm4_0 = offset & 0x1f;
                        return (imm11_5 << 25)
                            | (rs2 << 20)
                            | (2 << 15)
                            | (3 << 12)
                            | (imm4_0 << 7)
                            | 0x23;
                    }
                    _ => {} // Not happens
                };
            }
            _ => {} // Not happens
        };
        0xffffffff // Return invalid value
    }

    /// Disassembles an instruction pointed by Program Counter.
    pub fn disassemble_next_instruction(&mut self) -> String {
        // @TODO: Fetching can make a side effect,
        // for example updating page table entry or update peripheral hardware registers.
        // But ideally disassembling doesn't want to cause any side effect.
        // How can we avoid side effect?
        let mut original_word = match self.mmu.fetch_word(self.pc) {
            Ok(data) => data,
            Err(_e) => {
                return format!("PC:{:016x}, InstructionPageFault Trap!\n", self.pc);
            }
        };

        let word = match (original_word & 0x3) == 0x3 {
            true => original_word,
            false => {
                original_word &= 0xffff;
                self.uncompress(original_word)
            }
        };

        let inst = match RV32IMInstruction::decode(word, self.pc) {
            Ok(inst) => inst,
            Err(e) => {
                return format!(
                    "Unknown instruction PC:{:x} WORD:{:x}, {:?}",
                    self.pc, original_word, e
                );
            }
        };

        let name: &'static str = inst.into();
        let mut s = format!("PC:{:016x} ", self.unsigned_data(self.pc as i64));
        s += &format!("{original_word:08x} ");
        s += &format!("{name}");
        // s += &format!("{}", (inst.disassemble)(self, word, self.pc, true));
        s
    }

    /// Returns mutable `Mmu`
    pub fn get_mut_mmu(&mut self) -> &mut Mmu {
        &mut self.mmu
    }

    fn handle_jolt_cycle_marker(&mut self, ptr: u32, event: u32) -> Result<(), Trap> {
        match event {
            JOLT_CYCLE_MARKER_START => {
                let label = self.read_c_string(ptr)?; // guest NUL-string

                // Check if there's already an active marker with the same label
                let duplicate = self
                    .active_markers
                    .values()
                    .any(|marker| marker.label == label);
                if duplicate {
                    println!("Warning: Marker with label '{}' is already active", &label);
                }

                self.active_markers.insert(
                    ptr,
                    ActiveMarker {
                        label,
                        start_instrs: self.executed_instrs,
                        start_trace_len: self.trace_len,
                    },
                );
            }

            JOLT_CYCLE_MARKER_END => {
                if let Some(mark) = self.active_markers.remove(&ptr) {
                    let real = self.executed_instrs - mark.start_instrs;
                    let virt = self.trace_len - mark.start_trace_len;
                    println!(
                        "\"{}\": {} RV32IM cycles, {} virtual cycles",
                        mark.label, real, virt
                    );
                } else {
                    println!(
                        "Warning: Attempt to end a marker (ptr: 0x{ptr:x}) that was never started"
                    );
                }
            }
            _ => {
                panic!("Unexpected event: event must match either start or end marker.")
            }
        }
        Ok(())
    }

    /// Read a NUL-terminated guest string from memory.
    fn read_c_string(&mut self, mut addr: u32) -> Result<String, Trap> {
        let mut bytes = Vec::new();
        loop {
            let (b, _) = self.mmu.load(addr.into())?;
            if b == 0 {
                break;
            }
            bytes.push(b);
            addr += 1;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        if !self.active_markers.is_empty() {
            println!(
                "Warning: Found {} unclosed cycle tracking marker(s):",
                self.active_markers.len()
            );
            for (ptr, marker) in &self.active_markers {
                println!(
                    "  - '{}' (at ptr: 0x{:x}), started at {} RV32IM cycles",
                    marker.label, ptr, marker.start_instrs
                );
            }
        }
    }
}

fn get_register_name(num: usize) -> &'static str {
    match num {
        0 => "zero",
        1 => "ra",
        2 => "sp",
        3 => "gp",
        4 => "tp",
        5 => "t0",
        6 => "t1",
        7 => "t2",
        8 => "s0",
        9 => "s1",
        10 => "a0",
        11 => "a1",
        12 => "a2",
        13 => "a3",
        14 => "a4",
        15 => "a5",
        16 => "a6",
        17 => "a7",
        18 => "s2",
        19 => "s3",
        20 => "s4",
        21 => "s5",
        22 => "s6",
        23 => "s7",
        24 => "s8",
        25 => "s9",
        26 => "s10",
        27 => "s11",
        28 => "t3",
        29 => "t4",
        30 => "t5",
        31 => "t6",
        _ => panic!("Unknown register num {num}"),
    }
}

fn normalize_u64(value: u64, width: &Xlen) -> u64 {
    match width {
        Xlen::Bit32 => value as u32 as u64,
        Xlen::Bit64 => value,
    }
}

fn normalize_register(value: usize) -> u64 {
    value.try_into().unwrap()
}

#[cfg(test)]
mod test_cpu {
    use super::*;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;

    fn create_cpu() -> Cpu {
        Cpu::new(Box::new(DummyTerminal::new()))
    }

    #[test]
    fn initialize() {
        let _cpu = create_cpu();
    }

    #[test]
    fn update_pc() {
        let mut cpu = create_cpu();
        assert_eq!(0, cpu.read_pc());
        cpu.update_pc(1);
        assert_eq!(1, cpu.read_pc());
        cpu.update_pc(0xffffffffffffffff);
        assert_eq!(0xffffffffffffffff, cpu.read_pc());
    }

    #[test]
    fn update_xlen() {
        let mut cpu = create_cpu();
        assert!(matches!(cpu.xlen, Xlen::Bit64));
        cpu.update_xlen(Xlen::Bit32);
        assert!(matches!(cpu.xlen, Xlen::Bit32));
        cpu.update_xlen(Xlen::Bit64);
        assert!(matches!(cpu.xlen, Xlen::Bit64));
        // Note: cpu.update_xlen() updates cpu.mmu.xlen, too.
        // The test for mmu.xlen should be in Mmu?
    }

    #[test]
    fn read_register() {
        let mut cpu = create_cpu();
        // Initial register values are 0 other than 0xb th register.
        // Initial value of 0xb th register is temporal for Linux boot and
        // I'm not sure if the value is correct. Then skipping so far.
        for i in 0..31 {
            if i != 0xb {
                assert_eq!(0, cpu.read_register(i));
            }
        }

        for i in 0..31 {
            cpu.x[i] = i as i64 + 1;
        }

        for i in 0..31 {
            match i {
                // 0th register is hardwired zero
                0 => assert_eq!(0, cpu.read_register(i)),
                _ => assert_eq!(i as i64 + 1, cpu.read_register(i)),
            }
        }

        for i in 0..31 {
            cpu.x[i] = (0xffffffffffffffff - i) as i64;
        }

        for i in 0..31 {
            match i {
                // 0th register is hardwired zero
                0 => assert_eq!(0, cpu.read_register(i)),
                _ => assert_eq!(-(i as i64 + 1), cpu.read_register(i)),
            }
        }

        // @TODO: Should I test the case where the argument equals to or is
        // greater than 32?
    }

    #[test]
    fn tick() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        cpu.update_pc(DRAM_BASE);

        // Write non-compressed "addi x1, x1, 1" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00108093) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        // Write compressed "addi x8, x0, 8" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE + 4, 0x20) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };

        cpu.tick(None);

        assert_eq!(DRAM_BASE + 4, cpu.read_pc());
        assert_eq!(1, cpu.read_register(1));

        cpu.tick(None);

        assert_eq!(DRAM_BASE + 6, cpu.read_pc());
        assert_eq!(8, cpu.read_register(8));
    }

    #[test]
    fn tick_operate() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        cpu.update_pc(DRAM_BASE);
        // write non-compressed "addi a0, a0, 12" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0xc50513) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        assert_eq!(DRAM_BASE, cpu.read_pc());
        assert_eq!(0, cpu.read_register(10));
        match cpu.tick_operate(None) {
            Ok(_) => {}
            Err(_e) => panic!("tick_operate() unexpectedly did panic"),
        };
        // .tick_operate() increments the program counter by 4 for
        // non-compressed instruction.
        assert_eq!(DRAM_BASE + 4, cpu.read_pc());
        // "addi a0, a0, a12" instruction writes 12 to a0 register.
        assert_eq!(12, cpu.read_register(10));
        // @TODO: Test compressed instruction operation
    }

    #[test]
    fn fetch() {
        // .fetch() reads four bytes from the memory
        // at the address the program counter points to.
        // .fetch() doesn't increment the program counter.
        // .tick_operate() does.
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        cpu.update_pc(DRAM_BASE);
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0xaaaaaaaa) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        match cpu.fetch() {
            Ok(data) => assert_eq!(0xaaaaaaaa, data),
            Err(_e) => panic!("Failed to fetch"),
        };
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x55555555) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        match cpu.fetch() {
            Ok(data) => assert_eq!(0x55555555, data),
            Err(_e) => panic!("Failed to fetch"),
        };
        // @TODO: Write test cases where Trap happens
    }

    // #[test]
    // fn decode() {
    //     let mut cpu = create_cpu();
    //     // 0x13 is addi instruction
    //     match cpu.decode(0x13) {
    //         Ok(inst) => assert_eq!(inst.name, "ADDI"),
    //         Err(_e) => panic!("Failed to decode"),
    //     };
    //     // .decode() returns error for invalid word data.
    //     match cpu.decode(0x0) {
    //         Ok(_inst) => panic!("Unexpectedly succeeded in decoding"),
    //         Err(()) => assert!(true),
    //     };
    //     // @TODO: Should I test all instructions?
    // }

    // #[test]
    // fn uncompress() {
    //     let mut cpu = create_cpu();
    //     // .uncompress() doesn't directly return an instruction but
    //     // it returns uncompressed word. Then you need to call .decode().
    //     match cpu.decode(cpu.uncompress(0x20)) {
    //         Ok(inst) => assert_eq!(inst.name, "ADDI"),
    //         Err(_e) => panic!("Failed to decode"),
    //     };
    //     // @TODO: Should I test all compressed instructions?
    // }

    // #[test]
    // fn wfi() {
    //     let wfi_instruction = 0x10500073;
    //     let mut cpu = create_cpu();
    //     // Just in case
    //     match cpu.decode(wfi_instruction) {
    //         Ok(inst) => assert_eq!(inst.name, "WFI"),
    //         Err(_e) => panic!("Failed to decode"),
    //     };
    //     cpu.get_mut_mmu().init_memory(4);
    //     cpu.update_pc(DRAM_BASE);
    //     // write WFI instruction
    //     match cpu.get_mut_mmu().store_word(DRAM_BASE, wfi_instruction) {
    //         Ok(_) => {}
    //         Err(_e) => panic!("Failed to store"),
    //     };
    //     cpu.tick();
    //     assert_eq!(DRAM_BASE + 4, cpu.read_pc());
    //     for _i in 0..10 {
    //         // Until interrupt happens, .tick() does nothing
    //         // @TODO: Check accurately that the state is unchanged
    //         cpu.tick();
    //         assert_eq!(DRAM_BASE + 4, cpu.read_pc());
    //     }
    //     // Machine timer interrupt
    //     cpu.write_csr_raw(CSR_MIE_ADDRESS, MIP_MTIP);
    //     cpu.write_csr_raw(CSR_MIP_ADDRESS, MIP_MTIP);
    //     cpu.write_csr_raw(CSR_MSTATUS_ADDRESS, 0x8);
    //     cpu.write_csr_raw(CSR_MTVEC_ADDRESS, 0x0);
    //     cpu.tick();
    //     // Interrupt happened and moved to handler
    //     assert_eq!(0, cpu.read_pc());
    // }

    #[test]
    fn interrupt() {
        let handler_vector = 0x10000000;
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        // Write non-compressed "addi x0, x0, 1" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00100013) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        cpu.update_pc(DRAM_BASE);

        // Machine timer interrupt but mie in mstatus is not enabled yet
        cpu.write_csr_raw(CSR_MIE_ADDRESS, MIP_MTIP);
        cpu.write_csr_raw(CSR_MIP_ADDRESS, MIP_MTIP);
        cpu.write_csr_raw(CSR_MTVEC_ADDRESS, handler_vector);

        cpu.tick(None);

        // Interrupt isn't caught because mie is disabled
        assert_eq!(DRAM_BASE + 4, cpu.read_pc());

        cpu.update_pc(DRAM_BASE);
        // Enable mie in mstatus
        cpu.write_csr_raw(CSR_MSTATUS_ADDRESS, 0x8);

        cpu.tick(None);

        // Interrupt happened and moved to handler
        assert_eq!(handler_vector, cpu.read_pc());

        // CSR Cause register holds the reason what caused the interrupt
        assert_eq!(0x8000000000000007, cpu.read_csr_raw(CSR_MCAUSE_ADDRESS));

        // @TODO: Test post CSR status register
        // @TODO: Test xIE bit in CSR status register
        // @TODO: Test privilege levels
        // @TODO: Test delegation
        // @TODO: Test vector type handlers
    }

    #[test]
    fn exception() {
        let handler_vector = 0x10000000;
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        // Write ECALL instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00000073) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        cpu.write_csr_raw(CSR_MTVEC_ADDRESS, handler_vector);
        cpu.update_pc(DRAM_BASE);

        cpu.tick(None);

        // Interrupt happened and moved to handler
        assert_eq!(handler_vector, cpu.read_pc());

        // CSR Cause register holds the reason what caused the trap
        assert_eq!(0xb, cpu.read_csr_raw(CSR_MCAUSE_ADDRESS));

        // @TODO: Test post CSR status register
        // @TODO: Test privilege levels
        // @TODO: Test delegation
        // @TODO: Test vector type handlers
    }

    #[test]
    fn hardocded_zero() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(8);
        cpu.update_pc(DRAM_BASE);

        // Write non-compressed "addi x0, x0, 1" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00100013) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        // Write non-compressed "addi x1, x1, 1" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE + 4, 0x00108093) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };

        // Test x0
        assert_eq!(0, cpu.read_register(0));
        cpu.tick(None); // Execute  "addi x0, x0, 1"
                        // x0 is still zero because it's hardcoded zero
        assert_eq!(0, cpu.read_register(0));

        // Test x1
        assert_eq!(0, cpu.read_register(1));
        cpu.tick(None); // Execute  "addi x1, x1, 1"
                        // x1 is not hardcoded zero
        assert_eq!(1, cpu.read_register(1));
    }

    #[test]
    fn disassemble_next_instruction() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        cpu.update_pc(DRAM_BASE);

        // Write non-compressed "addi x0, x0, 1" instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00100013) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };

        assert_eq!(
            "PC:0000000080000000 00100013 ADDI zero:0,zero:0,1",
            cpu.disassemble_next_instruction()
        );

        // No effect to PC
        assert_eq!(DRAM_BASE, cpu.read_pc());
    }
}
