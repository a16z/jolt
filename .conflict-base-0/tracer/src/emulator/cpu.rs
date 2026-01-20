#[cfg(feature = "std")]
extern crate fnv;

#[cfg(feature = "std")]
use self::fnv::FnvHashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::btree_map::BTreeMap as FnvHashMap;
use common::constants::REGISTER_COUNT;
use tracing::{info, warn};

use crate::instruction::{uncompress_instruction, Cycle, Instruction};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::mmu::{AddressingMode, Mmu};
use super::terminal::Terminal;

use crate::instruction::format::NormalizedOperands;
use crate::utils::panic::CallFrame;
#[cfg(not(feature = "std"))]
use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};
use jolt_platform::{
    JOLT_CYCLE_MARKER_END, JOLT_CYCLE_MARKER_START, JOLT_CYCLE_TRACK_ECALL_NUM,
    JOLT_PRINT_ECALL_NUM, JOLT_PRINT_LINE, JOLT_PRINT_STRING,
};
#[cfg(feature = "std")]
use std::collections::VecDeque;

const CSR_CAPACITY: usize = 4096;
const MAX_CALL_STACK_DEPTH: usize = 32;

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
#[allow(dead_code)]
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

#[derive(Clone, Debug)]
struct ActiveMarker {
    label: String,
    start_instrs: u64,      // executed_instrs  at ‘start’
    start_trace_len: usize, // trace.len()      at ‘start’
}

/// Emulates a RISC-V CPU core
#[derive(Clone, Debug)]
pub struct Cpu {
    clock: u64,
    pub(crate) xlen: Xlen,
    pub(crate) privilege_mode: PrivilegeMode,
    wfi: bool,
    // using only lower 32bits of x, pc, and csr registers
    // for 32-bit mode
    pub x: [i64; REGISTER_COUNT as usize],
    #[allow(dead_code)]
    f: [f64; 32],
    pub(crate) pc: u64,
    csr: [u64; CSR_CAPACITY],
    pub mmu: Mmu,
    reservation: u64, // @TODO: Should support multiple address reservations
    is_reservation_set: bool,
    _dump_flag: bool,
    unsigned_data_mask: u64,
    // pub trace: Vec<Cycle>,
    pub trace_len: usize,
    executed_instrs: u64, // “real” RV64IMAC cycles
    active_markers: FnvHashMap<u32, ActiveMarker>,
    pub vr_allocator: VirtualRegisterAllocator,
    /// Call stack tracking (circular buffer)
    call_stack: VecDeque<CallFrame>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Xlen {
    Bit32,
    Bit64, // @TODO: Support Bit128
}

#[derive(Clone, Debug, Copy)]
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
        let mut cpu = Self {
            clock: 0,
            xlen: Xlen::Bit64,
            privilege_mode: PrivilegeMode::Machine,
            wfi: false,
            x: [0; REGISTER_COUNT as usize],
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
            vr_allocator: VirtualRegisterAllocator::new(),
            call_stack: VecDeque::with_capacity(MAX_CALL_STACK_DEPTH),
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
        self.xlen = xlen;
        self.unsigned_data_mask = match xlen {
            Xlen::Bit32 => 0xffffffff,
            Xlen::Bit64 => 0xffffffffffffffff,
        };
        self.mmu.update_xlen(xlen);
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
    pub fn tick(&mut self, trace: Option<&mut Vec<Cycle>>) {
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
    fn tick_operate(&mut self, trace: Option<&mut Vec<Cycle>>) -> Result<(), Trap> {
        if self.wfi {
            if (self.read_csr_raw(CSR_MIE_ADDRESS) & self.read_csr_raw(CSR_MIP_ADDRESS)) != 0 {
                self.wfi = false;
            }
            return Ok(());
        }

        let original_word = self.fetch()?;
        let instruction_address = normalize_u64(self.pc, &self.xlen);
        let is_compressed = (original_word & 0x3) != 0x3;
        let word = match is_compressed {
            false => {
                self.pc = self.pc.wrapping_add(4); // 32-bit length non-compressed instruction
                original_word
            }
            true => {
                self.pc = self.pc.wrapping_add(2); // 16-bit length compressed instruction
                uncompress_instruction(original_word & 0xffff, self.xlen)
            }
        };

        let instr = Instruction::decode(word, instruction_address, is_compressed)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to decode instruction: word=0x{word:08x}, address=0x{instruction_address:x}, compressed={is_compressed}: {e}"
                )
            });

        if trace.is_none() {
            instr.execute(self);
            self.trace_len += 1;
        } else {
            instr.trace(self, trace);
            self.trace_len += instr.inline_sequence(&self.vr_allocator, self.xlen).len();
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
                let marker_len = self.x[12] as u32; // a2
                let event_type = self.x[13] as u32; // a3

                // Read / update the per-label counters.
                //
                // Any fault raised while touching guest memory (e.g. a bad
                // string pointer) is swallowed here and will manifest as the
                // usual access-fault on the *next* instruction fetch.
                let _ = self.handle_jolt_cycle_marker(marker_ptr, marker_len, event_type);

                return false; // we don't take the trap
            } else if call_id == JOLT_PRINT_ECALL_NUM {
                let string_ptr = self.x[11] as u32; // a0
                let string_len = self.x[12] as u32; // a1
                let event_type = self.x[13] as u32; // a2

                // Any fault raised while touching guest memory (e.g. a bad
                // string pointer) is swallowed here and will manifest as the
                // usual access-fault on the *next* instruction fetch.
                let _ = self.handle_jolt_print(string_ptr, string_len, event_type as u8);

                return false;
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
        self.mmu.update_privilege_mode(self.privilege_mode);
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

    #[allow(dead_code)]
    fn has_csr_access_privilege(&self, address: u16) -> bool {
        let privilege = (address >> 8) & 0x3; // the lowest privilege level that can access the CSR
        privilege as u8 <= get_privilege_encoding(&self.privilege_mode)
    }

    #[allow(dead_code)]
    fn read_csr(&mut self, address: u16) -> Result<u64, Trap> {
        match self.has_csr_access_privilege(address) {
            true => Ok(self.read_csr_raw(address)),
            false => Err(Trap {
                trap_type: TrapType::IllegalInstruction,
                value: self.pc.wrapping_sub(4), // @TODO: Is this always correct?
            }),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
                    tracing::error!("Unknown addressing_mode {:x}", value >> 60);
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

        let is_compressed = (original_word & 0x3) != 0x3;
        let word = match is_compressed {
            false => original_word,
            true => {
                original_word &= 0xffff;
                uncompress_instruction(original_word, self.xlen)
            }
        };

        let inst = match Instruction::decode(word, self.pc, is_compressed) {
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
        s += name;
        // s += &format!("{}", (inst.disassemble)(self, word, self.pc, true));
        s
    }

    /// Returns mutable `Mmu`
    pub fn get_mut_mmu(&mut self) -> &mut Mmu {
        &mut self.mmu
    }

    fn handle_jolt_cycle_marker(&mut self, ptr: u32, len: u32, event: u32) -> Result<(), Trap> {
        match event {
            JOLT_CYCLE_MARKER_START => {
                let label = self.read_string(ptr, len)?; // guest NUL-string

                // Check if there's already an active marker with the same label
                let duplicate = self
                    .active_markers
                    .values()
                    .any(|marker| marker.label == label);
                if duplicate {
                    warn!("Marker with label '{}' is already active", &label);
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
                    info!(
                        "\"{}\": {} RV64IMAC cycles, {} virtual cycles",
                        mark.label, real, virt
                    );
                } else {
                    warn!("Attempt to end a marker (ptr: 0x{ptr:x}) that was never started");
                }
            }
            _ => {
                panic!("Unexpected event: event must match either start or end marker.")
            }
        }
        Ok(())
    }

    fn handle_jolt_print(&mut self, ptr: u32, len: u32, event_type: u8) -> Result<(), Trap> {
        let message = self.read_string(ptr, len)?;
        if event_type == JOLT_PRINT_STRING as u8 {
            print!("{message}");
        } else if event_type == JOLT_PRINT_LINE as u8 {
            println!("{message}");
        } else {
            panic!("Unexpected event type: {event_type}");
        }
        Ok(())
    }

    /// Read a NUL-terminated guest string from memory.
    fn read_string(&mut self, mut addr: u32, len: u32) -> Result<String, Trap> {
        let mut bytes = Vec::with_capacity(len as usize);
        for _ in 0..len {
            let (b, _) = self.mmu.load(addr.into())?;
            bytes.push(b);
            addr += 1;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Track a function call (JAL/JALR instruction that saves callsite information)
    /// Optimized for minimal overhead - just append to a circular buffer (VecDeque)
    #[inline]
    pub fn track_call(&mut self, return_address: u64, operands: NormalizedOperands) {
        // Simple circular buffer - if full, overwrite oldest
        if self.call_stack.len() >= MAX_CALL_STACK_DEPTH {
            self.call_stack.pop_front();
        }

        self.call_stack.push_back(CallFrame {
            call_site: return_address,
            x: self.x,
            operands,
            cycle_count: self.trace_len,
        });
    }

    /// Get the current call stack (for displaying on panic)
    pub fn get_call_stack(&self) -> &VecDeque<CallFrame> {
        &self.call_stack
    }
}

impl Cpu {
    pub fn save_state_with_empty_memory(&self) -> Cpu {
        Cpu {
            clock: self.clock,
            xlen: self.xlen,
            privilege_mode: self.privilege_mode,
            wfi: self.wfi,
            x: self.x,
            f: self.f,
            pc: self.pc,
            csr: self.csr,
            mmu: self.mmu.save_state_with_empty_memory(),
            reservation: self.reservation,
            is_reservation_set: self.is_reservation_set,
            _dump_flag: self._dump_flag,
            unsigned_data_mask: self.unsigned_data_mask,
            trace_len: self.trace_len,
            executed_instrs: self.executed_instrs,
            active_markers: self.active_markers.clone(),
            vr_allocator: self.vr_allocator.clone(),
            call_stack: self.call_stack.clone(),
        }
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        if !self.active_markers.is_empty() {
            warn!(
                "Warning: Found {} unclosed cycle tracking marker(s):",
                self.active_markers.len()
            );
            for (ptr, marker) in &self.active_markers {
                warn!(
                    "  - '{}' (at ptr: 0x{:x}), started at {} RV64IMAC cycles",
                    marker.label, ptr, marker.start_instrs
                );
            }
        }
    }
}

#[allow(dead_code)]
pub fn get_register_name(num: usize) -> &'static str {
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

#[cfg(test)]
mod test_cpu {
    use super::*;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;

    fn create_cpu() -> Cpu {
        Cpu::new(Box::new(DummyTerminal::default()))
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
        cpu.get_mut_mmu().init_memory(9);
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
    fn hardcoded_zero() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(9);
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
            "PC:0000000080000000 00100013 ADDI",
            cpu.disassemble_next_instruction()
        );

        // No effect to PC
        assert_eq!(DRAM_BASE, cpu.read_pc());
    }
}
