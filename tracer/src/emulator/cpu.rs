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

use super::mmu::Mmu;
use super::terminal::Terminal;

/// A FIFO queue for storing and retrieving advice data between emulation passes.
/// During the first emulation pass (with `compute_advice` feature), advice functions
/// write serialized data to this tape. During the second pass (without the feature),
/// advice functions read from this tape in the same order.
#[derive(Clone, Debug, Default)]
pub struct AdviceTape {
    data: Vec<u8>,
    read_position: usize,
}

impl AdviceTape {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a tape from raw bytes with the read cursor at 0.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            data,
            read_position: 0,
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Append bytes to the advice tape (called during first emulation pass)
    pub fn write(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    /// Read a specific number of bytes from the advice tape (called during second emulation pass)
    pub fn read(&mut self, num_bytes: usize) -> Option<u64> {
        if self.read_position + num_bytes > self.data.len() {
            return None;
        }

        let mut result = 0u64;
        for i in 0..num_bytes {
            result |= (self.data[self.read_position + i] as u64) << (i * 8);
        }
        self.read_position += num_bytes;
        Some(result)
    }

    pub fn reset_read_position(&mut self) {
        self.read_position = 0;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of bytes remaining to be read
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.read_position)
    }
}

pub fn advice_tape_write(cpu: &mut Cpu, bytes: &[u8]) {
    cpu.advice_tape.write(bytes);
}

pub fn advice_tape_read(cpu: &mut Cpu, num_bytes: usize) -> Option<u64> {
    cpu.advice_tape.read(num_bytes)
}

/// Get the number of bytes remaining to be read from the CPU's advice tape
pub fn advice_tape_remaining(cpu: &Cpu) -> usize {
    cpu.advice_tape.remaining()
}

use crate::utils::panic::CallFrame;
#[cfg(not(feature = "std"))]
use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, rc::Rc, string::String, vec::Vec};
use jolt_platform::{
    JOLT_CYCLE_MARKER_END, JOLT_CYCLE_MARKER_START, JOLT_PRINT_LINE, JOLT_PRINT_STRING,
};
#[cfg(feature = "field-inline")]
use jolt_program::field_inline::FieldEncodedValue;
#[cfg(feature = "std")]
use std::collections::VecDeque;

const CSR_CAPACITY: usize = 4096;
const MAX_CALL_STACK_DEPTH: usize = 32;

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug)]
pub struct FieldRegisterFile {
    registers: [FieldEncodedValue; jolt_riscv::FIELD_REGISTER_COUNT as usize],
}

#[cfg(feature = "field-inline")]
impl Default for FieldRegisterFile {
    fn default() -> Self {
        Self {
            registers: [FieldEncodedValue::zero(); jolt_riscv::FIELD_REGISTER_COUNT as usize],
        }
    }
}

#[cfg(feature = "field-inline")]
impl FieldRegisterFile {
    pub fn read(&self, register: u8) -> FieldEncodedValue {
        Self::check_register(register);
        self.registers[register as usize]
    }

    pub fn write(&mut self, register: u8, value: FieldEncodedValue) {
        Self::check_register(register);
        self.registers[register as usize] = value;
    }

    /// The 5-bit instruction encoding admits register indices the field register
    /// file does not have. Preprocessing rejects such rows at metadata construction;
    /// trapping here keeps the emulator and the proving pipeline in agreement instead
    /// of silently reading zero / dropping writes for an out-of-range index.
    fn check_register(register: u8) {
        assert!(
            (register as usize) < jolt_riscv::FIELD_REGISTER_COUNT as usize,
            "field register index {register} is out of range (count {})",
            jolt_riscv::FIELD_REGISTER_COUNT
        );
    }
}

const CSR_USTATUS_ADDRESS: u16 = 0x000;
const CSR_FFLAGS_ADDRESS: u16 = 0x001;
const CSR_FRM_ADDRESS: u16 = 0x002;
const CSR_FCSR_ADDRESS: u16 = 0x003;
const CSR_UIE_ADDRESS: u16 = 0x004;
const CSR_UTVEC_ADDRESS: u16 = 0x005;
const CSR_UEPC_ADDRESS: u16 = 0x041;
const CSR_UCAUSE_ADDRESS: u16 = 0x042;
const CSR_UTVAL_ADDRESS: u16 = 0x043;
const CSR_SSTATUS_ADDRESS: u16 = 0x100;
const CSR_SEDELEG_ADDRESS: u16 = 0x102;
const CSR_SIDELEG_ADDRESS: u16 = 0x103;
const CSR_SIE_ADDRESS: u16 = 0x104;
const CSR_STVEC_ADDRESS: u16 = 0x105;
const CSR_SEPC_ADDRESS: u16 = 0x141;
const CSR_SCAUSE_ADDRESS: u16 = 0x142;
const CSR_STVAL_ADDRESS: u16 = 0x143;
const CSR_SIP_ADDRESS: u16 = 0x144;
const CSR_MSTATUS_ADDRESS: u16 = 0x300;
const CSR_MISA_ADDRESS: u16 = 0x301;
const CSR_MEDELEG_ADDRESS: u16 = 0x302;
const CSR_MIDELEG_ADDRESS: u16 = 0x303;
const CSR_MIE_ADDRESS: u16 = 0x304;

const CSR_MTVEC_ADDRESS: u16 = 0x305;
const CSR_MEPC_ADDRESS: u16 = 0x341;
const CSR_MCAUSE_ADDRESS: u16 = 0x342;
const CSR_MTVAL_ADDRESS: u16 = 0x343;
const CSR_MIP_ADDRESS: u16 = 0x344;
const CSR_TIME_ADDRESS: u16 = 0xc01;

const MIP_MEIP: u64 = 0x800;
pub const MIP_MTIP: u64 = 0x080;
pub const MIP_MSIP: u64 = 0x008;
pub const MIP_SEIP: u64 = 0x200;
const MIP_STIP: u64 = 0x020;
const MIP_SSIP: u64 = 0x002;

#[derive(Clone, Debug)]
struct ActiveMarker {
    start_instrs: u64,
    start_trace_len: usize,
}

/// Host-side I/O mode. `Replay` suppresses effects that must happen exactly
/// once per program run — stdout prints, cycle-marker bookkeeping,
/// advice-tape appends, call-stack tracking — so a parallel-trace worker
/// re-executing a chunk does not repeat what pass-1 already did.
/// Guest-visible effects (JoltDevice loads/stores, advice reads) stay live
/// in both modes: trace rows depend on them.
///
/// Load-bearing assumption: a guest never appends to the advice tape and
/// reads those bytes back within the same run. The SDK's two-pass advice
/// design guarantees this (writes happen in `compute_advice` execute passes,
/// reads in trace passes); a same-chunk append-then-read would replay
/// wrongly under suppression, and the chunk-boundary paranoia compare flags
/// the tape divergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostIo {
    Live,
    Replay,
}

/// Architectural CPU state captured at a tick boundary — everything a
/// bit-exact trace-mode chunk replay needs, and nothing run-scoped: cycle
/// markers and the call stack stay with pass-1 (host-side logging), and the
/// virtual-register allocator is rebuilt fresh per worker (its guards are
/// strictly intra-tick, and cloning would share the `Arc<Mutex>` across
/// threads).
#[derive(Clone, Debug)]
pub(crate) struct ChunkCpuState {
    clock: u64,
    privilege_mode: PrivilegeMode,
    wfi: bool,
    x: [i64; REGISTER_COUNT as usize],
    f: [f64; 32],
    pc: u64,
    /// Boxed: 32 KB, keeps checkpoints cheap to move.
    csr: Box<[u64; CSR_CAPACITY]>,
    reservation: u64,
    is_reservation_set: bool,
    reservation_width: ReservationWidth,
    trace_len: usize,
    executed_instrs: u64,
    /// Unread advice-tape suffix; the replay tape is exactly this suffix
    /// with read position 0. Appends are suppressed in replay ([`HostIo`]).
    advice_suffix: Vec<u8>,
    #[cfg(feature = "field-inline")]
    field_registers: FieldRegisterFile,
}

impl ChunkCpuState {
    /// First difference between this captured boundary state and `cpu`'s
    /// current state (`None` = equal). Paranoia check: a worker finishing
    /// chunk k must land exactly on checkpoint k+1's capture. `trace_len` is
    /// row-uniform across modes and is compared too — a replay row-count
    /// drift shows up here at the boundary that caused it.
    pub(crate) fn diff_vs_cpu(&self, cpu: &Cpu) -> Option<String> {
        if self.trace_len != cpu.trace_len {
            return Some(format!(
                "trace_len: {} vs {}",
                self.trace_len, cpu.trace_len
            ));
        }
        if self.pc != cpu.pc {
            return Some(format!("pc: {:#x} vs {:#x}", self.pc, cpu.pc));
        }
        for i in 0..REGISTER_COUNT as usize {
            if self.x[i] != cpu.x[i] {
                return Some(format!("x[{i}]: {:#x} vs {:#x}", self.x[i], cpu.x[i]));
            }
        }
        for i in 0..CSR_CAPACITY {
            if self.csr[i] != cpu.csr[i] {
                return Some(format!(
                    "csr[{i:#x}]: {:#x} vs {:#x}",
                    self.csr[i], cpu.csr[i]
                ));
            }
        }
        if self.clock != cpu.clock {
            return Some(format!("clock: {} vs {}", self.clock, cpu.clock));
        }
        if self.wfi != cpu.wfi {
            return Some(format!("wfi: {} vs {}", self.wfi, cpu.wfi));
        }
        if core::mem::discriminant(&self.privilege_mode)
            != core::mem::discriminant(&cpu.privilege_mode)
        {
            return Some(format!(
                "privilege_mode: {:?} vs {:?}",
                self.privilege_mode, cpu.privilege_mode
            ));
        }
        if (
            self.reservation,
            self.is_reservation_set,
            self.reservation_width,
        ) != (
            cpu.reservation,
            cpu.is_reservation_set,
            cpu.reservation_width,
        ) {
            return Some(format!(
                "reservation: ({:#x}, {}, {:?}) vs ({:#x}, {}, {:?})",
                self.reservation,
                self.is_reservation_set,
                self.reservation_width,
                cpu.reservation,
                cpu.is_reservation_set,
                cpu.reservation_width
            ));
        }
        if self.executed_instrs != cpu.executed_instrs {
            return Some(format!(
                "executed_instrs: {} vs {}",
                self.executed_instrs, cpu.executed_instrs
            ));
        }
        let cpu_suffix = &cpu.advice_tape.data[cpu.advice_tape.read_position..];
        if self.advice_suffix != cpu_suffix {
            return Some(format!(
                "advice suffix: {} bytes vs {} bytes (or contents differ)",
                self.advice_suffix.len(),
                cpu_suffix.len()
            ));
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct Cpu {
    clock: u64,

    pub(crate) privilege_mode: PrivilegeMode,
    wfi: bool,
    pub x: [i64; REGISTER_COUNT as usize],
    #[allow(dead_code)]
    f: [f64; 32],
    pub(crate) pc: u64,
    csr: [u64; CSR_CAPACITY],
    pub mmu: Mmu,
    reservation: u64,
    is_reservation_set: bool,
    reservation_width: ReservationWidth,
    unsigned_data_mask: u64,
    pub trace_len: usize,
    executed_instrs: u64, // "real" RV64IMAC cycles
    active_markers: FnvHashMap<String, ActiveMarker>,
    pub vr_allocator: VirtualRegisterAllocator,
    call_stack: VecDeque<CallFrame>,
    /// Whether call frames snapshot the register file (JOLT_BACKTRACE=full).
    capture_backtrace_registers: bool,
    pub advice_tape: AdviceTape,
    /// Live in pass-1/serial runs; Replay in parallel-trace workers.
    host_io: HostIo,
    #[cfg(feature = "field-inline")]
    pub field_registers: FieldRegisterFile,
}

/// Width of an LR/SC reservation set. Ordered `Word < Doubleword` so
/// `reservation_covers` can compare with `>=` — an 8-byte reservation set
/// covers a 4-byte SC write, but not vice versa.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ReservationWidth {
    Word,
    Doubleword,
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

fn get_privilege_encoding(mode: &PrivilegeMode) -> u8 {
    match mode {
        PrivilegeMode::User => 0,
        PrivilegeMode::Supervisor => 1,
        PrivilegeMode::Reserved => panic!(),
        PrivilegeMode::Machine => 3,
    }
}

pub fn get_privilege_mode(encoding: u64) -> PrivilegeMode {
    match encoding {
        0 => PrivilegeMode::User,
        1 => PrivilegeMode::Supervisor,
        3 => PrivilegeMode::Machine,
        _ => panic!("Unknown privilege encoding"),
    }
}

fn get_trap_cause(trap: &Trap) -> u64 {
    let interrupt_bit = 0x8000000000000000_u64;
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
    pub fn new(terminal: Box<dyn Terminal>) -> Self {
        let mut cpu = Self {
            clock: 0,
            privilege_mode: PrivilegeMode::Machine,
            wfi: false,
            x: [0; REGISTER_COUNT as usize],
            f: [0.0; 32],
            pc: 0,
            csr: [0; CSR_CAPACITY],
            mmu: Mmu::new(terminal),
            reservation: 0,
            is_reservation_set: false,
            reservation_width: ReservationWidth::Word,
            unsigned_data_mask: 0xffffffffffffffff,
            trace_len: 0,
            executed_instrs: 0,
            active_markers: FnvHashMap::default(),
            vr_allocator: VirtualRegisterAllocator::new(),
            call_stack: VecDeque::with_capacity(MAX_CALL_STACK_DEPTH),
            capture_backtrace_registers: std::env::var("JOLT_BACKTRACE")
                .map(|v| v.eq_ignore_ascii_case("full"))
                .unwrap_or(false),
            advice_tape: AdviceTape::new(),
            host_io: HostIo::Live,
            #[cfg(feature = "field-inline")]
            field_registers: FieldRegisterFile::default(),
        };
        cpu.write_csr_raw(CSR_MISA_ADDRESS, 0x800000008014312f);
        cpu
    }

    /// Set the host-I/O mode (see [`HostIo`]). Workers replaying chunks run
    /// in `Replay` for their whole lifetime.
    pub(crate) fn set_host_io(&mut self, mode: HostIo) {
        self.host_io = mode;
    }

    #[inline(always)]
    pub fn raise_trap(&mut self, trap: Trap, faulting_pc: u64) {
        let _ = self.handle_trap(trap, faulting_pc, false);
    }

    pub fn update_pc(&mut self, value: u64) {
        self.pc = value;
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

    pub fn write_register(&mut self, reg: usize, write_value: i64) {
        debug_assert!(
            reg < REGISTER_COUNT as usize,
            "reg must be 0-{}. {reg}",
            REGISTER_COUNT - 1
        );
        match reg {
            0 => {
                // 0th register is hardwired zero
                debug_assert_eq!(self.x[reg], 0);
            }
            _ => self.x[reg] = write_value,
        }
    }

    pub fn read_pc(&self) -> u64 {
        self.pc
    }

    pub fn set_reservation(&mut self, address: u64, width: ReservationWidth) {
        self.reservation = address;
        self.is_reservation_set = true;
        self.reservation_width = width;
    }

    pub fn clear_reservation(&mut self) {
        self.is_reservation_set = false;
    }

    /// Returns true if a reservation is held at `address` whose reservation
    /// set is at least `min_width` wide. Per the RISC-V A spec (2024 ratified,
    /// §13.1.2): "SC succeeds only if the reservation is still valid and the
    /// reservation set contains the bytes being written." Concretely:
    ///   - LR.W + SC.W at same addr → succeeds (Word ≥ Word)
    ///   - LR.D + SC.W at same addr → succeeds (Doubleword ≥ Word, spec)
    ///   - LR.W + SC.D at same addr → fails    (Word < Doubleword)
    ///   - LR.D + SC.D at same addr → succeeds (Doubleword ≥ Doubleword)
    pub fn reservation_covers(&self, address: u64, min_width: ReservationWidth) -> bool {
        self.is_reservation_set
            && self.reservation == address
            && self.reservation_width >= min_width
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
        // Jolt guests have no interrupt sources (no CLINT/PLIC) and cannot
        // write MIP (unsupported CSR rejected at decode), so pending-interrupt
        // handling is gated on a single always-zero load. handle_interrupt is
        // a no-op when MIP is 0.
        if self.read_csr_raw(CSR_MIP_ADDRESS) != 0 {
            self.handle_interrupt(self.pc);
        }
        self.clock = self.clock.wrapping_add(1);
    }

    fn tick_operate(&mut self, trace: Option<&mut Vec<Cycle>>) -> Result<(), Trap> {
        if self.wfi {
            if (self.read_csr_raw(CSR_MIE_ADDRESS) & self.read_csr_raw(CSR_MIP_ADDRESS)) != 0 {
                self.wfi = false;
            }
            return Ok(());
        }

        let instr = match self.mmu.decode_cache.lookup(self.pc) {
            Some(cached) => {
                let instr = cached.instr;
                self.pc = self.pc.wrapping_add(cached.len as u64);
                instr
            }
            None => self.decode_and_cache()?,
        };

        match trace {
            None => {
                // Rows are counted inside the walk (`RISCVTrace::trace` with
                // no sink bumps trace_len once per suppressed row), keeping
                // trace_len row-uniform across modes.
                instr.execute(self);
            }
            Some(trace_vec) => {
                let rows_before = trace_vec.len();
                instr.trace(self, Some(&mut *trace_vec));
                self.trace_len += trace_vec.len() - rows_before;
            }
        }

        if instr.is_real() {
            self.executed_instrs += 1;
        }
        self.x[0] = 0; // hardwired zero

        Ok(())
    }

    /// Runs `f` with `source`'s inline sequence, cached per PC alongside the
    /// decoded instruction.
    ///
    /// Expansion is a pure function of the instruction: `inline_sequence`
    /// builds a fresh `ExpansionAllocator` per call and every virtual-register
    /// guard is released by the time it returns, so the first execution's
    /// sequence can be reused by later executions at the same PC. Callers that
    /// need per-execution advice values patch them into *copies* of the rows,
    /// never into the cached template.
    ///
    /// The rows are moved out of the cache entry while `f` runs (so `f` can
    /// borrow the CPU mutably) and put back afterwards. Text-store
    /// invalidation clears the decode slot, orphaning the entry along with its
    /// expansion, so a rewritten instruction can never be served stale rows.
    #[inline]
    pub(crate) fn with_cached_inline_sequence(
        &mut self,
        source: &Instruction,
        f: impl FnOnce(&mut Cpu, &[Instruction]),
    ) {
        let token = self.mmu.decode_cache.expansion_slot(source);
        let rows: Box<[Instruction]> = token
            .and_then(|index| self.mmu.decode_cache.take_expansion(index))
            .unwrap_or_else(|| {
                source
                    .inline_sequence(&self.vr_allocator)
                    .into_boxed_slice()
            });
        f(self, &rows);
        if let Some(index) = token {
            self.mmu.decode_cache.put_expansion(index, rows);
        }
    }

    fn decode_and_cache(&mut self) -> Result<Instruction, Trap> {
        let original_word = self.fetch()?;
        let instruction_address = self.pc;
        let is_compressed = (original_word & 0x3) != 0x3;
        let word = match is_compressed {
            false => {
                self.pc = self.pc.wrapping_add(4);
                original_word
            }
            true => {
                self.pc = self.pc.wrapping_add(2);
                uncompress_instruction(original_word & 0xffff)
            }
        };

        let instr = Instruction::decode(word, instruction_address, is_compressed)
            .unwrap_or_else(|e| decode_failure(word, instruction_address, is_compressed, e));
        self.mmu.decode_cache.insert(
            instruction_address,
            instr,
            if is_compressed { 2 } else { 4 },
        );
        Ok(instr)
    }

    fn handle_interrupt(&mut self, instruction_address: u64) {
        let minterrupt = self.read_csr_raw(CSR_MIP_ADDRESS) & self.read_csr_raw(CSR_MIE_ADDRESS);

        if (minterrupt & MIP_MEIP) != 0
            && self.handle_trap(
                Trap {
                    trap_type: TrapType::MachineExternalInterrupt,
                    value: self.pc,
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
                    value: self.pc,
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
                    value: self.pc,
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
                    value: self.pc,
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
                    value: self.pc,
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
                    value: self.pc,
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
        let current_privilege_encoding = get_privilege_encoding(&self.privilege_mode) as u64;
        let cause = get_trap_cause(&trap);

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

            let interrupt_enabled = match trap.trap_type {
                TrapType::UserSoftwareInterrupt => usie != 0,
                TrapType::SupervisorSoftwareInterrupt => ssie != 0,
                TrapType::MachineSoftwareInterrupt => msie != 0,
                TrapType::UserTimerInterrupt => utie != 0,
                TrapType::SupervisorTimerInterrupt => stie != 0,
                TrapType::MachineTimerInterrupt => mtie != 0,
                TrapType::UserExternalInterrupt => ueie != 0,
                TrapType::SupervisorExternalInterrupt => seie != 0,
                TrapType::MachineExternalInterrupt => meie != 0,
                _ => true,
            };
            if !interrupt_enabled {
                return false;
            }
        }

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
            PrivilegeMode::Reserved => panic!(),
        };
        true
    }

    fn fetch(&mut self) -> Result<u32, Trap> {
        let word = match self.mmu.fetch_word(self.pc) {
            Ok(word) => word,
            Err(e) => {
                self.pc = self.pc.wrapping_add(4);
                return Err(e);
            }
        };
        Ok(word)
    }

    // SSTATUS, SIE, and SIP are subsets of MSTATUS, MIE, and MIP
    pub fn read_csr_raw(&self, address: u16) -> u64 {
        match address {
            CSR_FFLAGS_ADDRESS => self.csr[CSR_FCSR_ADDRESS as usize] & 0x1f,
            CSR_FRM_ADDRESS => (self.csr[CSR_FCSR_ADDRESS as usize] >> 5) & 0x7,
            CSR_SSTATUS_ADDRESS => self.csr[CSR_MSTATUS_ADDRESS as usize] & 0x80000003000de162,
            CSR_SIE_ADDRESS => self.csr[CSR_MIE_ADDRESS as usize] & 0x222,
            CSR_SIP_ADDRESS => self.csr[CSR_MIP_ADDRESS as usize] & 0x222,
            CSR_TIME_ADDRESS => panic!("CLINT is unsupported."),
            _ => self.csr[address as usize],
        }
    }

    pub fn write_csr_raw(&mut self, address: u16, value: u64) {
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
                self.csr[address as usize] = value & 0x666;
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

    pub(crate) fn sign_extend(&self, value: i64) -> i64 {
        value
    }

    pub(crate) fn unsigned_data(&self, value: i64) -> u64 {
        (value as u64) & self.unsigned_data_mask
    }

    pub(crate) fn most_negative(&self) -> i64 {
        i64::MIN
    }

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
                uncompress_instruction(original_word)
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
        s
    }

    pub fn get_mut_mmu(&mut self) -> &mut Mmu {
        &mut self.mmu
    }

    pub fn handle_jolt_cycle_marker(&mut self, ptr: u32, len: u32, event: u32) -> Result<(), Trap> {
        if self.host_io == HostIo::Replay {
            return Ok(());
        }
        match event {
            JOLT_CYCLE_MARKER_START => {
                let label = self.read_string(ptr, len)?;
                let marker = ActiveMarker {
                    start_instrs: self.executed_instrs,
                    start_trace_len: self.trace_len,
                };
                if self.active_markers.insert(label.clone(), marker).is_some() {
                    warn!("Marker with label '{label}' is already active; restarting it");
                }
            }

            JOLT_CYCLE_MARKER_END => {
                let label = self.read_string(ptr, len)?;
                if let Some(mark) = self.active_markers.remove(&label) {
                    let real = self.executed_instrs - mark.start_instrs;
                    let total = self.trace_len - mark.start_trace_len;
                    let virtual_instrs = total - real as usize;
                    info!(
                        "\"{label}\": {real} RV64IMAC cycles + {virtual_instrs} virtual instructions = {total} total cycles"
                    );
                } else {
                    warn!("Attempt to end a marker '{label}' that was never started");
                }
            }
            _ => {
                panic!("Unexpected event: event must match either start or end marker.")
            }
        }
        Ok(())
    }

    pub fn handle_jolt_print(&mut self, ptr: u32, len: u32, event_type: u32) -> Result<(), Trap> {
        if self.host_io == HostIo::Replay {
            return Ok(());
        }
        let message = self.read_string(ptr, len)?;
        if event_type == JOLT_PRINT_STRING {
            print!("{message}");
        } else if event_type == JOLT_PRINT_LINE {
            println!("{message}");
        } else {
            panic!("Unexpected event type: {event_type}");
        }
        Ok(())
    }

    pub fn handle_advice_write(&mut self, ptr: u64, len: u64) -> Result<(), Trap> {
        // Pass-1 already appended these bytes; a replay append would corrupt
        // the worker's suffix-relative read offsets.
        if self.host_io == HostIo::Replay {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(len as usize);
        for i in 0..len {
            let (b, _) = self.mmu.load(ptr + i)?;
            bytes.push(b);
        }
        advice_tape_write(self, &bytes);
        Ok(())
    }

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
    pub fn track_call(&mut self, return_address: u64) {
        // Backtraces are pass-1's job; workers do not maintain a call stack.
        if self.host_io == HostIo::Replay {
            return;
        }
        if self.call_stack.len() >= MAX_CALL_STACK_DEPTH {
            self.call_stack.pop_front();
        }

        self.call_stack.push_back(CallFrame {
            call_site: return_address,
            // Register snapshots are only displayed by JOLT_BACKTRACE=full;
            // skip the bulk copy unless that mode was requested.
            x: self.capture_backtrace_registers.then(|| Box::new(self.x)),
            cycle_count: self.trace_len,
        });
    }

    pub fn get_call_stack(&self) -> &VecDeque<CallFrame> {
        &self.call_stack
    }
}

impl Cpu {
    pub fn save_state_with_empty_memory(&self) -> Cpu {
        Cpu {
            clock: self.clock,
            privilege_mode: self.privilege_mode,
            wfi: self.wfi,
            x: self.x,
            f: self.f,
            pc: self.pc,
            csr: self.csr,
            mmu: self.mmu.save_state_with_empty_memory(),
            reservation: self.reservation,
            is_reservation_set: self.is_reservation_set,
            reservation_width: self.reservation_width,
            unsigned_data_mask: self.unsigned_data_mask,
            trace_len: self.trace_len,
            executed_instrs: self.executed_instrs,
            active_markers: self.active_markers.clone(),
            vr_allocator: self.vr_allocator.clone(),
            call_stack: self.call_stack.clone(),
            capture_backtrace_registers: self.capture_backtrace_registers,
            advice_tape: self.advice_tape.clone(),
            host_io: self.host_io,
            #[cfg(feature = "field-inline")]
            field_registers: self.field_registers.clone(),
        }
    }

    /// Capture chunk-replay CPU state. Must be called at a tick boundary.
    ///
    /// The destructure is exhaustive on purpose: adding a `Cpu` field fails
    /// compilation here until the field is classified as captured, dropped,
    /// or worker-fresh.
    pub(crate) fn capture_chunk_state(&self) -> ChunkCpuState {
        let Cpu {
            clock,
            privilege_mode,
            wfi,
            x,
            f,
            pc,
            csr,
            // Memory image, JoltDevice and decode cache are captured by the
            // checkpoint layer (see `parallel::ChunkCheckpoint`).
            mmu: _,
            reservation,
            is_reservation_set,
            reservation_width,
            // Constants, re-established by worker construction.
            unsigned_data_mask: _,
            trace_len,
            executed_instrs,
            // Pass-1-only logging/bookkeeping.
            active_markers: _,
            // Fresh per worker: guards are strictly intra-tick, and cloning
            // would share the Arc<Mutex> across threads.
            vr_allocator,
            // Pass-1 owns panic backtraces.
            call_stack: _,
            capture_backtrace_registers: _,
            advice_tape,
            // Worker-owned mode flag.
            host_io: _,
            #[cfg(feature = "field-inline")]
            field_registers,
        } = self;
        debug_assert!(
            vr_allocator.is_quiescent(),
            "chunk capture with outstanding virtual-register guards (mid-tick capture?)"
        );
        ChunkCpuState {
            clock: *clock,
            privilege_mode: *privilege_mode,
            wfi: *wfi,
            x: *x,
            f: *f,
            pc: *pc,
            csr: Box::new(*csr),
            reservation: *reservation,
            is_reservation_set: *is_reservation_set,
            reservation_width: *reservation_width,
            trace_len: *trace_len,
            executed_instrs: *executed_instrs,
            advice_suffix: advice_tape.data[advice_tape.read_position..].to_vec(),
            #[cfg(feature = "field-inline")]
            field_registers: field_registers.clone(),
        }
    }

    /// Install captured chunk state into this (worker) CPU. Counterpart of
    /// [`Cpu::capture_chunk_state`]; the memory image, JoltDevice outputs and
    /// decode cache are installed by the checkpoint layer.
    pub(crate) fn install_chunk_state(&mut self, state: &ChunkCpuState) {
        debug_assert!(
            self.vr_allocator.is_quiescent(),
            "chunk install with outstanding virtual-register guards"
        );
        self.clock = state.clock;
        self.privilege_mode = state.privilege_mode;
        self.wfi = state.wfi;
        self.x = state.x;
        self.f = state.f;
        self.pc = state.pc;
        self.csr = *state.csr;
        self.reservation = state.reservation;
        self.is_reservation_set = state.is_reservation_set;
        self.reservation_width = state.reservation_width;
        self.trace_len = state.trace_len;
        self.executed_instrs = state.executed_instrs;
        self.active_markers.clear();
        self.call_stack.clear();
        self.advice_tape = AdviceTape::from_bytes(state.advice_suffix.clone());
        #[cfg(feature = "field-inline")]
        {
            self.field_registers = state.field_registers.clone();
        }
    }

    /// First architectural-state difference vs `other`, as a human-readable
    /// report (`None` = equal).
    ///
    /// Compares everything a trace-mode chunk replay depends on: pc, the full
    /// register file (including virtual registers), the CSR array, clock, wfi,
    /// privilege mode, the LR/SC reservation triple, the advice tape (data and
    /// read position), `executed_instrs`, `trace_len` (row-uniform across
    /// trace and execute modes), and JoltDevice outputs/panic. Excludes
    /// host-side bookkeeping (markers, call stack) and the dead `f`
    /// registers. Memory is not compared — callers hash it separately.
    pub fn arch_state_diff(&self, other: &Cpu) -> Option<String> {
        if self.trace_len != other.trace_len {
            return Some(format!(
                "trace_len: {} vs {}",
                self.trace_len, other.trace_len
            ));
        }
        if self.pc != other.pc {
            return Some(format!("pc: {:#x} vs {:#x}", self.pc, other.pc));
        }
        for i in 0..REGISTER_COUNT as usize {
            if self.x[i] != other.x[i] {
                return Some(format!("x[{i}]: {:#x} vs {:#x}", self.x[i], other.x[i]));
            }
        }
        for i in 0..CSR_CAPACITY {
            if self.csr[i] != other.csr[i] {
                return Some(format!(
                    "csr[{i:#x}]: {:#x} vs {:#x}",
                    self.csr[i], other.csr[i]
                ));
            }
        }
        if self.clock != other.clock {
            return Some(format!("clock: {} vs {}", self.clock, other.clock));
        }
        if self.wfi != other.wfi {
            return Some(format!("wfi: {} vs {}", self.wfi, other.wfi));
        }
        if core::mem::discriminant(&self.privilege_mode)
            != core::mem::discriminant(&other.privilege_mode)
        {
            return Some(format!(
                "privilege_mode: {:?} vs {:?}",
                self.privilege_mode, other.privilege_mode
            ));
        }
        if (
            self.reservation,
            self.is_reservation_set,
            self.reservation_width,
        ) != (
            other.reservation,
            other.is_reservation_set,
            other.reservation_width,
        ) {
            return Some(format!(
                "reservation: ({:#x}, {}, {:?}) vs ({:#x}, {}, {:?})",
                self.reservation,
                self.is_reservation_set,
                self.reservation_width,
                other.reservation,
                other.is_reservation_set,
                other.reservation_width
            ));
        }
        if self.advice_tape.read_position != other.advice_tape.read_position {
            return Some(format!(
                "advice_tape.read_position: {} vs {}",
                self.advice_tape.read_position, other.advice_tape.read_position
            ));
        }
        if self.advice_tape.data != other.advice_tape.data {
            return Some(format!(
                "advice_tape.data: {} bytes vs {} bytes (or contents differ)",
                self.advice_tape.data.len(),
                other.advice_tape.data.len()
            ));
        }
        if self.executed_instrs != other.executed_instrs {
            return Some(format!(
                "executed_instrs: {} vs {}",
                self.executed_instrs, other.executed_instrs
            ));
        }
        match (&self.mmu.jolt_device, &other.mmu.jolt_device) {
            (Some(a), Some(b)) => {
                if a.outputs != b.outputs {
                    return Some(format!(
                        "jolt_device.outputs: {} bytes vs {} bytes (or contents differ)",
                        a.outputs.len(),
                        b.outputs.len()
                    ));
                }
                if a.panic != b.panic {
                    return Some(format!("jolt_device.panic: {} vs {}", a.panic, b.panic));
                }
            }
            (None, None) => {}
            _ => return Some("jolt_device: present vs absent".to_string()),
        }
        None
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        if !self.active_markers.is_empty() {
            warn!(
                "Warning: Found {} unclosed cycle tracking marker(s):",
                self.active_markers.len()
            );
            for (label, marker) in &self.active_markers {
                warn!(
                    "  - '{}', started at {} RV64IMAC cycles",
                    label, marker.start_instrs
                );
            }
        }
    }
}

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

#[cold]
#[inline(never)]
fn decode_failure(word: u32, address: u64, compressed: bool, e: impl core::fmt::Display) -> ! {
    panic!(
        "Failed to decode instruction: word=0x{word:08x}, address=0x{address:x}, compressed={compressed}: {e}"
    )
}

#[cfg(test)]
mod test_cpu {
    use std::io::{Result as IoResult, Write};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;

    fn create_cpu() -> Cpu {
        Cpu::new(Box::new(DummyTerminal::default()))
    }

    #[test]
    fn read_register() {
        let mut cpu = create_cpu();
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
                0 => assert_eq!(0, cpu.read_register(i)),
                _ => assert_eq!(i as i64 + 1, cpu.read_register(i)),
            }
        }

        for i in 0..31 {
            cpu.x[i] = (0xffffffffffffffff - i) as i64;
        }

        for i in 0..31 {
            match i {
                0 => assert_eq!(0, cpu.read_register(i)),
                _ => assert_eq!(-(i as i64 + 1), cpu.read_register(i)),
            }
        }
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
    fn exception() {
        // ECALL executes through its inline sequence in both modes (execute
        // mode mirrors trace mode), so trap state lives in the CSR virtual
        // registers: vr34 = mtvec, vr36 = mepc, vr37 = mcause.
        let handler_vector = 0x10000000;
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(4);
        // Write ECALL instruction
        match cpu.get_mut_mmu().store_word(DRAM_BASE, 0x00000073) {
            Ok(_) => {}
            Err(_e) => panic!("Failed to store"),
        };
        cpu.x[34] = handler_vector as i64;
        cpu.update_pc(DRAM_BASE);

        cpu.tick(None);

        assert_eq!(handler_vector, cpu.read_pc());

        // mepc/mcause virtual registers hold the faulting pc and cause
        assert_eq!(DRAM_BASE as i64, cpu.x[36]);
        assert_eq!(0xb, cpu.x[37]);
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

        assert_eq!(0, cpu.read_register(0));
        cpu.tick(None);
        assert_eq!(0, cpu.read_register(0));

        assert_eq!(0, cpu.read_register(1));
        cpu.tick(None);
        assert_eq!(1, cpu.read_register(1));
    }

    #[test]
    fn advice_tape_reads_back_little_endian_in_fifo_order() {
        let mut tape = AdviceTape::new();
        assert!(tape.is_empty());
        assert_eq!(tape.read(1), None, "reading an empty tape yields None");

        tape.write(&[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(tape.len(), 4);
        assert_eq!(tape.remaining(), 4);

        assert_eq!(tape.read(2), Some(0x0201), "bytes assemble little-endian");
        assert_eq!(tape.remaining(), 2);
        assert_eq!(tape.read(4), None);
        assert_eq!(tape.remaining(), 2);
        assert_eq!(tape.read(2), Some(0x0403));
        assert_eq!(tape.remaining(), 0);

        tape.reset_read_position();
        assert_eq!(tape.read(4), Some(0x0403_0201));
    }

    #[test]
    fn handle_advice_write_copies_guest_memory_onto_the_tape() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(64);
        for (i, byte) in [0xde_u8, 0xad, 0xbe, 0xef].iter().enumerate() {
            cpu.get_mut_mmu().store_raw(DRAM_BASE + i as u64, *byte);
        }
        cpu.handle_advice_write(DRAM_BASE, 4).unwrap();
        assert_eq!(advice_tape_remaining(&cpu), 4);
        assert_eq!(advice_tape_read(&mut cpu, 4), Some(0xefbe_adde));
    }

    #[test]
    fn reservation_covers_follows_the_lr_sc_width_rules() {
        let mut cpu = create_cpu();
        let addr = DRAM_BASE + 64;
        assert!(!cpu.is_reservation_set());

        cpu.set_reservation(addr, ReservationWidth::Word);
        assert!(cpu.is_reservation_set());
        assert!(cpu.reservation_covers(addr, ReservationWidth::Word));
        assert!(!cpu.reservation_covers(addr, ReservationWidth::Doubleword));

        // LR.D covers both widths
        cpu.set_reservation(addr, ReservationWidth::Doubleword);
        assert!(cpu.reservation_covers(addr, ReservationWidth::Word));
        assert!(cpu.reservation_covers(addr, ReservationWidth::Doubleword));

        // Different address or a cleared reservation never covers
        assert!(!cpu.reservation_covers(addr + 8, ReservationWidth::Word));
        cpu.clear_reservation();
        assert!(!cpu.reservation_covers(addr, ReservationWidth::Word));
    }

    #[test]
    fn cycle_markers_track_real_and_virtual_instruction_counts() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(1 << 16);
        let label = b"my_marker";
        for (i, byte) in label.iter().enumerate() {
            cpu.get_mut_mmu().store_raw(DRAM_BASE + i as u64, *byte);
        }
        let ptr = DRAM_BASE as u32;

        cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
            .unwrap();
        assert_eq!(cpu.active_markers.len(), 1);
        assert!(cpu.active_markers.contains_key("my_marker"));

        // A second start with the same label restarts the span
        cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
            .unwrap();
        assert_eq!(cpu.active_markers.len(), 1);

        cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_END)
            .unwrap();
        assert!(cpu.active_markers.is_empty());

        // Ending a marker that was never started is tolerated
        cpu.handle_jolt_cycle_marker(ptr + 64, 0, JOLT_CYCLE_MARKER_END)
            .unwrap();
    }

    #[test]
    fn cycle_marker_end_matches_the_label_at_another_address() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(1 << 16);
        let label = b"span";
        for (i, byte) in label.iter().enumerate() {
            cpu.get_mut_mmu().store_raw(DRAM_BASE + i as u64, *byte);
            cpu.get_mut_mmu()
                .store_raw(DRAM_BASE + 32 + i as u64, *byte);
        }
        let start_ptr = DRAM_BASE as u32;
        let end_ptr = start_ptr + 32;

        cpu.handle_jolt_cycle_marker(start_ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
            .unwrap();
        cpu.handle_jolt_cycle_marker(end_ptr, label.len() as u32, JOLT_CYCLE_MARKER_END)
            .unwrap();
        assert!(cpu.active_markers.is_empty());
    }

    #[test]
    fn cycle_marker_label_reusing_an_address_keeps_the_earlier_span() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(1 << 16);
        let ptr = DRAM_BASE as u32;
        // Both labels are built in the same buffer, as a freed block reused by
        // a runtime-built label would be.
        for label in [b"aaaa", b"bbbb"] {
            for (i, byte) in label.iter().enumerate() {
                cpu.get_mut_mmu().store_raw(DRAM_BASE + i as u64, *byte);
            }
            cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
                .unwrap();
        }
        assert!(cpu.active_markers.contains_key("aaaa"));
        assert!(cpu.active_markers.contains_key("bbbb"));
    }

    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> IoResult<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> IoResult<()> {
            Ok(())
        }
    }

    #[test]
    fn cycle_marker_restart_overwrites_the_active_span_and_warns() {
        let mut cpu = create_cpu();
        cpu.get_mut_mmu().init_memory(1 << 16);
        let label = b"span";
        for (i, byte) in label.iter().enumerate() {
            cpu.get_mut_mmu().store_raw(DRAM_BASE + i as u64, *byte);
        }
        let ptr = DRAM_BASE as u32;
        let logs = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer({
                let logs = logs.clone();
                move || logs.clone()
            })
            .with_ansi(false)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
                .unwrap();
            cpu.executed_instrs = 3;
            cpu.trace_len = 5;
            cpu.handle_jolt_cycle_marker(ptr, label.len() as u32, JOLT_CYCLE_MARKER_START)
                .unwrap();
        });

        let marker = &cpu.active_markers["span"];
        assert_eq!((marker.start_instrs, marker.start_trace_len), (3, 5));
        let logs = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains("Marker with label 'span' is already active"),
            "{logs}"
        );
    }
}
