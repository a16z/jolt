//! Guest state plane shared between generated code and Rust helpers.
//!
//! Generated code addresses this struct through a pinned register with
//! hardcoded field offsets, so the layout is `repr(C)` and offsets are
//! asserted at compile time.

use common::constants::REGISTER_COUNT;
use common::jolt_device::JoltDevice;

/// Why generated code returned to the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum ExitReason {
    /// Still running (initial value; never observed on return).
    Running = 0,
    /// Guest terminated via the PC-stall convention (`j .`).
    Terminated = 1,
    /// Guest RAM access outside the memory plane.
    FaultOutOfBounds = 2,
    /// Indirect jump to an address that is not a compiled group start.
    FaultBadJumpTarget = 3,
    /// A host helper reported an error (e.g. device access violation).
    FaultHelper = 4,
}

/// State shared with generated code. Field offsets are load-bearing.
#[repr(C)]
pub struct GuestState {
    /// All 128 guest registers (32 architectural + 96 virtual), by index.
    pub x: [u64; REGISTER_COUNT as usize],
    /// Guest PC (source address of the current group). Written by generated
    /// code only at indirect control flow and on exit.
    pub pc: u64,
    /// Trace rows executed so far.
    pub trace_len: u64,
    /// [`ExitReason`] as a raw u64 (written by generated code).
    pub exit: u64,
    /// Faulting guest address when `exit` is a fault.
    pub fault_addr: u64,
    /// Host base of the guest RAM plane (also pinned in a register; stored
    /// here so helpers can reconstruct it from `&mut GuestState` alone).
    pub mem_base: u64,
    /// Size in bytes of the RAM plane.
    pub mem_size: u64,
    /// Host context for helper calls (device, advice tape, panic state).
    pub host: *mut HostContext,
    /// Runtime advice values for the group being executed, filled by the
    /// group's advice helper and read by its `VirtualAdvice` rows in order.
    pub advice_slots: [u64; ADVICE_SLOTS],
    /// Per-program advice-job table (borrowed from the compiled artifact);
    /// generated code passes a job index, helpers dereference it.
    pub advice_jobs: *const AdviceJob,
}

/// Maximum runtime advice values one source-instruction group can need
/// (largest today: the modular-division inlines at 8).
pub const ADVICE_SLOTS: usize = 16;

/// One group's advice computation, resolved at compile time.
pub enum AdviceJob {
    /// DIV/REM family: `code` selects the variant's formula.
    Div { code: u8, rs1: u8, rs2: u8 },
    /// A registered inline's `build_advice`, called through
    /// [`InlineAdviceContext`](tracer::InlineAdviceContext).
    Inline {
        registration: &'static tracer::InlineRegistration,
        operands: tracer::instruction::format::format_inline::FormatInline,
    },
}

pub const OFF_X: i32 = 0;
pub const OFF_PC: i32 = (REGISTER_COUNT as i32) * 8;
pub const OFF_TRACE_LEN: i32 = OFF_PC + 8;
pub const OFF_EXIT: i32 = OFF_TRACE_LEN + 8;
pub const OFF_FAULT_ADDR: i32 = OFF_EXIT + 8;
pub const OFF_MEM_BASE: i32 = OFF_FAULT_ADDR + 8;
pub const OFF_MEM_SIZE: i32 = OFF_MEM_BASE + 8;
pub const OFF_HOST: i32 = OFF_MEM_SIZE + 8;
pub const OFF_ADVICE_SLOTS: i32 = OFF_HOST + 8;

const _: () = {
    assert!(core::mem::offset_of!(GuestState, x) == OFF_X as usize);
    assert!(core::mem::offset_of!(GuestState, pc) == OFF_PC as usize);
    assert!(core::mem::offset_of!(GuestState, trace_len) == OFF_TRACE_LEN as usize);
    assert!(core::mem::offset_of!(GuestState, exit) == OFF_EXIT as usize);
    assert!(core::mem::offset_of!(GuestState, fault_addr) == OFF_FAULT_ADDR as usize);
    assert!(core::mem::offset_of!(GuestState, mem_base) == OFF_MEM_BASE as usize);
    assert!(core::mem::offset_of!(GuestState, mem_size) == OFF_MEM_SIZE as usize);
    assert!(core::mem::offset_of!(GuestState, host) == OFF_HOST as usize);
    assert!(core::mem::offset_of!(GuestState, advice_slots) == OFF_ADVICE_SLOTS as usize);
};

#[inline]
pub const fn advice_slot_offset(slot: usize) -> i32 {
    OFF_ADVICE_SLOTS + (slot as i32) * 8
}

#[inline]
pub const fn reg_offset(register: u8) -> i32 {
    OFF_X + (register as i32) * 8
}

/// Host-side context reachable from helper calls. Not addressed by generated
/// code directly (only through `extern "C"` helpers), so layout is free.
pub struct HostContext {
    pub device: JoltDevice,
    /// Runtime advice tape bytes (append-only; reads go through the cursor).
    pub advice_tape: Vec<u8>,
    /// Read cursor into `advice_tape` (advice-load kinds).
    pub advice_cursor: usize,
    /// Set when a helper encounters an unrecoverable condition; carries the
    /// message surfaced in the resulting `TraceError`.
    pub helper_error: Option<String>,
}
