//! Guest state plane shared between generated code and Rust helpers.
//!
//! Generated code addresses this struct through a pinned register with
//! hardcoded field offsets, so the layout is `repr(C)` and offsets are
//! asserted at compile time.

use common::constants::REGISTER_COUNT;
use common::jolt_device::JoltDevice;
use jolt_program::execution::TraceError;

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
    /// Record mode ran out of observation slots (the record pass emitted more
    /// rows than the fast pass counted, i.e. the two diverged).
    FaultObservationOverflow = 5,
    /// Paused at a group boundary having reached `row_limit`; `pc` holds the
    /// resume point, so calling the body again continues the execution.
    Paused = 6,
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
    /// Pause once `trace_len` reaches this, at the next group boundary (the
    /// only place a resumable PC is statically known). `u64::MAX` disables
    /// pausing, which is what the eager paths use.
    pub row_limit: u64,
    /// Record mode: next observation slot, bumped per emitted row.
    pub obs_cursor: *mut Observation,
    /// Record mode: one past the last writable slot.
    pub obs_end: *mut Observation,
}

impl GuestState {
    /// Translate the generated-code exit state into the backend error channel.
    #[expect(clippy::print_stderr)]
    pub fn check_exit(&self, host: &mut HostContext) -> Result<(), TraceError> {
        match self.exit {
            e if e == ExitReason::Terminated as u64 => {}
            e if e == ExitReason::FaultOutOfBounds as u64 => {
                return Err(TraceError::Backend("guest RAM access out of bounds"));
            }
            e if e == ExitReason::FaultBadJumpTarget as u64 => {
                return Err(TraceError::Backend(
                    "indirect jump to a non-compiled address",
                ));
            }
            e if e == ExitReason::FaultObservationOverflow as u64 => {
                return Err(TraceError::Backend(
                    "record pass overflowed the observation buffer (row-count divergence)",
                ));
            }
            _ => {
                if let Some(message) = host.helper_error.take() {
                    eprintln!("jolt-tracer-x86 helper error: {message}");
                }
                return Err(TraceError::Backend("host helper reported an error"));
            }
        }
        Ok(())
    }
}

/// One row's dynamic values, written by generated code in record mode.
///
/// Generated code cannot construct a `TraceRow` (its `Option` fields have no
/// guaranteed layout), so it writes this fixed POD instead and a Rust pass
/// reassembles rows afterwards, taking the static half from
/// `expanded_bytecode[row_index]`. 64 bytes keeps the cursor bump a shift and
/// the write pattern cache-friendly; the fields a given kind does not use are
/// simply ignored by the reassembly pass.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Observation {
    pub row_index: u64,
    pub rs1: u64,
    pub rs2: u64,
    pub rd_pre: u64,
    pub rd_post: u64,
    pub ram_address: u64,
    pub ram_pre: u64,
    pub ram_post: u64,
}

pub const OBSERVATION_SIZE: i32 = core::mem::size_of::<Observation>() as i32;

pub const OBS_ROW_INDEX: i32 = 0;
pub const OBS_RS1: i32 = 8;
pub const OBS_RS2: i32 = 16;
pub const OBS_RD_PRE: i32 = 24;
pub const OBS_RD_POST: i32 = 32;
pub const OBS_RAM_ADDRESS: i32 = 40;
pub const OBS_RAM_PRE: i32 = 48;
pub const OBS_RAM_POST: i32 = 56;

const _: () = {
    assert!(OBSERVATION_SIZE == 64);
    assert!(core::mem::offset_of!(Observation, row_index) == OBS_ROW_INDEX as usize);
    assert!(core::mem::offset_of!(Observation, rs1) == OBS_RS1 as usize);
    assert!(core::mem::offset_of!(Observation, rs2) == OBS_RS2 as usize);
    assert!(core::mem::offset_of!(Observation, rd_pre) == OBS_RD_PRE as usize);
    assert!(core::mem::offset_of!(Observation, rd_post) == OBS_RD_POST as usize);
    assert!(core::mem::offset_of!(Observation, ram_address) == OBS_RAM_ADDRESS as usize);
    assert!(core::mem::offset_of!(Observation, ram_pre) == OBS_RAM_PRE as usize);
    assert!(core::mem::offset_of!(Observation, ram_post) == OBS_RAM_POST as usize);
};

/// Maximum runtime advice values one source-instruction group can need
/// (largest today: the modular-division inlines at 8).
pub const ADVICE_SLOTS: usize = 16;

/// One group's advice computation, resolved at compile time.
pub struct AdviceJob {
    pub compute: AdviceCompute,
    /// Number of `VirtualAdvice` rows in the job's group, i.e. how many
    /// values the computation must provide. Patched in once the group's rows
    /// have been emitted; the runtime helper checks the provided count
    /// against it so a short advice vector fails loudly instead of leaving
    /// stale slots to be read (the interpreter panics on the same mismatch).
    pub advice_rows: usize,
}

/// How a group's advice values are computed.
pub enum AdviceCompute {
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
pub const OFF_ADVICE_JOBS: i32 = OFF_ADVICE_SLOTS + (ADVICE_SLOTS as i32) * 8;
pub const OFF_ROW_LIMIT: i32 = OFF_ADVICE_JOBS + 8;
pub const OFF_OBS_CURSOR: i32 = OFF_ROW_LIMIT + 8;
pub const OFF_OBS_END: i32 = OFF_OBS_CURSOR + 8;

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
    assert!(core::mem::offset_of!(GuestState, advice_jobs) == OFF_ADVICE_JOBS as usize);
    assert!(core::mem::offset_of!(GuestState, row_limit) == OFF_ROW_LIMIT as usize);
    assert!(core::mem::offset_of!(GuestState, obs_cursor) == OFF_OBS_CURSOR as usize);
    assert!(core::mem::offset_of!(GuestState, obs_end) == OFF_OBS_END as usize);
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
