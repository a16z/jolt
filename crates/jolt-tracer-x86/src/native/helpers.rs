//! `extern "C"` helpers called from generated code.
//!
//! ABI: sysv64, first argument is always `*mut GuestState`. Pinned registers
//! (r12–r15) are callee-saved, so helpers preserve them for free.
//!
//! Error protocol: a failing helper writes `GuestState::exit` (and
//! `HostContext::helper_error` / `fault_addr` as appropriate); generated code
//! tests `exit` after each helper call and jumps to the exit stub when
//! nonzero. Return values travel in rax.

use common::constants::{RAM_START_ADDRESS, STACK_CANARY_SIZE};

use super::state::{AdviceCompute, ExitReason, GuestState, HostContext};

use common::jolt_device::JoltDevice;
use jolt_platform::{
    JOLT_ADVICE_WRITE_CALL_ID, JOLT_CYCLE_MARKER_END, JOLT_CYCLE_MARKER_START,
    JOLT_CYCLE_TRACK_CALL_ID, JOLT_PRINT_CALL_ID,
};
use tracer::emulator::host_io::{
    print_terminator, read_host_bytes, write_guest_print, HostMemory, HOST_STRING_ADDRESS_LIMIT,
};
use tracer::emulator::mmu::identity_load_succeeds;
use tracing::warn;

fn host_context<'a>(state: *mut GuestState) -> (&'a mut GuestState, &'a mut HostContext) {
    // SAFETY: generated code passes the GuestState it was launched with; its
    // `host` pointer outlives the run (both are owned by the driver frame).
    unsafe { (&mut *state, &mut *(*state).host) }
}

fn fail(state: &mut GuestState, host: &mut HostContext, message: String) -> u64 {
    host.helper_error = Some(message);
    state.exit = ExitReason::FaultHelper as u64;
    0
}

/// Guest memory as `host_io` reads it. Readability follows the tracer's
/// [`identity_load_succeeds`], so this backend rejects exactly the host calls
/// the interpreter rejects.
struct HostIoMemory<'a> {
    state: &'a GuestState,
    device: &'a JoltDevice,
}

impl HostMemory for HostIoMemory<'_> {
    fn is_host_readable(&self, address: u64) -> bool {
        identity_load_succeeds(Some(self.device), address, |ea| {
            ea - RAM_START_ADDRESS < self.state.mem_size
        })
    }

    fn load_host_byte(&mut self, address: u64) -> Option<u8> {
        if address >= RAM_START_ADDRESS {
            let offset = address - RAM_START_ADDRESS;
            // SAFETY: `read_host_bytes` loads only addresses `is_host_readable`
            // accepted, which bounds `offset` by the plane size.
            return Some(unsafe {
                (self.state.mem_base as *const u8)
                    .add(offset as usize)
                    .read()
            });
        }
        Some(self.device.load(address))
    }
}

/// Reads guest bytes for a host call under the tracer's shared rules, logging
/// a rejected range.
fn read_host_call_bytes(
    state: &GuestState,
    host: &HostContext,
    ptr: u64,
    len: u64,
    address_limit: u64,
) -> Option<Vec<u8>> {
    let mut memory = HostIoMemory {
        state,
        device: &host.device,
    };
    read_host_bytes(&mut memory, ptr, len, address_limit)
        .inspect_err(|error| warn!("Ignoring host call: {error}"))
        .ok()
}

/// `VirtualHostIO`: call id in a0 (x10), args in a1–a3 (x11–x13). No rd
/// write, no RAM-access row (matches the interpreter). Like the interpreter's
/// handlers, it never fails: it logs and ignores an unknown event or an
/// unreadable range. Both passes run it, so a print or warning appears once
/// per pass.
pub extern "sysv64" fn host_io(state: *mut GuestState) -> u64 {
    let (state, host) = host_context(state);
    let call_id = state.x[10] as u32;
    if call_id == JOLT_ADVICE_WRITE_CALL_ID {
        let len = state.x[12];
        if let Some(bytes) = read_host_call_bytes(state, host, state.x[11], len, u64::MAX) {
            if host.advice_tape.try_reserve(bytes.len()).is_ok() {
                host.advice_tape.extend_from_slice(&bytes);
            } else {
                warn!("Ignoring advice write of {len} bytes: the advice tape cannot grow");
            }
        }
        return 0;
    }
    let ptr = u64::from(state.x[11] as u32);
    let len = u64::from(state.x[12] as u32);
    let event = state.x[13] as u32;
    match call_id {
        JOLT_CYCLE_TRACK_CALL_ID => {
            // The interpreter logs cycle counts; this backend only reads the
            // label, so it rejects and logs the same calls.
            if event == JOLT_CYCLE_MARKER_START || event == JOLT_CYCLE_MARKER_END {
                let _label = read_host_call_bytes(state, host, ptr, len, HOST_STRING_ADDRESS_LIMIT);
            } else {
                warn!("Ignoring cycle-marker host call with unknown event {event}");
            }
        }
        JOLT_PRINT_CALL_ID => {
            let Some(terminator) = print_terminator(event) else {
                warn!("Ignoring print host call with unknown event type {event}");
                return 0;
            };
            let Some(bytes) =
                read_host_call_bytes(state, host, ptr, len, HOST_STRING_ADDRESS_LIMIT)
            else {
                return 0;
            };
            if let Err(error) = write_guest_print(&mut std::io::stdout().lock(), &bytes, terminator)
            {
                warn!("Guest print to stdout failed: {error}");
            }
        }
        // The interpreter ignores unknown call ids too.
        _ => {}
    }
    0
}

/// `Ld`/`VirtualLoad*` slow path: device-region, unaligned, or out-of-bounds
/// addresses. Returns the loaded doubleword in rax on success.
pub extern "sysv64" fn slow_load_doubleword(state: *mut GuestState, address: u64) -> u64 {
    let (state, host) = host_context(state);
    if !address.is_multiple_of(8) {
        // The interpreter panics ("Unaligned load_doubleword"); surface as a
        // helper error instead (spec invariant 7 asymmetry).
        return fail(
            state,
            host,
            format!("unaligned load_doubleword: {address:#x}"),
        );
    }
    if address >= RAM_START_ADDRESS {
        // In range would have taken the fast path; this is out of bounds.
        state.fault_addr = address;
        state.exit = ExitReason::FaultOutOfBounds as u64;
        return 0;
    }
    let mut value = 0u64;
    for i in 0..8 {
        value |= (host.device.load(address + i) as u64) << (i * 8);
    }
    value
}

/// Record the pre-value of a device-region store into the current row's
/// observation slot. The fast RAM path captures this inline; the device path
/// cannot (the bytes live in `JoltDevice`), so the helper does it.
fn record_device_store_pre(state: &mut GuestState, host: &HostContext, address: u64) {
    if state.obs_cursor.is_null() || state.obs_cursor >= state.obs_end {
        return;
    }
    let mut pre = 0u64;
    for i in 0..8 {
        pre |= (host.device.load(address + i) as u64) << (i * 8);
    }
    // SAFETY: the cursor is in bounds (checked above) and points at the slot
    // generated code is currently filling for this row.
    unsafe { (*state.obs_cursor).ram_pre = pre };
}

/// `Sd`/`VirtualStore*` slow path: device-region, unaligned, stack-canary, or
/// out-of-bounds addresses. `JoltDevice::store` also handles the
/// panic/termination bits.
pub extern "sysv64" fn slow_store_doubleword(
    state: *mut GuestState,
    address: u64,
    value: u64,
) -> u64 {
    let (state, host) = host_context(state);
    if !address.is_multiple_of(8) {
        return fail(
            state,
            host,
            format!("unaligned store_doubleword: {address:#x}"),
        );
    }
    if address >= RAM_START_ADDRESS {
        state.fault_addr = address;
        let offset = address - RAM_START_ADDRESS;
        if offset.wrapping_sub(state.canary_offset) < STACK_CANARY_SIZE {
            // Same condition and wording as `Mmu::assert_effective_address`,
            // which panics here (spec invariant 7 asymmetry).
            let stack_size = host.device.memory_layout.stack_size;
            return fail(
                state,
                host,
                format!(
                    "Stack overflow: attempted to store {address:#X}, which is in the stack \
                     canary region. Increase stack_size in MemoryConfig (currently \
                     {stack_size} bytes)."
                ),
            );
        }
        state.exit = ExitReason::FaultOutOfBounds as u64;
        return 0;
    }
    record_device_store_pre(state, host, address);
    for i in 0..8 {
        host.device.store(address + i, (value >> (i * 8)) as u8);
    }
    0
}

/// A virtual assert or a floor-addressed row's alignment check failed. The
/// interpreter panics; the x86 backend surfaces a `TraceError` (accepted
/// asymmetry, spec invariant 7).
pub extern "sysv64" fn assert_failed(state: *mut GuestState, code: u64, value: u64) -> u64 {
    let (state, host) = host_context(state);
    let message = match code {
        0 => format!("RAM access is not halfword aligned: {value:#x}"),
        1 => format!("RAM access is not word aligned: {value:#x}"),
        2 => "VirtualAssertLTE failed".to_string(),
        3 => format!("VirtualAssertEQ failed (lhs {value:x})"),
        4 => format!("VirtualAssertValidDiv0 failed (quotient {value:x})"),
        5 => format!("VirtualAssertValidUnsignedRemainder failed (remainder {value:x})"),
        6 => "VirtualAssertMulUNoOverflow failed".to_string(),
        _ => format!("assert {code} failed with value {value:x}"),
    };
    fail(state, host, message)
}

/// Guest-state view handed to registered inline advice builders.
///
/// The same `build_advice` functions the interpreter uses run unchanged here
/// (the seam's purpose): reads go to this backend's register array and memory
/// plane instead of the interpreter's `Cpu`.
struct GuestAdviceContext<'a> {
    state: &'a mut GuestState,
    host: &'a HostContext,
}

impl tracer::InlineAdviceContext for GuestAdviceContext<'_> {
    fn register(&self, index: usize) -> u64 {
        self.state.x[index]
    }

    fn load_doubleword(&mut self, address: u64) -> Option<u64> {
        if !address.is_multiple_of(8) {
            return None;
        }
        if address >= RAM_START_ADDRESS {
            let offset = address - RAM_START_ADDRESS;
            if offset + 8 > self.state.mem_size {
                return None;
            }
            // SAFETY: offset + 8 is within the mapped plane.
            let mut bytes = [0u8; 8];
            // SAFETY: offset + 8 <= mem_size, so the read stays inside the
            // mapped guest-memory plane.
            unsafe {
                (self.state.mem_base as *const u8)
                    .add(offset as usize)
                    .copy_to(bytes.as_mut_ptr(), 8);
            }
            return Some(u64::from_le_bytes(bytes));
        }
        let mut value = 0u64;
        for i in 0..8 {
            value |= (self.host.device.load(address + i) as u64) << (i * 8);
        }
        Some(value)
    }
}

/// Compute one source-instruction group's runtime advice into
/// `GuestState::advice_slots`, before any of the group's rows execute (the
/// interpreter computes these from the same pre-group register state).
pub extern "sysv64" fn advice_compute(state: *mut GuestState, job_index: u64) -> u64 {
    let (state, host) = host_context(state);
    // SAFETY: `advice_jobs` points at the compiled program's job table, which
    // outlives the run; generated code only passes indices it emitted.
    let job = unsafe { &*state.advice_jobs.add(job_index as usize) };
    match &job.compute {
        AdviceCompute::Div { code, rs1, rs2 } => {
            let x = state.x[*rs1 as usize] as i64;
            let y = state.x[*rs2 as usize] as i64;
            // One advice value per group; formulas mirror the tracer's
            // per-variant `trace` implementations (div.rs, rem.rs, …).
            let quotient = match code {
                // DIV: signed quotient
                0 => {
                    if y == 0 {
                        u64::MAX
                    } else if x == i64::MIN && y == -1 {
                        x as u64
                    } else {
                        (x / y) as u64
                    }
                }
                // REM: quotient magnitude
                1 => {
                    if y == 0 {
                        0
                    } else if x == i64::MIN && y == -1 {
                        1 << 63
                    } else {
                        (x / y).unsigned_abs()
                    }
                }
                // DIVW: 32-bit signed quotient, sign-extended
                // (except the overflow case, zero-extended 0x8000_0000)
                2 => {
                    let x = x as i32;
                    let y = y as i32;
                    if y == 0 {
                        u64::MAX
                    } else if y == -1 && x == i32::MIN {
                        0x8000_0000
                    } else {
                        (x / y) as u64
                    }
                }
                // REMW: 32-bit quotient magnitude, zero-extended
                3 => {
                    let x = x as i32;
                    let y = y as i32;
                    if y == 0 {
                        0
                    } else if y == -1 && x == i32::MIN {
                        1 << 31
                    } else {
                        u64::from((x / y).unsigned_abs())
                    }
                }
                // DIVU / REMU
                4 => (x as u64).checked_div(y as u64).unwrap_or(u64::MAX),
                // DIVUW / REMUW: 32-bit, zero-extended
                _ => {
                    let x = x as u32;
                    let y = y as u32;
                    x.checked_div(y).map_or(u64::from(u32::MAX), u64::from)
                }
            };
            state.advice_slots[0] = quotient;
            0
        }
        AdviceCompute::Inline {
            registration,
            operands,
        } => {
            let operands = *operands;
            let build = registration.build_advice;
            let values = {
                let mut context = GuestAdviceContext { state, host };
                build(operands, &mut context)
            };
            // Advice faults (invalid guest pointer in an operand register) are
            // surfaced as helper errors here — this backend has an error
            // channel, unlike the reference tracer, which panics.
            let values = match values {
                Ok(values) => values,
                Err(e) => {
                    return fail(
                        state,
                        host,
                        format!("inline {} advice failed: {e}", registration.name),
                    );
                }
            };
            // The provided count must match the group's VirtualAdvice rows
            // exactly — a short vector would leave stale slots to be read as
            // advice. The interpreter panics on the same mismatch
            // ("did not provide enough values" / "provided too many values");
            // this backend errors (spec invariant 7 asymmetry).
            let provided = values.as_ref().map_or(0, std::collections::VecDeque::len);
            if provided != job.advice_rows {
                return fail(
                    state,
                    host,
                    format!(
                        "inline {} provided {provided} advice values; its group reads {}",
                        registration.name, job.advice_rows
                    ),
                );
            }
            let Some(values) = values else { return 0 };
            for (slot, value) in values.into_iter().enumerate() {
                state.advice_slots[slot] = value;
            }
            0
        }
    }
}

/// `VirtualAdviceLoad`: read `num_bytes` (1/2/4/8) little-endian from the
/// tape, zero-filled; exhaustion is a helper error (the interpreter panics).
pub extern "sysv64" fn advice_read(state: *mut GuestState, num_bytes: u64) -> u64 {
    let (state, host) = host_context(state);
    let n = num_bytes as usize;
    if host.advice_cursor + n > host.advice_tape.len() {
        return fail(state, host, "Failed to read from advice tape".to_string());
    }
    let mut value = 0u64;
    for i in 0..n {
        value |= (host.advice_tape[host.advice_cursor + i] as u64) << (i * 8);
    }
    host.advice_cursor += n;
    value
}
