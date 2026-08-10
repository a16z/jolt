//! `extern "C"` helpers called from generated code.
//!
//! ABI: sysv64, first argument is always `*mut GuestState`. Pinned registers
//! (r12–r15) are callee-saved, so helpers preserve them for free.
//!
//! Error protocol: a failing helper writes `GuestState::exit` (and
//! `HostContext::helper_error` / `fault_addr` as appropriate); generated code
//! tests `exit` after each helper call and jumps to the exit stub when
//! nonzero. Return values travel in rax.

use common::constants::RAM_START_ADDRESS;

use super::state::{AdviceCompute, ExitReason, GuestState, HostContext};

// jolt-platform host-IO ABI (mirrors tracer/src/emulator/cpu.rs handlers).
use jolt_platform::{
    JOLT_ADVICE_WRITE_CALL_ID, JOLT_CYCLE_MARKER_END, JOLT_CYCLE_MARKER_START,
    JOLT_CYCLE_TRACK_CALL_ID, JOLT_PRINT_CALL_ID, JOLT_PRINT_LINE, JOLT_PRINT_STRING,
};

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

/// Read one guest byte, routing RAM to the plane and lower addresses to the
/// device (mirrors `Mmu::load`).
fn read_guest_byte(state: &GuestState, host: &HostContext, address: u64) -> u8 {
    if address >= RAM_START_ADDRESS {
        let offset = address - RAM_START_ADDRESS;
        if offset < state.mem_size {
            // SAFETY: offset bounds-checked against the plane size.
            return unsafe { (state.mem_base as *const u8).add(offset as usize).read() };
        }
        return 0;
    }
    host.device.load(address)
}

/// `VirtualHostIO`: call id in a0 (x10), args in a1–a3 (x11–x13).
/// No rd write, no RAM-access row (matches the interpreter).
// Guest prints go to stdout by definition (mirrors the interpreter's
// handle_jolt_print).
#[expect(clippy::print_stdout)]
pub extern "sysv64" fn host_io(state: *mut GuestState) -> u64 {
    let (state, host) = host_context(state);
    let call_id = state.x[10] as u32;
    if call_id == JOLT_ADVICE_WRITE_CALL_ID {
        let ptr = state.x[11];
        let len = state.x[12];
        for i in 0..len {
            let byte = read_guest_byte(state, host, ptr.wrapping_add(i));
            host.advice_tape.push(byte);
        }
        return 0;
    }
    let ptr = state.x[11] as u32 as u64;
    let len = state.x[12] as u32 as usize;
    let event = state.x[13] as u32;
    match call_id {
        JOLT_CYCLE_TRACK_CALL_ID => {
            // Diagnostics only in the interpreter (logged cycle counts);
            // reproduce the validation, skip the logging.
            if event != JOLT_CYCLE_MARKER_START && event != JOLT_CYCLE_MARKER_END {
                return fail(state, host, format!("unknown cycle marker event {event}"));
            }
            0
        }
        JOLT_PRINT_CALL_ID => {
            let mut bytes = Vec::with_capacity(len);
            for i in 0..len as u64 {
                bytes.push(read_guest_byte(state, host, ptr.wrapping_add(i)));
            }
            let text = String::from_utf8_lossy(&bytes);
            match event {
                JOLT_PRINT_STRING => print!("{text}"),
                JOLT_PRINT_LINE => println!("{text}"),
                other => return fail(state, host, format!("unknown print event {other}")),
            }
            0
        }
        // Unknown call ids are silently ignored by the interpreter.
        _ => 0,
    }
}

/// `Ld` slow path: device-region, unaligned, or out-of-bounds effective
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

/// `Sd` slow path: device-region, unaligned, or out-of-bounds effective
/// addresses. `JoltDevice::store` also handles the panic/termination bits.
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
        state.exit = ExitReason::FaultOutOfBounds as u64;
        return 0;
    }
    record_device_store_pre(state, host, address);
    for i in 0..8 {
        host.device.store(address + i, (value >> (i * 8)) as u8);
    }
    0
}

/// A virtual assert failed. The interpreter panics; the x86 backend surfaces
/// a `TraceError` (accepted asymmetry, spec invariant 7).
pub extern "sysv64" fn assert_failed(state: *mut GuestState, code: u64, value: u64) -> u64 {
    let (state, host) = host_context(state);
    let message = match code {
        0 => format!("RAM access (LH or LHU) is not halfword aligned: {value:x}"),
        1 => format!("RAM access (LW or LWU) is not word aligned: {value:x}"),
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
            let (a, b) = match code {
                // DIV / REM: [quotient, |remainder|]
                0 => {
                    if y == 0 {
                        (u64::MAX, x.unsigned_abs())
                    } else if x == i64::MIN && y == -1 {
                        (x as u64, 0)
                    } else {
                        ((x / y) as u64, (x % y).unsigned_abs())
                    }
                }
                // DIVW / REMW: 32-bit, quotient sign-extended
                1 => {
                    let x = x as i32;
                    let y = y as i32;
                    let (q, r) = if y == 0 {
                        (-1i32, x.unsigned_abs())
                    } else if y == -1 && x == i32::MIN {
                        (i32::MIN, 0)
                    } else {
                        (x / y, (x % y).unsigned_abs())
                    };
                    (q as u64, r as u64)
                }
                // DIVU / REMU: [quotient]
                2 => ((x as u64).checked_div(y as u64).unwrap_or(u64::MAX), 0),
                // DIVUW / REMUW: 32-bit, zero-extended
                _ => {
                    let x = x as u32;
                    let y = y as u32;
                    (x.checked_div(y).map_or(u64::from(u32::MAX), u64::from), 0)
                }
            };
            state.advice_slots[0] = a;
            state.advice_slots[1] = b;
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

/// `VirtualAdviceLen`: bytes left on the advice tape.
pub extern "sysv64" fn advice_remaining(state: *mut GuestState) -> u64 {
    let (_state, host) = host_context(state);
    host.advice_tape.len().saturating_sub(host.advice_cursor) as u64
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
