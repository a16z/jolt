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

use super::state::{ExitReason, GuestState, HostContext};

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
        _ => format!("assert {code} failed with value {value:x}"),
    };
    fail(state, host, message)
}
