// Constants for our custom cycle tracking ECALL
pub const JOLT_CYCLE_TRACK_ECALL_NUM: i32 = 0xC7C1E; // "C Y C L E" in hex

// Marker event types
pub const JOLT_CYCLE_MARKER_START: i32 = 1;
pub const JOLT_CYCLE_MARKER_END: i32 = 2;
pub const JOLT_CYCLE_MARKER_SNAPSHOT: i32 = 3;

/// Start tracking cycles for a labeled section of code
pub fn start_cycle_tracking(marker_id: &str) {
    let marker_id = marker_id.as_ptr() as i32;
    emit_jolt_cycle_marker_ecall(marker_id, JOLT_CYCLE_MARKER_START);
}

/// End tracking cycles for a labeled section of code (prints the result)
pub fn end_cycle_tracking(marker_id: &str) {
    let marker_id = marker_id.as_ptr() as i32;
    emit_jolt_cycle_marker_ecall(marker_id, JOLT_CYCLE_MARKER_END);
}

// Internal function to emit the ECALL
#[inline(always)]
fn emit_jolt_cycle_marker_ecall(marker_id: i32, event_type: i32) {
    unsafe {
        core::arch::asm!(
            ".word 0x00000073", // ECALL instruction opcode
            // Pass the ECALL number in x10
            in("x10") JOLT_CYCLE_TRACK_ECALL_NUM,
            // Pass the marker_id in x11
            in("x11") marker_id,
            // Pass the event_type in x12
            in("x12") event_type,
            options(nostack)
        );
    }
}