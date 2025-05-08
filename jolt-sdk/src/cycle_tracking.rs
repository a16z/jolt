// Constants to signal the emulator
pub const JOLT_CYCLE_TRACK_ECALL_NUM: i32 = 0xC7C1E; // "C Y C L E"
pub const JOLT_CYCLE_MARKER_START: i32 = 1;
pub const JOLT_CYCLE_MARKER_END: i32 = 2;

/// Create a cycle tracking label
pub fn start_cycle_tracking(marker_id: &str) {
    let marker_id = marker_id.as_ptr() as i32;
    emit_jolt_cycle_marker_ecall(marker_id, JOLT_CYCLE_MARKER_START);
}

/// End a cycle tracking label
pub fn end_cycle_tracking(marker_id: &str) {
    let marker_id = marker_id.as_ptr() as i32;
    emit_jolt_cycle_marker_ecall(marker_id, JOLT_CYCLE_MARKER_END);
}

// inserts an ECALL directly into the compiled code
#[inline(always)]
fn emit_jolt_cycle_marker_ecall(marker_id: i32, event_type: i32) {
    unsafe {
        core::arch::asm!(
            ".word 0x00000073", // ECALL opcode
            in("x10") JOLT_CYCLE_TRACK_ECALL_NUM, //
            in("x11") marker_id, // we store ptr address of the label &str to recover it during emulation
            in("x12") event_type, // either start or end
            options(nostack)
        );
    }
}
