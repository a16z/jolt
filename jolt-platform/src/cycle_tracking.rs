//! Cycle tracking utility for emulated riscv cycles (both virtual and real)
//! Important for usage: often enough the rust compiler will optimize away
//! computations / other instructions when trying to profile cycles.
//! This will result in inaccurate measurements.
//! The easiest solution is to use the hint Module (https://doc.rust-lang.org/core/hint/index.html),
//! `black_box()` in particular can be used to prevent the compiler from moving your code.

// Constants to signal the emulator
pub const JOLT_CYCLE_TRACK_CALL_ID: u32 = 0xC7C1E; // "C Y C L E"
pub const JOLT_CYCLE_MARKER_START: u32 = 1;
pub const JOLT_CYCLE_MARKER_END: u32 = 2;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv_specific {
    use super::{JOLT_CYCLE_MARKER_END, JOLT_CYCLE_MARKER_START, JOLT_CYCLE_TRACK_CALL_ID};

    pub fn start_cycle_tracking(marker_id_str: &str) {
        let marker_id_ptr = marker_id_str.as_ptr() as usize;
        let marker_len = marker_id_str.len();
        emit_jolt_cycle_marker(
            marker_id_ptr as u32,
            marker_len as u32,
            JOLT_CYCLE_MARKER_START,
        );
    }

    pub fn end_cycle_tracking(marker_id_str: &str) {
        let marker_id_ptr = marker_id_str.as_ptr() as usize;
        let marker_len = marker_id_str.len();
        emit_jolt_cycle_marker(
            marker_id_ptr as u32,
            marker_len as u32,
            JOLT_CYCLE_MARKER_END,
        );
    }

    // Inserts a VirtualHostIO instruction directly into the compiled code.
    #[inline(always)]
    fn emit_jolt_cycle_marker(marker_id: u32, marker_len: u32, event_type: u32) {
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        unsafe {
            core::arch::asm!(
                ".insn i 0x5B, 2, x0, x0, 0", // VirtualHostIO (opcode=0x5B, funct3=2)
                in("x10") JOLT_CYCLE_TRACK_CALL_ID,
                in("x11") marker_id,
                in("x12") marker_len,
                in("x13") event_type,
                options(nostack, preserves_flags)
            );
        }
    }
}

#[allow(unused_variables)]
pub fn start_cycle_tracking(marker_id: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::start_cycle_tracking(marker_id);
}

#[allow(unused_variables)]
pub fn end_cycle_tracking(marker_id: &str) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    riscv_specific::end_cycle_tracking(marker_id);
}
