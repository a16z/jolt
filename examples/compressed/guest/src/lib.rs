#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

// Helper functions that demonstrate compressed instruction patterns
fn immediate_ops(x: u32) -> u32 {
    let y = 5; // C.LI
    let z = x + 1; // C.ADDI
    let w = x + 15; // C.ADDI
    y + z + w
}

fn register_ops(a: u32, b: u32) -> u32 {
    let c = a; // C.MV
    let d = a + b; // C.ADD
    let e = d - a; // C.SUB
    c + e
}

fn small_constants() -> u32 {
    let a = 0; // C.LI
    let b = 1; // C.LI
    let c = 31; // C.LI (max 6-bit signed immediate)
    let d = 16; // C.LI
    a + b + c + d
}

fn memory_ops(data: &mut [u32; 8]) -> u32 {
    data[0] = 42; // C.SW
    data[1] = data[0] + 1; // C.LW + C.ADDI + C.SW
    let val = data[2]; // C.LW
    data[3] = val; // C.SW
    data[0] + data[1]
}

fn shift_ops(x: u32) -> u32 {
    let a = x << 1; // C.SLLI
    let b = a >> 1; // C.SRLI
    let c = (a as i32) >> 1; // C.SRAI
    a + b + (c as u32)
}

fn control_flow(x: u32) -> u32 {
    if x == 0 {
        // C.BEQZ
        return 1;
    }

    if x != 0 {
        // C.BNEZ
        x + 2
    } else {
        0
    }
}

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn demo(input: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("demo");

    // Test various compressed instruction patterns
    let mut test_data = [0u32; 8];

    // Call all our test functions to ensure they generate instructions
    let imm_result = black_box(immediate_ops(input));
    let reg_result = black_box(register_ops(input, imm_result));
    let const_result = black_box(small_constants());
    let mem_result = black_box(memory_ops(&mut test_data));
    let shift_result = black_box(shift_ops(input));
    let ctrl_result = black_box(control_flow(input));

    // Combine results to ensure all functions are used
    black_box(imm_result + reg_result + const_result + mem_result + shift_result + ctrl_result);

    end_cycle_tracking("demo");
    0
}
