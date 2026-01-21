#![cfg_attr(feature = "guest", no_std)]
#![allow(unused_assignments, asm_sub_register)]

#[jolt::provable(guest_only, memory_size = 65536, max_trace_length = 65536)]
fn memory_ops() -> (i32, u32, i32, u32) {
    use core::arch::asm;

    let mut data: [u8; 8] = [0; 8];
    unsafe {
        let ptr = data.as_mut_ptr();

        // Store Byte (SB instruction)
        asm!(
            "sb {value}, 0({ptr})",
            ptr = in(reg) ptr,
            value = in(reg) 0x12,
        );

        // Load Byte Signed (LB instruction)
        let mut val_lb: i32 = 0;
        asm!(
            "lb {val}, 0({ptr})",
            ptr = in(reg) ptr,
            val = out(reg) val_lb,
        );

        // Load Byte Unsigned (LBU instruction)
        let mut val_lbu: u32 = 0;
        asm!(
            "lbu {val}, 1({ptr})",
            ptr = in(reg) ptr,
            val = out(reg) val_lbu,
        );

        // Store Halfword (SH instruction)
        asm!(
            "sh {value}, 2({ptr})",
            ptr = in(reg) ptr,
            value = in(reg) 0x3456,
        );

        // Load Halfword Signed (LH instruction)
        let mut val_lh: i32 = 0;
        asm!(
            "lh {val}, 2({ptr})",
            ptr = in(reg) ptr,
            val = out(reg) val_lh,
        );

        // Load Halfword Unsigned (LHU instruction)
        let mut val_lhu: u32 = 0;
        asm!(
            "lhu {val}, 4({ptr})",
            ptr = in(reg) ptr,
            val = out(reg) val_lhu,
        );

        // Return these values so that the load instructions
        // don't get optimized away
        (val_lb, val_lbu, val_lh, val_lhu)
    }
}
