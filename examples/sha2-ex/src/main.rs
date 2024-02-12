#![no_std]
#![no_main]

use core::arch::global_asm;
use core::panic::PanicInfo;

use sha2::{Sha256, Digest};

global_asm!(include_str!("entry.s"));

#[no_mangle]
pub extern "C" fn main() {
    const NUM_ITER: usize = 100;
    const TOTAL_BYTES: usize = NUM_ITER * 32;
    let inputs = [5u8; TOTAL_BYTES];
    let mut hasher = Sha256::new();
    hasher.update(inputs);
    let _result = hasher.finalize();
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
