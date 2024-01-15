#![no_std]
#![no_main]

use core::arch::global_asm;
use core::panic::PanicInfo;

use sha3::{Digest, Keccak256};

global_asm!(include_str!("entry.s"));

#[no_mangle]
pub extern "C" fn main() {
    let mut hasher = Keccak256::new();
    let inputs = [5u8; 256];
    hasher.update(inputs);
    let _result = hasher.finalize();
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
