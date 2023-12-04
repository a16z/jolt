#![no_std]
#![no_main]

use core::arch::global_asm;
use core::panic::PanicInfo;

global_asm!(include_str!("entry.s"));

#[no_mangle]
pub extern "C" fn main() {
    fib(5);
}

fn fib(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
