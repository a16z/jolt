#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use core::prelude::rust_2021::*;
#[macro_use]
extern crate core;
pub fn sha2(input: &[u8]) -> [u8; 32] {
    { jolt_inlines_sha2::Sha256::digest(input) }
}
use core::arch::global_asm;
static ALLOCATOR: jolt::BumpAllocator = jolt::BumpAllocator;
const _: () = {
    #[rustc_std_internal_symbol]
    unsafe fn __rust_alloc(size: usize, align: usize) -> *mut u8 {
        ::core::alloc::GlobalAlloc::alloc(
            &ALLOCATOR,
            ::core::alloc::Layout::from_size_align_unchecked(size, align),
        )
    }
    #[rustc_std_internal_symbol]
    unsafe fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize) -> () {
        ::core::alloc::GlobalAlloc::dealloc(
            &ALLOCATOR,
            ptr,
            ::core::alloc::Layout::from_size_align_unchecked(size, align),
        )
    }
    #[rustc_std_internal_symbol]
    unsafe fn __rust_realloc(
        ptr: *mut u8,
        size: usize,
        align: usize,
        new_size: usize,
    ) -> *mut u8 {
        ::core::alloc::GlobalAlloc::realloc(
            &ALLOCATOR,
            ptr,
            ::core::alloc::Layout::from_size_align_unchecked(size, align),
            new_size,
        )
    }
    #[rustc_std_internal_symbol]
    unsafe fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        ::core::alloc::GlobalAlloc::alloc_zeroed(
            &ALLOCATOR,
            ::core::alloc::Layout::from_size_align_unchecked(size, align),
        )
    }
};
#[no_mangle]
pub extern "C" fn main() {
    let mut offset = 0;
    let input_ptr = 2147467264u64 as *const u8;
    let input_slice = unsafe { core::slice::from_raw_parts(input_ptr, 4096usize) };
    let (input, input_slice) = jolt::postcard::take_from_bytes::<&[u8]>(input_slice)
        .unwrap();
    let to_return = (|| -> _ { { jolt_inlines_sha2::Sha256::digest(input) } })();
    let output_ptr = 2147471360u64 as *mut u8;
    let output_slice = unsafe { core::slice::from_raw_parts_mut(output_ptr, 4096usize) };
    jolt::postcard::to_slice::<[u8; 32]>(&to_return, output_slice).unwrap();
    unsafe {
        core::ptr::write_volatile(2147475464usize as *mut u8, 1);
    }
}
use core::panic::PanicInfo;
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe {
        core::ptr::write_volatile(2147475456u64 as *mut u8, 1);
    }
    loop {}
}
