#![cfg_attr(not(feature = "host"), no_std)]

extern crate jolt_sdk_macros;

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
};

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub mod host_utils;
#[cfg(feature = "host")]
pub use host_utils::*;

pub struct BumpAllocator {
    offset: UnsafeCell<usize>,
}

unsafe impl Sync for BumpAllocator {}

extern "C" {
    static _HEAP_PTR: u8;
}

fn heap_start() -> usize {
    unsafe { _HEAP_PTR as *const u8 as usize }
}

impl BumpAllocator {
    pub const fn new() -> Self {
        Self {
            offset: UnsafeCell::new(0),
        }
    }

    pub fn free_memory(&self) -> usize {
        heap_start() + (self.offset.get() as usize)
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let alloc_start = align_up(self.free_memory(), layout.align());
        let alloc_end = alloc_start + layout.size();
        *self.offset.get() = alloc_end - self.free_memory();

        alloc_start as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
