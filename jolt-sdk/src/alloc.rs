use core::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct BumpAllocator;

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_alloc(layout.size(), layout.align())
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

extern "C" {
    static _HEAP_PTR: u8;
}

static ALLOC_NEXT: AtomicUsize = AtomicUsize::new(0);

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn sys_alloc(size: usize, align: usize) -> *mut u8 {
    let mut next = get_next_alloc();

    if next == 0 {
        next = get_heap_ptr();
    }

    next = align_up(next, align);

    let ptr = next as *mut u8;
    next += size;

    set_next_alloc(next);
    ptr
}

fn get_heap_ptr() -> usize {
    unsafe { (&_HEAP_PTR) as *const u8 as usize }
}

fn get_next_alloc() -> usize {
    ALLOC_NEXT.load(Ordering::Relaxed)
}

fn set_next_alloc(next: usize) {
    ALLOC_NEXT.store(next, Ordering::Relaxed)
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
