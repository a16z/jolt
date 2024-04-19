use core::alloc::{GlobalAlloc, Layout};

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

static mut ALLOC_NEXT: usize = 0;

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn sys_alloc(size: usize, align: usize) -> *mut u8 {
    let mut next = unsafe { ALLOC_NEXT };

    if next == 0 {
        next = unsafe { (&_HEAP_PTR) as *const u8 as usize };
    }

    next = align_up(next, align);

    let ptr = next as *mut u8;
    next += size;

    unsafe { ALLOC_NEXT = next };
    ptr
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
