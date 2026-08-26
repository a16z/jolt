//! O(1) size-class guest heap allocator.
//!
//! Replaces `linked_list_allocator` (first-fit) as the no_std guest allocator.
//! In a zkVM every executed RISC-V instruction is a proving-cost cycle, and a
//! first-fit free-list walk is O(free blocks) per call. On an allocation-heavy
//! real workload (stateless Ethereum block validation: revm + MPT over ~1.5 GiB
//! heap), PC-sampling attributed **85.7% of all executed instructions** to the
//! linked-list allocator (58.6% `allocate_first_fit` + 27.1% `deallocate`);
//! swapping in this allocator cut total trace length ~6× (29.8B → 4.9B cycles).
//!
//! Design — every operation is a few dozen instructions, no searching:
//! - Allocations round up to a power-of-two size class (min 8 B).
//! - Each class keeps a singly-linked free list; freed blocks store the `next`
//!   pointer in their first word. Alloc pops the head, dealloc pushes.
//! - Class miss → bump the arena cursor (blocks are class-size aligned, so any
//!   `align <= size` request is satisfied; larger `align` rounds the class up).
//! - `realloc` grows the most recent bump allocation in place (Vec doubling
//!   hits this constantly); otherwise allocate-copy-free. Same-class or
//!   shrinking reallocs return the block unchanged.
//! - No splitting or coalescing. The cost is bounded internal fragmentation
//!   (< 2× per live block) and per-class high-water free lists — a deliberate
//!   memory-for-cycles trade appropriate for single-run guest heaps.
//!
//! Single-hart guests only: state is plain statics, no locking.

use core::alloc::Layout;
use core::ptr;

/// Size classes: 8, 16, 32, …, 2^(MIN_SHIFT + NUM_CLASSES − 1) bytes.
/// The top class (8 GiB) comfortably exceeds any single allocation a guest
/// heap can hold; larger requests simply fail.
const MIN_SHIFT: u32 = 3;
const NUM_CLASSES: usize = 31;

struct Arena {
    cursor: usize,
    end: usize,
    /// Head of the free list per class; freed blocks store `next` in word 0.
    free: [usize; NUM_CLASSES],
}

// Safety: guests are single-hart and interrupts do not preempt allocation;
// host-side unit tests serialize access.
static mut ARENA: Arena = Arena {
    cursor: 0,
    end: 0,
    free: [0; NUM_CLASSES],
};

/// Size class index for a layout: ceil(log2(max(size, align, 8))) − 3.
#[inline]
fn class_of(layout: Layout) -> u32 {
    let needed = layout.size().max(layout.align()).max(8);
    let shift = usize::BITS - (needed - 1).leading_zeros(); // ceil(log2)
    shift - MIN_SHIFT
}

/// Initialize the arena over `[heap_start, heap_start + heap_size)`.
pub fn init(heap_start: usize, heap_size: usize) {
    unsafe {
        let a = &mut *ptr::addr_of_mut!(ARENA);
        a.cursor = heap_start;
        a.end = heap_start + heap_size;
        a.free = [0; NUM_CLASSES];
    }
}

/// O(1): pop the class free list, else bump the cursor. Null on exhaustion.
pub fn alloc(layout: Layout) -> *mut u8 {
    let class = class_of(layout) as usize;
    if class >= NUM_CLASSES {
        return ptr::null_mut();
    }
    let size = 1usize << (class as u32 + MIN_SHIFT);
    unsafe {
        let a = &mut *ptr::addr_of_mut!(ARENA);

        // Fast path: recycle from the class free list. Blocks grown in place
        // (see `realloc`) are only guaranteed their ORIGINAL class alignment,
        // so re-check the requested alignment; on mismatch fall through to the
        // bump path (the head stays put for smaller-alignment requests).
        let head = a.free[class];
        if head != 0 && head & (layout.align() - 1) == 0 {
            a.free[class] = *(head as *const usize);
            return head as *mut u8;
        }

        // Bump path: blocks are class-size aligned.
        let start = (a.cursor + size - 1) & !(size - 1);
        let Some(new_cursor) = start.checked_add(size) else {
            return ptr::null_mut();
        };
        if new_cursor > a.end {
            return ptr::null_mut();
        }
        a.cursor = new_cursor;
        start as *mut u8
    }
}

/// O(1): push onto the class free list.
pub fn dealloc(ptr_in: *mut u8, layout: Layout) {
    if ptr_in.is_null() {
        return;
    }
    let class = class_of(layout) as usize;
    if class >= NUM_CLASSES {
        return;
    }
    unsafe {
        let a = &mut *ptr::addr_of_mut!(ARENA);
        *(ptr_in as *mut usize) = a.free[class];
        a.free[class] = ptr_in as usize;
    }
}

/// Same-class and shrinking reallocs are free; growth extends the newest bump
/// block in place when possible, else allocate-copy-free.
///
/// `ptr_in` must be null or a live block previously returned by this allocator
/// for `old_layout` — the standard `GlobalAlloc::realloc` contract.
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "installed as a safe fn pointer in zeroos MemoryOps; callers uphold the GlobalAlloc realloc contract"
)]
pub fn realloc(ptr_in: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
    if ptr_in.is_null() {
        return match Layout::from_size_align(new_size, old_layout.align()) {
            Ok(l) => alloc(l),
            Err(_) => ptr::null_mut(),
        };
    }
    if new_size == 0 {
        dealloc(ptr_in, old_layout);
        return ptr::null_mut();
    }
    let Ok(new_layout) = Layout::from_size_align(new_size, old_layout.align()) else {
        return ptr::null_mut();
    };

    let old_class = class_of(old_layout);
    let new_class = class_of(new_layout);
    if new_class as usize >= NUM_CLASSES {
        return ptr::null_mut();
    }

    // Same class (or shrink) → the existing block already fits. Shrinks keep
    // the larger footprint; the tail is orphaned when the block is later freed
    // into the smaller class (rare — `shrink_to_fit` — and bounded).
    if new_class <= old_class {
        return ptr_in;
    }

    unsafe {
        let a = &mut *ptr::addr_of_mut!(ARENA);
        // In-place growth: this block is the most recent bump allocation.
        let old_size = 1usize << (old_class + MIN_SHIFT);
        let grown_size = 1usize << (new_class + MIN_SHIFT);
        let addr = ptr_in as usize;
        if addr + old_size == a.cursor
            && addr & (new_layout.align() - 1) == 0
            && addr + grown_size <= a.end
        {
            a.cursor = addr + grown_size;
            return ptr_in;
        }
    }

    let new_ptr = alloc(new_layout);
    if !new_ptr.is_null() {
        let copy = old_layout.size().min(new_size);
        unsafe {
            ptr::copy_nonoverlapping(ptr_in, new_ptr, copy);
        }
        dealloc(ptr_in, old_layout);
    }
    new_ptr
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc as alloc_crate;
    use alloc_crate::vec;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    /// Single test — the arena is a shared static and cargo runs tests in
    /// parallel threads, so all scenarios run sequentially here.
    #[test]
    fn size_class_allocator_end_to_end() {
        let heap = vec![0u8; 1 << 20].leak();
        // Skew the base so class alignment is actually exercised.
        let base = (heap.as_ptr() as usize + 63) & !63;
        init(base, (1 << 20) - 128);

        // Basic alloc: aligned to class size, distinct, writable.
        let a = alloc(layout(24, 8)); // class 32
        let b = alloc(layout(32, 8)); // class 32
        assert!(!a.is_null() && !b.is_null() && a != b);
        assert_eq!(a as usize % 32, 0);
        unsafe {
            ptr::write_bytes(a, 0xAA, 24);
            ptr::write_bytes(b, 0xBB, 32);
        }

        // Free-list reuse is LIFO within a class.
        dealloc(a, layout(24, 8));
        let a2 = alloc(layout(25, 4)); // same class 32
        assert_eq!(a, a2);

        // Alignment beyond size bumps the class.
        let c = alloc(layout(8, 64));
        assert_eq!(c as usize % 64, 0);

        // Realloc within a class is a no-op pointer-wise.
        let d = alloc(layout(20, 8)); // class 32
        let d2 = realloc(d, layout(20, 8), 30);
        assert_eq!(d, d2);

        // Newest bump block grows in place.
        let e = alloc(layout(64, 8));
        let e2 = realloc(e, layout(64, 8), 128);
        assert_eq!(e, e2);

        // Non-newest growth copies content.
        let f = alloc(layout(16, 8));
        unsafe { ptr::write_bytes(f, 0xCD, 16) };
        let _g = alloc(layout(16, 8)); // f is no longer newest
        let f2 = realloc(f, layout(16, 8), 64);
        assert_ne!(f, f2);
        for i in 0..16 {
            assert_eq!(unsafe { *f2.add(i) }, 0xCD);
        }

        // Shrink keeps the pointer.
        let h = alloc(layout(256, 8));
        assert_eq!(realloc(h, layout(256, 8), 16), h);

        // Oversize and exhaustion return null.
        assert!(alloc(layout(1 << 40, 8)).is_null());
        assert!(alloc(layout(1 << 21, 8)).is_null()); // larger than arena

        // Churn: repeated alloc/free cycles stay within the arena and recycle.
        let mut ptrs = vec![];
        for round in 0..50 {
            for i in 0..64 {
                let p = alloc(layout(8 + (i * 7) % 500, 8));
                assert!(!p.is_null(), "round {round} alloc {i}");
                ptrs.push((p, 8 + (i * 7) % 500));
            }
            for (p, sz) in ptrs.drain(..) {
                dealloc(p, layout(sz, 8));
            }
        }
    }
}
