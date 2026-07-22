//! C malloc/free implementation for guest environment
//!
//! Provides standard C library memory allocation functions (malloc, free, realloc, calloc)
//! that C libraries like secp256k1 can link against. These functions use Rust's global
//! allocator as the underlying memory source.
extern crate alloc;

use core::{alloc::Layout, cmp, ffi::c_void, mem, ptr};

/// Alignment guaranteed to C code, matching `max_align_t` behavior.
/// RISC-V 64-bit: 16 bytes, 32-bit: 8 bytes. Over-aligning is safe.
const DEFAULT_ALIGN: usize = if cfg!(target_pointer_width = "64") {
    16
} else {
    8
};

/// Magic value used to validate that a pointer passed to free/realloc
/// was actually allocated by this shim.
const HEADER_MAGIC: usize = 0x4A4F_4C54; // "JOLT"

/// Allocation metadata stored before each allocated block.
/// Size must be a multiple of `DEFAULT_ALIGN` to keep payload properly aligned.
#[repr(C)]
#[cfg_attr(target_pointer_width = "64", repr(align(16)))]
#[cfg_attr(target_pointer_width = "32", repr(align(8)))]
struct AllocHeader {
    magic: usize,
    payload_size: usize,
}

/// Compile-time check: header size must be aligned to C requirements.
const _: () = {
    let header_aligned = (mem::size_of::<AllocHeader>() % DEFAULT_ALIGN) == 0;
    ["AllocHeader size must be multiple of DEFAULT_ALIGN"][!header_aligned as usize];
};

/// Creates memory layout for allocation: [AllocHeader][payload]
/// Both header and payload are aligned to C ABI requirements.
#[inline]
fn alloc_layout(payload_size: usize) -> Option<Layout> {
    let total_size = mem::size_of::<AllocHeader>().checked_add(payload_size)?;
    Layout::from_size_align(total_size, DEFAULT_ALIGN).ok()
}

/// Reads and validates the header for a payload pointer returned by malloc/calloc.
/// Returns None if the pointer is invalid (bad magic or failed layout reconstruction).
#[inline]
unsafe fn read_header(payload_ptr: *mut u8) -> Option<(Layout, *mut u8, usize)> {
    let header_ptr = payload_ptr.sub(mem::size_of::<AllocHeader>()) as *mut AllocHeader;
    let header = header_ptr.read();

    if header.magic != HEADER_MAGIC {
        return None;
    }

    let block_ptr = payload_ptr.sub(mem::size_of::<AllocHeader>());
    let layout = alloc_layout(header.payload_size)?;
    Some((layout, block_ptr, header.payload_size))
}

/// Standard C malloc - allocate memory block of given size.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    // Ensure non-zero allocation size for C compatibility
    let payload_size = size.max(1);
    let Some(layout) = alloc_layout(payload_size) else {
        return core::ptr::null_mut();
    };

    let block_ptr = alloc::alloc::alloc(layout);
    if block_ptr.is_null() {
        return core::ptr::null_mut();
    }

    // Write header with magic and payload size
    (block_ptr as *mut AllocHeader).write(AllocHeader {
        magic: HEADER_MAGIC,
        payload_size,
    });

    // Return pointer to payload (after header)
    block_ptr.add(mem::size_of::<AllocHeader>()) as *mut c_void
}

/// Standard C free - deallocate memory block.
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let payload_ptr = ptr as *mut u8;
    let Some((layout, block_ptr, _)) = read_header(payload_ptr) else {
        return;
    };

    // Poison the magic to prevent double-free
    let header_ptr = block_ptr as *mut AllocHeader;
    (*header_ptr).magic = 0;

    alloc::alloc::dealloc(block_ptr, layout);
}

/// Standard C realloc - resize memory block.
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    if ptr.is_null() {
        return malloc(new_size);
    }
    if new_size == 0 {
        free(ptr);
        return core::ptr::null_mut();
    }

    let old_payload_ptr = ptr as *mut u8;
    let Some((_, _, old_payload_size)) = read_header(old_payload_ptr) else {
        return core::ptr::null_mut();
    };

    let new_ptr = malloc(new_size);
    if new_ptr.is_null() {
        return core::ptr::null_mut();
    }

    // Copy existing data (up to smaller of old/new sizes)
    ptr::copy_nonoverlapping(
        old_payload_ptr,
        new_ptr as *mut u8,
        cmp::min(old_payload_size, new_size),
    );
    free(ptr);
    new_ptr
}

/// Standard C calloc - allocate zero-initialized memory for array.
#[no_mangle]
pub unsafe extern "C" fn calloc(elem_count: usize, elem_size: usize) -> *mut c_void {
    match elem_count.checked_mul(elem_size) {
        Some(total_size) if total_size > 0 => {
            // Use allocator's zero-initialization if available
            let Some(layout) = alloc_layout(total_size) else {
                return core::ptr::null_mut();
            };
            let block_ptr = alloc::alloc::alloc_zeroed(layout);
            if block_ptr.is_null() {
                return core::ptr::null_mut();
            }

            // Write header with magic and total payload size
            (block_ptr as *mut AllocHeader).write(AllocHeader {
                magic: HEADER_MAGIC,
                payload_size: total_size,
            });

            // Return pointer to zeroed payload
            block_ptr.add(mem::size_of::<AllocHeader>()) as *mut c_void
        }
        _ => core::ptr::null_mut(), // Overflow or zero size
    }
}
