//! Unified-memory buffers. On Apple Silicon the CPU and GPU share one
//! physical memory pool, so W1's whole buffer story is: every `MTLBuffer` is
//! `storageModeShared`, and host allocations are wrapped in place with
//! `newBufferWithBytesNoCopy` whenever they qualify — never a private+blit
//! staging path.
//!
//! # No-copy eligibility
//!
//! `newBufferWithBytesNoCopy` requires a page-aligned base pointer and a
//! page-multiple length, so wrapping a slice whose byte length is not a page
//! multiple means handing Metal the tail of the final page. That is sound
//! exactly when the backing allocation itself has page granularity, which
//! [`MetalContext::wrap_slice_nocopy`] establishes one of two ways:
//!
//! - the slice's byte length already is a page multiple (nothing beyond the
//!   slice is exposed), or
//! - the byte length is ≥ 32 KiB: macOS `malloc` serves large allocations
//!   (the `MALLOC_LARGE` zone, ≥ 32 KiB) via `vm_allocate`, page-aligned
//!   AND rounded up to whole pages, so the wrapped tail stays inside the
//!   allocation.
//!
//! Alignment is always checked at runtime, never assumed. Anything else
//! falls back to an allocate+copy shared buffer, tagged
//! [`DeviceBuffer::was_copied`] so callers can count copies.
//!
//! [`PageAlignedVec`] (`posix_memalign(16384)`, capacity rounded to page
//! multiples) is the kernel-tier-owned-table allocation: no-copy wrappable
//! at ANY length via [`PageAlignedVec::device_buffer`], which uses its
//! capacity guarantee instead of the size heuristic.
//!
//! # Ownership
//!
//! A no-copy `MTLBuffer` borrows its backing memory, so [`DeviceBuffer<'a>`]
//! carries the source borrow and [`ComputePass`](super::runtime::ComputePass)
//! requires every dispatched buffer to outlive the pass — a wrapped buffer
//! cannot outlive its backing slice, and the backing cannot be dropped (or,
//! for mutable wraps, aliased) while the GPU may touch it. All GPU work in
//! this tier is synchronous (`run` waits), so nothing escapes the borrow.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::error::MetalError;
use super::runtime::MetalContext;

/// Apple Silicon page size. Asserted against `sysconf` at first use in
/// [`PageAlignedVec`] construction (defensive: this crate hard-codes it in
/// eligibility checks and capacity rounding).
pub const PAGE_SIZE: usize = 16384;

/// macOS malloc's large-allocation threshold: at and above this size,
/// allocations are `vm_allocate`d with page granularity (the invariant the
/// no-copy size heuristic relies on).
const MALLOC_LARGE_THRESHOLD: usize = 32 * 1024;

const fn round_up_to_page(bytes: usize) -> usize {
    bytes.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

/// A `storageModeShared` `MTLBuffer`, either wrapping host memory in place
/// (no-copy) or device-allocated. `'a` is the backing borrow for wrapped
/// buffers (`'static` for device-allocated ones).
pub struct DeviceBuffer<'a> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len_bytes: usize,
    copied: bool,
    _backing: PhantomData<&'a [u8]>,
}

impl DeviceBuffer<'_> {
    pub(super) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// True when construction fell back to allocate+copy (the wrapped slice
    /// was not no-copy eligible).
    pub fn was_copied(&self) -> bool {
        self.copied
    }

    /// Logical length (the wrapped slice's byte length, not the
    /// page-rounded `MTLBuffer` length).
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Copy device-visible contents out (for device-allocated result
    /// buffers). Callers must only read after the producing pass's `run`
    /// returned — all work in this tier is synchronous.
    pub fn copy_to_u32s(&self, out: &mut [u32]) {
        assert!(
            size_of_val(out) <= self.len_bytes,
            "read past logical buffer length"
        );
        // SAFETY: `contents()` is the shared-storage base pointer, valid for
        // `len_bytes` ≥ the requested read; `out` is a disjoint &mut.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.contents().as_ptr().cast::<u32>().cast_const(),
                out.as_mut_ptr(),
                out.len(),
            );
        }
    }
}

impl MetalContext {
    /// Device-allocated shared buffer of `len_u32` words (uninitialized).
    pub fn alloc_u32s(&self, len_u32: usize) -> Result<DeviceBuffer<'static>, MetalError> {
        let len_bytes = len_u32 * 4;
        let raw = self
            .device()
            .newBufferWithLength_options(len_bytes.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Alloc { bytes: len_bytes })?;
        Ok(DeviceBuffer {
            raw,
            len_bytes,
            copied: false,
            _backing: PhantomData,
        })
    }

    /// Wrap `slice` in place if it is no-copy eligible (see module docs);
    /// `None` means the caller must use [`wrap_slice`](Self::wrap_slice)'s
    /// copy fallback (or a [`PageAlignedVec`]).
    pub fn wrap_slice_nocopy<'a, T: Copy>(&self, slice: &'a [T]) -> Option<DeviceBuffer<'a>> {
        let len_bytes = size_of_val(slice);
        let aligned = (slice.as_ptr() as usize).is_multiple_of(PAGE_SIZE);
        let page_granular =
            len_bytes.is_multiple_of(PAGE_SIZE) || len_bytes >= MALLOC_LARGE_THRESHOLD;
        if len_bytes == 0 || !aligned || !page_granular {
            return None;
        }
        // SAFETY: base is page-aligned and the page-rounded range stays
        // inside the backing allocation (page-multiple length exposes
        // nothing beyond the slice; otherwise the ≥ 32 KiB length puts the
        // allocation in malloc's page-granular large zone — module docs).
        // The GPU never writes through a shared `const` binding, and
        // `DeviceBuffer<'a>` keeps the borrow alive for the buffer's life.
        unsafe { self.wrap_raw_nocopy(NonNull::from(&slice[0]).cast(), len_bytes) }
    }

    /// Mutable no-copy wrap for kernel outputs: same eligibility as
    /// [`wrap_slice_nocopy`](Self::wrap_slice_nocopy), and the exclusive
    /// borrow guarantees the host neither reads nor writes the slice while
    /// the buffer (and thus any GPU write) is live.
    pub fn wrap_slice_mut_nocopy<'a, T: Copy>(
        &self,
        slice: &'a mut [T],
    ) -> Option<DeviceBuffer<'a>> {
        let len_bytes = size_of_val(slice);
        let aligned = (slice.as_mut_ptr() as usize).is_multiple_of(PAGE_SIZE);
        let page_granular =
            len_bytes.is_multiple_of(PAGE_SIZE) || len_bytes >= MALLOC_LARGE_THRESHOLD;
        if len_bytes == 0 || !aligned || !page_granular {
            return None;
        }
        // SAFETY: as in `wrap_slice_nocopy`; the &mut borrow moved into
        // `DeviceBuffer<'a>` makes host access impossible while GPU writes
        // may occur.
        unsafe { self.wrap_raw_nocopy(NonNull::from(&mut slice[0]).cast(), len_bytes) }
    }

    /// Wrap without copying when eligible, else allocate a shared buffer and
    /// copy (`was_copied() == true`).
    pub fn wrap_slice<'a, T: Copy>(&self, slice: &'a [T]) -> Result<DeviceBuffer<'a>, MetalError> {
        if let Some(buffer) = self.wrap_slice_nocopy(slice) {
            return Ok(buffer);
        }
        let len_bytes = size_of_val(slice);
        if len_bytes == 0 {
            return self.alloc_u32s(0);
        }
        let src: NonNull<c_void> = NonNull::from(&slice[0]).cast();
        // SAFETY: `src` points at `len_bytes` readable bytes; Metal copies
        // them into the new buffer during this call.
        let raw = unsafe {
            self.device().newBufferWithBytes_length_options(
                src,
                len_bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Alloc { bytes: len_bytes })?;
        Ok(DeviceBuffer {
            raw,
            len_bytes,
            copied: true,
            _backing: PhantomData,
        })
    }

    /// # Safety
    ///
    /// `ptr` must be page-aligned and `[ptr, ptr + round_up_to_page(len_bytes))`
    /// must lie inside one live allocation that outlives `'a`.
    unsafe fn wrap_raw_nocopy<'a>(
        &self,
        ptr: NonNull<c_void>,
        len_bytes: usize,
    ) -> Option<DeviceBuffer<'a>> {
        // SAFETY: caller contract (page-aligned base, page-rounded range
        // inside a live allocation for 'a); deallocator None means Metal
        // borrows rather than adopts the memory.
        let raw = unsafe {
            self.device()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    round_up_to_page(len_bytes),
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }?;
        Some(DeviceBuffer {
            raw,
            len_bytes,
            copied: false,
            _backing: PhantomData,
        })
    }
}

/// Page-aligned, page-granular owned storage for kernel-tier tables:
/// `posix_memalign(PAGE_SIZE)` with capacity rounded up to a page multiple,
/// so [`device_buffer`](Self::device_buffer) is no-copy at any length.
///
/// `T: Copy` keeps drop handling trivial (freeing the allocation never needs
/// to run element destructors).
pub struct PageAlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    cap_bytes: usize,
}

// SAFETY: the allocation is exclusively owned and never shared internally;
// thread transfer/sharing is exactly as safe as for T itself.
unsafe impl<T: Copy + Send> Send for PageAlignedVec<T> {}
// SAFETY: as above — &PageAlignedVec<T> only exposes &T.
unsafe impl<T: Copy + Sync> Sync for PageAlignedVec<T> {}

impl<T: Copy> PageAlignedVec<T> {
    /// Allocate `len` elements initialized by `f(index)`.
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        const {
            assert!(align_of::<T>() <= PAGE_SIZE);
            assert!(size_of::<T>() > 0, "zero-sized elements are pointless here");
        }
        // Defensive: the hard-coded page size must match the platform's.
        // SAFETY: sysconf is a trivially safe FFI query.
        let platform_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        debug_assert_eq!(platform_page, PAGE_SIZE as i64);
        let _ = platform_page;

        let cap_bytes = round_up_to_page((len * size_of::<T>()).max(1));
        let mut base: *mut c_void = std::ptr::null_mut();
        // SAFETY: PAGE_SIZE is a power of two ≥ size_of::<*mut c_void>();
        // base is a valid out-pointer.
        let rc = unsafe { libc::posix_memalign(&raw mut base, PAGE_SIZE, cap_bytes) };
        assert!(
            rc == 0 && !base.is_null(),
            "posix_memalign({cap_bytes}) failed: {rc}"
        );
        // SAFETY: null-checked just above.
        let ptr = unsafe { NonNull::new_unchecked(base.cast::<T>()) };
        for i in 0..len {
            // SAFETY: i < len and the allocation holds ≥ len elements.
            unsafe { ptr.as_ptr().add(i).write(f(i)) };
        }
        Self {
            ptr,
            len,
            cap_bytes,
        }
    }

    pub fn from_elem(value: T, len: usize) -> Self {
        Self::from_fn(len, |_| value)
    }

    pub fn from_slice(source: &[T]) -> Self {
        Self::from_fn(source.len(), |i| source[i])
    }

    /// No-copy device view of the initialized elements. Never falls back to
    /// copying: the capacity is page-granular by construction.
    pub fn device_buffer<'a>(&'a self, ctx: &MetalContext) -> Result<DeviceBuffer<'a>, MetalError> {
        let len_bytes = self.len * size_of::<T>();
        debug_assert!(round_up_to_page(len_bytes.max(1)) <= self.cap_bytes);
        // SAFETY: base is PAGE_SIZE-aligned and the page-rounded length is
        // ≤ cap_bytes, which this vec owns for 'a.
        unsafe { ctx.wrap_raw_nocopy(self.ptr.cast(), len_bytes) }
            .ok_or(MetalError::Alloc { bytes: len_bytes })
    }

    /// Mutable no-copy device view (for kernel outputs); the exclusive
    /// borrow blocks host access while the buffer is live.
    pub fn device_buffer_mut<'a>(
        &'a mut self,
        ctx: &MetalContext,
    ) -> Result<DeviceBuffer<'a>, MetalError> {
        let len_bytes = self.len * size_of::<T>();
        debug_assert!(round_up_to_page(len_bytes.max(1)) <= self.cap_bytes);
        // SAFETY: as in `device_buffer`, with exclusivity from &mut self.
        unsafe { ctx.wrap_raw_nocopy(self.ptr.cast(), len_bytes) }
            .ok_or(MetalError::Alloc { bytes: len_bytes })
    }
}

impl<T: Copy> Deref for PageAlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: ptr is valid for len initialized elements owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for PageAlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in Deref, with exclusivity from &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for PageAlignedVec<T> {
    fn drop(&mut self) {
        // SAFETY: ptr came from posix_memalign (malloc family) and is freed
        // exactly once; T: Copy, so elements need no drop.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}
