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
use std::sync::{Arc, Mutex, PoisonError};

use jolt_field::Fr;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::error::MetalError;
use super::runtime::MetalContext;

/// Apple Silicon page size. Asserted against `sysconf` at first use in
/// [`PageAlignedVec`] construction (defensive: this crate hard-codes it in
/// eligibility checks and capacity rounding).
pub const PAGE_SIZE: usize = 16384;

/// `JOLT_METAL_ALLOC_TRACE=1` live-set census: every device-owned
/// allocation ≥ 4 MiB emits paired `metal_alloc`/`metal_free` tracing
/// events (id, bytes, kind), so a Chrome trace carries the transient
/// buffers' exact lifetimes next to the stage spans. Id `0` = untraced;
/// zero work when the env is unset.
mod alloc_trace {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    const MIN_BYTES: usize = 4 << 20;

    pub(super) fn allocated(bytes: usize, kind: &'static str) -> u64 {
        let enabled = *ENABLED.get_or_init(|| {
            std::env::var("JOLT_METAL_ALLOC_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
        });
        if !enabled || bytes < MIN_BYTES {
            return 0;
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        tracing::info!(id, bytes, kind, "metal_alloc");
        id
    }

    pub(super) fn freed(id: u64) {
        if id != 0 {
            tracing::info!(id, "metal_free");
        }
    }
}

/// macOS malloc's large-allocation threshold: at and above this size,
/// allocations are `vm_allocate`d with page granularity (the invariant the
/// no-copy size heuristic relies on).
pub(super) const MALLOC_LARGE_THRESHOLD: usize = 32 * 1024;

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

    /// Borrow shared-storage contents as typed values after the producing
    /// command buffer has completed.
    pub(super) fn typed_slice<T>(&self, len: usize) -> &[T] {
        assert!(len * std::mem::size_of::<T>() <= self.len_bytes);
        // SAFETY: Metal shared allocations are page-aligned; the size check
        // bounds the slice, and callers only request POD arkworks layouts.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), len) }
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
        // Alloc-only census event: a `DeviceBuffer` cannot carry a `Drop`
        // impl (it would pin every pass-scoped view to scope end), and
        // Metal's actual release is deferred past the wrapper's drop anyway.
        let _ = alloc_trace::allocated(len_bytes, "device");
        Ok(DeviceBuffer {
            raw,
            len_bytes,
            copied: false,
            _backing: PhantomData,
        })
    }

    /// Device-owned shared buffer initialized from `words`.
    pub(super) fn copy_u32s(&self, words: &[u32]) -> Result<DeviceBuffer<'static>, MetalError> {
        // SAFETY: a u32 slice is a contiguous initialized byte range.
        let bytes =
            unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), size_of_val(words)) };
        self.copy_bytes(bytes)
    }

    /// Device-owned shared buffer initialized from raw bytes.
    pub(super) fn copy_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer<'static>, MetalError> {
        let len_bytes = bytes.len();
        if bytes.is_empty() {
            return self.alloc_u32s(0);
        }
        let src: NonNull<c_void> = NonNull::from(&bytes[0]).cast();
        // SAFETY: `src` spans `len_bytes` readable bytes; Metal copies them.
        let raw = unsafe {
            self.device().newBufferWithBytes_length_options(
                src,
                len_bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Alloc { bytes: len_bytes })?;
        let _ = alloc_trace::allocated(len_bytes, "device_copy");
        Ok(DeviceBuffer {
            raw,
            len_bytes,
            copied: true,
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
        // Alloc-only census event (see `alloc_u32s` on why no free pair).
        let _ = alloc_trace::allocated(len_bytes, "wrap_copy");
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

/// An `MTLBuffer` that OWNS its backing memory, for tables that live across
/// many compute passes (a sumcheck kernel's round tables). The borrow-scoped
/// [`DeviceBuffer<'a>`] cannot outlive a `prepare`-time slice, so slots that
/// keep device-visible state between rounds hold `OwnedDeviceBuffer`s and
/// mint per-pass [`DeviceBuffer`] views ([`device_buffer`]
/// (Self::device_buffer), a retain — no re-wrap, no copy).
///
/// Backing is the `Vec` handed to [`MetalContext::own_vec`] when it is
/// no-copy eligible (module docs), else a [`PageAlignedVec`] copy —
/// [`was_copied`](Self::was_copied) reports which. Both allocations are
/// address-stable across moves of this struct, so the wrap stays valid.
///
/// Host access ([`as_slice`](Self::as_slice) /
/// [`as_mut_slice`](Self::as_mut_slice)) is safe under this tier's
/// synchronous execution model: every pass blocks until GPU completion, and
/// the borrow rules make `as_mut_slice` unavailable while a minted
/// [`DeviceBuffer`] (and thus a pass that could touch the memory) is live.
pub struct OwnedDeviceBuffer<T: Copy> {
    backing: OwnedBacking<T>,
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    copied: bool,
    /// Census id for adopted-`Vec` backings; a `Page` backing traces its own
    /// allocation, so it stays `0` here.
    trace_id: u64,
}

impl<T: Copy> Drop for OwnedDeviceBuffer<T> {
    fn drop(&mut self) {
        alloc_trace::freed(self.trace_id);
    }
}

enum OwnedBacking<T: Copy> {
    Vec(Vec<T>),
    Page(PageAlignedVec<T>),
    /// A carved sub-range of a retired allocation (see [`ArenaSlab`]).
    Arena(ArenaLease<T>),
}

impl<T: Copy> OwnedDeviceBuffer<T> {
    pub fn as_slice(&self) -> &[T] {
        match &self.backing {
            OwnedBacking::Vec(v) => v,
            OwnedBacking::Page(v) => v,
            OwnedBacking::Arena(lease) => lease.as_slice(),
        }
    }

    /// Host-side mutable view. Unavailable (borrow rules) while any minted
    /// [`DeviceBuffer`] view is live, so host writes cannot race GPU work.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.backing {
            OwnedBacking::Vec(v) => v,
            OwnedBacking::Page(v) => v,
            OwnedBacking::Arena(lease) => lease.as_mut_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// True when construction fell back to a [`PageAlignedVec`] copy.
    pub fn was_copied(&self) -> bool {
        self.copied
    }

    /// A pass-scoped view of this buffer (an objc retain of the same
    /// `MTLBuffer`). The `'_` borrow keeps the backing immovable and blocks
    /// `as_mut_slice` while the view — and any pass dispatched over it —
    /// is alive.
    pub fn device_buffer(&self) -> DeviceBuffer<'_> {
        DeviceBuffer {
            raw: self.raw.clone(),
            len_bytes: std::mem::size_of_val(self.as_slice()),
            copied: self.copied,
            _backing: PhantomData,
        }
    }
}

impl MetalContext {
    /// Take ownership of `vec` as a device-visible buffer: wrapped in place
    /// when no-copy eligible (module docs), else copied into a
    /// [`PageAlignedVec`] (`was_copied() == true`; callers count copies).
    pub fn own_vec<T: Copy>(&self, vec: Vec<T>) -> Result<OwnedDeviceBuffer<T>, MetalError> {
        let len_bytes = std::mem::size_of_val(vec.as_slice());
        let aligned = (vec.as_ptr() as usize).is_multiple_of(PAGE_SIZE);
        let page_granular =
            len_bytes.is_multiple_of(PAGE_SIZE) || len_bytes >= MALLOC_LARGE_THRESHOLD;
        if len_bytes > 0 && aligned && page_granular {
            // SAFETY: same eligibility argument as `wrap_slice_nocopy`; the
            // Vec moves into the returned struct un-resized, so its heap
            // allocation (what the pointer references) outlives the wrap.
            let raw =
                unsafe { self.wrap_raw_nocopy_untracked(NonNull::from(&vec[0]).cast(), len_bytes) }
                    .ok_or(MetalError::Alloc { bytes: len_bytes })?;
            return Ok(OwnedDeviceBuffer {
                backing: OwnedBacking::Vec(vec),
                raw,
                copied: false,
                trace_id: alloc_trace::allocated(len_bytes, "vec_adopt"),
            });
        }
        let mut owned = self.own_page_aligned(PageAlignedVec::from_slice(&vec))?;
        owned.copied = true;
        Ok(owned)
    }

    /// Take ownership of a [`PageAlignedVec`] as a device-visible buffer —
    /// no-copy at any length via its capacity guarantee.
    pub fn own_page_aligned<T: Copy>(
        &self,
        vec: PageAlignedVec<T>,
    ) -> Result<OwnedDeviceBuffer<T>, MetalError> {
        let len_bytes = vec.len() * size_of::<T>();
        // SAFETY: base is PAGE_SIZE-aligned with page-granular capacity (the
        // PageAlignedVec construction invariant); the vec moves into the
        // returned struct, so the allocation outlives the wrap.
        let raw = unsafe { self.wrap_raw_nocopy_untracked(vec.ptr.cast(), len_bytes) }
            .ok_or(MetalError::Alloc { bytes: len_bytes })?;
        Ok(OwnedDeviceBuffer {
            backing: OwnedBacking::Page(vec),
            raw,
            copied: false,
            trace_id: 0,
        })
    }

    /// # Safety
    ///
    /// As for `wrap_raw_nocopy`, except the caller (not a `PhantomData`
    /// borrow) guarantees the allocation outlives the returned `MTLBuffer` —
    /// used by the owning wrappers, which move the allocation in next to it.
    unsafe fn wrap_raw_nocopy_untracked(
        &self,
        ptr: NonNull<c_void>,
        len_bytes: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        // SAFETY: forwarded caller contract.
        unsafe {
            self.device()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    round_up_to_page(len_bytes.max(1)),
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
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
    trace_id: u64,
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
            trace_id: alloc_trace::allocated(cap_bytes, "page"),
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
        alloc_trace::freed(self.trace_id);
        // SAFETY: ptr came from posix_memalign (malloc family) and is freed
        // exactly once; T: Copy, so elements need no drop.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}

/// A retired device-owned allocation serving later takers as a placement
/// arena: sub-ranges are carved out at page-aligned offsets instead of
/// allocating (and first-touch-faulting) fresh pages, so a proof's peak
/// transient footprint is the largest concurrent SET, not the sum of every
/// stage's allocations. Measured shape (`JOLT_METAL_ALLOC_TRACE` @2^23):
/// stage 5 retires a 2.3 + 1.15 GiB ping-pong pair that stage 6b's five
/// adoption families then re-cover almost exactly — whole-buffer handoff
/// served one family and the rest allocated ~3.5 GiB fresh.
///
/// Lifecycle: the retired pool holds a STRONG ref until the first carve,
/// then only leases keep the slab alive — when the last lease drops, the
/// memory frees at exactly the moment the un-pooled buffer would have
/// (stage-6b batch end), so later stages never see parked idle pages.
pub(super) struct ArenaSlab {
    /// The retired buffer, kept whole for its backing allocation (its own
    /// `MTLBuffer` wrap sits idle; leases mint their own at their offsets).
    _owner: OwnedDeviceBuffer<Fr>,
    base: NonNull<u8>,
    state: Mutex<SlabState>,
}

// SAFETY: the owner is a Metal buffer handle plus its host backing memory —
// `MTLBuffer` is an `MTLResource`, which Metal documents as thread-safe
// (only command encoders are not), and the backing is plain bytes. `base`
// is only dereferenced through leases, whose byte ranges the mutex-guarded
// free-list keeps disjoint.
unsafe impl Send for ArenaSlab {}
// SAFETY: as for `Send` — all range bookkeeping is behind the mutex.
unsafe impl Sync for ArenaSlab {}

struct SlabState {
    /// Free byte ranges: page-aligned starts, disjoint, unordered.
    free: Vec<std::ops::Range<usize>>,
    leased: usize,
    carved: bool,
}

impl ArenaSlab {
    /// Adopt a retired `Fr` buffer as a slab. `None` when the backing is not
    /// page-aligned (never the case for the pool's no-copy buffers — checked
    /// rather than assumed).
    pub(super) fn adopt(buffer: OwnedDeviceBuffer<Fr>) -> Option<Arc<Self>> {
        let slice = buffer.as_slice();
        let len_bytes = std::mem::size_of_val(slice);
        let base = NonNull::new(slice.as_ptr().cast_mut().cast::<u8>())?;
        if !(base.as_ptr() as usize).is_multiple_of(PAGE_SIZE) || len_bytes < PAGE_SIZE {
            return None;
        }
        Some(Arc::new(Self {
            _owner: buffer,
            base,
            state: Mutex::new(SlabState {
                free: std::iter::once(0..len_bytes).collect(),
                leased: 0,
                carved: false,
            }),
        }))
    }

    /// Total free bytes (for pool ordering; ranges may be fragmented).
    pub(super) fn free_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .free
            .iter()
            .map(|range| range.len())
            .sum()
    }

    /// True once every carved lease has been returned — the slab has served
    /// its purpose and the pool should release its reference.
    pub(super) fn exhausted(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.carved && state.leased == 0
    }

    /// Carve a `len`-element lease out of the smallest fitting free range,
    /// minting its own no-copy `MTLBuffer` at the leased offset. `None` when
    /// no range fits.
    pub(super) fn carve(
        self: &Arc<Self>,
        context: &MetalContext,
        len: usize,
    ) -> Option<OwnedDeviceBuffer<Fr>> {
        let len_bytes = len * size_of::<Fr>();
        // Page-rounded split point: the remainder must start page-aligned,
        // and the lease's own wrap rounds up to the same boundary.
        let take_bytes = round_up_to_page(len_bytes.max(1));
        let (offset, raw) = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let best = state
                .free
                .iter()
                .enumerate()
                .filter(|(_, range)| range.len() >= take_bytes)
                .min_by_key(|(_, range)| range.len())
                .map(|(index, _)| index)?;
            let range = state.free.swap_remove(best);
            let offset = range.start;
            if range.len() > take_bytes {
                state.free.push(range.start + take_bytes..range.end);
            }
            state.leased += 1;
            state.carved = true;
            // SAFETY: `offset` is page-aligned inside the owner allocation
            // and `take_bytes` stays inside the adopted range, which the
            // original no-copy wrap already proved page-granular.
            let raw = unsafe {
                context.wrap_raw_nocopy_untracked(
                    NonNull::new_unchecked(self.base.as_ptr().add(offset)).cast(),
                    len_bytes,
                )
            };
            let Some(raw) = raw else {
                // Roll the range back; the caller falls through to a fresh
                // allocation.
                state.free.push(offset..offset + take_bytes);
                state.leased -= 1;
                return None;
            };
            (offset, raw)
        };
        Some(OwnedDeviceBuffer {
            backing: OwnedBacking::Arena(ArenaLease {
                slab: Arc::clone(self),
                offset_bytes: offset,
                len,
                _elem: PhantomData,
            }),
            raw,
            copied: false,
            trace_id: 0,
        })
    }
}

/// One carved sub-range: returns itself to the slab's free-list on drop.
pub(super) struct ArenaLease<T: Copy> {
    slab: Arc<ArenaSlab>,
    offset_bytes: usize,
    len: usize,
    _elem: PhantomData<T>,
}

impl<T: Copy> ArenaLease<T> {
    fn as_slice(&self) -> &[T] {
        // SAFETY: the lease exclusively owns `[offset, offset + len·size)`
        // (free-list disjointness); page-aligned offset ⇒ aligned for T.
        unsafe {
            std::slice::from_raw_parts(
                self.slab.base.as_ptr().add(self.offset_bytes).cast::<T>(),
                self.len,
            )
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, with exclusivity from `&mut self` — no
        // other lease overlaps this byte range.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.slab.base.as_ptr().add(self.offset_bytes).cast::<T>(),
                self.len,
            )
        }
    }
}

impl<T: Copy> Drop for ArenaLease<T> {
    fn drop(&mut self) {
        let take_bytes = round_up_to_page((self.len * size_of::<T>()).max(1));
        let mut state = self
            .slab
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state
            .free
            .push(self.offset_bytes..self.offset_bytes + take_bytes);
        state.leased -= 1;
    }
}

/// W1D diagnostic (ignored): does `MADV_FREE_REUSABLE` actually remove
/// dirty anonymous pages from `phys_footprint` when the range is wrapped by
/// a live no-copy `MTLBuffer`? Pins the W4-U1 failure mechanism — the
/// campaign's retired-arena madvise returned 0 but the parked 30 GiB never
/// left the ledger. Run explicitly:
/// `cargo nextest run -p jolt-kernels madvise_reusable --features metal --run-ignored all`
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod madvise_probe {
    use super::*;

    fn phys_footprint_bytes() -> u64 {
        // `rusage_info_v4`: 16 UUID bytes then 35 u64 counters;
        // ri_phys_footprint at post-UUID index 7 — the layout jolt-profiling
        // pins with its `footprint_tracks_dirty_memory` test.
        #[repr(C)]
        struct RusageInfoV4 {
            uuid: [u8; 16],
            counters: [u64; 35],
        }
        extern "C" {
            fn proc_pid_rusage(
                pid: libc::c_int,
                flavor: libc::c_int,
                buffer: *mut RusageInfoV4,
            ) -> libc::c_int;
        }
        const RUSAGE_INFO_V4: libc::c_int = 4;
        const RI_PHYS_FOOTPRINT_INDEX: usize = 7;
        let mut info = RusageInfoV4 {
            uuid: [0; 16],
            counters: [0; 35],
        };
        // SAFETY: RUSAGE_INFO_V4 fills exactly a rusage_info_v4.
        let rc = unsafe { proc_pid_rusage(std::process::id() as i32, RUSAGE_INFO_V4, &mut info) };
        assert_eq!(rc, 0);
        info.counters[RI_PHYS_FOOTPRINT_INDEX]
    }

    struct Region {
        base: NonNull<u8>,
        bytes: usize,
    }

    impl Region {
        fn dirty(bytes: usize) -> Self {
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer, power-of-two alignment.
            let rc = unsafe { libc::posix_memalign(&raw mut raw, PAGE_SIZE, bytes) };
            assert_eq!(rc, 0);
            let base = NonNull::new(raw.cast::<u8>()).unwrap();
            // SAFETY: the allocation spans `bytes`.
            unsafe { std::ptr::write_bytes(base.as_ptr(), 0xa5, bytes) };
            Self { base, bytes }
        }

        fn reusable(&self) -> bool {
            // SAFETY: page-aligned range inside the live allocation.
            unsafe {
                libc::madvise(
                    self.base.as_ptr().cast(),
                    self.bytes,
                    libc::MADV_FREE_REUSABLE,
                ) == 0
            }
        }
    }

    impl Drop for Region {
        fn drop(&mut self) {
            // SAFETY: exactly the posix_memalign allocation, freed once.
            unsafe { libc::free(self.base.as_ptr().cast()) };
        }
    }

    /// Footprint delta (MiB) from applying REUSABLE to a freshly dirtied
    /// 1 GiB region under `configure`'s Metal wrapping choice.
    fn probe(context: &MetalContext, wrap: bool, release_before: bool) -> (bool, i64) {
        const BYTES: usize = 1 << 30;
        let region = Region::dirty(BYTES);
        let slice =
            // SAFETY: live dirtied allocation of BYTES.
            unsafe { std::slice::from_raw_parts(region.base.as_ptr(), region.bytes) };
        let buffer = wrap.then(|| context.wrap_slice_nocopy(slice).unwrap());
        let buffer = if release_before { None } else { buffer };
        let before = phys_footprint_bytes();
        let ok = region.reusable();
        let after = phys_footprint_bytes();
        drop(buffer);
        (ok, (after as i64 - before as i64) / (1 << 20))
    }

    #[test]
    #[ignore = "W1D diagnostic; prints footprint ledger deltas"]
    #[expect(clippy::print_stdout, reason = "diagnostic output is the deliverable")]
    fn madvise_reusable_vs_metal_wrap() {
        let _lock = super::super::testing::gpu_lock();
        let context = MetalContext::global().unwrap();

        // Ledger sanity: dirtying and freeing 1 GiB must move phys_footprint
        // by roughly that much, or the probe cannot be trusted.
        let base = phys_footprint_bytes();
        let region = Region::dirty(1 << 30);
        let dirtied = phys_footprint_bytes() as i64 - base as i64;
        drop(region);
        let freed = phys_footprint_bytes() as i64 - base as i64;
        println!(
            "ledger sanity: dirty 1 GiB => +{} MiB, free => {:+} MiB residual",
            dirtied / (1 << 20),
            freed / (1 << 20)
        );
        assert!(
            dirtied > (900 << 20),
            "footprint ledger did not track dirty"
        );

        let (ok_plain, delta_plain) = probe(context, false, false);
        let (ok_wrapped, delta_wrapped) = probe(context, true, false);
        let (ok_released, delta_released) = probe(context, true, true);

        println!("REUSABLE on 1 GiB dirty region, footprint delta:");
        println!("  plain malloc:           ok={ok_plain} delta={delta_plain} MiB");
        println!("  live MTLBuffer wrap:    ok={ok_wrapped} delta={delta_wrapped} MiB");
        println!("  wrap released first:    ok={ok_released} delta={delta_released} MiB");
    }
}
