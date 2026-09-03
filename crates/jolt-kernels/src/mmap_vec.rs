//! Fixed-capacity, mmap-backed storage for the proof's giant per-cycle
//! lanes.
//!
//! `Vec`-backed lanes die at their designed drop sites but their pages stay
//! RSS/phys_footprint-resident in libmalloc until an allocator-internal
//! trigger returns them — at 2^27 a ~30 GiB corpse pile rides into stage 6b
//! and the kernel compresses it mid-stage when ambient memory is tight. An
//! anonymous-mmap backing munmaps on drop, so the ledger release is
//! deterministic and immediate at the drop site. `madvise`-based decommit is
//! NOT an alternative: it is a silent no-op on any range ever wrapped by a
//! no-copy `MTLBuffer`, while munmap works even on live-wrapped ranges.
//!
//! The mapping is page-aligned by construction, so no-copy Metal wrapping
//! eligibility is guaranteed (the page-rounded wrap range stays inside the
//! page-rounded mapping). Anonymous mmap pages are kernel-zero-filled on
//! first touch — `zeroed` costs no memset.

use std::ops::{Deref, DerefMut};

/// Fixed-capacity contiguous storage; munmaps on drop (unix). `T: Copy`
/// keeps drop glue out of the picture — elements are plain data.
pub struct MmapVec<T: Copy> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    capacity: usize,
}

// SAFETY: MmapVec owns its mapping exclusively; access is through &self /
// &mut self exactly like Vec.
unsafe impl<T: Copy + Send> Send for MmapVec<T> {}
// SAFETY: as above.
unsafe impl<T: Copy + Sync> Sync for MmapVec<T> {}

#[cfg(unix)]
fn map_bytes(bytes: usize) -> std::ptr::NonNull<u8> {
    // SAFETY: anonymous private mapping, no fd; length checked nonzero by
    // callers.
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert!(
        raw != libc::MAP_FAILED,
        "mmap of {bytes} bytes failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: MAP_FAILED excluded above; mmap never returns null otherwise.
    unsafe { std::ptr::NonNull::new_unchecked(raw.cast::<u8>()) }
}

#[cfg(unix)]
fn unmap_bytes(ptr: std::ptr::NonNull<u8>, bytes: usize) {
    // SAFETY: exactly the range map_bytes returned, unmapped once.
    let rc = unsafe { libc::munmap(ptr.as_ptr().cast(), bytes) };
    debug_assert_eq!(rc, 0, "munmap failed: {}", std::io::Error::last_os_error());
}

// Non-unix fallback: leak-free heap mapping through the global allocator.
// (jolt-kernels' production targets are unix; this keeps the crate portable.)
#[cfg(not(unix))]
fn map_bytes(bytes: usize) -> std::ptr::NonNull<u8> {
    let layout = std::alloc::Layout::from_size_align(bytes, page_size()).expect("layout");
    // SAFETY: nonzero size guaranteed by callers.
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    std::ptr::NonNull::new(raw).expect("allocation failed")
}

#[cfg(not(unix))]
fn unmap_bytes(ptr: std::ptr::NonNull<u8>, bytes: usize) {
    let layout = std::alloc::Layout::from_size_align(bytes, page_size()).expect("layout");
    // SAFETY: exactly the map_bytes allocation, freed once.
    unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
}

fn page_size() -> usize {
    // Apple Silicon pages are 16 KiB; 64 KiB rounding stays correct (and
    // no-copy-wrap eligible) on any unix page size up to 64 KiB.
    1 << 16
}

#[expect(clippy::expect_used, reason = "capacity overflow is unrecoverable")]
fn byte_capacity<T>(capacity: usize) -> usize {
    size_of::<T>()
        .checked_mul(capacity)
        .and_then(|bytes| bytes.checked_next_multiple_of(page_size()))
        .expect("MmapVec capacity overflow")
}

impl<T: Copy> MmapVec<T> {
    /// An empty vec that can grow to exactly `capacity` elements via
    /// [`push`](Self::push).
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(align_of::<T>() <= page_size());
        let bytes = byte_capacity::<T>(capacity.max(1));
        Self {
            ptr: map_bytes(bytes).cast::<T>(),
            len: 0,
            capacity,
        }
    }

    /// `len` zeroed elements (kernel zero-fill; no memset).
    ///
    /// The zero pattern must be a valid `T`; for the lane scalars (integers,
    /// packed rows) it is.
    pub fn zeroed(len: usize) -> Self {
        let mut vec = Self::with_capacity(len);
        vec.len = len;
        vec
    }

    /// `len` copies of `value`.
    pub fn filled(len: usize, value: T) -> Self {
        let mut vec = Self::zeroed(len);
        vec.fill(value);
        vec
    }

    /// Append `value`. Panics past the fixed capacity — lane fills know
    /// their exact length up front.
    #[inline]
    pub fn push(&mut self, value: T) {
        assert!(
            self.len < self.capacity,
            "MmapVec::push past fixed capacity"
        );
        // SAFETY: len < capacity keeps the write inside the mapping.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Append every item; panics past the fixed capacity like
    /// [`push`](Self::push).
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<T: Copy> Deref for MmapVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: len elements initialized (zero-filled mapping or pushes).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for MmapVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in Deref, with exclusivity from &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for MmapVec<T> {
    fn drop(&mut self) {
        unmap_bytes(
            self.ptr.cast::<u8>(),
            byte_capacity::<T>(self.capacity.max(1)),
        );
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for MmapVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T: Copy + PartialEq> PartialEq for MmapVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a, T: Copy> IntoIterator for &'a MmapVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Copy> FromIterator<T> for MmapVec<T> {
    /// Collect an exact-size iterator (lane builds); over-long iterators
    /// panic at the capacity assert.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);
        let mut vec = Self::with_capacity(capacity);
        for value in iter {
            vec.push(value);
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_filled_push_roundtrip() {
        let z = MmapVec::<u64>::zeroed(1000);
        assert_eq!(z.len(), 1000);
        assert!(z.iter().all(|&x| x == 0));

        let f = MmapVec::<u8>::filled(4097, 0xa5);
        assert!(f.iter().all(|&x| x == 0xa5));

        let mut p = MmapVec::<u32>::with_capacity(3);
        p.push(1);
        p.push(2);
        p.push(3);
        assert_eq!(&p[..], &[1, 2, 3]);

        let c: MmapVec<u16> = (0..100u16).collect();
        assert_eq!(c.len(), 100);
        assert_eq!(c[99], 99);
    }

    #[test]
    fn page_aligned_for_nocopy_wrap() {
        let v = MmapVec::<u64>::zeroed(3);
        assert_eq!(v.as_slice().as_ptr() as usize % 16384, 0);
    }

    #[test]
    #[should_panic(expected = "past fixed capacity")]
    fn push_past_capacity_panics() {
        let mut v = MmapVec::<u8>::with_capacity(1);
        v.push(0);
        v.push(1);
    }
}
