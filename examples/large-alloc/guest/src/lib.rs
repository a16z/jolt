use libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE};

/// std-guest exercise of ZeroOS's mmap/munmap region accounting.
///
/// Round 1 (mallocng path): every allocation is past musl mallocng's
/// individual-mmap threshold (~128 KiB), so the malloc is served by `mmap`
/// and the `drop` routes through a whole-mapping `munmap`.
///
/// Round 2 (POSIX repeated-unmap path): unmap the same range twice. POSIX
/// allows a munmap range with no live mappings ("if there are no mappings in
/// the specified address range, munmap() has no effect"), so the second call
/// must succeed as a no-op. The guest kernel serves mappings from its heap
/// allocator, which only understands whole-region frees — a repeated unmap
/// must not reach it.
///
/// Round 3 (POSIX partial-unmap path): map a 16-page region, unmap its
/// 8-page tail (allocators use this to trim reservations), keep using the
/// live half, then release the whole original range.
///
/// The guest must survive all frees and keep allocating afterwards.
#[jolt::provable(
    heap_size = 16777216,
    stack_size = 1048576,
    max_trace_length = 16777216
)]
fn large_alloc_roundtrip() -> u64 {
    let mut acc: u64 = 0;
    for round in 0..4u64 {
        // 256 KiB + a per-round page so the mappings differ in size.
        let len = (256 << 10) + (round as usize) * 4096;
        let mut buffer = vec![0u8; len];
        let mut i = 0;
        while i < len {
            buffer[i] = (round as u8).wrapping_add(i as u8);
            i += 4096;
        }
        acc = acc.wrapping_add(buffer.iter().map(|&byte| u64::from(byte)).sum::<u64>());
        drop(buffer);
    }

    const PAGE: usize = 4096;
    unsafe {
        // Round 2: repeated unmap of the same range.
        let base = libc::mmap(
            core::ptr::null_mut(),
            16 * PAGE,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(base, MAP_FAILED, "anonymous mmap failed");
        base.cast::<u8>().write(0xA5);
        assert_eq!(libc::munmap(base, 16 * PAGE), 0);
        assert_eq!(
            libc::munmap(base, 16 * PAGE),
            0,
            "repeated munmap must no-op"
        );

        // Round 3: trim a reservation's tail, then release the whole range.
        let base = libc::mmap(
            core::ptr::null_mut(),
            16 * PAGE,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(base, MAP_FAILED, "anonymous mmap failed");
        let base = base.cast::<u8>();
        base.write(0xA5);
        assert_eq!(libc::munmap(base.add(8 * PAGE).cast(), 8 * PAGE), 0);
        base.add(7 * PAGE).write(0x5A);
        acc = acc.wrapping_add(u64::from(base.read()) + u64::from(base.add(7 * PAGE).read()));
        // Release the whole original range; the already-unmapped tail makes
        // this a partially-live range, which POSIX also allows.
        assert_eq!(libc::munmap(base.cast(), 16 * PAGE), 0);
    }

    // The heap must still serve large allocations after the raw unmaps.
    let tail = vec![7u8; 512 << 10];
    acc = acc.wrapping_add(tail.iter().map(|&byte| u64::from(byte)).sum::<u64>());
    acc
}
