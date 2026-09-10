//! Word-wise `memcpy`, `memset`, `memcmp` for RISC-V guests.
//!
//! libc's generic routines move one byte per iteration on this target; in a
//! zkVM every instruction is a proving-cost cycle, and the verifier-as-guest
//! workloads copy megabytes of setup and proof bytes. These definitions take
//! precedence over the libc archive members at link time.
//!
//! Long copies use aligned wide loads and stores; when source and destination
//! disagree on alignment, the copy reads aligned source words and assembles
//! each destination word from two neighbours with shifts (the classic libc
//! approach). Byte loops only ever run for the unaligned head and the tail:
//! the zkVM expands every byte-granular or misaligned access into several
//! trace rows, and a misaligned word costs more than the bytes it replaces.

use core::ptr;

const WORD: usize = 8;

#[inline(always)]
fn misalignment(p: usize) -> usize {
    (WORD - (p % WORD)) % WORD
}
/// # Safety
/// `dst` and `src` are valid for `n` bytes and do not overlap.
#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, mut n: usize) -> *mut u8 {
    let mut d = dst;
    let mut s = src;
    if n >= 2 * WORD {
        // Align the destination first; the source then sits at a fixed byte
        // offset `off` from alignment.
        let head = misalignment(d as usize);
        for _ in 0..head {
            *d = *s;
            d = d.add(1);
            s = s.add(1);
        }
        n -= head;
        let mut dw = d.cast::<u64>();
        let off = (s as usize) % WORD;
        let mut words = n / WORD;
        if off == 0 {
            let mut sw = s.cast::<u64>();
            while words >= 4 {
                let a = ptr::read(sw);
                let b = ptr::read(sw.add(1));
                let c = ptr::read(sw.add(2));
                let e = ptr::read(sw.add(3));
                ptr::write(dw, a);
                ptr::write(dw.add(1), b);
                ptr::write(dw.add(2), c);
                ptr::write(dw.add(3), e);
                dw = dw.add(4);
                sw = sw.add(4);
                words -= 4;
            }
            while words > 0 {
                ptr::write(dw, ptr::read(sw));
                dw = dw.add(1);
                sw = sw.add(1);
                words -= 1;
            }
            s = sw.cast::<u8>();
        } else if words > 0 {
            // Little-endian: destination word i = bytes [8i, 8i+8) of the
            // source = the top (8-off) bytes of aligned word i and the low
            // off bytes of aligned word i+1. The final aligned word read
            // holds the last needed byte, so it stays inside the source's
            // 8-byte-rounded extent (as libc's copies do).
            let shift = (off * 8) as u32;
            let mut sw = s.sub(off).cast::<u64>();
            let mut prev = ptr::read(sw);
            while words > 0 {
                sw = sw.add(1);
                let next = ptr::read(sw);
                ptr::write(dw, (prev >> shift) | (next << (64 - shift)));
                prev = next;
                dw = dw.add(1);
                words -= 1;
            }
            s = sw.cast::<u8>().add(off);
        }
        d = dw.cast::<u8>();
        n %= WORD;
    }
    while n > 0 {
        *d = *s;
        d = d.add(1);
        s = s.add(1);
        n -= 1;
    }
    dst
}

/// # Safety
/// `dst` is valid for `n` bytes.
#[no_mangle]
pub unsafe extern "C" fn memset(dst: *mut u8, byte: i32, mut n: usize) -> *mut u8 {
    let value = byte as u8;
    let mut d = dst;
    let head = misalignment(d as usize).min(n);
    for _ in 0..head {
        *d = value;
        d = d.add(1);
    }
    n -= head;
    let word = u64::from_ne_bytes([value; WORD]);
    let mut dw = d.cast::<u64>();
    let mut words = n / WORD;
    while words >= 4 {
        ptr::write(dw, word);
        ptr::write(dw.add(1), word);
        ptr::write(dw.add(2), word);
        ptr::write(dw.add(3), word);
        dw = dw.add(4);
        words -= 4;
    }
    while words > 0 {
        ptr::write(dw, word);
        dw = dw.add(1);
        words -= 1;
    }
    d = dw.cast::<u8>();
    n %= WORD;
    while n > 0 {
        *d = value;
        d = d.add(1);
        n -= 1;
    }
    dst
}

/// # Safety
/// `a` and `b` are valid for `n` bytes.
#[no_mangle]
pub unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, mut n: usize) -> i32 {
    let mut pa = a;
    let mut pb = b;
    if (pa as usize) % WORD == (pb as usize) % WORD {
        let head = misalignment(pa as usize).min(n);
        for _ in 0..head {
            if *pa != *pb {
                return i32::from(*pa) - i32::from(*pb);
            }
            pa = pa.add(1);
            pb = pb.add(1);
        }
        n -= head;
        while n >= WORD {
            if ptr::read(pa.cast::<u64>()) != ptr::read(pb.cast::<u64>()) {
                break;
            }
            pa = pa.add(WORD);
            pb = pb.add(WORD);
            n -= WORD;
        }
    }
    while n > 0 {
        if *pa != *pb {
            return i32::from(*pa) - i32::from(*pb);
        }
        pa = pa.add(1);
        pb = pb.add(1);
        n -= 1;
    }
    0
}

/// # Safety
/// `a` and `b` are valid for `n` bytes.
#[no_mangle]
pub unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    memcmp(a, b, n)
}
