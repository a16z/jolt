//! Word-wise `memcpy`, `memset` and `memcmp` for RISC-V guests.
//!
//! libc's generic routines move one byte per iteration on this target; in a
//! zkVM every instruction is a proving-cost cycle, and a verifier-as-guest
//! copies megabytes of setup and proof bytes. These definitions take
//! precedence over the libc archive members at link time.
//!
//! Long copies use aligned word loads and stores. When source and destination
//! disagree on alignment, the copy reads aligned source words and assembles
//! each destination word from two neighbours with shifts. Byte loops only run
//! for the unaligned head and the tail: the zkVM expands every byte-granular or
//! misaligned access into several trace rows.
//!
//! `memcpy` must copy arbitrary object bytes, including uninitialized padding,
//! and the shifted path reads whole aligned words that extend past either end
//! of the source range. A Rust typed read may do neither, so `memcpy` reads
//! source memory only through the opaque loads below. Every word they read
//! contains at least one in-range byte, so it lies in memory the caller mapped:
//! no mapping boundary falls inside an aligned word.

use core::arch::asm;
use core::ptr;

const WORD: usize = 8;

#[inline(always)]
fn misalignment(p: usize) -> usize {
    (WORD - (p % WORD)) % WORD
}

/// Read the aligned word at `p` without asserting anything about its bytes.
///
/// # Safety
/// `p` is 8-byte aligned and its word contains a byte the caller may read.
#[inline(always)]
unsafe fn load_word(p: *const u8) -> u64 {
    let value: u64;
    asm!("ld {value}, 0({p})", value = lateout(reg) value, p = in(reg) p,
        options(nostack, readonly, preserves_flags));
    value
}

/// Read the byte at `p` without asserting that it is initialized.
///
/// # Safety
/// `p` is valid for a one-byte read.
#[inline(always)]
unsafe fn load_byte(p: *const u8) -> u8 {
    let value: u8;
    asm!("lbu {value}, 0({p})", value = lateout(reg) value, p = in(reg) p,
        options(nostack, readonly, preserves_flags));
    value
}

/// # Safety
/// `dst` and `src` are valid for `n` bytes and do not overlap.
#[no_mangle]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, mut n: usize) -> *mut u8 {
    let mut d = dst;
    let mut s = src;
    if n >= 2 * WORD {
        // Align the destination; the source then sits at a fixed offset `off`
        // from alignment.
        for _ in 0..misalignment(d as usize) {
            *d = load_byte(s);
            d = d.add(1);
            s = s.add(1);
            n -= 1;
        }
        let mut dw = d.cast::<u64>();
        let off = (s as usize) % WORD;
        let mut words = n / WORD;
        if off == 0 {
            while words >= 4 {
                let a = load_word(s);
                let b = load_word(s.add(WORD));
                let c = load_word(s.add(2 * WORD));
                let e = load_word(s.add(3 * WORD));
                ptr::write(dw, a);
                ptr::write(dw.add(1), b);
                ptr::write(dw.add(2), c);
                ptr::write(dw.add(3), e);
                dw = dw.add(4);
                s = s.add(4 * WORD);
                words -= 4;
            }
            while words > 0 {
                ptr::write(dw, load_word(s));
                dw = dw.add(1);
                s = s.add(WORD);
                words -= 1;
            }
        } else if words > 0 {
            // Little-endian: destination word `i` is the top `8 - off` bytes of
            // aligned source word `i` and the low `off` bytes of word `i + 1`.
            // Each aligned word read holds a byte of the source range.
            let shift = (off * 8) as u32;
            let mut sw = s.sub(off);
            let mut prev = load_word(sw);
            while words > 0 {
                sw = sw.add(WORD);
                let next = load_word(sw);
                ptr::write(dw, (prev >> shift) | (next << (64 - shift)));
                prev = next;
                dw = dw.add(1);
                words -= 1;
            }
            s = sw.add(off);
        }
        d = dw.cast::<u8>();
        n %= WORD;
    }
    while n > 0 {
        *d = load_byte(s);
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
/// `a` and `b` are valid for reads of `n` initialized bytes.
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
/// `a` and `b` are valid for reads of `n` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    memcmp(a, b, n)
}
