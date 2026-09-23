//! Word-wise `memset` and `memcmp` for RISC-V guests.
//!
//! `memcpy` stays with the runtime: Rust typed loads cannot copy arbitrary
//! uninitialized object padding, and rounded loads may exceed the source range.
//! These routines access only complete in-range words and byte heads/tails.

use core::ptr;

const WORD: usize = 8;

#[inline(always)]
fn misalignment(p: usize) -> usize {
    (WORD - (p % WORD)) % WORD
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
