#![cfg_attr(feature = "guest", no_std)]

#[cfg(feature = "guest")]
use core::arch::asm;

/// Three doublewords whose bytes all have the high bit set, so signed loads
/// sign-extend. The access base points at the middle doubleword, so
/// immediates from -8 to 15 reach every byte.
#[cfg(feature = "guest")]
#[repr(align(8))]
struct Scratch([u8; 24]);

#[cfg(feature = "guest")]
impl Scratch {
    fn new() -> Self {
        let mut bytes = [0; 24];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = 0x80 | (i as u8).wrapping_mul(37);
        }
        Self(bytes)
    }

    fn base(&mut self) -> *mut u8 {
        self.0.as_mut_ptr().wrapping_add(8)
    }

    fn words(&self) -> [u64; 3] {
        let mut words = [0; 3];
        for (word, bytes) in words.iter_mut().zip(self.0.chunks_exact(8)) {
            let mut le = [0; 8];
            le.copy_from_slice(bytes);
            *word = u64::from_le_bytes(le);
        }
        words
    }
}

/// Folds the value `op` loads at each offset into `hash`.
#[cfg(feature = "guest")]
macro_rules! load {
    ($hash:ident, $base:ident, $op:literal: $($offset:literal),+) => {$(
        let value: u64;
        asm!(
            concat!($op, " {value}, ", $offset, "({base})"),
            value = out(reg) value,
            base = in(reg) $base,
            options(nostack, readonly, preserves_flags),
        );
        $hash = $hash.rotate_left(7) ^ value;
    )+};
}

/// Stores a fresh `value` with `op` at each offset, then folds the three
/// doublewords into `hash`.
#[cfg(feature = "guest")]
macro_rules! store {
    ($hash:ident, $base:ident, $value:ident, $op:literal: $($offset:literal),+) => {$(
        $value = $value.rotate_left(13) ^ 0x9e37_79b9_7f4a_7c15;
        asm!(
            concat!($op, " {value}, ", $offset, "({base})"),
            value = in(reg) $value,
            base = in(reg) $base,
            options(nostack, preserves_flags),
        );
        load!($hash, $base, "ld": "-8", "0", "8");
    )+};
}

/// One accessor per load and store kind, each running its access from a single
/// PC with immediate 0: walking the base across the scratch takes that PC
/// through every alignment.
#[cfg(feature = "guest")]
macro_rules! single_pc_accessors {
    (loads: $($load:ident = $load_op:literal),+; stores: $($store:ident = $store_op:literal),+) => {
        $(
            #[inline(never)]
            unsafe fn $load(address: *const u8) -> u64 {
                let value: u64;
                asm!(
                    concat!($load_op, " {value}, 0({address})"),
                    value = out(reg) value,
                    address = in(reg) address,
                    options(nostack, readonly, preserves_flags),
                );
                value
            }
        )+
        $(
            #[inline(never)]
            unsafe fn $store(address: *mut u8, value: u64) {
                asm!(
                    concat!($store_op, " {value}, 0({address})"),
                    value = in(reg) value,
                    address = in(reg) address,
                    options(nostack, preserves_flags),
                );
            }
        )+
    };
}

#[cfg(feature = "guest")]
single_pc_accessors!(
    loads: walk_lb = "lb", walk_lbu = "lbu", walk_lh = "lh", walk_lhu = "lhu", walk_lw = "lw",
        walk_lwu = "lwu", walk_ld = "ld";
    stores: walk_sb = "sb", walk_sh = "sh", walk_sw = "sw", walk_sd = "sd"
);

#[cfg(feature = "guest")]
type WalkLoad = unsafe fn(*const u8) -> u64;
#[cfg(feature = "guest")]
type WalkStore = unsafe fn(*mut u8, u64);

/// Runs every load and store from its single-PC accessor, the base walking the
/// scratch by the access width or, when `misaligned`, byte by byte, and returns
/// `hash` with every loaded value folded in.
#[cfg(feature = "guest")]
unsafe fn walk(base: *mut u8, mut hash: u64, mut value: u64, misaligned: bool) -> u64 {
    let loads: [(WalkLoad, isize); 7] = [
        (walk_lb, 1),
        (walk_lbu, 1),
        (walk_lh, 2),
        (walk_lhu, 2),
        (walk_lw, 4),
        (walk_lwu, 4),
        (walk_ld, 8),
    ];
    for (load, width) in loads {
        for offset in (-8..=16 - width).filter(|offset| misaligned || offset % width == 0) {
            hash = hash.rotate_left(7) ^ load(base.wrapping_offset(offset));
        }
    }
    let stores: [(WalkStore, isize); 4] = [(walk_sb, 1), (walk_sh, 2), (walk_sw, 4), (walk_sd, 8)];
    for (store, width) in stores {
        for offset in (-8..=16 - width).filter(|offset| misaligned || offset % width == 0) {
            value = value.rotate_left(13) ^ 0x9e37_79b9_7f4a_7c15;
            store(base.wrapping_offset(offset), value);
            load!(hash, base, "ld": "-8", "0", "8");
        }
    }
    hash
}

/// Every load and store width at every naturally aligned offset inside the
/// scratch: first reached with negative and non-negative immediates, then from
/// one PC per kind with the base walking. Returns the hash of every loaded
/// value and the final scratch doublewords.
#[jolt::provable(guest_only, heap_size = 65536, max_trace_length = 65536)]
fn memory_ops() -> (u64, [u64; 3]) {
    let mut scratch = Scratch::new();
    let base = scratch.base();
    let mut hash = 0u64;
    let mut value = 0u64;
    unsafe {
        load!(hash, base, "lb": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        load!(hash, base, "lbu": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        load!(hash, base, "lh": "-8", "-6", "-4", "-2", "0", "2", "4", "6", "8", "10", "12", "14");
        load!(hash, base, "lhu": "-8", "-6", "-4", "-2", "0", "2", "4", "6", "8", "10", "12", "14");
        load!(hash, base, "lw": "-8", "-4", "0", "4", "8", "12");
        load!(hash, base, "lwu": "-8", "-4", "0", "4", "8", "12");
        load!(hash, base, "ld": "-8", "0", "8");
        store!(hash, base, value, "sb": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        store!(hash, base, value, "sh": "-8", "-6", "-4", "-2", "0", "2", "4", "6", "8", "10", "12", "14");
        store!(hash, base, value, "sw": "-8", "-4", "0", "4", "8", "12");
        store!(hash, base, value, "sd": "-8", "0", "8");
        hash = walk(base, hash, value, false);
    }
    (hash, scratch.words())
}

/// `memory_ops` at every offset inside the scratch, aligned or not. Proving it
/// needs the `zicclsm` feature.
#[jolt::provable(guest_only, heap_size = 65536, max_trace_length = 65536)]
fn misaligned_memory_ops() -> (u64, [u64; 3]) {
    let mut scratch = Scratch::new();
    let base = scratch.base();
    let mut hash = 0u64;
    let mut value = 0u64;
    unsafe {
        load!(hash, base, "lb": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        load!(hash, base, "lbu": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        load!(hash, base, "lh": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14");
        load!(hash, base, "lhu": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14");
        load!(hash, base, "lw": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12");
        load!(hash, base, "lwu": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12");
        load!(hash, base, "ld": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8");
        store!(hash, base, value, "sb": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");
        store!(hash, base, value, "sh": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14");
        store!(hash, base, value, "sw": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12");
        store!(hash, base, value, "sd": "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8");
        hash = walk(base, hash, value, true);
    }
    (hash, scratch.words())
}

/// One access misaligned for its width, chosen by `kind`: `LH`, `LHU`, `LW`,
/// `LWU`, `SH`, `SW`, `LD` or `SD`, some within a doubleword and some crossing
/// one. Under `zicclsm` each runs its misaligned sequence, and otherwise the
/// tracer aborts on it; soundness tests force its aligned sequence instead and
/// check that the proof fails.
#[jolt::provable(guest_only, heap_size = 65536, max_trace_length = 65536)]
fn misaligned_access(kind: u8) -> u64 {
    let mut scratch = Scratch::new();
    let base = scratch.base();
    let mut value = 0x0123_4567_89ab_cdef_u64;
    unsafe {
        match kind {
            0 => asm!("lh {value}, 1({base})", value = out(reg) value, base = in(reg) base),
            1 => asm!("lhu {value}, 3({base})", value = out(reg) value, base = in(reg) base),
            2 => asm!("lw {value}, 2({base})", value = out(reg) value, base = in(reg) base),
            3 => asm!("lwu {value}, 7({base})", value = out(reg) value, base = in(reg) base),
            4 => asm!("sh {value}, 5({base})", value = in(reg) value, base = in(reg) base),
            5 => asm!("sw {value}, 6({base})", value = in(reg) value, base = in(reg) base),
            6 => asm!("ld {value}, 3({base})", value = out(reg) value, base = in(reg) base),
            _ => asm!("sd {value}, 5({base})", value = in(reg) value, base = in(reg) base),
        }
    }
    value ^ scratch.words()[1]
}
