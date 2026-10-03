//! Conformance checks for the SDK's field-register limb conversions,
//! `jolt::field_to_limbs!` and `jolt::field_from_limbs!`, in whichever proof
//! field the guest is built for. Any failed check panics the guest.

#![cfg_attr(feature = "guest", no_std)]

use jolt::field_inline::{FieldInlineModulus, MODULUS};

/// Imports `value`, wider than either modulus, into a field register and reads
/// it back `readout_width` limbs wide, zero-extended to six; then runs the
/// fixed checks. Hosts drive the import-policy, width, and dishonest-advice
/// cases through the inputs: this readout is the guest's first, so injected
/// advice lands on it.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn check_limb_conversions(value: [u64; 6], readout_width: u8) -> [u64; 6] {
    jolt::field_from_limbs!(15, value);
    let readout = match readout_width {
        1 => widen(jolt::field_to_limbs!(15, 1)),
        2 => widen(jolt::field_to_limbs!(15, 2)),
        4 => widen(jolt::field_to_limbs!(15, 4)),
        6 => jolt::field_to_limbs!(15),
        _ => panic!("unsupported readout width"),
    };
    round_trips();
    arithmetic_matches_limbs();
    widths();
    registers_survive_readouts();
    imports_replace_the_destination();
    readout
}

/// Canonical in both supported fields; the multi-limb values cross every limb
/// boundary of a 128-bit integer.
const SHARED: [[u64; 4]; 7] = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [u64::MAX, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [u64::MAX, u64::MAX - 1, 0, 0],
    [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 0, 0],
];

/// At least the 128-bit prime, and below the BN254 modulus.
const BN254_ONLY: [[u64; 4]; 4] = [
    [u64::MAX, u64::MAX, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 1],
    [u64::MAX, u64::MAX, u64::MAX, 0x3064_4e72_e131_a028],
];

fn round_trips() {
    for value in SHARED {
        assert!(MODULUS.is_canonical(&value));
        round_trip(value);
    }
    for value in BN254_ONLY {
        let canonical = MODULUS == FieldInlineModulus::Bn254;
        assert_eq!(MODULUS.is_canonical(&value), canonical);
        if canonical {
            round_trip(value);
        }
    }
    let p_minus_one = modulus_plus(-1);
    assert!(MODULUS.is_canonical(&p_minus_one));
    assert!(!MODULUS.is_canonical(&modulus_plus(0)));
    assert!(!MODULUS.is_canonical(&modulus_plus(1)));
    round_trip(p_minus_one);
}

fn round_trip(value: [u64; 4]) {
    jolt::field_from_limbs!(1, value);
    let limbs: [u64; 4] = jolt::field_to_limbs!(1);
    assert_eq!(limbs, value);
    assert_eq!(jolt::field_to_limbs!(1, 6), widen(value));
}

/// −1 computed in the field reads out as p − 1, and p − 1 imported plus one is
/// zero.
fn arithmetic_matches_limbs() {
    jolt::field_load_imm!(2, 0);
    jolt::field_load_imm!(3, 1);
    jolt::field_sub!(4, 2, 3);
    let minus_one: [u64; 4] = jolt::field_to_limbs!(4);
    assert_eq!(minus_one, modulus_plus(-1));
    jolt::field_from_limbs!(5, minus_one);
    jolt::field_add!(6, 5, 3);
    jolt::field_assert_zero!(6);
}

fn widths() {
    jolt::field_from_limbs!(7, [42]);
    assert_eq!(jolt::field_to_limbs!(7, 1), [42]);
    assert_eq!(jolt::field_to_limbs!(7, 2), [42, 0]);
    assert_eq!(jolt::field_to_limbs!(7, 6), [42, 0, 0, 0, 0, 0]);
    jolt::field_from_limbs!(7, [5, 7]);
    assert_eq!(jolt::field_to_limbs!(7, 2), [5, 7]);
}

/// A distinct two-limb value for each register, canonical in both fields.
fn live_value(register: u64) -> [u64; 2] {
    [
        0x0101_0101_0101_0101 * (register + 1),
        0x0f0f_0f0f_0f0f_0f0f ^ register,
    ]
}

macro_rules! each_register {
    ($apply:ident) => {
        $apply!(0);
        $apply!(1);
        $apply!(2);
        $apply!(3);
        $apply!(4);
        $apply!(5);
        $apply!(6);
        $apply!(7);
        $apply!(8);
        $apply!(9);
        $apply!(10);
        $apply!(11);
        $apply!(12);
        $apply!(13);
        $apply!(14);
        $apply!(15);
    };
}

/// Every readout consumes its register in place and restores it while the
/// other fifteen registers hold live values; reading each register twice and
/// then all of them again shows the source and every bystander survive.
fn registers_survive_readouts() {
    macro_rules! load {
        ($register:literal) => {
            jolt::field_from_limbs!($register, live_value($register))
        };
    }
    macro_rules! check {
        ($register:literal) => {
            assert_eq!(jolt::field_to_limbs!($register, 2), live_value($register))
        };
    }
    each_register!(load);
    each_register!(check);
    each_register!(check);
}

fn imports_replace_the_destination() {
    jolt::field_from_limbs!(9, [3, 1]);
    jolt::field_from_limbs!(9, [4]);
    assert_eq!(jolt::field_to_limbs!(9, 2), [4, 0]);
    jolt::field_from_limbs!(9, [6, 2]);
    assert_eq!(jolt::field_to_limbs!(9, 2), [6, 2]);
}

/// `p + offset` for a small offset, as four little-endian limbs.
fn modulus_plus(offset: i64) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    limbs[..MODULUS.limbs().len()].copy_from_slice(MODULUS.limbs());
    let mut carry = i128::from(offset);
    for limb in &mut limbs {
        let sum = i128::from(*limb) + carry;
        *limb = sum as u64;
        carry = sum >> 64;
    }
    limbs
}

fn widen<const N: usize, const W: usize>(limbs: [u64; N]) -> [u64; W] {
    let mut wide = [0; W];
    wide[..N].copy_from_slice(&limbs);
    wide
}
