//! Checked conversions between field-inline registers and little-endian `u64`
//! limbs, behind [`field_to_limbs!`](crate::field_to_limbs) and
//! [`field_from_limbs!`](crate::field_from_limbs).
//!
//! The conversions run only on the riscv64 Jolt guest target. Anywhere else
//! they panic: there is no field register file to read or write, and returning
//! placeholder limbs would misrepresent a conversion that never happened. Guest
//! crates still compile natively, so a native call of a guest function that
//! converts limbs panics.

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
use core::arch::asm;

#[cfg(target_arch = "riscv64")]
use jolt_platform::spoil_proof;
pub use jolt_platform::FieldInlineModulus;

use crate::field_register;
#[cfg(target_arch = "riscv64")]
use crate::{
    field_inline_i_word, FIELD_INLINE_ADVICE_LIMB_FUNCT3, FIELD_INLINE_ADVICE_LIMB_FUNCT7,
    FIELD_INLINE_ASSERT_ZERO_FUNCT3, FIELD_INLINE_ASSERT_ZERO_FUNCT7,
    FIELD_INLINE_LOAD_ACCUMULATE_FROM_MEMORY_FUNCT7, FIELD_INLINE_LOAD_IMM_FUNCT3,
};
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
use crate::{
    field_inline_r_word, FIELD_INLINE_BRIDGE_X_REGISTER,
    FIELD_INLINE_LOAD_ACCUMULATE_FROM_REGISTER_FUNCT3, FIELD_INLINE_R_TYPE_FUNCT7,
};

/// The proof field this guest build assumes, meaningful only in guest builds.
///
/// jolt-host's field-inline guest builds receive the tracer's execution field
/// through [`FIELD_INLINE_MODULUS_ENV`](crate::FIELD_INLINE_MODULUS_ENV); builds
/// without it assume BN254, the field of the SDK's Dory prover. Every limb
/// conversion proves in-guest that this is the field executing the guest, so
/// a guest built for the wrong field cannot be proven.
// `option_env!` takes a literal: this spells `FIELD_INLINE_MODULUS_ENV`. A
// misspelling would make Akita guests assume BN254 and fail that binding.
pub const MODULUS: FieldInlineModulus = match option_env!("JOLT_FIELD_INLINE_MODULUS") {
    None => FieldInlineModulus::Bn254,
    Some(name) => match FieldInlineModulus::from_name(name) {
        Some(modulus) => modulus,
        None => panic!("JOLT_FIELD_INLINE_MODULUS must be `bn254` or `fp128`"),
    },
};

/// Limbs the modulus occupies. Readouts always split exactly this many.
#[cfg(target_arch = "riscv64")]
const MODULUS_LIMBS: usize = MODULUS.limbs().len();

/// The x-register FIELD_LOAD_ACCUMULATE_FROM_MEMORY writes each loaded word to
/// (`a1`), pinned by the asm operand constraints of [`bind_modulus`].
#[cfg(target_arch = "riscv64")]
const MEMORY_SCRATCH_X_REGISTER: u32 = 11;

#[doc(hidden)]
pub fn to_limbs<const REGISTER: u32, const N: usize>() -> [u64; N] {
    const { assert!(N > 0, "field_to_limbs! needs at least one limb") };
    let _ = const { field_register(REGISTER) };
    #[cfg(target_arch = "riscv64")]
    {
        let mut canonical = [0u64; MODULUS_LIMBS];
        for limb in &mut canonical[..MODULUS_LIMBS - 1] {
            *limb = advice_limb::<REGISTER>();
        }
        canonical[MODULUS_LIMBS - 1] = last_advice_limb::<REGISTER>();
        bind_modulus::<REGISTER>();
        // The advice rows fix the integer only modulo p: the limbs of p also
        // encode zero. A dishonest prover must not be able to finish a proof.
        if !MODULUS.is_canonical(&canonical) {
            spoil_proof();
        }
        // The limbs are now unique, so an oversized value is the guest's own
        // condition, not prover advice.
        assert!(
            canonical.iter().skip(N).all(|&limb| limb == 0),
            "field_to_limbs!: the field value does not fit in {N} limbs"
        );
        for &limb in canonical.iter().rev() {
            load_accumulate_from_register::<REGISTER>(limb);
        }
        let mut limbs = [0u64; N];
        for (limb, &canonical) in limbs.iter_mut().zip(&canonical) {
            *limb = canonical;
        }
        limbs
    }
    #[cfg(not(target_arch = "riscv64"))]
    unsupported_target()
}

#[doc(hidden)]
pub fn from_limbs<const REGISTER: u32, const N: usize>(limbs: [u64; N]) {
    const { assert!(N > 0, "field_from_limbs! needs at least one limb") };
    let _ = const { field_register(REGISTER) };
    #[cfg(target_arch = "riscv64")]
    {
        assert!(
            MODULUS.is_canonical(&limbs),
            "field_from_limbs!: the limbs encode an integer not below the field modulus"
        );
        load_zero::<REGISTER>();
        bind_modulus::<REGISTER>();
        // Canonical limbs past the modulus width are zero.
        for &limb in limbs.iter().take(MODULUS_LIMBS).rev() {
            load_accumulate_from_register::<REGISTER>(limb);
        }
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        let _ = limbs;
        unsupported_target()
    }
}

/// Updates field register `REGISTER` to `old * 2^64 + value`; see
/// [`field_load_accumulate_from_register!`](crate::field_load_accumulate_from_register).
#[doc(hidden)]
#[inline(always)]
pub fn load_accumulate_from_register<const REGISTER: u32>(value: u64) {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    // SAFETY: emits one fixed field-inline instruction word; its only register
    // contract is the value living in a0 for the duration of the block, which
    // the operand constraint provides. No memory is touched.
    unsafe {
        asm!(
            ".word {word}",
            word = const field_inline_r_word(
                FIELD_INLINE_R_TYPE_FUNCT7,
                FIELD_INLINE_LOAD_ACCUMULATE_FROM_REGISTER_FUNCT3,
                field_register(REGISTER),
                FIELD_INLINE_BRIDGE_X_REGISTER,
                0,
            ),
            in("x10") value,
            options(nostack),
        );
    }
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    let _ = value;
}

/// FIELD_ADVICE_LIMB in place: the low limb of `REGISTER` lands in a0 and the
/// quotient replaces `REGISTER`.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn advice_limb<const REGISTER: u32>() -> u64 {
    let limb: u64;
    // SAFETY: emits one fixed field-inline instruction word that writes only
    // field register REGISTER and a0, the declared output. No memory is
    // touched.
    unsafe {
        asm!(
            ".word {advice}",
            advice = const advice_limb_word(REGISTER),
            out("x10") limb,
            options(nostack),
        );
    }
    limb
}

/// The last FIELD_ADVICE_LIMB of a readout, then FIELD_ASSERT_ZERO on the
/// quotient it leaves. With the earlier rows, the zero residual pins
/// `value = Σ limb_i·2^(64·i) (mod p)` with every limb below 2^64. The
/// assertion directly follows the limb, so a failure names the residual.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn last_advice_limb<const REGISTER: u32>() -> u64 {
    let limb: u64;
    // SAFETY: emits two fixed field-inline instruction words: the first writes
    // only field register REGISTER and a0, the declared output; the second
    // writes nothing. No memory is touched.
    unsafe {
        asm!(
            ".word {advice}",
            ".word {assert_zero}",
            advice = const advice_limb_word(REGISTER),
            assert_zero = const assert_zero_word(REGISTER),
            out("x10") limb,
            options(nostack),
        );
    }
    limb
}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn load_zero<const REGISTER: u32>() {
    // SAFETY: emits one fixed field-inline instruction word that writes only
    // field register REGISTER. No memory is touched.
    unsafe {
        asm!(
            ".word {load_imm}",
            load_imm = const field_inline_i_word(
                FIELD_INLINE_LOAD_IMM_FUNCT3,
                field_register(REGISTER),
                0,
            ),
            options(nostack),
        );
    }
}

/// Proves [`MODULUS`] is the executing field while `REGISTER` holds zero, and
/// leaves it zero.
///
/// Accumulating the guest's modulus limbs into a zero register yields
/// `p_guest mod p_active`, and asserting zero proves `p_active` divides
/// `p_guest`. Both are prime, so the fields are equal; the tracer pins
/// [`FieldInlineModulus::limbs`] to the proof-field moduli. The limbs are read
/// from the program image, which the verifier fixes, one memory-sourced
/// accumulation per limb.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn bind_modulus<const REGISTER: u32>() {
    let modulus = MODULUS.limbs().as_ptr();
    // SAFETY: every load reads one word of the 'static modulus table at a0, at
    // an offset below its length; a1 receives each loaded word and is declared
    // clobbered. Only field register REGISTER changes, and the closing
    // assertion leaves it zero.
    unsafe {
        match MODULUS {
            FieldInlineModulus::Bn254 => asm!(
                ".word {limb3}",
                ".word {limb2}",
                ".word {limb1}",
                ".word {limb0}",
                ".word {assert_zero}",
                limb3 = const memory_accumulate_word(REGISTER, 3),
                limb2 = const memory_accumulate_word(REGISTER, 2),
                limb1 = const memory_accumulate_word(REGISTER, 1),
                limb0 = const memory_accumulate_word(REGISTER, 0),
                assert_zero = const assert_zero_word(REGISTER),
                in("x10") modulus,
                out("x11") _,
                options(nostack, readonly),
            ),
            FieldInlineModulus::Fp128 => asm!(
                ".word {limb1}",
                ".word {limb0}",
                ".word {assert_zero}",
                limb1 = const memory_accumulate_word(REGISTER, 1),
                limb0 = const memory_accumulate_word(REGISTER, 0),
                assert_zero = const assert_zero_word(REGISTER),
                in("x10") modulus,
                out("x11") _,
                options(nostack, readonly),
            ),
        }
    }
}

#[cfg(target_arch = "riscv64")]
const fn advice_limb_word(register: u32) -> u32 {
    field_inline_r_word(
        FIELD_INLINE_ADVICE_LIMB_FUNCT7,
        FIELD_INLINE_ADVICE_LIMB_FUNCT3,
        FIELD_INLINE_BRIDGE_X_REGISTER,
        field_register(register),
        field_register(register),
    )
}

#[cfg(target_arch = "riscv64")]
const fn assert_zero_word(register: u32) -> u32 {
    field_inline_r_word(
        FIELD_INLINE_ASSERT_ZERO_FUNCT7,
        FIELD_INLINE_ASSERT_ZERO_FUNCT3,
        0,
        field_register(register),
        0,
    )
}

/// FIELD_LOAD_ACCUMULATE_FROM_MEMORY of the word at `a0 + 8 * offset` into
/// `register`, through the `a1` scratch register.
#[cfg(target_arch = "riscv64")]
const fn memory_accumulate_word(register: u32, offset: u32) -> u32 {
    assert!(
        offset < 1 << 5,
        "memory-sourced accumulation offsets are 5 bits"
    );
    field_inline_r_word(
        FIELD_INLINE_LOAD_ACCUMULATE_FROM_MEMORY_FUNCT7 | offset,
        FIELD_INLINE_LOAD_ACCUMULATE_FROM_REGISTER_FUNCT3,
        MEMORY_SCRATCH_X_REGISTER,
        FIELD_INLINE_BRIDGE_X_REGISTER,
        field_register(register),
    )
}

#[cfg(not(target_arch = "riscv64"))]
fn unsupported_target() -> ! {
    panic!("field-inline register conversions run only on the riscv64 Jolt guest target")
}

#[cfg(all(test, not(target_arch = "riscv64")))]
mod tests {
    #[test]
    #[should_panic(expected = "run only on the riscv64 Jolt guest target")]
    fn native_readout_returns_no_placeholder() {
        let _: [u64; 4] = crate::field_to_limbs!(0);
    }

    #[test]
    #[should_panic(expected = "run only on the riscv64 Jolt guest target")]
    fn native_import_is_not_silently_skipped() {
        crate::field_from_limbs!(0, [1u64]);
    }
}
