//! Guest-facing `Fr` type with `add / sub / mul / inv` methods.
//!
//! `Fr` is a natural-form `[u64; 4]` wrapper. Each arithmetic method
//! dispatches to one of three implementations depending on compile-time
//! configuration:
//!
//! 1. **Host** (`feature = "host"` or non-RISC-V target): delegates to
//!    `ark_bn254::Fr` arithmetic.
//! 2. **Pass-1 / compute_advice** (`feature = "compute_advice"`, RISC-V):
//!    computes via ark-bn254 and writes the 4 result limbs to the host's
//!    advice tape using `VirtualHostIO`. The macro-built compute_advice
//!    ELF runs this path to populate the advice buffer.
//! 3. **Pass-2 / normal RISC-V** (no host/compute_advice flags, RISC-V):
//!    emits the v2 BN254 Fr coprocessor instruction sequence — load `a`/`b`
//!    via 7-cycle Horner sequences (`FieldMov` + `FieldSLL64/128/192` +
//!    `FieldAdd`), one `FieldOp` cycle, then 4 × `ADVICE_LD` + 7-cycle
//!    reconstruction + `FieldAssertEq` to bind the FieldOp output to the
//!    advice limbs.
//!
//! The two-pass advice-tape mechanism is set up by `#[jolt::provable]`'s
//! macro (`build_with_features(target_dir, &["compute_advice"])`), so guest
//! crates that use this SDK only need to enable the `compute_advice`
//! feature on `jolt-inlines-bn254-fr` from their own `compute_advice`
//! feature.
//!
//! ## Per-op cycle cost (RISC-V Pass 2)
//!
//! | Op    | Load A | Load B | FieldOp | Extract | Total |
//! |-------|-------:|-------:|--------:|--------:|------:|
//! | add   |     7  |     7  |       1 |     12  |   27  |
//! | sub   |     7  |     7  |       1 |     12  |   27  |
//! | mul   |     7  |     7  |       1 |     12  |   27  |
//! | inv   |     7  |     —  |       1 |     12  |   20  |
//!
//! v1's per-op cost was 13 (binary) / 9 (unary) using the dedicated
//! `FMov-I2F` / `FMov-F2I` per-limb transfer instructions — those don't
//! exist in v2 (the Bridge ABI is composite), so v2's naive boundary cost
//! is structurally larger. Amortizing over many in-field ops (pinned-slot
//! API) recovers + exceeds v1; that API is future work.

/// A natural-form BN254 Fr field element, stored as 4 little-endian u64
/// limbs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Fr {
    pub limbs: [u64; 4],
}

impl Fr {
    /// Zero element.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { limbs: [0; 4] }
    }

    /// One element.
    #[inline(always)]
    pub const fn one() -> Self {
        Self {
            limbs: [1, 0, 0, 0],
        }
    }

    /// Wrap raw limbs without validation. Caller asserts `value < p`.
    #[inline(always)]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self { limbs }
    }

    /// Raw limbs (copy).
    #[inline(always)]
    pub const fn to_limbs(&self) -> [u64; 4] {
        self.limbs
    }

    /// Returns `self + rhs mod p`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = Fr::zero();
        binary_op::<{ crate::FUNCT3_FADD }>(self, rhs, &mut out);
        out
    }

    /// Returns `self − rhs mod p`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = Fr::zero();
        binary_op::<{ crate::FUNCT3_FSUB }>(self, rhs, &mut out);
        out
    }

    /// Returns `self · rhs mod p`.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = Fr::zero();
        binary_op::<{ crate::FUNCT3_FMUL }>(self, rhs, &mut out);
        out
    }

    /// Returns `self⁻¹ mod p`, or `Fr::zero()` when `self == 0`.
    #[inline]
    pub fn inv(&self) -> Self {
        let mut out = Fr::zero();
        unary_op::<{ crate::FUNCT3_FINV }>(self, &mut out);
        out
    }
}

// ============================================================================
// Pass 2 / normal RISC-V: emit v2 FR coprocessor instructions, read advice.
// ============================================================================

#[cfg(all(
    target_arch = "riscv64",
    not(feature = "host"),
    not(feature = "compute_advice")
))]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    use crate::encode::*;

    // Horner-load a (x10..x13) into FR slot 1, scratching slots 4 and 5.
    const A_MOV: u32 = encode_field_mov(4, 10);
    const A_SLL64: u32 = encode_field_sll64(5, 11);
    const A_ADD1: u32 = encode_fadd(4, 4, 5);
    const A_SLL128: u32 = encode_field_sll128(5, 12);
    const A_ADD2: u32 = encode_fadd(4, 4, 5);
    const A_SLL192: u32 = encode_field_sll192(5, 13);
    const A_FINAL: u32 = encode_fadd(1, 4, 5);

    // Horner-load b (x14..x17) into FR slot 2.
    const B_MOV: u32 = encode_field_mov(4, 14);
    const B_SLL64: u32 = encode_field_sll64(5, 15);
    const B_ADD1: u32 = encode_fadd(4, 4, 5);
    const B_SLL128: u32 = encode_field_sll128(5, 16);
    const B_ADD2: u32 = encode_fadd(4, 4, 5);
    const B_SLL192: u32 = encode_field_sll192(5, 17);
    const B_FINAL: u32 = encode_fadd(2, 4, 5);

    // FieldOp f3 = op(f1, f2). FUNCT3 picks FADD/FSUB/FMUL.
    const fn op_word<const F: u32>() -> u32 {
        encode_r(BN254_FR_FUNCT7, 2, 1, F, 3, FIELD_OP_OPCODE)
    }

    // Reconstruct advice (now in x18..x21) into FR slot 4, then assert
    // f3 == f4 (catches advice that doesn't match the FieldOp output).
    const R_MOV: u32 = encode_field_mov(4, 18);
    const R_SLL64: u32 = encode_field_sll64(5, 19);
    const R_ADD1: u32 = encode_fadd(4, 4, 5);
    const R_SLL128: u32 = encode_field_sll128(5, 20);
    const R_ADD2: u32 = encode_fadd(4, 4, 5);
    const R_SLL192: u32 = encode_field_sll192(5, 21);
    const R_FINAL: u32 = encode_fadd(4, 4, 5);
    const ASSERT: u32 = encode_field_assert_eq(3, 4);

    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    unsafe {
        core::arch::asm!(
            ".word {a_mov}", ".word {a_sll64}", ".word {a_add1}",
            ".word {a_sll128}", ".word {a_add2}",
            ".word {a_sll192}", ".word {a_final}",
            ".word {b_mov}", ".word {b_sll64}", ".word {b_add1}",
            ".word {b_sll128}", ".word {b_add2}",
            ".word {b_sll192}", ".word {b_final}",
            ".word {op}",
            // 4 × ADVICE_LD reading the result limbs into x18..x21.
            // ADVICE_LD lives at jolt-sdk's CUSTOM_OPCODE 0x5B funct3 0b110,
            // I-type with rd as the destination, rs1=x0 (unused), imm=0.
            ".insn i 0x5B, 0b110, x18, x0, 0",
            ".insn i 0x5B, 0b110, x19, x0, 0",
            ".insn i 0x5B, 0b110, x20, x0, 0",
            ".insn i 0x5B, 0b110, x21, x0, 0",
            ".word {r_mov}", ".word {r_sll64}", ".word {r_add1}",
            ".word {r_sll128}", ".word {r_add2}",
            ".word {r_sll192}", ".word {r_final}",
            ".word {assert}",
            a_mov = const A_MOV, a_sll64 = const A_SLL64, a_add1 = const A_ADD1,
            a_sll128 = const A_SLL128, a_add2 = const A_ADD2,
            a_sll192 = const A_SLL192, a_final = const A_FINAL,
            b_mov = const B_MOV, b_sll64 = const B_SLL64, b_add1 = const B_ADD1,
            b_sll128 = const B_SLL128, b_add2 = const B_ADD2,
            b_sll192 = const B_SLL192, b_final = const B_FINAL,
            op = const op_word::<FUNCT3>(),
            r_mov = const R_MOV, r_sll64 = const R_SLL64, r_add1 = const R_ADD1,
            r_sll128 = const R_SLL128, r_add2 = const R_ADD2,
            r_sll192 = const R_SLL192, r_final = const R_FINAL,
            assert = const ASSERT,
            in("x10") a.limbs[0],
            in("x11") a.limbs[1],
            in("x12") a.limbs[2],
            in("x13") a.limbs[3],
            in("x14") b.limbs[0],
            in("x15") b.limbs[1],
            in("x16") b.limbs[2],
            in("x17") b.limbs[3],
            lateout("x18") r0,
            lateout("x19") r1,
            lateout("x20") r2,
            lateout("x21") r3,
        );
    }

    out.limbs = [r0, r1, r2, r3];
}

#[cfg(all(
    target_arch = "riscv64",
    not(feature = "host"),
    not(feature = "compute_advice")
))]
#[inline]
fn unary_op<const FUNCT3: u32>(a: &Fr, out: &mut Fr) {
    use crate::encode::*;

    const A_MOV: u32 = encode_field_mov(4, 10);
    const A_SLL64: u32 = encode_field_sll64(5, 11);
    const A_ADD1: u32 = encode_fadd(4, 4, 5);
    const A_SLL128: u32 = encode_field_sll128(5, 12);
    const A_ADD2: u32 = encode_fadd(4, 4, 5);
    const A_SLL192: u32 = encode_field_sll192(5, 13);
    const A_FINAL: u32 = encode_fadd(1, 4, 5);

    // FINV ignores frs2.
    const fn op_word<const F: u32>() -> u32 {
        encode_r(BN254_FR_FUNCT7, 0, 1, F, 3, FIELD_OP_OPCODE)
    }

    const R_MOV: u32 = encode_field_mov(4, 18);
    const R_SLL64: u32 = encode_field_sll64(5, 19);
    const R_ADD1: u32 = encode_fadd(4, 4, 5);
    const R_SLL128: u32 = encode_field_sll128(5, 20);
    const R_ADD2: u32 = encode_fadd(4, 4, 5);
    const R_SLL192: u32 = encode_field_sll192(5, 21);
    const R_FINAL: u32 = encode_fadd(4, 4, 5);
    const ASSERT: u32 = encode_field_assert_eq(3, 4);

    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    unsafe {
        core::arch::asm!(
            ".word {a_mov}", ".word {a_sll64}", ".word {a_add1}",
            ".word {a_sll128}", ".word {a_add2}",
            ".word {a_sll192}", ".word {a_final}",
            ".word {op}",
            ".insn i 0x5B, 0b110, x18, x0, 0",
            ".insn i 0x5B, 0b110, x19, x0, 0",
            ".insn i 0x5B, 0b110, x20, x0, 0",
            ".insn i 0x5B, 0b110, x21, x0, 0",
            ".word {r_mov}", ".word {r_sll64}", ".word {r_add1}",
            ".word {r_sll128}", ".word {r_add2}",
            ".word {r_sll192}", ".word {r_final}",
            ".word {assert}",
            a_mov = const A_MOV, a_sll64 = const A_SLL64, a_add1 = const A_ADD1,
            a_sll128 = const A_SLL128, a_add2 = const A_ADD2,
            a_sll192 = const A_SLL192, a_final = const A_FINAL,
            op = const op_word::<FUNCT3>(),
            r_mov = const R_MOV, r_sll64 = const R_SLL64, r_add1 = const R_ADD1,
            r_sll128 = const R_SLL128, r_add2 = const R_ADD2,
            r_sll192 = const R_SLL192, r_final = const R_FINAL,
            assert = const ASSERT,
            in("x10") a.limbs[0],
            in("x11") a.limbs[1],
            in("x12") a.limbs[2],
            in("x13") a.limbs[3],
            lateout("x18") r0,
            lateout("x19") r1,
            lateout("x20") r2,
            lateout("x21") r3,
        );
    }

    out.limbs = [r0, r1, r2, r3];
}

// ============================================================================
// Pass 1 / compute_advice (RISC-V): compute via ark-bn254 + write 4 advice u64s.
// ============================================================================

#[cfg(all(
    target_arch = "riscv64",
    not(feature = "host"),
    feature = "compute_advice"
))]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    use ark_bn254::Fr as ArkFr;
    let af = limbs_to_ark(&a.limbs);
    let bf = limbs_to_ark(&b.limbs);
    let r: ArkFr = match FUNCT3 {
        crate::FUNCT3_FMUL => af * bf,
        crate::FUNCT3_FADD => af + bf,
        crate::FUNCT3_FSUB => af - bf,
        _ => panic!("binary_op: unsupported funct3"),
    };
    let limbs = ark_to_limbs(&r);
    out.limbs = limbs;
    advice_write_4_u64(&limbs);
}

#[cfg(all(
    target_arch = "riscv64",
    not(feature = "host"),
    feature = "compute_advice"
))]
#[inline]
fn unary_op<const FUNCT3: u32>(a: &Fr, out: &mut Fr) {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::Field;
    let af = limbs_to_ark(&a.limbs);
    let r: ArkFr = match FUNCT3 {
        crate::FUNCT3_FINV => af.inverse().unwrap_or(ArkFr::from(0u64)),
        _ => panic!("unary_op: unsupported funct3"),
    };
    let limbs = ark_to_limbs(&r);
    out.limbs = limbs;
    advice_write_4_u64(&limbs);
}

/// Write 4 u64 limbs to the advice tape via VirtualHostIO. Mirrors
/// `jolt::AdviceWriter::write_u64`'s call sequence; inlined here so the
/// inlines crate doesn't need a jolt-sdk dependency.
///
/// The host-side advice-write call ID is `0xADBABE` (defined in
/// `jolt-platform/src/advice.rs::JOLT_ADVICE_WRITE_CALL_ID`). Vendored
/// here as a const so this crate doesn't take a jolt-platform dependency;
/// if upstream changes, the runtime advice-tape write fails.
#[cfg(all(
    target_arch = "riscv64",
    not(feature = "host"),
    feature = "compute_advice"
))]
#[inline]
fn advice_write_4_u64(limbs: &[u64; 4]) {
    const JOLT_ADVICE_WRITE_CALL_ID: u64 = 0xADBABE;
    for limb in limbs {
        let bytes = limb.to_le_bytes();
        let src_ptr = bytes.as_ptr() as u64;
        let len = 8u64;
        unsafe {
            core::arch::asm!(
                ".insn i 0x5B, 2, x0, x0, 0",  // VirtualHostIO
                in("a0") JOLT_ADVICE_WRITE_CALL_ID,
                in("a1") src_ptr,
                in("a2") len,
                options(nostack, preserves_flags)
            );
        }
    }
}

// ============================================================================
// Host (or non-RISC-V): direct ark-bn254. No advice-tape interaction.
// ============================================================================

#[cfg(feature = "host")]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    use ark_bn254::Fr as ArkFr;
    let af = limbs_to_ark(&a.limbs);
    let bf = limbs_to_ark(&b.limbs);
    let r: ArkFr = match FUNCT3 {
        crate::FUNCT3_FMUL => af * bf,
        crate::FUNCT3_FADD => af + bf,
        crate::FUNCT3_FSUB => af - bf,
        _ => panic!("binary_op: unsupported funct3 {:#x}", FUNCT3),
    };
    out.limbs = ark_to_limbs(&r);
}

#[cfg(feature = "host")]
#[inline]
fn unary_op<const FUNCT3: u32>(a: &Fr, out: &mut Fr) {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::Field;
    let af = limbs_to_ark(&a.limbs);
    let r: ArkFr = match FUNCT3 {
        crate::FUNCT3_FINV => af.inverse().unwrap_or(ArkFr::from(0u64)),
        _ => panic!("unary_op: unsupported funct3 {:#x}", FUNCT3),
    };
    out.limbs = ark_to_limbs(&r);
}

// ============================================================================
// Non-RISC-V, no-host fallback: panic at runtime. Don't expect anyone to
// call Fr arithmetic in this configuration; if they do they need to enable
// `--features host`.
// ============================================================================

#[cfg(all(not(target_arch = "riscv64"), not(feature = "host")))]
#[inline]
fn binary_op<const FUNCT3: u32>(_a: &Fr, _b: &Fr, _out: &mut Fr) {
    panic!(
        "Fr arithmetic on non-RISC-V targets requires --features host. \
         (FUNCT3={FUNCT3:#x})"
    );
}

#[cfg(all(not(target_arch = "riscv64"), not(feature = "host")))]
#[inline]
fn unary_op<const FUNCT3: u32>(_a: &Fr, _out: &mut Fr) {
    panic!(
        "Fr arithmetic on non-RISC-V targets requires --features host. \
         (FUNCT3={FUNCT3:#x})"
    );
}

// ============================================================================
// Limb ↔ ark conversion shared between host + compute_advice paths.
// ============================================================================

#[cfg(any(feature = "host", feature = "compute_advice"))]
#[inline]
// Dead-code allow gated on the cfg combo where it fires (RISC-V check on
// non-RISC-V host with compute_advice feature flips on the helper but not
// the caller). Workspace policy is #[expect] not #[allow], so use cfg_attr.
#[cfg_attr(
    all(not(target_arch = "riscv64"), feature = "compute_advice", not(feature = "host")),
    expect(dead_code, reason = "consumed only on RISC-V compute_advice build")
)]
fn limbs_to_ark(limbs: &[u64; 4]) -> ark_bn254::Fr {
    use ark_ff::PrimeField;
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
}

#[cfg(any(feature = "host", feature = "compute_advice"))]
#[inline]
#[allow(dead_code)]
fn ark_to_limbs(fr: &ark_bn254::Fr) -> [u64; 4] {
    use ark_ff::{BigInteger, PrimeField};
    let bytes = fr.into_bigint().to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = core::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}
