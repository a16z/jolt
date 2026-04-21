//! Guest-facing `Fr` type with add / sub / mul / inv methods.
//!
//! `Fr` is a natural-form `[u64; 4]` wrapper. Arithmetic methods emit the
//! BN254 Fr coprocessor's RISC-V instructions when compiled for RISC-V
//! (no-std guest target); on the host they delegate to `ark-bn254` for
//! parity.
//!
//! ## ABI
//!
//! The limb-register ABI moves each `Fr` value between integer registers
//! (x[i]) and field registers (field_regs[f]) via FMovIntToFieldLimb (4
//! cycles) before each FieldOp, then back via FMovFieldToIntLimb (4 cycles)
//! after. Asymptotic cost per `Fr::mul`: 4 + 1 + 4 = 9 traced cycles (+
//! the schoolbook-check R1CS chain — task #59).
//!
//! ## Use
//!
//! ```no_run
//! use jolt_inlines_bn254_fr::Fr;
//! let a = Fr::from_limbs([5, 0, 0, 0]);
//! let b = Fr::from_limbs([7, 0, 0, 0]);
//! let c = a.mul(&b);
//! assert_eq!(c.to_limbs(), [35, 0, 0, 0]);
//! ```

/// A natural-form BN254 Fr field element, stored as 4 little-endian u64 limbs.
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
        Self::binary_op::<{ crate::FUNCT3_FADD }>(self, rhs)
    }

    /// Returns `self - rhs mod p`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::binary_op::<{ crate::FUNCT3_FSUB }>(self, rhs)
    }

    /// Returns `self * rhs mod p`.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self::binary_op::<{ crate::FUNCT3_FMUL }>(self, rhs)
    }

    /// Returns `self^{-1} mod p`, or `Fr::zero()` when `self == 0`.
    #[inline]
    pub fn inv(&self) -> Self {
        let mut out = Fr::zero();
        unary_op::<{ crate::FUNCT3_FINV }>(self, &mut out);
        out
    }

    #[inline]
    fn binary_op<const FUNCT3: u32>(a: &Self, b: &Self) -> Self {
        let mut out = Fr::zero();
        binary_op::<FUNCT3>(a, b, &mut out);
        out
    }
}

// -------- Guest (RISC-V) and host (ark-bn254) dispatch ------------------------
//
// # ABI for the BN254 Fr native-field coprocessor
//
// Every `Fr::{add,sub,mul,inv}` call emits a single inline-asm block that
// holds the entire load-compute-store sequence. The compiler binds:
//
//   a.limbs[0..4]  →  a0..a3     (x10..x13)
//   b.limbs[0..4]  →  a4..a7     (x14..x17)    [binary ops only]
//   out.limbs[0..4]←  a8..a11    (x18..x21)
//
// Those register assignments are load-bearing for **task #52**'s limb-to-Fr
// bridge sumcheck. Keeping all eight source limbs live in x10..x17 across
// the FieldOp instruction means the bridge can state its identity as:
//
//   FieldOpA(r_cycle)  ==  Σ_{k=0..3}  RegVal(10+k, r_cycle) · 2^{64·k}
//   FieldOpB(r_cycle)  ==  Σ_{k=0..3}  RegVal(14+k, r_cycle) · 2^{64·k}
//
// on every FieldOp cycle, where `r_cycle` is a single opening point and the
// register reads come from the existing Registers Twist. No FMov-cycle
// traversal, no auxiliary per-limb witness polys. Single sumcheck over
// cycles, gated by the FieldOp circuit flag.
//
// Any change to the register allocation below MUST be mirrored in the
// bridge Module's constraint — see `specs/native-field-registers.md`
// Phase 3.

/// Emit a FieldOp with two Fr sources (FADD / FSUB / FMUL).
///
/// Guest: loads `a` → field_regs[1] (from a0..a3), `b` → field_regs[2] (from
/// a4..a7), invokes the FieldOp writing field_regs[3], reads field_regs[3]
/// back into a8..a11. Host: delegates to ark-bn254.
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    // Instruction-word encodings. Each `limb_idx` / `rs1` / `frd` field is
    // fixed by the ABI, so these resolve to compile-time constants.
    const LA0: u32 = i2f_word(10, 0, 1);
    const LA1: u32 = i2f_word(11, 1, 1);
    const LA2: u32 = i2f_word(12, 2, 1);
    const LA3: u32 = i2f_word(13, 3, 1);
    const LB0: u32 = i2f_word(14, 0, 2);
    const LB1: u32 = i2f_word(15, 1, 2);
    const LB2: u32 = i2f_word(16, 2, 2);
    const LB3: u32 = i2f_word(17, 3, 2);
    const SR0: u32 = f2i_word(18, 3, 0);
    const SR1: u32 = f2i_word(19, 3, 1);
    const SR2: u32 = f2i_word(20, 3, 2);
    const SR3: u32 = f2i_word(21, 3, 3);
    // FieldOp binary form: frs1=1, frs2=2, frd=3. const-evaluable because
    // `field_op_word` is a `const fn` and FUNCT3 is a const generic.
    const fn op_word<const F: u32>() -> u32 {
        field_op_word::<F>(1, 2, 3)
    }

    unsafe {
        core::arch::asm!(
            // Load a into field_regs[1] from x10..x13
            ".word {la0}", ".word {la1}", ".word {la2}", ".word {la3}",
            // Load b into field_regs[2] from x14..x17
            ".word {lb0}", ".word {lb1}", ".word {lb2}", ".word {lb3}",
            // field_regs[3] = op(field_regs[1], field_regs[2])
            ".word {op}",
            // Store field_regs[3] back into x18..x21
            ".word {wr0}", ".word {wr1}", ".word {wr2}", ".word {wr3}",
            la0 = const LA0, la1 = const LA1, la2 = const LA2, la3 = const LA3,
            lb0 = const LB0, lb1 = const LB1, lb2 = const LB2, lb3 = const LB3,
            op = const op_word::<FUNCT3>(),
            wr0 = const SR0, wr1 = const SR1, wr2 = const SR2, wr3 = const SR3,
            // Rust inline asm on RISC-V uses x-prefixed physical register
            // names, not ABI mnemonics (a0→x10, a4→x14, a8→x18, ...).
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

    out.limbs[0] = r0;
    out.limbs[1] = r1;
    out.limbs[2] = r2;
    out.limbs[3] = r3;
}

/// Emit a FieldOp with one Fr source (FINV). Layout mirrors `binary_op` but
/// skips the b-loading cycles; frs2 is ignored by the FieldOp decoder.
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn unary_op<const FUNCT3: u32>(a: &Fr, out: &mut Fr) {
    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    const LA0: u32 = i2f_word(10, 0, 1);
    const LA1: u32 = i2f_word(11, 1, 1);
    const LA2: u32 = i2f_word(12, 2, 1);
    const LA3: u32 = i2f_word(13, 3, 1);
    const SR0: u32 = f2i_word(18, 3, 0);
    const SR1: u32 = f2i_word(19, 3, 1);
    const SR2: u32 = f2i_word(20, 3, 2);
    const SR3: u32 = f2i_word(21, 3, 3);
    // FINV unary form: frs1=1, frs2=0, frd=3.
    const fn op_word<const F: u32>() -> u32 {
        field_op_word::<F>(1, 0, 3)
    }

    unsafe {
        core::arch::asm!(
            ".word {la0}", ".word {la1}", ".word {la2}", ".word {la3}",
            ".word {op}",
            ".word {wr0}", ".word {wr1}", ".word {wr2}", ".word {wr3}",
            la0 = const LA0, la1 = const LA1, la2 = const LA2, la3 = const LA3,
            op = const op_word::<FUNCT3>(),
            wr0 = const SR0, wr1 = const SR1, wr2 = const SR2, wr3 = const SR3,
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

    out.limbs[0] = r0;
    out.limbs[1] = r1;
    out.limbs[2] = r2;
    out.limbs[3] = r3;
}

/// Encode a FMovIntToFieldLimb instruction word with the given rs1 (0..31),
/// limb index (0..3), and frd (0..15).
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
const fn i2f_word(rs1: u32, limb_idx: u32, frd: u32) -> u32 {
    (crate::BN254_FR_FUNCT7 << 25)
        | (limb_idx << 20)
        | (rs1 << 15)
        | (crate::FUNCT3_FMOV_I2F << 12)
        | (frd << 7)
        | crate::INLINE_OPCODE
}

/// Encode a FMovFieldToIntLimb instruction word with the given rd, frs1, and
/// limb index.
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
const fn f2i_word(rd: u32, frs1: u32, limb_idx: u32) -> u32 {
    (crate::BN254_FR_FUNCT7 << 25)
        | (limb_idx << 20)
        | (frs1 << 15)
        | (crate::FUNCT3_FMOV_F2I << 12)
        | (rd << 7)
        | crate::INLINE_OPCODE
}

/// Encode a FieldOp instruction word. `FUNCT3` selects FMUL/FADD/FSUB/FINV.
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
const fn field_op_word<const FUNCT3: u32>(frs1: u32, frs2: u32, frd: u32) -> u32 {
    (crate::BN254_FR_FUNCT7 << 25)
        | (frs2 << 20)
        | (frs1 << 15)
        | (FUNCT3 << 12)
        | (frd << 7)
        | crate::INLINE_OPCODE
}

// -------- Host implementation (for prover / bench / host tests) --------------

#[cfg(feature = "host")]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::{BigInteger, PrimeField};
    let af = limbs_to_ark(&a.limbs);
    let bf = limbs_to_ark(&b.limbs);
    let r: ArkFr = match FUNCT3 {
        crate::FUNCT3_FMUL => af * bf,
        crate::FUNCT3_FADD => af + bf,
        crate::FUNCT3_FSUB => af - bf,
        _ => panic!("binary_op: unsupported funct3 {:#x}", FUNCT3),
    };
    out.limbs = ark_to_limbs(&r);
    let _ = <ArkFr as PrimeField>::MODULUS.to_bytes_le(); // type anchor
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

#[cfg(feature = "host")]
#[inline]
fn limbs_to_ark(limbs: &[u64; 4]) -> ark_bn254::Fr {
    use ark_ff::PrimeField;
    let mut bytes = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
}

#[cfg(feature = "host")]
#[inline]
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

// Stubs for builds that are neither RISC-V guest nor `host`-feature host.
#[cfg(all(
    not(feature = "host"),
    not(all(target_arch = "riscv64", not(feature = "host")))
))]
#[inline]
fn binary_op<const FUNCT3: u32>(_a: &Fr, _b: &Fr, _out: &mut Fr) {
    panic!(
        "jolt-inlines-bn254-fr: binary_op called without RISC-V guest or `host` feature enabled"
    );
}

#[cfg(all(
    not(feature = "host"),
    not(all(target_arch = "riscv64", not(feature = "host")))
))]
#[inline]
fn unary_op<const FUNCT3: u32>(_a: &Fr, _out: &mut Fr) {
    panic!(
        "jolt-inlines-bn254-fr: unary_op called without RISC-V guest or `host` feature enabled"
    );
}

// -------- Host-side tests --------------------------------------------------

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;

    #[test]
    fn add_small() {
        let a = Fr::from_limbs([5, 0, 0, 0]);
        let b = Fr::from_limbs([7, 0, 0, 0]);
        assert_eq!(a.add(&b).to_limbs(), [12, 0, 0, 0]);
    }

    #[test]
    fn mul_small() {
        let a = Fr::from_limbs([5, 0, 0, 0]);
        let b = Fr::from_limbs([7, 0, 0, 0]);
        assert_eq!(a.mul(&b).to_limbs(), [35, 0, 0, 0]);
    }

    #[test]
    fn inv_one_is_one() {
        let one = Fr::one();
        assert_eq!(one.inv().to_limbs(), [1, 0, 0, 0]);
    }

    #[test]
    fn inv_times_self_is_one() {
        let a = Fr::from_limbs([42, 0, 0, 0]);
        let inv_a = a.inv();
        let prod = a.mul(&inv_a);
        assert_eq!(prod.to_limbs(), [1, 0, 0, 0]);
    }

    #[test]
    fn sub_wraps_via_modulus() {
        // 0 - 1 mod p = p - 1
        let zero = Fr::zero();
        let one = Fr::one();
        let p_minus_one = zero.sub(&one).to_limbs();
        assert_eq!(
            p_minus_one,
            [
                crate::BN254_FR_MODULUS[0] - 1,
                crate::BN254_FR_MODULUS[1],
                crate::BN254_FR_MODULUS[2],
                crate::BN254_FR_MODULUS[3],
            ]
        );
    }
}
