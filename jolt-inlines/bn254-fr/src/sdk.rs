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

/// Emit a FieldOp with two Fr sources.
///
/// Guest: loads `a`/`b` into field_regs[1]/[2], invokes the FieldOp, reads
/// field_regs[3] back into `out`. Host: delegates to ark-bn254.
#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn binary_op<const FUNCT3: u32>(a: &Fr, b: &Fr, out: &mut Fr) {
    // Layout: field_regs[1] ← a, field_regs[2] ← b, FieldOp, field_regs[3] → out.
    load_fr_into_field_reg::<1>(a);
    load_fr_into_field_reg::<2>(b);
    unsafe {
        // .insn r opcode, funct3, funct7, rd, rs1, rs2
        // Encoded manually as a .word to avoid assembler limitations on
        // custom-opcode immediates.
        let word: u32 = (crate::BN254_FR_FUNCT7 << 25)
            | (2u32 << 20)  // frs2 = 2
            | (1u32 << 15)  // frs1 = 1
            | (FUNCT3 << 12)
            | (3u32 << 7)   // frd  = 3
            | crate::INLINE_OPCODE;
        core::arch::asm!(".word {w}", w = const word);
    }
    store_field_reg_into_fr::<3>(out);
}

#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn unary_op<const FUNCT3: u32>(a: &Fr, out: &mut Fr) {
    load_fr_into_field_reg::<1>(a);
    unsafe {
        let word: u32 = (crate::BN254_FR_FUNCT7 << 25)
            | (0u32 << 20)
            | (1u32 << 15)
            | (FUNCT3 << 12)
            | (3u32 << 7)
            | crate::INLINE_OPCODE;
        core::arch::asm!(".word {w}", w = const word);
    }
    store_field_reg_into_fr::<3>(out);
}

#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn load_fr_into_field_reg<const FRD: u32>(a: &Fr) {
    for limb_idx in 0..4u32 {
        let limb = a.limbs[limb_idx as usize];
        unsafe {
            // FMovIntToFieldLimb: rd=frd, rs1=(any int reg holding limb),
            // rs2=limb_idx. We use inline-asm register binding to stash
            // the limb into a scratch integer register that the tracer
            // reads as x[rs1].
            let word: u32 = (crate::BN254_FR_FUNCT7 << 25)
                | (limb_idx << 20)
                | (10u32 << 15)        // rs1 = x10 (a0) – bound below
                | (crate::FUNCT3_FMOV_I2F << 12)
                | (FRD << 7)
                | crate::INLINE_OPCODE;
            core::arch::asm!(
                "mv a0, {limb}",
                ".word {w}",
                limb = in(reg) limb,
                w = const word,
                out("a0") _,
            );
        }
    }
}

#[cfg(all(target_arch = "riscv64", not(feature = "host")))]
#[inline]
fn store_field_reg_into_fr<const FRS1: u32>(out: &mut Fr) {
    for limb_idx in 0..4u32 {
        let mut limb: u64;
        unsafe {
            let word: u32 = (crate::BN254_FR_FUNCT7 << 25)
                | (limb_idx << 20)
                | (FRS1 << 15)
                | (crate::FUNCT3_FMOV_F2I << 12)
                | (10u32 << 7)         // rd = x10 (a0)
                | crate::INLINE_OPCODE;
            core::arch::asm!(
                ".word {w}",
                "mv {out}, a0",
                w = const word,
                out = out(reg) limb,
                out("a0") _,
            );
        }
        out.limbs[limb_idx as usize] = limb;
    }
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
