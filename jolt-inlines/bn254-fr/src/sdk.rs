//! Low-level SDK primitives for emitting BN254 Fr coprocessor instructions.
//!
//! These functions emit the R-type asm directly and take field-register indices
//! `frd/frs1/frs2 ∈ 0..16` as encoded register operands. Callers are responsible
//! for allocating registers (a higher-level `Fr` newtype with register-allocation
//! tracking lives on top of these primitives — to be added when downstream code
//! needs it).

#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
macro_rules! emit_field_op_r {
    ($funct7:expr, $funct3:expr, $rd:expr, $rs1:expr, $rs2:expr) => {{
        let word: u32 = ($funct7 << 25)
            | (($rs2 & 0x1f) << 20)
            | (($rs1 & 0x1f) << 15)
            | ($funct3 << 12)
            | (($rd & 0x1f) << 7)
            | $crate::FIELD_OP_OPCODE;
        core::arch::asm!(
            ".4byte {word}",
            word = const word,
            options(nostack, preserves_flags)
        );
    }};
}

/// FMUL: `FReg[frd] = FReg[frs1] · FReg[frs2]` over BN254 Fr.
///
/// # Safety
/// `frd`, `frs1`, `frs2` must be valid field-register indices (`0..16`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn fmul(frd: u32, frs1: u32, frs2: u32) {
    emit_field_op_r!(
        crate::BN254_FR_FUNCT7,
        crate::FUNCT3_FMUL,
        frd,
        frs1,
        frs2
    );
}

/// FADD: `FReg[frd] = FReg[frs1] + FReg[frs2]` over BN254 Fr.
///
/// # Safety
/// `frd`, `frs1`, `frs2` must be valid field-register indices (`0..16`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn fadd(frd: u32, frs1: u32, frs2: u32) {
    emit_field_op_r!(
        crate::BN254_FR_FUNCT7,
        crate::FUNCT3_FADD,
        frd,
        frs1,
        frs2
    );
}

/// FSUB: `FReg[frd] = FReg[frs1] − FReg[frs2]` over BN254 Fr.
///
/// # Safety
/// `frd`, `frs1`, `frs2` must be valid field-register indices (`0..16`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn fsub(frd: u32, frs1: u32, frs2: u32) {
    emit_field_op_r!(
        crate::BN254_FR_FUNCT7,
        crate::FUNCT3_FSUB,
        frd,
        frs1,
        frs2
    );
}

/// FINV: `FReg[frd] = FReg[frs1]⁻¹` over BN254 Fr (with `0⁻¹ = 0`).
///
/// # Safety
/// `frd`, `frs1` must be valid field-register indices (`0..16`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn finv(frd: u32, frs1: u32) {
    emit_field_op_r!(crate::BN254_FR_FUNCT7, crate::FUNCT3_FINV, frd, frs1, 0);
}

/// FieldAssertEq: assert `FReg[frs1] == FReg[frs2]`; no write.
///
/// # Safety
/// `frs1`, `frs2` must be valid field-register indices (`0..16`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn field_assert_eq(frs1: u32, frs2: u32) {
    emit_field_op_r!(
        crate::BN254_FR_FUNCT7,
        crate::FUNCT3_FIELD_ASSERT_EQ,
        0,
        frs1,
        frs2
    );
}

/// FieldMov: `FReg[frd] = [XReg[rs1], 0, 0, 0]` (low-limb load).
///
/// # Safety
/// `frd` must be a valid field-register index (`0..16`); `rs1` must be a valid
/// integer-register index (`0..32`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn field_mov(frd: u32, rs1: u32) {
    emit_field_op_r!(
        crate::BN254_FR_FUNCT7,
        crate::FUNCT3_FIELD_MOV,
        frd,
        rs1,
        0
    );
}

/// FieldSLL64: `FReg[frd] = XReg[rs1] · 2^64` (lands in limb 1).
///
/// # Safety
/// `frd` must be a valid field-register index (`0..16`); `rs1` must be a valid
/// integer-register index (`0..32`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn field_sll64(frd: u32, rs1: u32) {
    emit_field_op_r!(
        crate::BN254_FR_SLL_FUNCT7,
        crate::FUNCT3_FIELD_SLL64,
        frd,
        rs1,
        0
    );
}

/// FieldSLL128: `FReg[frd] = XReg[rs1] · 2^128` (lands in limb 2).
///
/// # Safety
/// `frd` must be a valid field-register index (`0..16`); `rs1` must be a valid
/// integer-register index (`0..32`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn field_sll128(frd: u32, rs1: u32) {
    emit_field_op_r!(
        crate::BN254_FR_SLL_FUNCT7,
        crate::FUNCT3_FIELD_SLL128,
        frd,
        rs1,
        0
    );
}

/// FieldSLL192: `FReg[frd] = XReg[rs1] · 2^192` (lands in limb 3).
///
/// # Safety
/// `frd` must be a valid field-register index (`0..16`); `rs1` must be a valid
/// integer-register index (`0..32`).
#[cfg(all(
    not(feature = "host"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[inline(always)]
pub unsafe fn field_sll192(frd: u32, rs1: u32) {
    emit_field_op_r!(
        crate::BN254_FR_SLL_FUNCT7,
        crate::FUNCT3_FIELD_SLL192,
        frd,
        rs1,
        0
    );
}

#[cfg(feature = "host")]
pub use host::*;

#[cfg(feature = "host")]
mod host {
    //! Host-side facade. Computes the same Fr semantics over `ark-bn254::Fr`
    //! against a stack-allocated 16-register state. Used by tests and by host
    //! tooling that wants to mirror the guest's FR register file without going
    //! through the tracer.

    use ark_bn254::Fr;
    use ark_ff::{Field, PrimeField, Zero};

    fn fr_from_limbs(limbs: [u64; 4]) -> Fr {
        let mut bytes = [0u8; 32];
        for (i, limb) in limbs.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        Fr::from_le_bytes_mod_order(&bytes)
    }

    /// A 16-register BN254 Fr file, mirroring the tracer's `field_regs` state.
    #[derive(Clone, Default)]
    pub struct FieldRegFile {
        regs: [Fr; super::super::FIELD_REG_COUNT],
    }

    impl FieldRegFile {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn get(&self, idx: usize) -> Fr {
            self.regs[idx]
        }

        pub fn fmul(&mut self, frd: usize, frs1: usize, frs2: usize) {
            self.regs[frd] = self.regs[frs1] * self.regs[frs2];
        }

        pub fn fadd(&mut self, frd: usize, frs1: usize, frs2: usize) {
            self.regs[frd] = self.regs[frs1] + self.regs[frs2];
        }

        pub fn fsub(&mut self, frd: usize, frs1: usize, frs2: usize) {
            self.regs[frd] = self.regs[frs1] - self.regs[frs2];
        }

        pub fn finv(&mut self, frd: usize, frs1: usize) {
            self.regs[frd] = self.regs[frs1].inverse().unwrap_or_else(Fr::zero);
        }

        pub fn field_assert_eq(&self, frs1: usize, frs2: usize) {
            assert_eq!(self.regs[frs1], self.regs[frs2]);
        }

        pub fn field_mov(&mut self, frd: usize, x: u64) {
            self.regs[frd] = fr_from_limbs([x, 0, 0, 0]);
        }

        pub fn field_sll64(&mut self, frd: usize, x: u64) {
            self.regs[frd] = fr_from_limbs([0, x, 0, 0]);
        }

        pub fn field_sll128(&mut self, frd: usize, x: u64) {
            self.regs[frd] = fr_from_limbs([0, 0, x, 0]);
        }

        pub fn field_sll192(&mut self, frd: usize, x: u64) {
            self.regs[frd] = fr_from_limbs([0, 0, 0, x]);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fmul_matches_native_fr() {
            let mut regs = FieldRegFile::new();
            regs.field_mov(0, 7);
            regs.field_mov(1, 11);
            regs.fmul(2, 0, 1);
            assert_eq!(regs.get(2), Fr::from(77u64));
        }

        #[test]
        fn fadd_fsub_round_trip() {
            let mut regs = FieldRegFile::new();
            regs.field_mov(0, 100);
            regs.field_mov(1, 42);
            regs.fadd(2, 0, 1);
            regs.fsub(3, 2, 1);
            assert_eq!(regs.get(3), Fr::from(100u64));
        }

        #[test]
        fn finv_of_zero_is_zero() {
            let mut regs = FieldRegFile::new();
            regs.finv(0, 0);
            assert_eq!(regs.get(0), Fr::zero());
        }

        #[test]
        fn sll_lands_in_correct_limb() {
            let mut regs = FieldRegFile::new();
            regs.field_sll64(0, 1);
            regs.field_sll128(1, 1);
            regs.field_sll192(2, 1);
            let two_pow_64 = Fr::from(2u64).pow([64u64]);
            let two_pow_128 = Fr::from(2u64).pow([128u64]);
            let two_pow_192 = Fr::from(2u64).pow([192u64]);
            assert_eq!(regs.get(0), two_pow_64);
            assert_eq!(regs.get(1), two_pow_128);
            assert_eq!(regs.get(2), two_pow_192);
        }
    }
}
