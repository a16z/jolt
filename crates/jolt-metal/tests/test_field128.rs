//! Test-only 128-bit Montgomery field (p = 2^127 - 1, Mersenne prime M127).
//!
//! Provides minimal CPU field arithmetic and [`GpuFieldConfig`] for testing
//! that the Metal MSL generator works with N=4 limbs. Not intended for
//! production use — CPU performance is irrelevant.

use jolt_field::GpuFieldConfig;

/// p = 2^127 - 1
const MODULUS: [u32; 4] = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0x7FFF_FFFF];
/// INV32 * p ≡ -1 (mod 2^32). Since p mod 2^32 = 0xffffffff ≡ -1, INV32 = 1.
const INV32: u32 = 1;
/// R mod p = 2^128 mod (2^127 - 1) = 2
const ONE: [u32; 4] = [2, 0, 0, 0];
/// R^2 mod p = (2^128)^2 mod (2^127 - 1) = 4
const R2: [u32; 4] = [4, 0, 0, 0];

/// 128-bit test field element in Montgomery form (4 x u32, little-endian).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F128 {
    pub limbs: [u32; 4],
}

/// Marker type for GpuFieldConfig — never instantiated.
pub struct F128Config;

impl GpuFieldConfig for F128Config {
    const NUM_U32_LIMBS: usize = 4;
    const ACC_U32_LIMBS: usize = 10; // 2*4 + 2
    const FIELD_BYTE_SIZE: usize = 16;

    fn modulus_u32() -> &'static [u32] {
        &MODULUS
    }

    fn inv32() -> u32 {
        INV32
    }

    fn r2_u32() -> &'static [u32] {
        &R2
    }

    fn one_u32() -> &'static [u32] {
        &ONE
    }
}

impl F128 {
    pub const ZERO: Self = Self { limbs: [0; 4] };

    pub fn one() -> Self {
        Self { limbs: ONE }
    }

    pub fn from_u64(val: u64) -> Self {
        let raw = Self {
            limbs: [val as u32, (val >> 32) as u32, 0, 0],
        };
        raw.to_mont()
    }

    fn gte_modulus(a: &[u32; 4]) -> bool {
        for i in (0..4).rev() {
            if a[i] > MODULUS[i] {
                return true;
            }
            if a[i] < MODULUS[i] {
                return false;
            }
        }
        true
    }

    fn reduce(a: [u32; 4]) -> [u32; 4] {
        if !Self::gte_modulus(&a) {
            return a;
        }
        let mut result = [0u32; 4];
        let mut borrow: u64 = 0;
        for i in 0..4 {
            let diff = (a[i] as u64).wrapping_sub(MODULUS[i] as u64).wrapping_sub(borrow);
            result[i] = diff as u32;
            borrow = (diff >> 32) & 1;
        }
        result
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        let mut result = [0u32; 4];
        let mut carry: u64 = 0;
        for (i, r) in result.iter_mut().enumerate() {
            let s = self.limbs[i] as u64 + other.limbs[i] as u64 + carry;
            *r = s as u32;
            carry = s >> 32;
        }
        if carry != 0 {
            let mut borrow: u64 = 0;
            for (r, &m) in result.iter_mut().zip(MODULUS.iter()) {
                let diff = (*r as u64).wrapping_sub(m as u64).wrapping_sub(borrow);
                *r = diff as u32;
                borrow = (diff >> 32) & 1;
            }
        } else {
            result = Self::reduce(result);
        }
        Self { limbs: result }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        let mut result = [0u32; 4];
        let mut borrow: u64 = 0;
        for (i, r) in result.iter_mut().enumerate() {
            let diff = (self.limbs[i] as u64)
                .wrapping_sub(other.limbs[i] as u64)
                .wrapping_sub(borrow);
            *r = diff as u32;
            borrow = (diff >> 32) & 1;
        }
        if borrow != 0 {
            let mut carry: u64 = 0;
            for (r, &m) in result.iter_mut().zip(MODULUS.iter()) {
                let s = *r as u64 + m as u64 + carry;
                *r = s as u32;
                carry = s >> 32;
            }
        }
        Self { limbs: result }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    /// CIOS Montgomery multiplication.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        let mut t = [0u64; 5];

        for j in 0..4 {
            let bj = other.limbs[j] as u64;
            let mut carry: u64 = 0;
            for (ti, &ai) in t[..4].iter_mut().zip(self.limbs.iter()) {
                let prod = ai as u64 * bj + *ti + carry;
                *ti = prod & 0xFFFF_FFFF;
                carry = prod >> 32;
            }
            t[4] += carry;

            let m = ((t[0] as u32).wrapping_mul(INV32)) as u64;
            let mut carry: u64 = {
                let prod = m * MODULUS[0] as u64 + t[0];
                prod >> 32
            };
            for i in 1..4 {
                let prod = m * MODULUS[i] as u64 + t[i] + carry;
                t[i - 1] = prod & 0xFFFF_FFFF;
                carry = prod >> 32;
            }
            let s = t[4] + carry;
            t[3] = s & 0xFFFF_FFFF;
            t[4] = s >> 32;
        }

        let mut result = [0u32; 4];
        for (r, ti) in result.iter_mut().zip(t.iter()) {
            *r = *ti as u32;
        }
        Self {
            limbs: Self::reduce(result),
        }
    }

    pub fn sqr(self) -> Self {
        self.mul(self)
    }

    fn to_mont(self) -> Self {
        let r2 = Self { limbs: R2 };
        self.mul(r2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f128_basic_arithmetic() {
        let one = F128::one();
        let two = one.add(one);
        let three = two.add(one);

        assert_eq!(two.mul(three), F128::from_u64(6));
        assert_eq!(three.sub(two), one);
        assert_eq!(one.neg().add(one), F128::ZERO);
    }

    #[test]
    fn f128_from_u64() {
        let a = F128::from_u64(42);
        let b = F128::from_u64(100);
        assert_eq!(a.mul(b), F128::from_u64(4200));
    }

    #[test]
    fn f128_gpu_config_invariants() {
        assert_eq!(F128Config::NUM_U32_LIMBS, 4);
        assert_eq!(F128Config::ACC_U32_LIMBS, 10);
        assert_eq!(F128Config::FIELD_BYTE_SIZE, 16);
        assert_eq!(F128Config::modulus_u32().len(), 4);
        assert_eq!(F128Config::r2_u32().len(), 4);
        assert_eq!(F128Config::one_u32().len(), 4);
        // CIOS safety: top limb < 2^31
        assert!(F128Config::modulus_u32()[3] < (1u32 << 31));
    }

    #[test]
    fn f128_squaring() {
        let five = F128::from_u64(5);
        assert_eq!(five.sqr(), F128::from_u64(25));
    }
}
