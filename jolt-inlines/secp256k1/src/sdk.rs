//! secp256k1 operations optimized for Jolt zkVM.

use ark_ff::{AdditiveGroup, BigInt, Field, Zero};
use ark_secp256k1::{Fq, Fr};

/// secp256k1 base field element
/// in montgomery form
/// as a wrapper around the arkworks implementation
/// so that various operations can be replaced with inlines
#[derive(Clone, PartialEq, Debug)]
pub struct Secp256k1Fq {
    e: ark_secp256k1::Fq,
}

impl Secp256k1Fq {
    /// creates a new Secp256k1Fq element from an Fq element
    #[inline(always)]
    pub fn new(e: Fq) -> Self {
        Secp256k1Fq { e }
    }
    /// creates a new Secp256k1Fq element from a [u64; 4] array
    #[inline(always)]
    pub fn from_u64_arr(arr: &[u64; 4]) -> Self {
        Secp256k1Fq {
            e: Fq::new_unchecked(BigInt { 0: *arr }),
        }
    }
    /// get inner Fq type
    #[inline(always)]
    pub fn fq(&self) -> Fq {
        self.e
    }
    /// returns the additive identity element (0)
    #[inline(always)]
    pub fn zero() -> Self {
        Secp256k1Fq { e: Fq::zero() }
    }
    /// returns seven
    #[inline(always)]
    pub fn seven() -> Self {
        // derived from Fq::from(7)
        // precomputed to avoid recomputation in point doubling
        Secp256k1Fq {
            e: Fq::new_unchecked(BigInt {
                0: [30064777911u64, 0u64, 0u64, 0u64],
            }),
        }
    }
    /// returns true if the element is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.e.is_zero()
    }
    /// returns -self
    #[inline(always)]
    pub fn neg(&self) -> Self {
        Secp256k1Fq { e: -self.e }
    }
    /// returns self + other
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e + other.e,
        }
    }
    /// returns self - other
    #[inline(always)]
    pub fn sub(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e - other.e,
        }
    }
    /// returns 2*self
    #[inline(always)]
    pub fn dbl(&self) -> Self {
        Secp256k1Fq { e: self.e.double() }
    }
    /// returns 3*self
    #[inline(always)]
    pub fn tpl(&self) -> Self {
        Secp256k1Fq {
            e: self.e.double() + self.e,
        }
    }
    /// returns self * other
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e * other.e,
        }
    }
    /// returns self^2
    #[inline(always)]
    pub fn square(&self) -> Self {
        Secp256k1Fq { e: self.e.square() }
    }
    /// returns self / other
    /// uses custom inline for performance
    /// costs nearly the same as multiplication
    #[cfg(all(
        not(feature = "host"),
        any(target_arch = "riscv32", target_arch = "riscv64")
    ))]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fq) -> Self {
        // get inverse as pure advice
        let mut c = Secp256k1Fq::zero();
        unsafe {
            use crate::{INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_FUNCT7};
            core::arch::asm!(
                ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                opcode = const INLINE_OPCODE,
                funct3 = const SECP256K1_DIVQ_ADV_FUNCT3,
                funct7 = const SECP256K1_FUNCT7,
                rd = in(reg) c.e.0.0.as_mut_ptr(),
                rs1 = in(reg) self.e.0.0.as_ptr(),
                rs2 = in(reg) other.e.0.0.as_ptr(),
                options(nostack)
            );
        }
        // compute tmp = other * c
        let tmp = other.mul(&c);
        // if greater than or equal to p, or other * c is not equal to self, panic
        if (c.e.0 .0[3] == 0xFFFFFFFFFFFFFFFF
            && c.e.0 .0[2] == 0xFFFFFFFFFFFFFFFF
            && c.e.0 .0[1] == 0xFFFFFFFFFFFFFFFF
            && c.e.0 .0[0] >= 0xFFFFFFFFFFFFFC2F)
            || (tmp.e.0 .0[0] != self.e.0 .0[0]
                || tmp.e.0 .0[1] != self.e.0 .0[1]
                || tmp.e.0 .0[2] != self.e.0 .0[2]
                || tmp.e.0 .0[3] != self.e.0 .0[3])
        {
            // literal assembly to induce a panic (spoils the proof)
            // merely using assert_eq! here is insufficient as it doesn't
            // spoil the proof
            unsafe {
                let u = 0u64;
                let v = 1u64;
                core::arch::asm!(
                    ".insn b {opcode}, {funct3}, {rs1}, {rs2}, 0",
                    opcode = const 0x5B, // virtual instruction opcode
                    funct3 = const 0b001, // VirtualAssertEQ funct3
                    rs1 = in(reg) u,
                    rs2 = in(reg) v,
                    options(nostack)
                );
            }
        }
        c
    }
    #[cfg(all(
        not(feature = "host"),
        not(any(target_arch = "riscv32", target_arch = "riscv64"))
    ))]
    pub fn div(&self, _other: &Secp256k1Fq) -> Self {
        panic!("Secp256k1Fq::div called on non-RISC-V target without host feature");
    }
    #[cfg(feature = "host")]
    #[inline(always)]
    pub fn div(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e / other.e,
        }
    }
}

/// secp256k1 point in affine form
/// (as a pair of base field elements in montgomery form)
/// infinity is represented as (0, 0) because this point is not on the curve
#[derive(Clone, PartialEq, Debug)]
pub struct Secp256k1Point {
    x: Secp256k1Fq,
    y: Secp256k1Fq,
}

impl Secp256k1Point {
    /// creates a new Secp256k1Point from two Secp256k1Fq elements
    /// performs no checks to ensure that the point is on the curve
    /// #[inline(always)]
    pub fn new_unchecked(x: Secp256k1Fq, y: Secp256k1Fq) -> Self {
        Secp256k1Point { x, y }
    }
    /// converts the point to a [u64; 8] array
    #[inline(always)]
    pub fn to_u64_arr(&self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        arr[0..4].copy_from_slice(&self.x.e.0 .0);
        arr[4..8].copy_from_slice(&self.y.e.0 .0);
        arr
    }
    /// creates a Secp256k1Point from a [u64; 8] array
    /// performs no checks to ensure that the point is on the curve
    #[inline(always)]
    pub fn from_u64_arr_unchecked(arr: &[u64; 8]) -> Self {
        let x = Secp256k1Fq::from_u64_arr(&[arr[0], arr[1], arr[2], arr[3]]);
        let y = Secp256k1Fq::from_u64_arr(&[arr[4], arr[5], arr[6], arr[7]]);
        Secp256k1Point { x, y }
    }
    /// get x coordinate
    #[inline(always)]
    pub fn x(&self) -> Secp256k1Fq {
        self.x.clone()
    }
    /// get y coordinate
    #[inline(always)]
    pub fn y(&self) -> Secp256k1Fq {
        self.y.clone()
    }
    /// get generator
    #[inline(always)]
    pub fn generator() -> Self {
        Secp256k1Point {
            x: Secp256k1Fq::new(ark_secp256k1::G_GENERATOR_X),
            y: Secp256k1Fq::new(ark_secp256k1::G_GENERATOR_Y),
        }
    }
    /// returns the point at infinity (0, 0)
    #[inline(always)]
    pub fn infinity() -> Self {
        Secp256k1Point {
            x: Secp256k1Fq::zero(),
            y: Secp256k1Fq::zero(),
        }
    }
    /// returns true if the point is the point at infinity (0, 0)
    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.x.e.is_zero() && self.y.e.is_zero()
    }
    /// negates a point on the secp256k1 curve
    #[inline(always)]
    pub fn neg(&self) -> Self {
        if self.is_infinity() {
            Secp256k1Point::infinity()
        } else {
            Secp256k1Point {
                x: self.x.clone(),
                y: self.y.neg(),
            }
        }
    }
    /// doubles a point on the secp256k1 curve
    #[inline(always)]
    pub fn double(&self) -> Self {
        if self.y.is_zero() {
            Secp256k1Point::infinity()
        } else {
            let s = self.x.square().tpl().div(&self.y.dbl());
            let x2 = s.square().sub(&self.x.dbl());
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            Secp256k1Point { x: x2, y: y2 }
        }
    }
    /// adds two points on the secp256k1 curve
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Point) -> Self {
        // if either point is at infinity, return the other point
        if self.is_infinity() {
            other.clone()
        } else if other.is_infinity() {
            self.clone()
        // if the points are equal, perform point doubling
        } else if self.x == other.x && self.y == other.y {
            self.double()
        // if the x coordinates are equal but the y coordinates are not, return the point at infinity
        } else if self.x == other.x && self.y != other.y {
            Secp256k1Point::infinity()
        // if the x coordinates are not equal and not infinity, perform standard point addition
        } else {
            let s = (self.y.sub(&other.y)).div(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x).sub(&other.x);
            let y2 = s.mul(&(self.x.sub(&x2))).sub(&self.y);
            Secp256k1Point { x: x2, y: y2 }
        }
    }
    // specialty routine for computing res = 2*self + other
    // tries to avoid doubling as much as possible
    // and avoids computing unnecessary intermediate values
    #[inline(always)]
    pub fn double_and_add(&self, other: &Secp256k1Point) -> Self {
        // if self is infinity, then return other
        if self.is_infinity() {
            return other.clone();
        // if other is infinity, then return 2*self
        } else if other.is_infinity() {
            return self.add(self);
        // if self is equal to other, naive double and add
        } else if self.x == other.x && self.y == other.y {
            return self.add(self).add(other);
        // if self and other are inverses, return self
        } else if self.x == other.x && self.y != other.y {
            return self.clone();
        // general case, compute (self + other) + self
        // saving an operation in the middle
        // note that (self + other) cannot equal infinity or self here
        // so no special cases needed
        } else {
            let s = (self.y.sub(&other.y)).div(&self.x.sub(&other.x));
            let x2 = s.square().sub(&self.x).sub(&other.x);
            let t = self.y.dbl().div(&self.x.sub(&x2)).sub(&s);
            let x3 = t.square().sub(&self.x).sub(&x2);
            let y3 = t.mul(&(self.x.sub(&x3))).sub(&self.y);
            Secp256k1Point { x: x3, y: y3 }
        }
    }
    // returns lambda * self
    // where lambda is 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    #[inline(always)]
    pub fn endomorphism(&self) -> Self {
        if self.is_infinity() {
            Secp256k1Point::infinity()
        } else {
            // beta = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
            let beta = Secp256k1Fq::from_u64_arr(&[
                6387289667796044110u64,
                287633767014301871u64,
                17936018142961481989u64,
                8811915745022393683u64,
            ]);
            Secp256k1Point {
                x: self.x.mul(&beta),
                y: self.y.clone(),
            }
        }
    }
    // given k, return k1 and k2 such that
    // k = k1 + k2 * lambda
    // and |k1|, |k2| < 2^128
    // based on the implementation found in ec/src/scalar_mul/glv.rs in arkworks
    // This is a lazy and slow implementation for now
    // it will be folded into an inline later
    #[inline(always)]
    pub fn decompose_scalar(k: &Fr) -> [(bool, u128); 2] {
        use ark_ff::PrimeField;
        use num_bigint::BigInt as NBigInt;
        use num_bigint::Sign;
        use num_integer::Integer;
        let k: NBigInt = k.into_bigint().into();
        let r = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                65, 65, 54, 208, 140, 94, 210, 191, 59, 160, 72, 175, 230, 220, 174, 186, 254, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        );
        let a1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                21, 235, 132, 146, 228, 144, 108, 232, 205, 107, 212, 167, 33, 210, 134, 48,
            ],
        );
        let b1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                195, 228, 191, 10, 169, 127, 84, 111, 40, 136, 14, 1, 214, 126, 67, 228,
            ],
        );
        let a2 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                216, 207, 68, 157, 141, 16, 193, 87, 246, 243, 226, 168, 247, 80, 202, 20, 1,
            ],
        );
        let beta_1 = {
            let (mut div, rem) = (&k * &a1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let beta_2 = {
            let (mut div, rem) = (&k * &b1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let k1 = &k - &beta_1 * &a1 - &beta_2 * &a2;
        let k2 = &beta_1 * &b1 - &beta_2 * &a1;
        // return as (sign, abs_value) pairs
        let to_sign_abs = |n: NBigInt| -> (bool, u128) {
            let (sign, bytes) = n.to_bytes_le();
            // pad bytes to 16 bytes
            let mut bytes_padded = bytes.clone();
            while bytes_padded.len() < 16 {
                bytes_padded.push(0u8);
            }
            let abs_value = u128::from_le_bytes(bytes_padded[..16].try_into().unwrap());
            (sign == Sign::Minus, abs_value)
        };
        [to_sign_abs(k1), to_sign_abs(k2)]
    }
}
