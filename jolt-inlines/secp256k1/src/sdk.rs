//! secp256k1 operations optimized for Jolt zkVM.

use ark_ff::{AdditiveGroup, BigInt, Field, Zero};
use ark_secp256k1::Fq;

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
    /// returns self + other mod p
    #[inline(always)]
    pub fn add(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e + other.e,
        }
    }
    /// returns self - other mod p
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
    /// returns self * other mod p
    #[inline(always)]
    pub fn mul(&self, other: &Secp256k1Fq) -> Self {
        Secp256k1Fq {
            e: self.e * other.e,
        }
    }
    /// returns self^2 mod p
    #[inline(always)]
    pub fn square(&self) -> Self {
        Secp256k1Fq { e: self.e.square() }
    }
    /// returns self / other mod p
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
        let x = Secp256k1Fq::new(ark_secp256k1::Fq::new_unchecked(ark_ff::BigInt {
            0: [arr[0], arr[1], arr[2], arr[3]],
        }));
        let y = Secp256k1Fq::new(ark_secp256k1::Fq::new_unchecked(ark_ff::BigInt {
            0: [arr[4], arr[5], arr[6], arr[7]],
        }));
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
}
