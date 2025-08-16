use crate::utils::u64_and_sign::{U128AndSign, U64AndSign};
use ark_ff::BigInt;
use std::ops::{Mul, Sub};

// --- LEVEL 1 TYPES (STATIC) ---
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzType {
    U5, // Absolute value at most 31
    U64,
    U64AndSign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzType {
    I8,
    U64,
    U64AndSign,
    I128,
    U128AndSign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzType {
    Zero,
    I8,
    U64,
    U64AndSign,
    U128AndSign,
}

// --- LEVEL 1 VALUES (RUNTIME) ---
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    U5(i8),
    U64(u64),
    U64AndSign(U64AndSign),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    I8(i8),
    U64(u64),
    U64AndSign(U64AndSign),
    I128(i128),
    U128AndSign(U128AndSign),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzValue {
    Zero,
    I8(i8),
    U64(u64),
    U64AndSign(U64AndSign),
    U128AndSign(U128AndSign),
}

// --- LEVEL 2: SVO EXTENDED EVALUATION VALUES ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzExtendedEval {
    I8(i8),
    I128(i128),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzExtendedEval {
    L1 { val: u64, is_positive: bool },
    L2 { val: [u64; 2], is_positive: bool },
    L3 { val: [u64; 3], is_positive: bool },
}

// --- SVO PRODUCT TYPES ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SVOProductValue {
    L1 { val: u64, is_positive: bool },
    L2 { val: [u64; 2], is_positive: bool },
    L3 { val: [u64; 3], is_positive: bool },
    L4 { val: [u64; 4], is_positive: bool },
}

pub type UnreducedProduct = BigInt<8>;

// --- ARITHMETIC IMPLEMENTATIONS ---

impl Sub for AzExtendedEval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        use AzExtendedEval::*;
        match (self, rhs) {
            // As instructed, I8 - I8 results in an I8.
            (I8(a), I8(b)) => I8(a.wrapping_sub(b)),

            // Any other combination promotes to I128 for safety.
            (I128(a), I128(b)) => I128(a.wrapping_sub(b)),
            (I128(a), I8(b)) => I128(a.wrapping_sub(b as i128)),
            (I8(a), I128(b)) => I128((a as i128).wrapping_sub(b)),
        }
    }
}

impl Mul<BzExtendedEval> for AzExtendedEval {
    type Output = SVOProductValue;
    fn mul(self, _rhs: BzExtendedEval) -> Self::Output {
        unimplemented!("Multi-limb multiplication logic for AzExtendedEval * BzExtendedEval");
    }
}
