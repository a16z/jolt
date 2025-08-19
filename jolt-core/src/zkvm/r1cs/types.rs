use crate::zkvm::JoltField;
use ark_ff::BigInt;

// =============================
// Small scalar domain for raw witnesses
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmallScalar {
    U8(u8),
    U64(u64),
    I64(i64),
    U128(u128),
    I128(i128),
}

impl SmallScalar {
    pub fn as_i128(self) -> i128 {
        match self {
            SmallScalar::U8(v) => v as i128,
            SmallScalar::U64(v) => v as i128,
            SmallScalar::I64(v) => v as i128,
            SmallScalar::U128(v) => v as i128, // lossy if > i128::MAX; callers choose variant appropriately
            SmallScalar::I128(v) => v,
        }
    }

    pub fn as_u64_clamped(self) -> u64 {
        match self {
            SmallScalar::U8(v) => v as u64,
            SmallScalar::U64(v) => v,
            SmallScalar::I64(v) => v.max(0) as u64,
            SmallScalar::U128(v) => v as u64,
            SmallScalar::I128(v) => {
                if v >= 0 {
                    v as u64
                } else {
                    0
                }
            }
        }
    }

    pub fn to_i8(self) -> i8 {
        match self {
            SmallScalar::U8(v) => v as i8,
            SmallScalar::U64(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I64(v) => v.clamp(i8::MIN as i64, i8::MAX as i64) as i8,
            SmallScalar::U128(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I128(v) => v.clamp(i8::MIN as i128, i8::MAX as i128) as i8,
        }
    }
}

// =============================
// Per-row operand domains
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    I8(i8),
    U64AndSign { magnitude: u64, is_positive: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    U64(u64),
    U64AndSign { magnitude: u64, is_positive: bool },
    U128AndSign { magnitude: u128, is_positive: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzValue {
    Zero,
    U64(u64),
    U64AndSign { magnitude: u64, is_positive: bool },
    U128AndSign { magnitude: u128, is_positive: bool },
}

// =============================
// Extended evaluations
// =============================

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SVOProductValue {
    L1 { val: u64, is_positive: bool },
    L2 { val: [u64; 2], is_positive: bool },
    L3 { val: [u64; 3], is_positive: bool },
    L4 { val: [u64; 4], is_positive: bool },
}

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = BigInt<8>; // 512-bit unsigned integer

// =============================
// Limb/product helpers (sign-aware)
// =============================

#[allow(unused_variables)]
pub fn add_with_sign_u64(a_mag: u64, a_pos: bool, b_mag: u64, b_pos: bool) -> (u64, bool) {
    if a_pos == b_pos {
        (a_mag.saturating_add(b_mag), a_pos)
    } else if a_mag >= b_mag {
        (a_mag - b_mag, a_pos)
    } else {
        (b_mag - a_mag, b_pos)
    }
}

#[allow(unused_variables)]
pub fn add_limbs<const N: usize>(a: ([u64; N], bool), b: ([u64; N], bool)) -> ([u64; N], bool) {
    let (a_arr, a_pos) = a;
    let (b_arr, b_pos) = b;
    if a_pos == b_pos {
        let mut out = [0u64; N];
        let mut carry: u128 = 0;
        for i in 0..N {
            let s = (a_arr[i] as u128) + (b_arr[i] as u128) + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        (out, a_pos)
    } else {
        if a_arr >= b_arr {
            let mut out = [0u64; N];
            let mut borrow: i128 = 0;
            for i in 0..N {
                let d = (a_arr[i] as i128) - (b_arr[i] as i128) - borrow;
                out[i] = d as u64;
                borrow = if d < 0 { 1 } else { 0 };
            }
            (out, a_pos)
        } else {
            let mut out = [0u64; N];
            let mut borrow: i128 = 0;
            for i in 0..N {
                let d = (b_arr[i] as i128) - (a_arr[i] as i128) - borrow;
                out[i] = d as u64;
                borrow = if d < 0 { 1 } else { 0 };
            }
            (out, b_pos)
        }
    }
}

#[allow(unused_variables)]
pub fn mul_az_bz(az: AzExtendedEval, bz: BzExtendedEval) -> SVOProductValue {
    match (az, bz) {
        (AzExtendedEval::I8(v), BzExtendedEval::L1 { val, is_positive }) => {
            let sign = (v >= 0) == is_positive;
            let mag = (v as i128).unsigned_abs() as u64;
            SVOProductValue::L1 {
                val: mag.saturating_mul(val),
                is_positive: sign,
            }
        }
        _ => SVOProductValue::L2 {
            val: [0, 0],
            is_positive: true,
        },
    }
}

#[allow(unused_variables)]
pub fn field_mul_product<F: JoltField, const K: usize>(
    field: F,
    product: [u64; K],
) -> UnreducedProduct {
    // Placeholder: return zero; implement proper multi-precision mul
    BigInt::from(0u8)
}

#[allow(unused_variables)]
pub fn reduce_unreduced_to_field<F>(x: &UnreducedProduct) -> F {
    unimplemented!("reduce_unreduced_to_field needs field modulus and conversion")
}
