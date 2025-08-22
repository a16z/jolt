use crate::zkvm::JoltField;
use crate::utils::signed_bigint::SignedBigInt;
use ark_ff::BigInt;

// =============================
// Per-row operand domains
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    I8(i8),
    S64(SignedBigInt<1>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    U64(u64),
    S64(SignedBigInt<1>),
    S128(SignedBigInt<2>),
    S192(SignedBigInt<3>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzValue {
    Zero,
    U64(u64),
    S64(SignedBigInt<1>),
    S128(SignedBigInt<2>),
    S192(SignedBigInt<3>),
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
    L1(SignedBigInt<1>),
    L2(SignedBigInt<2>),
    L3(SignedBigInt<3>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SVOProductValue {
    L1(SignedBigInt<1>),
    L2(SignedBigInt<2>),
    L3(SignedBigInt<3>),
    L4(SignedBigInt<4>),
}

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = BigInt<8>; // 512-bit unsigned integer

// =============================
// Limb/product helpers (sign-aware)
// =============================



#[allow(unused_variables)]
pub fn mul_az_bz(az: AzExtendedEval, bz: BzExtendedEval) -> SVOProductValue {
    match (az, bz) {
        (AzExtendedEval::I8(v), BzExtendedEval::L1(signed_bigint)) => {
            let sign = (v >= 0) == signed_bigint.is_positive;
            let mag = (v as i128).unsigned_abs() as u64;
            SVOProductValue::L1(SignedBigInt::from_u64(
                mag.saturating_mul(signed_bigint.magnitude.0[0]),
                sign,
            ))
        }
        _ => SVOProductValue::L2(SignedBigInt::new([0, 0], true)),
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

