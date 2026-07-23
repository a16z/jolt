#![no_main]

//! Differential check of the optimized `Fr` accumulators against their naive
//! counterparts: the same op sequence through `FrSmallScalarAccumulator` /
//! `FrSignedProductAccumulator` and `NaiveSignedScalarAccumulator` /
//! `NaiveSignedProductAccumulator` must reduce to the same field element.

use jolt_field::limbs::Limbs;
use jolt_field::signed::S256;
use jolt_field::{
    Fr, FrSignedProductAccumulator, FrSmallScalarAccumulator, NaiveSignedProductAccumulator,
    NaiveSignedScalarAccumulator, ReducingBytes, SignedProductAccumulator,
    SignedScalarAccumulator,
};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const MAX_OPS: usize = 64;

fuzz_target!(|data: &[u8]| {
    let mut small = FrSmallScalarAccumulator::default();
    let mut small_naive = NaiveSignedScalarAccumulator::<Fr>::default();
    let mut product = FrSignedProductAccumulator::default();
    let mut product_naive = NaiveSignedProductAccumulator::<Fr>::default();

    let mut cursor = 0;
    let mut ops = 0;
    while cursor < data.len() && ops < MAX_OPS {
        let tag = data[cursor];
        cursor += 1;
        if cursor + SCALAR_BYTES > data.len() {
            break;
        }
        let value =
            <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[cursor..cursor + SCALAR_BYTES]);
        cursor += SCALAR_BYTES;

        match tag % 4 {
            0 => {
                small.add(value);
                small_naive.add(value);
            }
            1 => {
                if cursor + 8 > data.len() {
                    break;
                }
                let scalar = u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                small.fmadd_u64(value, scalar);
                small_naive.fmadd_u64(value, scalar);
            }
            2 => {
                if cursor + 8 > data.len() {
                    break;
                }
                let scalar = i64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                small.fmadd_i64(value, scalar);
                small_naive.fmadd_i64(value, scalar);
            }
            _ => {
                if cursor + 17 > data.len() {
                    break;
                }
                let magnitude =
                    u128::from_le_bytes(data[cursor..cursor + 16].try_into().unwrap());
                let is_positive = data[cursor + 16] % 2 == 0;
                cursor += 17;
                let scalar = S256::from_limbs(
                    Limbs::new([magnitude as u64, (magnitude >> 64) as u64, 0, 0]),
                    is_positive,
                );
                product.fmadd_s256(value, &scalar);
                product_naive.fmadd_s256(value, &scalar);
            }
        }
        ops += 1;
    }

    assert_eq!(
        small.reduce(),
        small_naive.reduce(),
        "small-scalar accumulator disagrees with naive reference"
    );
    assert_eq!(
        product.reduce(),
        product_naive.reduce(),
        "signed-product accumulator disagrees with naive reference"
    );
});
