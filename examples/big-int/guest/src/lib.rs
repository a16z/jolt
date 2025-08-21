#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::{end_cycle_tracking, start_cycle_tracking};


// Import field elements for mul_assign benchmark
use ark_bn254::Fr;
/// Benchmarks the mul_assign function from Montgomery backend
/// Takes two field elements as 4 u64 limbs each and returns the result as 4 u64 limbs
#[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
fn benchmark_mul_assign(
    a_0: u64, a_1: u64, a_2: u64, a_3: u64,
    b_0: u64, b_1: u64, b_2: u64, b_3: u64
) -> (u64, u64, u64, u64) {
    // Black box the inputs to prevent optimization
    let a_0 = black_box(a_0);
    let a_1 = black_box(a_1);
    let a_2 = black_box(a_2);
    let a_3 = black_box(a_3);
    let b_0 = black_box(b_0);
    let b_1 = black_box(b_1);
    let b_2 = black_box(b_2);
    let b_3 = black_box(b_3);
    
    // Create field elements from the input limbs
    // Fr uses 4 u64 limbs in little-endian order
    // Construct a BigInteger from the limbs and then create Fr from it
    use ark_ff::BigInteger256;
    
    let a_bigint = BigInteger256::new([a_0, a_1, a_2, a_3]);
    let mut a = Fr::from(a_bigint);
    
    let b_bigint = BigInteger256::new([b_0, b_1, b_2, b_3]);
    let b = Fr::from(b_bigint);
    
    // Black box before the operation
    a = black_box(a);
    let mut b = black_box(b);
    
    // Perform mul_assign with cycle tracking
    start_cycle_tracking("mul_assign");
    a *= black_box(b);  // This calls mul_assign internally
    // b *= black_box(a);  // This calls mul_assign internally
    // a *= black_box(b);  // This calls mul_assign internally
    // b *= black_box(a);  // This calls mul_assign internally
    // a *= black_box(b);  // This calls mul_assign internally
    // b *= black_box(a);  // This calls mul_assign internally
    // a *= black_box(b);  // This calls mul_assign internally
    // b *= black_box(a);  // This calls mul_assign internally
    // a *= black_box(a);  // This calls mul_assign internally
    // a *= black_box(b);  // This calls mul_assign internally
    let result = black_box(a);
    end_cycle_tracking("mul_assign");
    // 3117
    // 3517 RV32IM cycles
    // inline default: 3487
    // inline assembly: 2580
    
    // Black box the result
    
    
    
    // Convert result back to u64 limbs
    // Use the Into trait to get BigInt from Fr
    let result_bigint: BigInteger256 = result.into();
    let bytes = result_bigint.to_bytes_le();
    
    // Extract the 4 u64 limbs (32 bytes total, 8 bytes per limb)
    let limb_0 = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]);
    let limb_1 = u64::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]);
    let limb_2 = u64::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23]]);
    let limb_3 = u64::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31]]);
    
    (
        black_box(limb_0),
        black_box(limb_1),
        black_box(limb_2),
        black_box(limb_3)
    )
}
// "mul_assign": 3225 RV32IM cycles, 0 virtual cycles




// use ruint::aliases::{U256, U512};
// /// Benchmarks the mul_redc function from ruint
// /// Takes two 256-bit numbers as 4 u64 limbs each and returns the Montgomery reduced result
// #[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
// fn benchmark_mul_assign(
//     a_0: u64, a_1: u64, a_2: u64, a_3: u64,
//     b_0: u64, b_1: u64, b_2: u64, b_3: u64
// ) -> (u64, u64, u64, u64) {
//     // Black box the inputs to prevent optimization
//     let a_0 = black_box(a_0);
//     let a_1 = black_box(a_1);
//     let a_2 = black_box(a_2);
//     let a_3 = black_box(a_3);
//     let b_0 = black_box(b_0);
//     let b_1 = black_box(b_1);
//     let b_2 = black_box(b_2);
//     let b_3 = black_box(b_3);
    
//     // Construct arrays for mul_redc
//     let a = U256::from_limbs([a_0, a_1, a_2, a_3]);
//     let b = U256::from_limbs([b_0, b_1, b_2, b_3]);
    
//     // Black box before the operation
//     let a = black_box(a);
//     let b = black_box(b);

//     // Perform mul_redc with cycle tracking
//     start_cycle_tracking("mul");
//     let result: U512 = black_box(a).widening_mul(black_box(b));
//     end_cycle_tracking("mul");
    
//     // Extract limbs from result
//     let limbs = result.as_limbs();
//     (
//         black_box(limbs[0]),
//         black_box(limbs[1]),
//         black_box(limbs[2]),
//         black_box(limbs[3])
//     )
// }
// // "mul": 475 RV32IM cycles, 0 virtual cycles

// use ruint::aliases::U256;
// /// Benchmarks the mul_redc function from ruint
// /// Takes two 256-bit numbers as 4 u64 limbs each and returns the Montgomery reduced result
// #[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
// fn benchmark_mul_assign(
//     a_0: u64, a_1: u64, a_2: u64, a_3: u64,
//     b_0: u64, b_1: u64, b_2: u64, b_3: u64
// ) -> (u64, u64, u64, u64) {
//     // Black box the inputs to prevent optimization
//     let a_0 = black_box(a_0);
//     let a_1 = black_box(a_1);
//     let a_2 = black_box(a_2);
//     let a_3 = black_box(a_3);
//     let b_0 = black_box(b_0);
//     let b_1 = black_box(b_1);
//     let b_2 = black_box(b_2);
//     let b_3 = black_box(b_3);
    
//     // Construct arrays for mul_redc
//     let a = U256::from_limbs([a_0, a_1, a_2, a_3]);
//     let b = U256::from_limbs([b_0, b_1, b_2, b_3]);
    
//     // Black box before the operation
//     let a = black_box(a);
//     let b = black_box(b);

//     // Perform mul_redc with cycle tracking
//     start_cycle_tracking("short_mul");
//     let result = black_box(black_box(a) * black_box(b));
//     end_cycle_tracking("short_mul");
    
//     // Extract limbs from result
//     let limbs = result.as_limbs();
//     (
//         black_box(limbs[0]),
//         black_box(limbs[1]),
//         black_box(limbs[2]),
//         black_box(limbs[3])
//     )
// }
// // "short_mul": 72 RV32IM cycles, 0 virtual cycles

// use ruint::algorithms::mul_redc;
// /// Benchmarks the mul_redc function from ruint
// /// Takes two 256-bit numbers as 4 u64 limbs each and returns the Montgomery reduced result
// #[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
// fn benchmark_mul_assign(
//     a_0: u64, a_1: u64, a_2: u64, a_3: u64,
//     b_0: u64, b_1: u64, b_2: u64, b_3: u64
// ) -> (u64, u64, u64, u64) {
//     // Black box the inputs to prevent optimization
//     let a_0 = black_box(a_0);
//     let a_1 = black_box(a_1);
//     let a_2 = black_box(a_2);
//     let a_3 = black_box(a_3);
//     let b_0 = black_box(b_0);
//     let b_1 = black_box(b_1);
//     let b_2 = black_box(b_2);
//     let b_3 = black_box(b_3);
    
//     // Construct arrays for mul_redc
//     let a = [a_0, a_1, a_2, a_3];
//     let b = [b_0, b_1, b_2, b_3];
    
//     // BN254's Fr field modulus (scalar field prime r)
//     // r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//     const MODULUS: [u64; 4] = [
//         0x43e1f593f0000001,  // Least significant limb
//         0x2833e84879b97091,
//         0xb85045b68181585d,
//         0x30644e72e131a029,  // Most significant limb
//     ];
    
//     // Montgomery inverse for BN254's Fr field
//     // This is -MODULUS^{-1} mod 2^64
//     const INV: u64 = 0xc2e1f593efffffff;
    
//     // Black box before the operation
//     let a = black_box(a);
//     let b = black_box(b);
//     let modulus = black_box(MODULUS);
//     let inv = black_box(INV);
    
//     // Perform mul_redc with cycle tracking
//     start_cycle_tracking("mul_redc");
//     let result = mul_redc(a, b, modulus, inv);
//     let result = mul_redc(a, result, modulus, inv);
//     let result = mul_redc(a, result, modulus, inv);
//     let result = mul_redc(result, b, modulus, inv);
//     let result = mul_redc(a, result, modulus, inv);
//     end_cycle_tracking("mul_redc");
    
//     // Black box the result
//     let result = black_box(result);
    
//     (
//         black_box(result[0]),
//         black_box(result[1]),
//         black_box(result[2]),
//         black_box(result[3])
//     )
// }
// // "mul_redc": 4511 RV32IM cycles, 0 virtual cycles (5 executions)




use ark_ff::{BigInteger, BigInteger256};

// Commented out existing BigInteger multiplication benchmark
// use ark_ff::{BigInteger, BigInteger256};

// /// Original benchmark: Multiplies two 256-bit integers (each represented as two u128 limbs) and returns the 512-bit result as 8 u64 limbs
// #[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
// fn mul_u256(a_lo: u128, a_hi: u128, b_lo: u128, b_hi: u128) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
//     // Black box the inputs to prevent optimization
//     let a_lo = black_box(a_lo);
//     let a_hi = black_box(a_hi);
//     let b_lo = black_box(b_lo);
//     let b_hi = black_box(b_hi);
    
//     // Create BigInteger256 from the input limbs
//     // BigInteger256 uses 4 u64 limbs in little-endian order
//     // Convert u128 to u64 limbs: each u128 becomes 2 u64s
//     let a_limbs = [
//         (a_lo & 0xFFFFFFFFFFFFFFFF) as u64,         // a_lo low 64 bits
//         ((a_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // a_lo high 64 bits
//         (a_hi & 0xFFFFFFFFFFFFFFFF) as u64,         // a_hi low 64 bits
//         ((a_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // a_hi high 64 bits
//     ];
    
//     let b_limbs = [
//         (b_lo & 0xFFFFFFFFFFFFFFFF) as u64,         // b_lo low 64 bits
//         ((b_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // b_lo high 64 bits
//         (b_hi & 0xFFFFFFFFFFFFFFFF) as u64,         // b_hi low 64 bits
//         ((b_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // b_hi high 64 bits
//     ];
    
//     let a = black_box(BigInteger256::new(a_limbs));
//     let b = black_box(BigInteger256::new(b_limbs));
    
//     // Multiply using mul - returns (low_bits, high_bits) as two BigInteger256
//     start_cycle_tracking("big_int_mul");
//     let (low, high) = black_box(a.mul(&b));
//     end_cycle_tracking("big_int_mul");
    
//     // Get the result limbs (they're stored in little-endian order)
//     let low_limbs = low.0;
//     let high_limbs = high.0;
    
//     // Return all 8 u64 limbs (4 from low, 4 from high)
//     (
//         black_box(low_limbs[0]), 
//         black_box(low_limbs[1]), 
//         black_box(low_limbs[2]), 
//         black_box(low_limbs[3]),
//         black_box(high_limbs[0]),
//         black_box(high_limbs[1]),
//         black_box(high_limbs[2]),
//         black_box(high_limbs[3])
//     )
// }