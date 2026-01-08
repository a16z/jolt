//! Test Poseidon hash outputs for comparison with Go implementation

use ark_bn254::Fr;
use jolt_core::transcripts::{FrParams, PoseidonParams};
use light_poseidon::PoseidonHasher;

fn main() {
    println!("=== Rust Poseidon Test Vectors ===\n");

    // Test hash([0, 0, 0])
    let mut hasher = FrParams::poseidon();
    let inputs = [Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];
    let result = hasher.hash(&inputs).unwrap();
    println!("hash([0, 0, 0]) = {:?}", result);

    // Test hash([1, 2, 3])
    let mut hasher2 = FrParams::poseidon();
    let inputs2 = [Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
    let result2 = hasher2.hash(&inputs2).unwrap();
    println!("hash([1, 2, 3]) = {:?}", result2);

    // Test hash([42, 0, 0]) - simulating transcript append
    let mut hasher3 = FrParams::poseidon();
    let inputs3 = [Fr::from(42u64), Fr::from(0u64), Fr::from(0u64)];
    let result3 = hasher3.hash(&inputs3).unwrap();
    println!("hash([42, 0, 0]) = {:?}", result3);
}
