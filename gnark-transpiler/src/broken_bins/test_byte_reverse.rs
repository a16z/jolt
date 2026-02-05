#![allow(dead_code, unused_variables, unused_imports)]
//! Test ByteReverse computation
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

fn main() {
    // Test: 4096
    let value = Fr::from(4096u64);
    println!("Input: {:?}", value);
    
    // Serialize to LE bytes (like Rust serialize_uncompressed)
    let mut le_bytes = Vec::new();
    value.serialize_uncompressed(&mut le_bytes).unwrap();
    println!("LE bytes (32): {:02x?}", &le_bytes);
    
    // Reverse the bytes
    let mut reversed = le_bytes.clone();
    reversed.reverse();
    println!("Reversed: {:02x?}", &reversed);
    
    // from_le_bytes_mod_order
    let result = Fr::from_le_bytes_mod_order(&reversed);
    println!("Result: {:?}", result);
    
    // Expected from Go: 28269553036454149273332760011886696253239742350009903329945699220681916416
    println!("\nExpected from Go: 28269553036454149273332760011886696253239742350009903329945699220681916416");
}
