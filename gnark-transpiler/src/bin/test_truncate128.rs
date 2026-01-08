#![allow(dead_code, unused_variables, unused_imports)]
//! Test Truncate128Reverse computation
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use std::str::FromStr;

fn main() {
    // Test with hash of [0,0,0]
    let value_str = "5317387130258456662214331362918410991734007599705406860481038345552731150762";
    let value = Fr::from_str(value_str).unwrap();
    println!("Input: {:?}", value);
    
    // Serialize to LE bytes
    let mut le_bytes = Vec::new();
    value.serialize_uncompressed(&mut le_bytes).unwrap();
    println!("LE bytes (32): {:02x?}", &le_bytes);
    
    // Take first 16 bytes
    let le16: Vec<u8> = le_bytes[..16].to_vec();
    println!("LE16 (first 16): {:02x?}", &le16);
    
    // Reverse the 16 bytes
    let mut reversed = le16.clone();
    reversed.reverse();
    println!("Reversed: {:02x?}", &reversed);
    
    // from_le_bytes_mod_order (for 16 bytes)
    let result = Fr::from_le_bytes_mod_order(&reversed);
    println!("Result: {:?}", result);
    
    // Expected from Go: 226766529495067233327141020382834624226
    println!("\nExpected from Go: 226766529495067233327141020382834624226");
}
