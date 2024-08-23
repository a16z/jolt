#![allow(dead_code)]

pub fn convert_endianness_64(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(32)
        .flat_map(|b| b.iter().copied().rev().collect::<Vec<u8>>())
        .collect::<Vec<u8>>()
}

pub fn convert_endianness_128(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(64)
        .flat_map(|b| b.iter().copied().rev().collect::<Vec<u8>>())
        .collect::<Vec<u8>>()
}
