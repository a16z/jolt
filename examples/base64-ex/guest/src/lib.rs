#![cfg_attr(feature = "guest", no_std)]
extern crate alloc;

const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[jolt::provable(memory_size = 65536, max_trace_length = 65536)]
fn base64_encode(input: &[u8]) -> ([u8; 32], [u8; 32]) {
    let mut out = [0u8; 64];
    let mut buf: u32 = 0;
    let mut bits = 0;
    let mut len = 0;

    for &b in input {
        buf = (buf << 8) | b as u32;
        bits += 8;
        while bits >= 6 && len < 64 {
            bits -= 6;
            let idx = ((buf >> bits) & 0x3F) as usize;
            out[len] = BASE64_TABLE[idx];
            len += 1;
        }
    }

    if bits > 0 && len < 64 {
        buf <<= 6 - bits;
        let idx = (buf & 0x3F) as usize;
        out[len] = BASE64_TABLE[idx];
        len += 1;
    }

    while len % 4 != 0 && len < 64 {
        out[len] = b'=';
        len += 1;
    }

    let mut chunk1 = [0u8; 32];
    let mut chunk2 = [0u8; 32];
    chunk1.copy_from_slice(&out[..32]);
    chunk2.copy_from_slice(&out[32..64]);
    (chunk1, chunk2)
}
