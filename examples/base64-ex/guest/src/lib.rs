#![no_main]

use serde::{Serialize, Deserialize, Serializer, Deserializer};

/// New-type so we can impl Serialize/Deserialize for 64-byte arrays
#[derive(Copy, Clone)]
pub struct B64Array(pub [u8; 64]);

impl Serialize for B64Array {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for B64Array {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de>,
    {
        let slice: &[u8] = serde_bytes::deserialize(deserializer)?;
        if slice.len() != 64 {
            return Err(serde::de::Error::invalid_length(slice.len(), &"64 bytes"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(slice);
        Ok(B64Array(arr))
    }
}

const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[jolt::provable(memory_size = 65536, max_trace_length = 65536)]
fn base64_encode(input: &[u8]) -> B64Array {
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

    B64Array(out)
}