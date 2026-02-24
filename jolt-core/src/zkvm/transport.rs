//! Lightweight framed transport encoding helpers.
//!
//! This module is intentionally small and dependency-free so we can use it in verifier-facing
//! deserialization paths with explicit caps (DoS resistance) and strict parsing invariants.
//!
//! Design:
//! - Streams begin with a short fixed signature (header bytes).
//! - Then a sequence of frames: (tag: u8, len: varint u64, payload: len bytes).
//! - Decoders should be strict by default: reject unknown tags, reject duplicates for singleton
//!   sections, and require full consumption of each payload.

use std::io::{self, Read, Write};

/// Maximum number of bytes in a u64 varint (LEB128-style).
const VARINT_U64_MAX_BYTES: usize = 10;

// Recursion bundle framing constants (shared between host and guest).
pub const BUNDLE_SIGNATURE: &[u8; 8] = b"JOLTBDL\0";
pub const BUNDLE_TAG_PREPROCESSING: u8 = 1;
pub const BUNDLE_TAG_RECORD: u8 = 2;
pub const RECORD_TAG_DEVICE: u8 = 1;
pub const RECORD_TAG_PROOF: u8 = 2;

#[inline]
pub fn signature_check<R: Read>(r: &mut R, expected: &[u8]) -> io::Result<()> {
    let mut got = vec![0u8; expected.len()];
    r.read_exact(&mut got)?;
    if got != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid signature",
        ));
    }
    Ok(())
}

#[inline]
pub fn signature_write<W: Write>(w: &mut W, signature: &[u8]) -> io::Result<()> {
    w.write_all(signature)
}

#[inline]
pub fn write_varint_u64<W: Write>(w: &mut W, mut x: u64) -> io::Result<()> {
    while x >= 0x80 {
        w.write_all(&[((x as u8) & 0x7F) | 0x80])?;
        x >>= 7;
    }
    w.write_all(&[x as u8])
}

#[inline]
pub fn varint_u64_len(mut x: u64) -> usize {
    let mut n = 1usize;
    while x >= 0x80 {
        n += 1;
        x >>= 7;
    }
    n
}

#[inline]
pub fn read_varint_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut x = 0u64;
    let mut shift = 0u32;
    for _ in 0..VARINT_U64_MAX_BYTES {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        let byte = b[0];
        x |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Ok(x);
        }
        shift += 7;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "varint overflow",
    ))
}

#[inline]
pub fn read_u8_opt<R: Read>(r: &mut R) -> io::Result<Option<u8>> {
    let mut b = [0u8; 1];
    match r.read(&mut b) {
        Ok(0) => Ok(None),
        Ok(1) => Ok(Some(b[0])),
        Ok(_) => unreachable!(),
        Err(e) => Err(e),
    }
}

#[inline]
pub fn write_frame_header<W: Write>(w: &mut W, tag: u8, len: u64) -> io::Result<()> {
    w.write_all(&[tag])?;
    write_varint_u64(w, len)
}

#[inline]
pub fn read_frame_header<R: Read>(r: &mut R, max_len: u64) -> io::Result<Option<(u8, u64)>> {
    let Some(tag) = read_u8_opt(r)? else {
        return Ok(None);
    };
    let len = read_varint_u64(r)?;
    if len > max_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    Ok(Some((tag, len)))
}

#[inline]
pub fn skip_exact<R: Read>(r: &mut R, mut n: u64) -> io::Result<()> {
    let mut buf = [0u8; 4096];
    while n > 0 {
        let k = (n as usize).min(buf.len());
        r.read_exact(&mut buf[..k])?;
        n -= k as u64;
    }
    Ok(())
}
