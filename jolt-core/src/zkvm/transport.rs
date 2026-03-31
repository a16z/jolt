//! Lightweight length-prefixed transport encoding helpers.
//!
//! Wire format: `[magic: 4B][version: 1B][flags: 1B][section₀][section₁]…`
//! where each section is `[varint payload_len][payload bytes]`.
//! Sections are sequential and untagged; the deserializer reads them in a
//! fixed order defined by the proof schema.

use std::io::{self, Read, Write};

const VARINT_U64_MAX_BYTES: usize = 10;

#[inline]
pub fn write_magic_version<W: Write>(w: &mut W, magic: &[u8], version: u8) -> io::Result<()> {
    w.write_all(magic)?;
    w.write_all(&[version])
}

#[inline]
pub fn read_magic_version<R: Read>(r: &mut R, magic: &[u8; 4]) -> io::Result<u8> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    if buf != *magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid proof magic",
        ));
    }
    let mut v = [0u8; 1];
    r.read_exact(&mut v)?;
    Ok(v[0])
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
        let payload = (byte & 0x7F) as u64;
        if shift == 63 && payload > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint overflow",
            ));
        }
        x |= payload << shift;
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

/// Reads a varint-prefixed section, enforcing a maximum payload length.
/// Returns a `Take` reader limited to exactly the declared payload length.
/// Callers should check `limited.limit() == 0` after reading to detect trailing bytes.
#[inline]
pub fn read_section<R: Read>(r: &mut R, max_len: u64) -> io::Result<io::Take<&mut R>> {
    let len = read_varint_u64(r)?;
    if len > max_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "section too large",
        ));
    }
    Ok(r.take(len))
}

#[inline]
pub fn read_section_bytes<R: Read>(r: &mut R, max_len: u64) -> io::Result<Vec<u8>> {
    let mut limited = read_section(r, max_len)?;
    let mut bytes = vec![0u8; limited.limit() as usize];
    limited.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip_edge_cases() {
        let cases: &[u64] = &[0, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX];
        for &val in cases {
            let mut buf = Vec::new();
            write_varint_u64(&mut buf, val).unwrap();
            let decoded = read_varint_u64(&mut buf.as_slice()).unwrap();
            assert_eq!(decoded, val, "roundtrip failed for {val}");
        }
    }

    #[test]
    fn varint_u64_len_matches_encoding() {
        let cases: &[u64] = &[0, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX];
        for &val in cases {
            let mut buf = Vec::new();
            write_varint_u64(&mut buf, val).unwrap();
            assert_eq!(
                buf.len(),
                varint_u64_len(val),
                "varint_u64_len mismatch for {val}"
            );
        }
    }

    #[test]
    fn varint_overflow_rejected() {
        // 11 continuation bytes — exceeds VARINT_U64_MAX_BYTES
        let bad = vec![0x80u8; 11];
        let res = read_varint_u64(&mut bad.as_slice());
        assert!(res.is_err());
    }

    #[test]
    fn varint_10th_byte_overflow_rejected() {
        // 9 continuation bytes + a 10th byte with payload 2 (only 0 or 1 is valid at shift=63)
        let mut bad = vec![0x80u8; 9];
        bad.push(0x02);
        let res = read_varint_u64(&mut bad.as_slice());
        assert!(res.is_err());
    }

    #[test]
    fn magic_version_roundtrip() {
        let magic = b"JOLT";
        let version = 1u8;
        let mut buf = Vec::new();
        write_magic_version(&mut buf, magic, version).unwrap();
        assert_eq!(buf.len(), 5);

        let decoded_version = read_magic_version(&mut buf.as_slice(), magic).unwrap();
        assert_eq!(decoded_version, version);
    }

    #[test]
    fn wrong_magic_rejected() {
        let mut buf = Vec::new();
        write_magic_version(&mut buf, b"JOLT", 1).unwrap();
        let res = read_magic_version(&mut buf.as_slice(), b"BAAD");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid proof magic"));
    }

    #[test]
    fn wrong_version_readable() {
        let mut buf = Vec::new();
        write_magic_version(&mut buf, b"JOLT", 2).unwrap();
        let version = read_magic_version(&mut buf.as_slice(), b"JOLT").unwrap();
        assert_eq!(version, 2);
    }

    #[test]
    fn read_section_enforces_cap() {
        let mut buf = Vec::new();
        write_varint_u64(&mut buf, 1000).unwrap();
        buf.extend_from_slice(&[0u8; 1000]);

        let mut too_small = buf.as_slice();
        let res = read_section(&mut too_small, 999);
        assert!(res.is_err());

        let mut cursor = buf.as_slice();
        let limited = read_section(&mut cursor, 1000).unwrap();
        assert_eq!(limited.limit(), 1000);
    }
}
