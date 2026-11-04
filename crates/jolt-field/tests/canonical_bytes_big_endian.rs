use jolt_field::CanonicalBytes;

struct LittleEndianBytes<const N: usize>([u8; N]);

impl<const N: usize> CanonicalBytes for LittleEndianBytes<N> {
    const NUM_BYTES: usize = N;

    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0);
    }
}

fn write_big_endian<T: CanonicalBytes>(value: &T, out: &mut [u8]) {
    value.to_bytes_be(out);
}

#[test]
fn canonical_bytes_default_big_endian_matches_wire_order() {
    let value = LittleEndianBytes([0x78, 0x56, 0x34, 0x12]);
    let mut out = [0u8; 4];
    write_big_endian(&value, &mut out);
    assert_eq!(out, [0x12, 0x34, 0x56, 0x78]);
    assert_eq!(value.to_bytes_le_vec(), [0x78, 0x56, 0x34, 0x12]);
}

#[test]
fn canonical_bytes_default_big_endian_accepts_empty_encoding() {
    let value = LittleEndianBytes([]);
    let mut out = [];
    write_big_endian(&value, &mut out);
    assert!(out.is_empty());
}
