//! Decode-time bounds regression for the wire claim structs (v12 #116236).
//!
//! The proof-facing claim structs carry `Vec<C>` fields and are decoded from
//! untrusted bytes with `bincode::serde` (varint lengths). A forged length
//! prefix must not drive a pre-allocation of the claimed length: serde caps the
//! `Vec` size hint (1 MiB) and each `Fr` element consumes 32 input bytes, so
//! decode cost stays proportional to the supplied buffer. If either side of
//! that contract regresses, the huge-length decode below attempts an
//! exbibyte-scale allocation and aborts instead of returning `Err`.

use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
use jolt_field::{Fr, FromPrimitiveInt};

fn decode(
    bytes: &[u8],
) -> Result<(InstructionReadRafOutputClaims<Fr>, usize), bincode::error::DecodeError> {
    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
}

/// Pins the wire layout the forged buffer below relies on: the encoding leads
/// with the first `Vec` field's varint length.
#[test]
#[expect(clippy::unwrap_used)]
fn round_trip_pins_wire_layout() {
    let claims = InstructionReadRafOutputClaims {
        lookup_table_flags: vec![Fr::from_u64(1), Fr::from_u64(2)],
        instruction_ra: vec![Fr::from_u64(3)],
        instruction_raf_flag: Fr::from_u64(4),
    };
    let bytes = bincode::serde::encode_to_vec(&claims, bincode::config::standard()).unwrap();
    assert_eq!(bytes[0], 2, "leading byte is lookup_table_flags's length");
    let (decoded, consumed) = decode(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(decoded, claims);
}

#[test]
fn forged_huge_vec_length_prefix_fails_cleanly() {
    // varint(2^61) = 0xFD marker + u64 LE: lookup_table_flags claims 2^61
    // elements while the buffer carries two.
    let mut bytes = vec![0xFD];
    bytes.extend_from_slice(&(1u64 << 61).to_le_bytes());
    // Two canonical field elements (32 zero bytes each), then EOF.
    bytes.extend_from_slice(&[0u8; 64]);
    assert!(decode(&bytes).is_err());
}
