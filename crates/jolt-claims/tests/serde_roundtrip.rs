//! Wire-format guarantees for the output-claim structs that cross the
//! prover-verifier boundary: bincode round-trips, the field-declaration-order
//! byte layout, and pinned golden vectors so a silent wire-format change (a
//! reordered field, an added prefix, a serde impl swap) fails loudly instead
//! of round-tripping.

#![expect(
    clippy::expect_used,
    reason = "tests unwrap infallible serialization of well-formed values"
)]

use jolt_claims::protocols::jolt::relations::spartan::{
    OuterRemainderOutputClaims, OuterUniskipOutputClaims, ProductRemainderOutputClaims,
    ProductUniskipOutputClaims, SpartanShiftOutputClaims,
};
use jolt_claims::{OutputClaims, SumcheckDomain};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Fr's serde form is its canonical 32-byte little-endian array; bincode's
/// standard config adds no framing for fixed arrays or structs.
const FR_WIRE_BYTES: usize = 32;

fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .expect("bincode serialization should not fail")
}

fn assert_roundtrip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + core::fmt::Debug,
{
    let bytes = encode(value);
    let (decoded, consumed): (T, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("bincode deserialization should not fail");
    assert_eq!(consumed, bytes.len(), "decoder must consume every byte");
    assert_eq!(&decoded, value);
}

fn random_claims<T: OutputClaims<Fr>>(rng: &mut ChaCha20Rng) -> T {
    T::from_opening_values(|_| Some(Fr::random(rng)))
        .expect("every declared opening id resolves against the constant source")
}

#[test]
fn output_claim_structs_roundtrip_bincode_exactly() {
    let mut rng = ChaCha20Rng::from_seed([17; 32]);
    assert_roundtrip(&random_claims::<OuterUniskipOutputClaims<Fr>>(&mut rng));
    assert_roundtrip(&random_claims::<OuterRemainderOutputClaims<Fr>>(&mut rng));
    assert_roundtrip(&random_claims::<ProductUniskipOutputClaims<Fr>>(&mut rng));
    assert_roundtrip(&random_claims::<ProductRemainderOutputClaims<Fr>>(&mut rng));
    assert_roundtrip(&random_claims::<SpartanShiftOutputClaims<Fr>>(&mut rng));
}

/// The wire encoding of an output-claim struct is exactly the canonical
/// (field-declaration) order concatenation of 32-byte little-endian scalars —
/// no length prefixes, no per-field framing. Fills the struct with the
/// counter values 1..=N so any reordering or duplication shows up in the
/// byte stream.
#[test]
fn outer_remainder_wire_layout_is_declaration_order_le_scalars() {
    let mut counter = 0u64;
    let claims = OuterRemainderOutputClaims::<Fr>::from_opening_values(|_| {
        counter += 1;
        Some(Fr::from_u64(counter))
    })
    .expect("every declared opening id resolves against the counter source");
    let field_count = claims.canonical_order().len();

    let bytes = encode(&claims);
    assert_eq!(bytes.len(), field_count * FR_WIRE_BYTES);

    for index in 0..field_count {
        let mut expected = [0u8; FR_WIRE_BYTES];
        expected[..8].copy_from_slice(&(index as u64 + 1).to_le_bytes());
        assert_eq!(
            &bytes[index * FR_WIRE_BYTES..(index + 1) * FR_WIRE_BYTES],
            expected,
            "field {index} is not the canonical-order LE scalar",
        );
    }
}

/// Pinned golden vector: `ProductUniskipOutputClaims` with a known scalar is
/// exactly the scalar's 32 canonical little-endian bytes. A change to Fr's
/// serde form, to bincode framing, or to the struct shape breaks this even
/// though a round-trip would still pass.
#[test]
fn product_uniskip_output_claims_match_pinned_golden_bytes() {
    let claims = ProductUniskipOutputClaims {
        uniskip: Fr::from_u64(0x0123_4567_89ab_cdef),
    };
    let expected: [u8; 32] = [
        0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(encode(&claims), expected);
    assert_roundtrip(&claims);
}

/// Pinned golden vectors for `SumcheckDomain`: variant tag then payload, one
/// varint byte each under bincode's standard config. Reordering the enum's
/// variants silently changes the wire format — this pins it.
#[test]
fn sumcheck_domain_matches_pinned_golden_bytes() {
    assert_eq!(encode(&SumcheckDomain::BooleanHypercube), [0]);
    assert_eq!(
        encode(&SumcheckDomain::CenteredInteger { domain_size: 10 }),
        [1, 10],
    );
    assert_roundtrip(&SumcheckDomain::BooleanHypercube);
    assert_roundtrip(&SumcheckDomain::centered_integer(10));
}

/// Adversarial wire input: a scalar encoding of the BN254 modulus `r` (the
/// smallest non-canonical representative) and a truncated scalar must both be
/// rejected, not silently reduced or zero-padded.
#[test]
fn non_canonical_and_truncated_scalars_are_rejected() {
    // r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001,
    // little-endian.
    let modulus_le: [u8; 32] = [
        0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8, 0x33,
        0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e,
        0x64, 0x30,
    ];
    assert!(
        bincode::serde::decode_from_slice::<ProductUniskipOutputClaims<Fr>, _>(
            &modulus_le,
            bincode::config::standard(),
        )
        .is_err(),
        "the field modulus is not a canonical scalar and must not deserialize",
    );

    assert!(
        bincode::serde::decode_from_slice::<ProductUniskipOutputClaims<Fr>, _>(
            &modulus_le[..31],
            bincode::config::standard(),
        )
        .is_err(),
        "a truncated scalar must not deserialize",
    );
}
