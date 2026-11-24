// The bundled Dory proof uses the standard instruction profile's wire layout.
#![cfg(not(any(feature = "akita", feature = "field-inline")))]
#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "encoding regression tests should fail loudly"
)]

use ark_serialize::CanonicalSerialize;
use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, JoltGroup, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_verifier::{
    deserialize_verifier_object, serialize_verifier_object, JoltProof, JoltVerifierPreprocessing,
    VerifierObjectDecodeError,
};

type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type Bundle = (Preprocessing, JoltDevice, Proof, Option<DoryCommitment>);

const FIXTURE: &[u8] = include_bytes!("../fuzz/fixtures/muldiv-bundle.bin");

fn fixture_proof() -> Proof {
    let bundle: Bundle = deserialize_verifier_object(FIXTURE).expect("canonical fixture bundle");
    bundle.2
}

#[test]
fn accepted_proof_rejects_overlong_varint_widths() {
    let proof = fixture_proof();
    assert_eq!(proof.narg.as_slice(), &[]);
    let canonical = serialize_verifier_object(&proof).expect("serialize fixture proof");
    assert_eq!(canonical.last(), Some(&0), "canonical empty NARG length");

    let overlong_zeroes = [
        vec![251, 0, 0],
        vec![252, 0, 0, 0, 0],
        vec![253, 0, 0, 0, 0, 0, 0, 0, 0],
    ];
    for overlong in overlong_zeroes {
        let mut tampered = canonical[..canonical.len() - 1].to_vec();
        tampered.extend_from_slice(&overlong);

        let (decoded, consumed): (Proof, usize) =
            bincode::serde::decode_from_slice(&tampered, bincode::config::standard())
                .expect("bincode accepts the overlong NARG length");
        assert_eq!(consumed, tampered.len());
        assert_eq!(decoded, proof);
        assert!(matches!(
            deserialize_verifier_object::<Proof>(&tampered),
            Err(VerifierObjectDecodeError::NonCanonicalEncoding)
        ));
    }
}

#[test]
fn accepted_proof_rejects_trailing_byte() {
    let proof = fixture_proof();
    let canonical = serialize_verifier_object(&proof).expect("serialize fixture proof");
    let mut tampered = canonical.clone();
    tampered.push(0);

    let (decoded, consumed): (Proof, usize) =
        bincode::serde::decode_from_slice(&tampered, bincode::config::standard())
            .expect("bincode accepts a trailing byte");
    assert_eq!(consumed, canonical.len());
    assert_eq!(decoded, proof);
    assert!(matches!(
        deserialize_verifier_object::<Proof>(&tampered),
        Err(VerifierObjectDecodeError::NonCanonicalEncoding)
    ));
}

#[test]
fn accepted_proof_rejects_identity_with_nonzero_x() {
    let proof = fixture_proof();
    let bytes = serialize_verifier_object(&proof).expect("serialize fixture proof");
    let mut identity = Vec::new();
    Bn254G1::identity()
        .into_inner()
        .serialize_compressed(&mut identity)
        .unwrap();
    let positions: Vec<_> = bytes
        .windows(identity.len())
        .enumerate()
        .filter_map(|(position, window)| (window == identity).then_some(position))
        .collect();

    let mut equivalent_encodings = 0;
    for position in positions {
        let mut tampered = bytes.clone();
        tampered[position] = 1;
        let Ok((decoded, consumed)) =
            bincode::serde::decode_from_slice::<Proof, _>(&tampered, bincode::config::standard())
        else {
            continue;
        };
        if consumed == tampered.len() && decoded == proof {
            equivalent_encodings += 1;
            assert!(matches!(
                deserialize_verifier_object::<Proof>(&tampered),
                Err(VerifierObjectDecodeError::NonCanonicalEncoding)
            ));
        }
    }
    assert_eq!(
        equivalent_encodings, 1,
        "the accepted fixture should contain one reachable G1 identity"
    );
}
