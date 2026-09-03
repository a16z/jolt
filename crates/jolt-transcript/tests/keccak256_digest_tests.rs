use jolt_field::{CanonicalBytes, CanonicalEncoding, Fr, Ring};
use jolt_transcript::{Keccak256Transcript, Label, Transcript};
use sha3::{Digest, Keccak256};

fn hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn round_word(round: u32) -> [u8; 32] {
    let mut word = [0u8; 32];
    word[28..].copy_from_slice(&round.to_be_bytes());
    word
}

#[test]
fn chained_keccak_encoding_is_evm_replayable() {
    let mut label = [0u8; 32];
    label[..7].copy_from_slice(b"wrapper");
    let state0 = hash(&[&label]);
    let mut transcript = Keccak256Transcript::<Fr>::new(b"wrapper");
    assert_eq!(transcript.state(), state0);

    let value = Fr::from_u64(0x0102_0304);
    transcript.append(&value);
    let mut field_bytes = [0u8; 32];
    value.to_bytes_le(&mut field_bytes);
    field_bytes.reverse();
    let state1 = hash(&[&state0, &round_word(0), &field_bytes]);
    assert_eq!(transcript.state(), state1);

    transcript.append(&Label(b"round"));
    let mut padded_label = [0u8; 32];
    padded_label[..5].copy_from_slice(b"round");
    let state2 = hash(&[&state1, &round_word(1), &padded_label]);
    assert_eq!(transcript.state(), state2);

    let challenge_digest = hash(&[&state2, &round_word(2)]);
    let challenge = transcript.challenge();
    assert_eq!(challenge, Fr::from_challenge_bytes(&challenge_digest[..16]));
    assert_eq!(transcript.state(), challenge_digest);
}
