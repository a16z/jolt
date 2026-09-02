//! Tests for the streaming `Blake3Transcript`: byte-level spec of the keyed
//! chain and the Fiat-Shamir properties the shared suite checks (minus the
//! empty-append rule — an empty append is a no-op here by design).

use std::collections::HashSet;

use blake3::Hasher;
use jolt_field::{CanonicalEncoding, Fr};
use jolt_transcript::{Blake3Transcript, Transcript};

type B3 = Blake3Transcript<Fr>;

fn padded_label(label: &[u8]) -> [u8; 32] {
    let mut padded = [0u8; 32];
    padded[..label.len()].copy_from_slice(label);
    padded
}

fn root_output(key: &[u8; 32], segment: &[u8]) -> [u8; 64] {
    let mut hasher = Hasher::new_keyed(key);
    let _ = hasher.update(segment);
    let mut out = [0u8; 64];
    hasher.finalize_xof().fill(&mut out);
    out
}

fn split(out: &[u8; 64]) -> ([u8; 32], [u8; 16]) {
    let mut state = [0u8; 32];
    let mut challenge = [0u8; 16];
    state.copy_from_slice(&out[..32]);
    challenge.copy_from_slice(&out[32..48]);
    (state, challenge)
}

fn keyed(key: &[u8; 32], bytes: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, bytes).as_bytes()
}

fn challenge_after(label: &'static [u8], f: impl FnOnce(&mut B3)) -> Fr {
    let mut t = B3::new(label);
    f(&mut t);
    t.challenge()
}

#[test]
fn squeeze_is_keyed_root_output_of_pending_bytes() {
    let a: Vec<u8> = (0..40u8).collect();
    let b: Vec<u8> = (0..100u8).map(|i| i.wrapping_mul(7)).collect();

    let mut t = B3::new(b"chain");
    t.append_bytes(&a);
    t.append_bytes(&b);
    let c1: Fr = t.challenge();
    let c2: Fr = t.challenge_scalar();

    let s0 = *blake3::hash(&padded_label(b"chain")).as_bytes();
    let ab = [a, b].concat();
    let (s1, ch1) = split(&root_output(&s0, &ab));
    assert_eq!(c1, Fr::from_challenge_bytes(&ch1));
    assert_eq!(s1, keyed(&s0, &ab));

    let (s2, ch2) = split(&root_output(&s1, &[]));
    assert_eq!(c2, Fr::from_scalar_challenge_bytes(&ch2));
    assert_eq!(t.state(), keyed(&s2, &[]));
}

#[test]
fn full_chunk_is_closed_before_more_bytes_arrive() {
    let s0 = *blake3::hash(&padded_label(b"chunk")).as_bytes();
    let first: Vec<u8> = (0..1024u32).map(|i| i as u8).collect();

    let mut exact = B3::new(b"chunk");
    exact.append_bytes(&first);
    let _: Fr = exact.challenge();
    let (s_exact, _) = split(&root_output(&s0, &first));
    assert_eq!(exact.state(), keyed(&s_exact, &[]));

    let mut over = B3::new(b"chunk");
    over.append_bytes(&first);
    over.append_bytes(&[0xAB]);
    let _: Fr = over.challenge();
    let (s_over, _) = split(&root_output(&keyed(&s0, &first), &[0xAB]));
    assert_eq!(over.state(), keyed(&s_over, &[]));

    let mut split_appends = B3::new(b"chunk");
    split_appends.append_bytes(&first[..1000]);
    split_appends.append_bytes(&[&first[1000..], &[0xAB]].concat());
    let _: Fr = split_appends.challenge();
    assert_eq!(split_appends.state(), over.state());
}

#[test]
fn determinism_and_sensitivity() {
    let base = challenge_after(b"fs", |t| {
        t.append_bytes(&1u64.to_be_bytes());
        t.append_bytes(&2u64.to_be_bytes());
    });
    let again = challenge_after(b"fs", |t| {
        t.append_bytes(&1u64.to_be_bytes());
        t.append_bytes(&2u64.to_be_bytes());
    });
    assert_eq!(base, again);

    let swapped = challenge_after(b"fs", |t| {
        t.append_bytes(&2u64.to_be_bytes());
        t.append_bytes(&1u64.to_be_bytes());
    });
    assert_ne!(base, swapped);
    assert_ne!(base, challenge_after(b"fs", |_| {}));
    assert_ne!(base, challenge_after(b"other", |_| {}));
}

#[test]
fn consecutive_squeezes_are_distinct() {
    let mut t = B3::new(b"squeeze");
    let mut seen = HashSet::new();
    for _ in 0..1_000 {
        let c: Fr = t.challenge();
        assert!(seen.insert(c));
    }
}
