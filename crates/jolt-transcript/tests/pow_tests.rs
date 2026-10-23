//! Proof of work on the transcript sponge, as a standalone step.

#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_field::{CanonicalBytes, F8};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, BytesMsg, PoseidonSponge, Probe,
    ProverTranscript, VerifierTranscript, MAX_POW_BITS,
};
use spongefish::{
    instantiations::Keccak, DuplexSpongeInit, DuplexSpongeInterface, ProverState, VerifierState,
};

const SESSION: &[u8] = b"jolt/pow/v1";
const INSTANCE: [u8; 32] = [0x24; 32];

fn prover() -> ProverState<Keccak> {
    let mut prover = prover_transcript::<Keccak>(SESSION, INSTANCE);
    prover.prover_message(&BytesMsg(b"before".to_vec()));
    prover
}

fn verifier(narg: &[u8]) -> VerifierState<'_, Keccak> {
    let mut verifier = verifier_transcript::<Keccak>(SESSION, INSTANCE, narg);
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
    verifier
}

fn bytes(values: &[F8]) -> Vec<u8> {
    let mut out = vec![0u8; values.len()];
    for (value, byte) in values.iter().zip(out.chunks_exact_mut(1)) {
        value.to_bytes_le(byte);
    }
    out
}

/// The canonical LEB128 encoding of `value`.
fn leb128(mut value: u32) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(low);
            return out;
        }
        out.push(low | 0x80);
    }
}

/// Decodes a LEB128 encoding, without canonicality checks.
fn decode(encoding: &[u8]) -> u32 {
    encoding
        .iter()
        .rev()
        .fold(0, |value, byte| (value << 7) | u32::from(byte & 0x7f))
}

/// The low `bits` bits of the little-endian `check` bytes are zero.
fn passes(bits: u32, check: &[u8]) -> bool {
    let mut word = [0u8; 4];
    word[..check.len()].copy_from_slice(check);
    u32::from_le_bytes(word) & ((1u32 << bits) - 1) == 0
}

/// Rebuilds the step by hand from the layout: the canonical LEB128 nonce,
/// one prover message per byte, then `ceil(bits / 8)` check bytes with the
/// low `bits` bits zero. `F8` decodes raw bytes, so the challenge message
/// drawn next is the next `n` squeezed bytes. The nonce must be the least one
/// that passes.
fn manual_check(bits: u32, n: usize) {
    let word_len = (bits as usize).div_ceil(8);

    let mut jolt = prover();
    let start = jolt.narg_string().len();
    ProverTranscript::grind(&mut jolt, bits).unwrap();
    let nonce = jolt.narg_string()[start..].to_vec();
    let values = ProverTranscript::field_challenges::<F8>(&mut jolt, n).unwrap();
    let tail: [u8; 8] = jolt.verifier_message();
    let value = decode(&nonce);
    assert_eq!(nonce, leb128(value), "bits {bits}");
    assert!(u64::from(value) < 1 << (bits + 7), "bits {bits}");

    let mut manual = prover();
    for byte in &nonce {
        manual.prover_message(byte);
    }
    let mut check = vec![0u8; word_len];
    for byte in &mut check {
        *byte = manual.verifier_message();
    }
    assert!(passes(bits, &check), "bits {bits}");
    let mut expected = vec![0u8; n];
    for byte in &mut expected {
        *byte = manual.verifier_message();
    }
    assert_eq!(bytes(&values), expected, "bits {bits}, n {n}");
    assert_eq!(manual.verifier_message::<[u8; 8]>(), tail);
    assert_eq!(manual.narg_string(), jolt.narg_string());

    let base = prover();
    for smaller in 0..value {
        let mut probe = base.duplex_sponge_state.clone();
        let mut check = vec![0u8; word_len];
        let _ = probe.absorb(&leb128(smaller)).squeeze(&mut check);
        assert!(
            !passes(bits, &check),
            "nonce {smaller} also passes at bits {bits}"
        );
    }

    let mut verifier = verifier(jolt.narg_string());
    VerifierTranscript::check_grind(&mut verifier, bits).unwrap();
    let read = VerifierTranscript::field_challenges::<F8>(&mut verifier, n).unwrap();
    assert_eq!(read, values);
    assert_eq!(verifier.verifier_message::<[u8; 8]>(), tail);
    verifier.check_eof().unwrap();
}

#[test]
fn layout_matches_the_manual_transcript() {
    for bits in [1, 7, 8, 9, 16] {
        for n in [0, 1, 32, 200] {
            manual_check(bits, n);
        }
    }
}

fn sponge_prover<H: DuplexSpongeInit<U = u8>>() -> ProverState<H> {
    let mut prover = prover_transcript::<H>(SESSION, INSTANCE);
    prover.prover_message(&BytesMsg(b"before".to_vec()));
    prover
}

fn sponge_verifier<H: DuplexSpongeInit<U = u8>>(narg: &[u8]) -> VerifierState<'_, H> {
    let mut verifier = verifier_transcript::<H>(SESSION, INSTANCE, narg);
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
    verifier
}

/// Runs `check_grind` on `before ‖ nonce ‖ after` and returns the verifier's
/// result, the bytes it squeezes next and whether the rest still verifies.
fn verify_nonce<H: DuplexSpongeInit<U = u8>>(
    prefix: &[u8],
    nonce: &[u8],
    bits: u32,
) -> (bool, [u8; 16], bool) {
    let mut after = sponge_prover::<H>();
    after.prover_message(&BytesMsg(b"after".to_vec()));
    let suffix = &after.narg_string()[prefix.len()..];
    let narg = [prefix, nonce, suffix].concat();
    let mut verifier = sponge_verifier::<H>(&narg);
    let passed = VerifierTranscript::check_grind(&mut verifier, bits).is_ok();
    let next = verifier.verifier_message::<[u8; 16]>();
    let rest = verifier
        .prover_message::<BytesMsg>()
        .is_ok_and(|message| message.0 == b"after")
        && verifier.check_eof().is_ok();
    (passed, next, rest)
}

/// For each sponge, including Poseidon, which absorbs a length with every
/// call: the search replays the real sends, the honest nonce round-trips and
/// is the least one, a larger passing nonce in range is accepted (proof bytes
/// are not unique), and a passing nonce out of range is rejected and poisons.
fn sponge_checks<H: DuplexSpongeInit<U = u8> + Sync>() {
    for bits in [1, 2, 9] {
        let word_len = (bits as usize).div_ceil(8);
        let base = sponge_prover::<H>();
        let prefix = base.narg_string().to_vec();
        let search = ProverTranscript::message_search(&base);
        let probe_passes = |value: u32| {
            let mut check = vec![0u8; word_len];
            search.probe(&leb128(value), &mut check);
            passes(bits, &check)
        };

        let mut prover = sponge_prover::<H>();
        ProverTranscript::grind(&mut prover, bits).unwrap();
        let nonce = prover.narg_string()[prefix.len()..].to_vec();
        let value = decode(&nonce);
        assert_eq!(nonce, leb128(value));
        let challenge = prover.verifier_message::<[u8; 16]>();

        let mut sent = sponge_prover::<H>();
        for byte in &nonce {
            sent.prover_message(byte);
        }
        let mut probed = [0u8; 40];
        search.probe(&nonce, &mut probed);
        assert_eq!(sent.verifier_message::<[u8; 40]>(), probed);

        assert_eq!(
            verify_nonce::<H>(&prefix, &nonce, bits),
            (true, challenge, true)
        );
        for smaller in 0..value {
            assert!(
                !probe_passes(smaller),
                "nonce {smaller} passes at bits {bits}"
            );
        }
        for smaller in [0, value / 2, value.saturating_sub(1)] {
            if smaller < value {
                assert!(!verify_nonce::<H>(&prefix, &leb128(smaller), bits).0);
            }
        }

        let range = 1u32 << (bits + 7);
        let larger = (value + 1..range)
            .find(|&value| probe_passes(value))
            .unwrap();
        let (passed, _, rest) = verify_nonce::<H>(&prefix, &leb128(larger), bits);
        assert!(passed && rest, "bits {bits}");

        let beyond = (range..range * 4)
            .find(|&value| probe_passes(value))
            .unwrap();
        let (passed, _, rest) = verify_nonce::<H>(&prefix, &leb128(beyond), bits);
        assert!(!passed && !rest, "bits {bits}");
    }
}

#[test]
fn every_sponge_grinds_and_rejects_forgeries() {
    sponge_checks::<Keccak>();
    sponge_checks::<Blake2b512>();
    sponge_checks::<PoseidonSponge>();
}

#[test]
fn zero_bits_does_nothing() {
    let mut plain = prover();
    let mut pow = prover();
    ProverTranscript::grind(&mut pow, 0).unwrap();
    assert_eq!(pow.narg_string(), plain.narg_string());
    assert_eq!(
        pow.verifier_message::<[u8; 16]>(),
        plain.verifier_message::<[u8; 16]>()
    );

    let narg = prover().narg_string().to_vec();
    let mut plain = verifier(&narg);
    let mut pow = verifier(&narg);
    VerifierTranscript::check_grind(&mut pow, 0).unwrap();
    assert_eq!(
        pow.verifier_message::<[u8; 16]>(),
        plain.verifier_message::<[u8; 16]>()
    );
}

#[test]
fn message_search_matches_sending_then_squeezing_and_leaves_the_transcript() {
    let mut live = prover();
    let search = ProverTranscript::message_search(&live);
    let mut probed = [0u8; 200];
    search.probe(b"candidate", &mut probed);

    let mut sent = prover();
    for byte in b"candidate" {
        sent.prover_message(byte);
    }
    assert_eq!(sent.verifier_message::<[u8; 200]>(), probed);

    let mut untouched = prover();
    assert_eq!(live.narg_string(), untouched.narg_string());
    assert_eq!(
        live.verifier_message::<[u8; 16]>(),
        untouched.verifier_message::<[u8; 16]>()
    );
}

/// The prover sends the least accepted nonce in its canonical encoding, so a
/// smaller nonce, a non-canonical encoding and a truncated nonce all fail,
/// and each failure poisons the verifier.
#[test]
fn rejects_smaller_non_canonical_and_truncated_nonces() {
    let bits = 16;
    let start = prover().narg_string().len();
    let mut prover = prover();
    ProverTranscript::grind(&mut prover, bits).unwrap();
    let narg = prover.narg_string().to_vec();
    let (prefix, nonce) = narg.split_at(start);
    let value = decode(nonce);
    assert!(
        value > 1,
        "pick another session if the least nonce is small"
    );

    let forged = |encoding: &[u8]| [prefix, encoding].concat();
    let mut rejected = [0, value / 2, value - 1]
        .map(|smaller| forged(&leb128(smaller)))
        .to_vec();
    // The same nonce with a trailing zero byte.
    let mut padded = nonce.to_vec();
    *padded.last_mut().unwrap() |= 0x80;
    padded.push(0);
    rejected.push(forged(&padded));
    for cut in 1..=nonce.len() {
        rejected.push(narg[..narg.len() - cut].to_vec());
    }
    for forged in rejected {
        let mut verifier = verifier(&forged);
        assert!(VerifierTranscript::check_grind(&mut verifier, bits).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

#[test]
fn rejects_unsupported_difficulties_before_touching_the_sponge() {
    for bits in [MAX_POW_BITS + 1, u32::MAX] {
        let mut prover = prover();
        let before = prover.narg_string().len();
        assert!(ProverTranscript::grind(&mut prover, bits).is_err());
        assert_eq!(prover.narg_string().len(), before);
        let mut untouched = self::prover();
        assert_eq!(
            prover.verifier_message::<[u8; 16]>(),
            untouched.verifier_message::<[u8; 16]>()
        );

        let narg = self::prover().narg_string().to_vec();
        let mut verifier = verifier(&narg);
        assert!(VerifierTranscript::check_grind(&mut verifier, bits).is_err());
        let mut fresh = self::verifier(&narg);
        assert_eq!(
            verifier.verifier_message::<[u8; 16]>(),
            fresh.verifier_message::<[u8; 16]>()
        );
        assert!(verifier.check_eof().is_err());
    }
}
