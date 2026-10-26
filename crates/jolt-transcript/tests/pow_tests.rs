//! Proof of work on the transcript sponge, as a standalone step.

#![expect(clippy::unwrap_used, reason = "tests")]

use jolt_field::{CanonicalBytes, F8};
use jolt_transcript::FieldTranscript;
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, Blake2b512, BytesMsg, PoseidonSponge,
    Probe, ProverState, ProverTranscript, VerifierTranscript, MAX_POW_BITS,
};
use spongefish::{instantiations::Keccak, DuplexSpongeInit, DuplexSpongeInterface, VerifierState};

const SESSION: [u8; 32] = protocol_tag(b"jolt/pow/v1");
const INSTANCE: [u8; 32] = [0x24; 32];

fn prover() -> ProverState<Keccak> {
    let mut prover = prover_transcript::<Keccak>(&SESSION, &INSTANCE);
    prover.send(&BytesMsg(b"before".to_vec()));
    prover
}

fn verifier(narg: &[u8]) -> VerifierState<'_, Keccak> {
    let mut verifier = verifier_transcript::<Keccak>(&SESSION, &INSTANCE, narg);
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
fn leb128(mut value: u64) -> Vec<u8> {
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
fn decode(encoding: &[u8]) -> u64 {
    encoding
        .iter()
        .rev()
        .fold(0, |value, byte| (value << 7) | u64::from(byte & 0x7f))
}

/// The absorbed form of a nonce: one `u64` little-endian word.
fn word(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

/// The low `bits` bits of the eight little-endian `check` bytes are zero.
fn passes(bits: u32, check: &[u8]) -> bool {
    let word: [u8; 8] = check.try_into().unwrap();
    u64::from_le_bytes(word) & ((1u64 << bits) - 1) == 0
}

/// Sends `value` as the step does: one prover message, canonical LEB128 in
/// the NARG string and one `u64` word in the sponge.
fn send_nonce<H: DuplexSpongeInterface<U = u8>>(prover: &mut ProverState<H>, value: u64) {
    prover.prover_message_with(
        &value,
        |&value| word(value),
        |&value, narg| narg.extend_from_slice(&leb128(value)),
    );
}

/// Rebuilds the step by hand from the layout: the nonce as one prover
/// message (canonical LEB128 on the wire, a `u64` word absorbed), then eight
/// check bytes with the low `bits` bits zero. `F8` decodes raw bytes from a
/// squeeze padded to a word, so the challenge message drawn next is the first
/// `n` of the next `ceil(n / 8)` squeezed words. The nonce must be the least
/// one that passes.
fn manual_check(bits: u32, n: usize) {
    let mut jolt = prover();
    let start = jolt.narg_string().len();
    ProverTranscript::grind(&mut jolt, bits).unwrap();
    let nonce = jolt.narg_string()[start..].to_vec();
    let values = FieldTranscript::<F8>::challenge_vector(&mut jolt, n).unwrap();
    let tail: [u8; 8] = jolt.verifier_message();
    let value = decode(&nonce);
    assert_eq!(nonce, leb128(value), "bits {bits}");
    assert!(value < 1 << (bits + 7), "bits {bits}");

    let mut manual = prover();
    send_nonce(&mut manual, value);
    let check: [u8; 8] = manual.verifier_message();
    assert!(passes(bits, &check), "bits {bits}");
    let mut expected = vec![0u8; n.div_ceil(8) * 8];
    for byte in &mut expected {
        *byte = manual.verifier_message();
    }
    expected.truncate(n);
    assert_eq!(bytes(&values), expected, "bits {bits}, n {n}");
    assert_eq!(manual.verifier_message::<[u8; 8]>(), tail);
    assert_eq!(manual.narg_string(), jolt.narg_string());

    let base = prover();
    for smaller in 0..value {
        let mut probe = base.duplex_sponge_state.clone();
        let mut check = [0u8; 8];
        let _ = probe.absorb(&word(smaller)).squeeze(&mut check);
        assert!(
            !passes(bits, &check),
            "nonce {smaller} also passes at bits {bits}"
        );
    }

    let mut verifier = verifier(jolt.narg_string());
    VerifierTranscript::check_grind(&mut verifier, bits).unwrap();
    let read = FieldTranscript::<F8>::challenge_vector(&mut verifier, n).unwrap();
    assert_eq!(read, values);
    assert_eq!(verifier.verifier_message::<[u8; 8]>(), tail);
    verifier.check_eof().unwrap();
}

#[test]
fn layout_matches_the_manual_transcript() {
    for bits in [1, 7, 8, 9, 16] {
        for n in [0, 1, 5, 32, 200] {
            manual_check(bits, n);
        }
    }
}

fn sponge_prover<H: DuplexSpongeInit<U = u8> + Sync>() -> ProverState<H> {
    let mut prover = prover_transcript::<H>(&SESSION, &INSTANCE);
    prover.send(&BytesMsg(b"before".to_vec()));
    prover
}

fn sponge_verifier<H: DuplexSpongeInit<U = u8>>(narg: &[u8]) -> VerifierState<'_, H> {
    let mut verifier = verifier_transcript::<H>(&SESSION, &INSTANCE, narg);
    assert_eq!(verifier.prover_message::<BytesMsg>().unwrap().0, b"before");
    verifier
}

/// Runs `check_grind` on `before ‖ nonce ‖ after` and returns the verifier's
/// result, the bytes it squeezes next and whether the rest still verifies.
fn verify_nonce<H: DuplexSpongeInit<U = u8> + Sync>(
    prefix: &[u8],
    nonce: &[u8],
    bits: u32,
) -> (bool, [u8; 16], bool) {
    let mut after = sponge_prover::<H>();
    after.send(&BytesMsg(b"after".to_vec()));
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
/// call: the search replays the real send (one absorb of the nonce's word),
/// the honest nonce round-trips and is the least one, a larger passing nonce
/// in range is accepted (proof bytes are not unique), and a passing nonce out
/// of range is rejected and poisons.
fn sponge_checks<H: DuplexSpongeInit<U = u8> + Sync>() {
    for bits in [1, 2, 9] {
        let base = sponge_prover::<H>();
        let prefix = base.narg_string().to_vec();
        let search = ProverTranscript::message_search(&base);
        let probe_passes = |value: u64| {
            let mut check = [0u8; 8];
            search.probe(&word(value), &mut check);
            passes(bits, &check)
        };

        let mut prover = sponge_prover::<H>();
        ProverTranscript::grind(&mut prover, bits).unwrap();
        let nonce = prover.narg_string()[prefix.len()..].to_vec();
        let value = decode(&nonce);
        assert_eq!(nonce, leb128(value));
        let challenge = prover.verifier_message::<[u8; 16]>();

        let mut sent = sponge_prover::<H>();
        send_nonce(&mut sent, value);
        assert_eq!(sent.narg_string(), prover.narg_string());
        let mut probed = [0u8; 40];
        search.probe(&word(value), &mut probed);
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

        let range = 1u64 << (bits + 7);
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

    // One prover message of the candidate's bytes, absorbed in one call.
    let mut sent = prover();
    sent.send(b"candidate");
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

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

/// Grinds at `bits` after a fixed prefix and checks the nonce's NARG bytes,
/// the public nonce and the 32 bytes squeezed next against pinned values,
/// for both the NARG and the public form of the step.
fn golden<H: DuplexSpongeInit<U = u8> + Sync>(bits: u32, narg_nonce: &str, next: &str) {
    use jolt_transcript::Transcript;

    let session = protocol_tag(b"jolt/pow/vectors/v1");
    let start = || {
        let mut prover = prover_transcript::<H>(&session, b"statement");
        prover.send(&BytesMsg(b"before".to_vec()));
        prover
    };
    let (narg_nonce, next) = (unhex(narg_nonce), unhex(next));

    let mut prover = start();
    let offset = prover.narg_string().len();
    ProverTranscript::grind(&mut prover, bits).unwrap();
    assert_eq!(prover.narg_string()[offset..], narg_nonce, "bits {bits}");
    assert_eq!(
        prover.verifier_message::<[u8; 32]>()[..],
        next,
        "bits {bits}"
    );
    let narg = prover.narg_string().to_vec();
    let mut verifier = verifier_transcript::<H>(&session, b"statement", &narg);
    let _ = verifier.prover_message::<BytesMsg>().unwrap();
    VerifierTranscript::check_grind(&mut verifier, bits).unwrap();
    assert_eq!(verifier.verifier_message::<[u8; 32]>()[..], next);
    verifier.check_eof().unwrap();

    let mut public = start();
    let nonce = ProverTranscript::public_grind(&mut public, bits).unwrap();
    assert_eq!(nonce, Some(decode(&narg_nonce)), "bits {bits}");
    assert_eq!(
        public.verifier_message::<[u8; 32]>()[..],
        next,
        "bits {bits}"
    );
    let mut checked = start();
    checked.check_public_grind(bits, nonce).unwrap();
    assert_eq!(checked.verifier_message::<[u8; 32]>()[..], next);
}

#[cfg_attr(
    feature = "keccak256-digests",
    ignore = "pinned for SHA3-256 digests; keccak256-digests changes the session identifier"
)]
/// Pinned proofs of work, recorded when the cap was 25 bits with a `u32`
/// nonce. Raising the cap to 57 bits with a `u64` nonce must leave every
/// proof at 25 bits or fewer byte-identical, so these must never change.
#[test]
fn proofs_below_the_old_cap_are_pinned() {
    golden::<Keccak>(
        1,
        "01",
        "6e4fe197c703d9eefacec5b04b3de131bb4bb7d0be04e3b823afe062ee757d79",
    );
    golden::<Keccak>(
        9,
        "9601",
        "d7eeaec7d6506aebe60594fdde2d17b9cd20e314efaf1b88290d9c1987095698",
    );
    golden::<Keccak>(
        16,
        "cf10",
        "7fc406f9a14c9d44b7f31b02949e1bcedaabd985ca19cde57fde896ca8d75300",
    );
    golden::<Keccak>(
        18,
        "cf10",
        "7fc406f9a14c9d44b7f31b02949e1bcedaabd985ca19cde57fde896ca8d75300",
    );
    golden::<Blake2b512>(
        8,
        "27",
        "245b8f4362b2f535fbe6d082d48411f9bac2f0b3c84714461249d61e2759f686",
    );
    golden::<Blake2b512>(
        12,
        "27",
        "245b8f4362b2f535fbe6d082d48411f9bac2f0b3c84714461249d61e2759f686",
    );
    golden::<PoseidonSponge>(
        6,
        "ba01",
        "9a34afc198d2655a0adebf24dd29cb2602ad809773612100b5b565c1dbfc4a10",
    );
    golden::<PoseidonSponge>(
        11,
        "b541",
        "775eef584f43c79fdd35fcaf2ff8f1e1fc8f074ce6b3cc137488fabf6f5808b8",
    );
}

/// A nonce of `2^32` or more, past the old `u32` range, is in range from 26
/// bits on and is absorbed as its full `u64` word: the verifier accepts it
/// exactly when the check bytes after that word pass. Out of range, at 25
/// bits, it is rejected whatever its check bytes.
#[test]
fn public_nonces_past_u32_are_absorbed_as_full_words() {
    use jolt_transcript::Transcript;

    let search = ProverTranscript::message_search(&prover());
    let check = |value: u64| {
        let mut check = [0u8; 8];
        search.probe(&word(value), &mut check);
        check
    };
    // Nonces whose check passes at 1 bit and, separately, fails at 1 bit.
    let pass = (1u64 << 32..1 << 33)
        .find(|&value| passes(1, &check(value)))
        .unwrap();
    let fail = (1u64 << 32..1 << 33)
        .find(|&value| !passes(1, &check(value)))
        .unwrap();
    for bits in [26, MAX_POW_BITS] {
        for nonce in [pass, fail, u64::MAX >> (MAX_POW_BITS - bits)] {
            let mut verifier = prover();
            assert_eq!(
                verifier.check_public_grind(bits, Some(nonce)).is_ok(),
                passes(bits, &check(nonce)),
                "bits {bits}, nonce {nonce}"
            );
        }
        // The word's high half reaches the sponge.
        assert_ne!(check(pass), check(pass & u64::from(u32::MAX)));
    }
    let mut verifier = prover();
    assert!(verifier.check_public_grind(25, Some(1 << 32)).is_err());
    let mut verifier = prover();
    assert!(verifier.check_public_grind(26, Some(1 << 33)).is_err());
}
