//! Common test utilities and standardized test suite for transcript sponges.

/// Standardized test suite for a sponge `H` with uniform output: the
/// prover's and verifier's spongefish states over `H`, compared through the
/// `Fr` challenges they draw.
#[macro_export]
macro_rules! transcript_tests {
    ($sponge:ty) => {
        use jolt_field::Fr;
        use jolt_transcript::{
            protocol_tag, prover_transcript, verifier_transcript, FieldTranscript, ProverState,
            Transcript,
        };
        use std::collections::HashSet;

        const SESSION: [u8; 32] = protocol_tag(b"jolt-transcript/tests");

        fn transcript(statement: &[u8; 4]) -> ProverState<$sponge> {
            prover_transcript::<$sponge>(&SESSION, statement)
        }

        fn challenge_after<F: FnOnce(&mut ProverState<$sponge>)>(statement: &[u8; 4], f: F) -> Fr {
            let mut t = transcript(statement);
            f(&mut t);
            t.challenge().unwrap()
        }

        #[test]
        fn test_determinism() {
            let c1 = challenge_after(b"same", |t| {
                t.append(&42u64);
                t.append_bytes(b"hello world");
            });
            let c2 = challenge_after(b"same", |t| {
                t.append(&42u64);
                t.append_bytes(b"hello world");
            });
            assert_eq!(
                c1, c2,
                "Identical operations must yield identical challenges"
            );
        }

        #[test]
        fn test_session_and_statement_separation() {
            let base = challenge_after(b"stmt", |_| {});
            assert_ne!(base, challenge_after(b"STMT", |_| {}));
            let mut other =
                prover_transcript::<$sponge>(&protocol_tag(b"another session"), b"stmt");
            assert_ne!(base, other.challenge().unwrap());
        }

        #[test]
        fn test_prover_and_verifier_agree() {
            let mut prover = transcript(b"pair");
            let mut verifier = verifier_transcript::<$sponge>(&SESSION, b"pair", &[]);
            schedule(&mut prover);
            schedule(&mut verifier);
            let p: Vec<Fr> = prover.challenge_vector(3).unwrap();
            let v: Vec<Fr> = verifier.challenge_vector(3).unwrap();
            assert_eq!(p, v);
            verifier.check_eof().unwrap();
        }

        /// The same message schedule, on either side.
        fn schedule<T: FieldTranscript<Fr>>(t: &mut T) {
            t.append(&Fr::from(7u64));
            t.append_values(&[Fr::from(1u64), Fr::from(2u64)]);
            let _: Fr = t.challenge().unwrap();
            t.append_bytes(b"tail");
        }

        #[test]
        fn test_challenge_uniqueness() {
            let mut transcript = transcript(b"uniq");
            let mut challenges = HashSet::new();
            for i in 0..10_000 {
                let c: Fr = transcript.challenge().unwrap();
                assert!(
                    challenges.insert(c),
                    "Duplicate challenge found at iteration {i}"
                );
            }
        }

        #[test]
        fn test_order_matters() {
            let c1 = challenge_after(b"swap", |t| {
                t.append(&1u64);
                t.append(&2u64);
            });
            let c2 = challenge_after(b"swap", |t| {
                t.append(&2u64);
                t.append(&1u64);
            });
            assert_ne!(c1, c2, "Order of appends must affect challenge");
        }

        #[test]
        fn test_data_sensitivity() {
            let c1 = challenge_after(b"data", |t| t.append(&0u64));
            let c2 = challenge_after(b"data", |t| t.append(&1u64));
            assert_ne!(c1, c2, "Different data must produce different challenges");
        }

        #[test]
        fn test_bytes_framing() {
            // Each byte string carries its length, so splitting or extending
            // the bytes changes the absorbed words.
            let empty = challenge_after(b"empt", |t| t.append_bytes(&[]));
            assert_ne!(empty, challenge_after(b"empt", |t| t.append_bytes(&[0])));
            let split = challenge_after(b"empt", |t| {
                t.append_bytes(b"a");
                t.append_bytes(b"b");
            });
            assert_ne!(split, challenge_after(b"empt", |t| t.append_bytes(b"ab")));
            let leading_empty = challenge_after(b"empt", |t| {
                t.append_bytes(&[]);
                t.append_bytes(b"x");
            });
            assert_ne!(
                leading_empty,
                challenge_after(b"empt", |t| t.append_bytes(b"x"))
            );
        }

        #[test]
        #[should_panic(expected = "exceeds 32 bytes")]
        fn test_protocol_tag_too_long() {
            let long_name = std::hint::black_box([b'x'; 33]);
            let _ = protocol_tag(&long_name);
        }

        #[test]
        fn test_challenge_vector() {
            let challenges: Vec<Fr> = transcript(b"vect").challenge_vector(5).unwrap();
            assert_eq!(challenges.len(), 5);
            let unique: HashSet<_> = challenges.iter().collect();
            assert_eq!(unique.len(), 5, "All challenges in vector should be unique");
        }

        #[test]
        fn test_challenge_powers() {
            let powers: Vec<Fr> = transcript(b"powr").challenge_powers(4).unwrap();
            let gamma: Fr = transcript(b"powr").challenge().unwrap();
            assert_eq!(
                powers,
                vec![Fr::from(1u64), gamma, gamma * gamma, gamma * gamma * gamma]
            );
        }

        /// A short challenge is the little-endian integer of the first
        /// `ceil(bits / 8)` squeezed bytes with the bits above its width
        /// cleared. The squeeze is padded to whole words and takes nothing
        /// more. The verifier draws the same value.
        #[test]
        fn test_challenge_short_scalar() {
            use jolt_field::CanonicalEncoding;

            for (loss, bits) in [(1u64, 128u32), (2, 129), (3, 130), (4, 130)] {
                let mut widest = 0;
                for round in 0..64u32 {
                    let statement = round.to_be_bytes();
                    let mut short = transcript(&statement);
                    let mut manual = transcript(&statement);
                    let value: Fr = short.challenge_short_scalar(loss).unwrap();

                    let len = bits.div_ceil(8) as usize;
                    let mut bytes =
                        manual.verifier_message_as(len.next_multiple_of(8), <[u8]>::to_vec);
                    bytes.truncate(len);
                    let excess = 8 * len as u32 - bits;
                    *bytes.last_mut().unwrap() &= u8::MAX >> excess;
                    assert_eq!(value, Fr::from_bytes_le_reduced(&bytes));
                    let next: Fr = short.challenge().unwrap();
                    assert_eq!(next, manual.challenge().unwrap());

                    let mut verifier = verifier_transcript::<$sponge>(&SESSION, &statement, &[]);
                    let verified: Fr = verifier.challenge_short_scalar(loss).unwrap();
                    assert_eq!(verified, value);
                    widest = widest.max(value.num_bits());
                }
                // The top bit is set with probability 1/2 per draw.
                assert_eq!(widest, bits, "loss={loss}");
            }
        }
    };
}
