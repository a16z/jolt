//! Common test utilities and standardized test suite for transcript sponges.

/// Standardized test suite for a sponge `H` with uniform output: the
/// prover's and verifier's spongefish states over `H`, compared through the
/// `Fr` challenges they draw.
#[macro_export]
macro_rules! transcript_tests {
    ($sponge:ty) => {
        use jolt_field::Fr;
        use jolt_transcript::{
            prover_transcript, verifier_transcript, FieldTranscript, Label, ProverState, Transcript,
        };
        use std::collections::HashSet;

        const SESSION: &[u8] = b"jolt-transcript/tests";

        fn transcript(statement: &[u8; 4]) -> ProverState<$sponge> {
            prover_transcript::<$sponge>(SESSION, statement)
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
            let mut other = prover_transcript::<$sponge>(b"another session", b"stmt");
            assert_ne!(base, other.challenge().unwrap());
        }

        #[test]
        fn test_prover_and_verifier_agree() {
            let mut prover = transcript(b"pair");
            let mut verifier = verifier_transcript::<$sponge>(SESSION, b"pair", &[]);
            schedule(&mut prover);
            schedule(&mut verifier);
            let p: Vec<Fr> = prover.challenge_vector(3).unwrap();
            let v: Vec<Fr> = verifier.challenge_vector(3).unwrap();
            assert_eq!(p, v);
            verifier.check_eof().unwrap();
        }

        /// The same message schedule, on either side.
        fn schedule<T: FieldTranscript<Fr>>(t: &mut T) {
            t.append_labeled(b"value", &Fr::from(7u64));
            t.append_values(b"values", &[Fr::from(1u64), Fr::from(2u64)]);
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
        fn test_label_framing() {
            let labeled = challenge_after(b"labl", |t| t.append_labeled(b"x", &1u64));
            let explicit = challenge_after(b"labl", |t| {
                t.append(&Label(b"x"));
                t.append(&1u64);
            });
            assert_eq!(labeled, explicit);
            assert_ne!(
                labeled,
                challenge_after(b"labl", |t| t.append_labeled(b"y", &1u64))
            );
        }

        #[test]
        #[should_panic(expected = "exceeds 32 bytes")]
        fn test_label_too_long() {
            let long_label: &'static [u8; 33] = &[b'x'; 33];
            transcript(b"long").append(&Label(long_label));
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
    };
}
