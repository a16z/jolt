//! Common test utilities and standardized test suite for transcript implementations.

/// Standardized test suite macro for any `Transcript` implementation.
///
/// All comparisons are done through `challenge()` outputs since the
/// underlying spongefish duplex sponges do not expose internal state.
#[macro_export]
macro_rules! transcript_tests {
    ($transcript_type:ty) => {
        use jolt_transcript::Transcript;
        use std::collections::HashSet;

        // Helper: drive a transcript through a closure and squeeze a challenge.
        fn challenge_after<F: FnOnce(&mut $transcript_type)>(
            label: &'static [u8],
            f: F,
        ) -> <$transcript_type as Transcript>::Challenge {
            let mut t = <$transcript_type>::new(label);
            f(&mut t);
            t.challenge()
        }

        #[test]
        fn test_determinism() {
            let c1 = challenge_after(b"determinism_test", |t| {
                t.append_bytes(&42u64.to_be_bytes());
                t.append_bytes(b"hello world");
            });
            let c2 = challenge_after(b"determinism_test", |t| {
                t.append_bytes(&42u64.to_be_bytes());
                t.append_bytes(b"hello world");
            });
            assert_eq!(
                c1, c2,
                "Identical operations must yield identical challenges"
            );
        }

        #[test]
        fn test_domain_separation() {
            let c1 = challenge_after(b"protocol_a", |_| {});
            let c2 = challenge_after(b"protocol_b", |_| {});
            assert_ne!(c1, c2, "Different labels must produce different challenges");
        }

        #[test]
        fn test_challenge_uniqueness() {
            let mut transcript = <$transcript_type>::new(b"uniqueness_test");
            let mut challenges = HashSet::new();

            for i in 0..10_000 {
                let c = transcript.challenge();
                assert!(
                    challenges.insert(c),
                    "Duplicate challenge found at iteration {i}"
                );
            }
        }

        #[test]
        fn test_append_changes_state() {
            let baseline = challenge_after(b"mutation_test", |_| {});
            let after_append = challenge_after(b"mutation_test", |t| {
                t.append_bytes(&1u64.to_be_bytes());
            });
            assert_ne!(
                baseline, after_append,
                "append must change observable challenge"
            );
        }

        #[test]
        fn test_order_matters() {
            let c1 = challenge_after(b"order_test", |t| {
                t.append_bytes(&1u64.to_be_bytes());
                t.append_bytes(&2u64.to_be_bytes());
            });
            let c2 = challenge_after(b"order_test", |t| {
                t.append_bytes(&2u64.to_be_bytes());
                t.append_bytes(&1u64.to_be_bytes());
            });
            assert_ne!(c1, c2, "Order of appends must affect challenge");
        }

        #[test]
        fn test_data_sensitivity() {
            let c1 = challenge_after(b"data_test", |t| {
                t.append_bytes(&0u64.to_be_bytes());
            });
            let c2 = challenge_after(b"data_test", |t| {
                t.append_bytes(&1u64.to_be_bytes());
            });
            assert_ne!(c1, c2, "Different data must produce different challenges");
        }

        #[test]
        fn test_empty_bytes() {
            let baseline = challenge_after(b"empty_test", |_| {});
            let with_empty = challenge_after(b"empty_test", |t| {
                t.append_bytes(&[]);
            });
            assert_ne!(
                baseline, with_empty,
                "append_bytes(&[]) must observably change challenge"
            );
            // Determinism for empty appends.
            let with_empty_again = challenge_after(b"empty_test", |t| {
                t.append_bytes(&[]);
            });
            assert_eq!(with_empty, with_empty_again);
        }

        #[test]
        fn test_large_data() {
            let mut transcript = <$transcript_type>::new(b"large_data_test");
            let large_data = vec![0xABu8; 10_000];

            transcript.append_bytes(&large_data);
            let _ = transcript.challenge();
        }

        #[test]
        fn test_prover_verifier_consistency() {
            let mut prover = <$transcript_type>::new(b"protocol");
            prover.append_bytes(&42u64.to_be_bytes());
            prover.append_bytes(b"commitment");
            let prover_challenge = prover.challenge();

            let mut verifier = <$transcript_type>::new(b"protocol");
            verifier.append_bytes(&42u64.to_be_bytes());
            verifier.append_bytes(b"commitment");
            let verifier_challenge = verifier.challenge();

            assert_eq!(
                prover_challenge, verifier_challenge,
                "Prover and verifier must derive identical challenges"
            );
        }

        #[test]
        fn test_default_delegates_to_new() {
            let mut default_transcript = <$transcript_type>::default();
            let mut new_transcript = <$transcript_type>::new(b"");
            assert_eq!(
                default_transcript.challenge(),
                new_transcript.challenge(),
                "Default must delegate to new(b\"\")"
            );
        }

        #[test]
        #[should_panic(expected = "label must be at most 32 bytes")]
        fn test_label_too_long() {
            let long_label: &[u8; 33] = &[b'x'; 33];
            let _ = <$transcript_type>::new(long_label);
        }

        #[test]
        fn test_max_valid_label() {
            let max_label: &[u8; 32] = &[b'L'; 32];
            let mut t1 = <$transcript_type>::new(max_label);
            let mut t2 = <$transcript_type>::new(max_label);
            assert_eq!(t1.challenge(), t2.challenge());
        }

        #[test]
        fn test_challenge_vector() {
            let mut transcript = <$transcript_type>::new(b"vector_test");
            let challenges = transcript.challenge_vector(5);

            assert_eq!(challenges.len(), 5);

            let unique: HashSet<_> = challenges.iter().collect();
            assert_eq!(unique.len(), 5, "All challenges in vector should be unique");
        }
    };
}
