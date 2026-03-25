//! Common test utilities and standardized test suite for transcript implementations.

/// Standardized test suite macro for any `Transcript` implementation.
///
/// This macro generates a comprehensive test suite that verifies the core
/// properties required of any Fiat-Shamir transcript implementation:
///
/// - Determinism: Same inputs produce same outputs
/// - Domain separation: Different labels produce different transcripts
/// - Challenge uniqueness: Sequential challenges are unique
/// - State mutation: Appending data changes the state
/// - Prover/verifier consistency: Both sides derive identical challenges
#[macro_export]
macro_rules! transcript_tests {
    ($transcript_type:ty) => {
        use jolt_transcript::Transcript;
        use std::collections::HashSet;

        #[test]
        fn test_determinism() {
            let mut t1 = <$transcript_type>::new(b"determinism_test");
            let mut t2 = <$transcript_type>::new(b"determinism_test");

            t1.append_bytes(&42u64.to_be_bytes());
            t2.append_bytes(&42u64.to_be_bytes());
            assert_eq!(
                t1.state(),
                t2.state(),
                "States should match after identical operations"
            );

            t1.append_bytes(b"hello world");
            t2.append_bytes(b"hello world");
            assert_eq!(t1.state(), t2.state());

            assert_eq!(
                t1.challenge(),
                t2.challenge(),
                "Challenges should be identical for identical transcripts"
            );
        }

        #[test]
        fn test_domain_separation() {
            let mut t1 = <$transcript_type>::new(b"protocol_a");
            let mut t2 = <$transcript_type>::new(b"protocol_b");

            assert_ne!(
                t1.state(),
                t2.state(),
                "Different labels should produce different initial states"
            );

            assert_ne!(
                t1.challenge(),
                t2.challenge(),
                "Different labels should produce different challenges"
            );
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
            let mut transcript = <$transcript_type>::new(b"mutation_test");
            let initial_state = *transcript.state();

            transcript.append_bytes(&1u64.to_be_bytes());
            assert_ne!(
                *transcript.state(),
                initial_state,
                "append should change state"
            );

            let state_after_append = *transcript.state();
            transcript.append_bytes(b"test");
            assert_ne!(
                *transcript.state(),
                state_after_append,
                "append_bytes should change state"
            );
        }

        #[test]
        fn test_challenge_changes_state() {
            let mut transcript = <$transcript_type>::new(b"challenge_mutation");
            let initial_state = *transcript.state();

            let _ = transcript.challenge();
            assert_ne!(
                *transcript.state(),
                initial_state,
                "challenge should change state"
            );
        }

        #[test]
        fn test_order_matters() {
            let mut t1 = <$transcript_type>::new(b"order_test");
            let mut t2 = <$transcript_type>::new(b"order_test");

            t1.append_bytes(&1u64.to_be_bytes());
            t1.append_bytes(&2u64.to_be_bytes());

            t2.append_bytes(&2u64.to_be_bytes());
            t2.append_bytes(&1u64.to_be_bytes());

            assert_ne!(
                t1.state(),
                t2.state(),
                "Order of operations should affect state"
            );
        }

        #[test]
        fn test_data_sensitivity() {
            let mut t1 = <$transcript_type>::new(b"data_test");
            let mut t2 = <$transcript_type>::new(b"data_test");

            t1.append_bytes(&0u64.to_be_bytes());
            t2.append_bytes(&1u64.to_be_bytes());

            assert_ne!(
                t1.state(),
                t2.state(),
                "Different data should produce different states"
            );
        }

        #[test]
        fn test_empty_bytes() {
            let mut t1 = <$transcript_type>::new(b"empty_test");
            let mut t2 = <$transcript_type>::new(b"empty_test");
            let initial_state = *t1.state();

            t1.append_bytes(&[]);
            assert_ne!(
                *t1.state(),
                initial_state,
                "Empty bytes should change state"
            );

            t2.append_bytes(&[]);
            assert_eq!(t1.state(), t2.state());
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
                "Prover and verifier should derive identical challenges"
            );
        }

        #[test]
        fn test_clone_independence() {
            let mut original = <$transcript_type>::new(b"clone_test");
            original.append_bytes(&1u64.to_be_bytes());

            let mut cloned = original.clone();

            cloned.append_bytes(&2u64.to_be_bytes());

            let original_challenge = original.challenge();

            let mut fresh = <$transcript_type>::new(b"clone_test");
            fresh.append_bytes(&1u64.to_be_bytes());
            fresh.append_bytes(&2u64.to_be_bytes());
            let fresh_challenge = fresh.challenge();

            assert_ne!(
                original_challenge, fresh_challenge,
                "Clone mutation should not affect original"
            );
        }

        #[test]
        fn test_debug_impl() {
            let transcript = <$transcript_type>::new(b"debug_test");
            let debug_str = format!("{:?}", transcript);

            assert!(
                debug_str.contains("state"),
                "Debug output should contain state"
            );
            assert!(
                debug_str.contains("n_rounds"),
                "Debug output should contain n_rounds"
            );
        }

        #[test]
        fn test_default_vs_new() {
            let default_transcript = <$transcript_type>::default();
            let new_transcript = <$transcript_type>::new(b"");

            assert_ne!(
                default_transcript.state(),
                new_transcript.state(),
                "Default and new with empty label should differ"
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
            let transcript = <$transcript_type>::new(max_label);
            assert!(!transcript.state().iter().all(|&b| b == 0));
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
