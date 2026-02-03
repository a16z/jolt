//! Poseidon parameter generation for BN254 Fq
//!
//! Uses poseidon-paramgen (arkworks 0.5 fork) to generate parameters.
//! This module exists to:
//! 1. Provide transparency - show exactly how Fq params are derived
//! 2. Enable verification - test that hardcoded params match generated ones
//! 3. Allow regeneration - if parameters ever need updating
//!
//! The parameters were originally generated with poseidon-paramgen v0.4.0
//! (audited by NCC Group, Summer 2022). We use an arkworks 0.5 compatible fork
//! at https://github.com/defi-wonderland/poseidon377 that preserves the
//! cryptographic logic while updating only API calls.

use ark_bn254::Fq;
use ark_ff::PrimeField;
use poseidon_paramgen::v1::generate;
use poseidon_parameters::v1::PoseidonParameters;

/// Generate Poseidon parameters for BN254 Fq (base field)
///
/// Configuration:
/// - Width: 4 (state size, allows 3 inputs with rate = width - 1)
/// - Security: 128 bits
/// - Alpha: 5 (S-box exponent, auto-selected based on field)
///
/// These parameters are used for Poseidon hashing over the base field,
/// which is needed for SNARK recursion where the verifier operates in Fq.
pub fn generate_fq_params() -> PoseidonParameters<Fq> {
    let width = 4;
    let security_bits = 128;
    generate::<Fq>(security_bits, width, Fq::MODULUS, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::poseidon_fq_params::*;
    use poseidon_parameters::v1::{Alpha, MatrixOperations};

    /// Verifies that the hardcoded Poseidon parameters in `poseidon_fq_params.rs`
    /// exactly match what `poseidon-paramgen` generates for BN254 Fq.
    ///
    /// This test compares:
    /// - Round counts: full rounds (8) and partial rounds (56)
    /// - Alpha (S-box exponent): 5
    /// - MDS matrix: all 16 elements (4×4)
    /// - Round constants: all 256 values (64 rounds × 4 width)
    ///
    /// If any value doesn't match, the test fails with a descriptive message
    /// indicating which parameter diverged. This serves as a correctness check
    /// to ensure the hardcoded values weren't accidentally modified and match
    /// the audited `poseidon-paramgen` output.
    #[test]
    fn verify_hardcoded_fq_params_match_generated() {
        let generated = generate_fq_params();

        // Verify round counts
        assert_eq!(
            generated.rounds.full(),
            FQ_FULL_ROUNDS,
            "Full rounds mismatch"
        );
        assert_eq!(
            generated.rounds.partial(),
            FQ_PARTIAL_ROUNDS,
            "Partial rounds mismatch"
        );

        // Verify alpha (S-box exponent)
        let expected_alpha = Alpha::Exponent(FQ_ALPHA as u32);
        assert_eq!(generated.alpha, expected_alpha, "Alpha mismatch");

        // Verify MDS matrix
        for (i, row) in generated.mds.iter_rows().enumerate() {
            for (j, elem) in row.iter().enumerate() {
                let expected = Fq::from_be_bytes_mod_order(&hex_to_bytes(FQ_MDS[i][j]));
                assert_eq!(*elem, expected, "MDS mismatch at [{i}][{j}]");
            }
        }

        // Verify round constants
        let ark: Vec<Fq> = generated.arc.iter_rows().flatten().cloned().collect();
        assert_eq!(
            ark.len(),
            FQ_ROUND_CONSTANTS.len(),
            "Round constant count mismatch"
        );
        for (i, c) in ark.iter().enumerate() {
            let expected = Fq::from_be_bytes_mod_order(&hex_to_bytes(FQ_ROUND_CONSTANTS[i]));
            assert_eq!(*c, expected, "Round constant mismatch at index {i}");
        }
    }

    #[test]
    fn print_generated_params_summary() {
        let params = generate_fq_params();
        println!("Generated BN254 Fq Poseidon parameters:");
        println!("  Width: {}", params.t);
        println!("  Full rounds: {}", params.rounds.full());
        println!("  Partial rounds: {}", params.rounds.partial());
        println!("  Alpha: {:?}", params.alpha);
        println!("  MDS matrix size: {}x{}", params.t, params.t);
        println!(
            "  Total round constants: {}",
            params.rounds.total() * params.t
        );
    }

    /// Test that generated params work with light-poseidon (the library Jolt uses)
    /// This is an end-to-end integration test verifying the params are correctly
    /// formatted for use in PoseidonTranscriptFq.
    #[test]
    fn test_generated_params_work_with_light_poseidon() {
        use ark_std::Zero;
        use crate::transcripts::poseidon::{FqParams, PoseidonParams};
        use light_poseidon::PoseidonHasher;

        // Create a Poseidon hasher using the hardcoded Fq params (via FqParams trait)
        let mut hasher = FqParams::poseidon();

        // Hash some test inputs - this will fail if params are malformed
        let input1 = Fq::from(123u64);
        let input2 = Fq::from(456u64);
        let input3 = Fq::from(789u64);

        let result = hasher.hash(&[input1, input2, input3]);
        assert!(result.is_ok(), "Poseidon hash failed with Fq params");

        let hash_output = result.unwrap();
        assert!(!hash_output.is_zero(), "Hash output should not be zero");

        // Verify determinism - same inputs produce same output
        let mut hasher2 = FqParams::poseidon();
        let result2 = hasher2.hash(&[input1, input2, input3]).unwrap();
        assert_eq!(hash_output, result2, "Poseidon hash should be deterministic");
    }

    /// Test that PoseidonTranscriptFq produces consistent challenges
    #[test]
    fn test_fq_transcript_determinism() {
        use crate::transcripts::poseidon::PoseidonTranscriptFq;
        use crate::transcripts::Transcript;

        let mut t1: PoseidonTranscriptFq = Transcript::new(b"test_determinism");
        t1.append_u64(42);
        t1.append_bytes(b"hello world");

        let mut t2: PoseidonTranscriptFq = Transcript::new(b"test_determinism");
        t2.append_u64(42);
        t2.append_bytes(b"hello world");

        // Same inputs should produce same state
        assert_eq!(t1.state, t2.state, "Fq transcript should be deterministic");
    }
}
