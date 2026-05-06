//! Streaming (chunked) commitment for the Dory scheme.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::DoryRoutines;
use jolt_field::Fr;
use jolt_openings::StreamingCommitment;

use crate::scheme::{ark_to_jolt_g1, jolt_fr_to_ark, ArkFr};
use crate::types::DoryPartialCommitment;

impl StreamingCommitment for crate::DoryScheme {
    type PartialCommitment = DoryPartialCommitment;

    fn begin(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryPartialCommitment {
            row_commitments: Vec::new(),
            commit_blind: Fr::default(),
        }
    }

    /// Commits one full row of the polynomial as `MSM(g1_bases[..chunk.len()], chunk)`,
    /// matching the per-row work in [`DoryScheme::commit`](crate::DoryScheme::commit)'s
    /// dense path. Caller must feed every row at the same chunk width.
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed")]
    fn feed(partial: &mut Self::PartialCommitment, chunk: &[Fr], setup: &Self::ProverSetup) {
        assert!(
            chunk.len().is_power_of_two(),
            "streaming: chunk length ({}) must be a power of two",
            chunk.len(),
        );
        assert!(
            chunk.len() <= setup.0.g1_vec.len(),
            "streaming: chunk length ({}) exceeds Dory SRS size ({})",
            chunk.len(),
            setup.0.g1_vec.len(),
        );

        let g1_bases = &setup.0.g1_vec[..chunk.len()];
        let scalars: Vec<ArkFr> = chunk.iter().map(jolt_fr_to_ark).collect();
        let row_commitment = G1Routines::msm(g1_bases, &scalars);
        partial.row_commitments.push(ark_to_jolt_g1(row_commitment));
    }

    /// Aggregates row commitments into the final tier-2 commitment, matching
    /// [`DoryScheme::commit`](crate::DoryScheme::commit). Asserts that the
    /// streamed row count is a power of two.
    ///
    /// In ZK mode (feature = "zk") the tier-2 commitment is blinded. The blind
    /// scalar is discarded here; use [`crate::DoryScheme::finish_with_hint`] when
    /// you need the blind for a subsequent `open_zk` call.
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish")]
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        crate::DoryScheme::finish_with_hint(partial, setup).0
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::RandomSampling;
    #[cfg(not(feature = "zk"))]
    use jolt_openings::CommitmentScheme;
    use jolt_openings::StreamingCommitment;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_field::Fr;

    use crate::DoryScheme;

    // Direct == streaming only holds without blinding; in ZK mode each commit()
    // draws a fresh random blind so the two GT elements will differ.
    #[cfg(not(feature = "zk"))]
    #[test]
    fn streaming_matches_direct() {
        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let num_rows = 1usize << (num_vars - num_vars.div_ceil(2));
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let evals: Vec<Fr> = (0..num_rows * num_cols)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();

        let poly = jolt_poly::Polynomial::new(evals.clone());
        let (direct, _) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals.chunks(num_cols) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let streamed = DoryScheme::finish(partial, &prover_setup);

        assert_eq!(
            direct, streamed,
            "streaming and direct commitments must match"
        );
    }

    #[cfg(feature = "zk")]
    #[test]
    fn zk_streaming_open_zk_round_trip() {
        use jolt_openings::ZkOpeningScheme;
        use jolt_transcript::Transcript;

        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let mut rng = ChaCha20Rng::seed_from_u64(101);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);

        let evals: Vec<Fr> = (0..1usize << num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let poly = jolt_poly::Polynomial::new(evals.clone());
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals.chunks(num_cols) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let (commitment, hint) = DoryScheme::finish_with_hint(partial, &prover_setup);

        let mut pt = jolt_transcript::Blake2bTranscript::new(b"zk-streaming-rt");
        let (proof, _eval_com, _blind) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, Some(hint), &mut pt);

        let mut vt = jolt_transcript::Blake2bTranscript::new(b"zk-streaming-rt");
        let result = DoryScheme::verify_zk(&commitment, &point, &proof, &verifier_setup, &mut vt);
        assert!(
            result.is_ok(),
            "ZK streaming round-trip verification failed: {result:?}"
        );
    }
}
