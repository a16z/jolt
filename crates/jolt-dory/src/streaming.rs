//! Streaming (chunked) commitment for the Dory scheme.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::DoryRoutines;
use jolt_field::Fr;
use jolt_openings::StreamingCommitment;

use crate::scheme::{
    ark_to_jolt_fr, ark_to_jolt_g1, ark_to_jolt_g1_vec, ark_to_jolt_gt, commit_rows_tier_2,
    jolt_fr_to_ark, jolt_g1_vec_to_ark, ArkFr,
};
use crate::types::{DoryCommitment, DoryHint, DoryPartialCommitment, DoryProverSetup};

impl crate::DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish_zk")]
    pub fn finish_zk(
        partial: DoryPartialCommitment,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint) {
        validate_row_count(partial.row_commitments.len(), setup);
        let row_commitments = jolt_g1_vec_to_ark(partial.row_commitments);
        let (tier_2, commit_blind) = commit_rows_tier_2::<dory::ZK>(&row_commitments, setup);
        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(
                ark_to_jolt_g1_vec(row_commitments),
                ark_to_jolt_fr(&commit_blind),
            ),
        )
    }
}

impl StreamingCommitment for crate::DoryScheme {
    type PartialCommitment = DoryPartialCommitment;

    fn begin(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryPartialCommitment {
            row_commitments: Vec::new(),
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
    /// streamed row count is a power of two (the layout `DoryScheme::commit`
    /// produces).
    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish")]
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        let num_rows = partial.row_commitments.len();
        validate_row_count(num_rows, setup);

        let ark_rows = jolt_g1_vec_to_ark(partial.row_commitments);
        let (tier_2, _) = commit_rows_tier_2::<dory::Transparent>(&ark_rows, setup);
        DoryCommitment(ark_to_jolt_gt(&tier_2))
    }
}

fn validate_row_count(num_rows: usize, setup: &DoryProverSetup) {
    assert!(
        num_rows.is_power_of_two(),
        "streaming: row count ({num_rows}) must be a power of two",
    );
    assert!(
        num_rows <= setup.0.g2_vec.len(),
        "streaming: row count ({}) exceeds Dory SRS size ({})",
        num_rows,
        setup.0.g2_vec.len(),
    );
}

#[cfg(test)]
mod tests {
    use jolt_field::RandomSampling;
    use jolt_openings::{CommitmentScheme, StreamingCommitment};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use jolt_field::Fr;

    use crate::DoryScheme;

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
}
