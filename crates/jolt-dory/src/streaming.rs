//! Streaming (chunked) commitment for the Dory scheme.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::{DoryRoutines, PairingCurve};
use jolt_field::Fr;
use jolt_openings::StreamingCommitment;
use rayon::prelude::*;

use crate::scheme::{
    ark_to_jolt_g1, ark_to_jolt_g1_vec, ark_to_jolt_gt, jolt_fr_to_ark, jolt_g1_vec_to_ark, ArkFr,
};
use crate::types::{DoryCommitment, DoryHint, DoryPartialCommitment, DoryProverSetup};

type InnerBN254 = dory::backends::arkworks::BN254;

impl crate::DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::commit_evaluations_with_row_len")]
    pub fn commit_evaluations_with_row_len(
        data: &[Fr],
        row_len: usize,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint) {
        assert!(row_len > 0, "Dory row length must be nonzero");

        let g1_bases = &setup.0.g1_vec[..row_len];
        let row_commitments: Vec<_> = data
            .par_chunks(row_len)
            .map(|chunk| {
                let scalars: Vec<ArkFr> = chunk.iter().map(jolt_fr_to_ark).collect();
                G1Routines::msm(&g1_bases[..chunk.len()], &scalars)
            })
            .collect();

        let g2_bases = &setup.0.g2_vec[..row_commitments.len()];
        let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);
        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint(ark_to_jolt_g1_vec(row_commitments)),
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

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_feed")]
    fn feed(partial: &mut Self::PartialCommitment, chunk: &[Fr], setup: &Self::ProverSetup) {
        let g1_bases = &setup.0.g1_vec[..chunk.len()];
        let scalars: Vec<ArkFr> = chunk.iter().map(jolt_fr_to_ark).collect();
        let row_commitment = G1Routines::msm(g1_bases, &scalars);
        partial.row_commitments.push(ark_to_jolt_g1(row_commitment));
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::stream_finish")]
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        let ark_rows = jolt_g1_vec_to_ark(partial.row_commitments);
        let g2_bases = &setup.0.g2_vec[..ark_rows.len()];
        let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&ark_rows, g2_bases);
        DoryCommitment(ark_to_jolt_gt(&tier_2))
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::Field;
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
            .map(|_| <Fr as Field>::random(&mut rng))
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

    #[test]
    fn batched_rows_match_streaming_and_direct() {
        let num_vars: usize = 6;
        let row_len = 1usize << num_vars.div_ceil(2);
        let mut rng = ChaCha20Rng::seed_from_u64(101);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let evals: Vec<Fr> = (0..(1usize << num_vars))
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();
        let poly = jolt_poly::Polynomial::new(evals.clone());

        let (direct, direct_hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let (batched, batched_hint) =
            DoryScheme::commit_evaluations_with_row_len(&evals, row_len, &prover_setup);

        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals.chunks(row_len) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let streamed = DoryScheme::finish(partial, &prover_setup);

        assert_eq!(batched, direct);
        assert_eq!(batched, streamed);
        assert_eq!(batched_hint.0, direct_hint.0);
    }

    #[test]
    fn batched_rows_allow_short_final_row() {
        let num_vars: usize = 5;
        let row_len = 1usize << num_vars.div_ceil(2);
        let short_len = (1usize << num_vars) - 1;
        let mut rng = ChaCha20Rng::seed_from_u64(102);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let evals: Vec<Fr> = (0..short_len)
            .map(|_| <Fr as Field>::random(&mut rng))
            .collect();

        let (batched, batched_hint) =
            DoryScheme::commit_evaluations_with_row_len(&evals, row_len, &prover_setup);
        let mut partial = DoryScheme::begin(&prover_setup);
        for row in evals.chunks(row_len) {
            DoryScheme::feed(&mut partial, row, &prover_setup);
        }
        let streaming_hint = partial.row_commitments.clone();
        let streamed = DoryScheme::finish(partial, &prover_setup);

        assert_eq!(batched, streamed);
        assert_eq!(batched_hint.0, streaming_hint);
    }
}
