//! Streaming (chunked) commitment for the Dory scheme.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::{DoryRoutines, PairingCurve};
use jolt_field::Fr;
use jolt_openings::StreamingCommitment;

use crate::scheme::{ark_to_jolt_g1, ark_to_jolt_gt, jolt_fr_to_ark, jolt_g1_vec_to_ark, ArkFr};
use crate::types::{DoryCommitment, DoryPartialCommitment};

type InnerBN254 = dory::backends::arkworks::BN254;

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
}
