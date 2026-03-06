//! Streaming (chunked) commitment for the Dory scheme.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::{DoryRoutines, PairingCurve};
use jolt_crypto::Bn254GT;
use jolt_field::Fr;
use jolt_openings::StreamingCommitment;

use crate::scheme::jolt_fr_to_ark;
use crate::types::{DoryCommitment, DoryPartialCommitment};

type InnerFr = dory::backends::arkworks::ArkFr;
type InnerBN254 = dory::backends::arkworks::BN254;

impl StreamingCommitment for crate::DoryScheme {
    type PartialCommitment = DoryPartialCommitment;

    fn begin(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryPartialCommitment {
            row_commitments: Vec::new(),
        }
    }

    fn feed(partial: &mut Self::PartialCommitment, chunk: &[Fr], setup: &Self::ProverSetup) {
        let g1_bases = &setup.0.g1_vec[..chunk.len()];
        let scalars: Vec<InnerFr> = chunk.iter().map(jolt_fr_to_ark).collect();
        let row_commitment = G1Routines::msm(g1_bases, &scalars);
        // SAFETY: ArkG1 is repr(transparent) over G1Projective, same as Bn254G1.
        let jolt_commitment: jolt_crypto::Bn254G1 = unsafe { std::mem::transmute(row_commitment) };
        partial.row_commitments.push(jolt_commitment);
    }

    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output {
        // SAFETY: Bn254G1 is repr(transparent) over G1Projective, same as ArkG1.
        let ark_row_commitments: Vec<dory::backends::arkworks::ArkG1> = unsafe {
            std::mem::transmute(partial.row_commitments)
        };
        let g2_bases = &setup.0.g2_vec[..ark_row_commitments.len()];
        let tier_2 =
            <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&ark_row_commitments, g2_bases);
        // SAFETY: ArkGT is repr(transparent) over Fq12, same as Bn254GT.
        let gt: Bn254GT = unsafe { std::mem::transmute(tier_2) };
        DoryCommitment(gt)
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
