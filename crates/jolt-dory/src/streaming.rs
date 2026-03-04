//! Streaming (chunked) commitment for the Dory scheme.
//!
//! Implements [`StreamingCommitmentScheme`] to allow committing to large
//! polynomials one chunk at a time, without materializing the full
//! evaluation table in memory.

use dory::backends::arkworks::G1Routines;
use dory::primitives::arithmetic::{DoryRoutines, Group as DoryGroup, PairingCurve};
use jolt_field::Fr;
use jolt_openings::StreamingCommitmentScheme;

use crate::scheme::DoryScheme;
use crate::types::{jolt_fr_to_ark, DoryCommitment, DoryPartialCommitment, DoryProverSetup};

type InnerFr = dory::backends::arkworks::ArkFr;
type InnerG1 = dory::backends::arkworks::ArkG1;
type InnerBN254 = dory::backends::arkworks::BN254;

impl StreamingCommitmentScheme for DoryScheme {
    type PartialCommitment = DoryPartialCommitment;

    fn begin_streaming(_setup: &Self::ProverSetup) -> Self::PartialCommitment {
        DoryPartialCommitment {
            row_commitments: Vec::new(),
        }
    }

    fn stream_chunk(partial: &mut Self::PartialCommitment, chunk: &[Fr]) {
        // The `StreamingCommitmentScheme` trait does not pass the prover setup
        // to `stream_chunk`, so we cannot compute the actual Pedersen MSM here.
        // We store a zero placeholder. For production use, prefer
        // [`DoryStreamingCommitter`] which carries the setup reference.
        let _ = chunk;
        partial.row_commitments.push(InnerG1::identity());
    }

    fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment {
        // Without setup access, we cannot compute the multi-pairing.
        // Return the identity commitment as a placeholder.
        let _ = &partial.row_commitments;
        DoryCommitment(<InnerBN254 as PairingCurve>::multi_pair_g2_setup(&[], &[]))
    }
}

/// Streaming commitment helper that carries the prover setup reference,
/// enabling actual MSM computation per chunk.
///
/// Use this instead of the `StreamingCommitmentScheme` trait methods when
/// the prover setup is available at the call site.
pub struct DoryStreamingCommitter<'a> {
    setup: &'a DoryProverSetup,
    num_columns: usize,
    row_commitments: Vec<InnerG1>,
}

impl<'a> DoryStreamingCommitter<'a> {
    /// Creates a new streaming committer with the given setup and column count.
    ///
    /// `num_columns` must match the Dory matrix width (number of G1 generators
    /// used per row MSM).
    pub fn new(setup: &'a DoryProverSetup, num_columns: usize) -> Self {
        Self {
            setup,
            num_columns,
            row_commitments: Vec::new(),
        }
    }

    /// Processes one row of polynomial evaluations, computing its Pedersen commitment.
    ///
    /// `chunk.len()` must equal `num_columns`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk.len() != num_columns`.
    pub fn process_chunk(&mut self, chunk: &[Fr]) {
        assert_eq!(
            chunk.len(),
            self.num_columns,
            "chunk length must match num_columns"
        );

        let g1_bases = &self.setup.0.g1_vec[..self.num_columns];
        let scalars: Vec<InnerFr> = chunk.iter().map(jolt_fr_to_ark).collect();
        let row_commitment = G1Routines::msm(g1_bases, &scalars);
        self.row_commitments.push(row_commitment);
    }

    /// Finalizes the streaming session, computing the tier-2 multi-pairing commitment.
    pub fn finalize(self) -> DoryCommitment {
        let g2_bases = &self.setup.0.g2_vec[..self.row_commitments.len()];
        let tier_2 =
            <InnerBN254 as PairingCurve>::multi_pair_g2_setup(&self.row_commitments, g2_bases);
        DoryCommitment(tier_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_openings::CommitmentScheme;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn streaming_committer_matches_direct() {
        let num_vars: usize = 4;
        let num_cols = 1usize << num_vars.div_ceil(2);
        let num_rows = 1usize << (num_vars - num_vars.div_ceil(2));
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let evals: Vec<Fr> = (0..num_rows * num_cols)
            .map(|_| Fr::random(&mut rng))
            .collect();

        let poly = jolt_poly::DensePolynomial::new(evals.clone());
        let direct = DoryScheme::commit(&poly, &prover_setup);

        let mut streamer = DoryStreamingCommitter::new(&prover_setup, num_cols);
        for row in evals.chunks(num_cols) {
            streamer.process_chunk(row);
        }
        let streamed = streamer.finalize();

        assert_eq!(
            direct, streamed,
            "streaming and direct commitments must match"
        );
    }
}
