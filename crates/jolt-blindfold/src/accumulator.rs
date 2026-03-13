//! BlindFold accumulator: collects ZK stage data for deferred verification.
//!
//! During the Jolt proof, each sumcheck stage produces
//! [`CommittedRoundData`] containing committed round polynomials, their
//! coefficients, and blinding factors. The `BlindFoldAccumulator` collects
//! this data across all stages, then constructs the BlindFold verifier R1CS
//! and Nova folding proof.

use jolt_crypto::JoltCommitment;
use jolt_field::Field;

use crate::proof::CommittedRoundData;

/// Per-stage data stored in the accumulator.
///
/// Each sumcheck stage contributes its committed round data plus
/// metadata describing the claim structure (constraints, degrees,
/// public inputs).
#[derive(Clone, Debug)]
pub struct ZkStageData<F: Field, VC: JoltCommitment> {
    /// Committed round polynomials, coefficients, and blinding factors.
    pub round_data: CommittedRoundData<F, VC>,
}

/// Accumulates ZK stage data across all sumcheck stages for deferred
/// BlindFold verification.
///
/// The orchestrator calls [`push_stage`](Self::push_stage) after each
/// committed sumcheck stage completes. After all stages, the accumulated
/// data is used to build the BlindFold verifier R1CS, perform Nova
/// folding, and produce the final BlindFold proof.
///
/// Generic over [`JoltCommitment`] — the same accumulator works with
/// Pedersen, hash-based, or lattice-based commitment schemes.
#[derive(Clone, Debug)]
pub struct BlindFoldAccumulator<F: Field, VC: JoltCommitment> {
    stages: Vec<ZkStageData<F, VC>>,
}

impl<F: Field, VC: JoltCommitment> Default for BlindFoldAccumulator<F, VC> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field, VC: JoltCommitment> BlindFoldAccumulator<F, VC> {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Pre-allocates for `capacity` stages to avoid reallocation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stages: Vec::with_capacity(capacity),
        }
    }

    /// Appends committed round data from a completed sumcheck stage.
    pub fn push_stage(&mut self, round_data: CommittedRoundData<F, VC>) {
        self.stages.push(ZkStageData { round_data });
    }

    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    pub fn stages(&self) -> &[ZkStageData<F, VC>] {
        &self.stages
    }

    /// Consumes the accumulator, returning ownership of the stage data
    /// for BlindFold proof construction.
    pub fn into_stages(self) -> Vec<ZkStageData<F, VC>> {
        self.stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::arkworks::bn254::Bn254G1;
    use jolt_crypto::Pedersen;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};

    type TestVC = Pedersen<Bn254G1>;

    #[test]
    fn accumulator_push_and_retrieve() {
        let mut acc = BlindFoldAccumulator::<Fr, TestVC>::new();
        assert_eq!(acc.num_stages(), 0);

        let round_data = CommittedRoundData {
            round_commitments: vec![Bn254G1::default(); 3],
            poly_coeffs: vec![vec![Fr::one(), Fr::zero()]; 3],
            blinding_factors: vec![Fr::from_u64(42); 3],
            poly_degrees: vec![1; 3],
            challenges: vec![Fr::from_u64(7); 3],
        };

        acc.push_stage(round_data);
        assert_eq!(acc.num_stages(), 1);
        assert_eq!(acc.stages()[0].round_data.round_commitments.len(), 3);
    }

    #[test]
    fn accumulator_with_capacity() {
        let acc = BlindFoldAccumulator::<Fr, TestVC>::with_capacity(8);
        assert_eq!(acc.num_stages(), 0);
    }

    #[test]
    fn accumulator_into_stages() {
        let mut acc = BlindFoldAccumulator::<Fr, TestVC>::new();
        for i in 0..3 {
            let round_data = CommittedRoundData {
                round_commitments: vec![Bn254G1::default(); i + 1],
                poly_coeffs: vec![vec![Fr::one()]; i + 1],
                blinding_factors: vec![Fr::zero(); i + 1],
                poly_degrees: vec![1; i + 1],
                challenges: vec![Fr::from_u64(1); i + 1],
            };
            acc.push_stage(round_data);
        }

        let stages = acc.into_stages();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].round_data.round_commitments.len(), 1);
        assert_eq!(stages[2].round_data.round_commitments.len(), 3);
    }
}
