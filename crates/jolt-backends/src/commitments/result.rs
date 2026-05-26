use jolt_openings::CommitmentScheme;
use jolt_witness::{OracleDescriptor, OracleRef, ViewRequirement, WitnessNamespace};

use super::CommitmentSlot;

#[derive(Clone)]
pub struct CommittedPolynomialOutput<N: WitnessNamespace, PCS: CommitmentScheme> {
    pub slot: CommitmentSlot,
    pub oracle: OracleRef<N>,
    pub rows: usize,
    pub commitment: PCS::Output,
    pub opening_hint: PCS::OpeningHint,
}

impl<N: WitnessNamespace, PCS: CommitmentScheme> CommittedPolynomialOutput<N, PCS> {
    pub const fn new(
        slot: CommitmentSlot,
        oracle: OracleRef<N>,
        rows: usize,
        commitment: PCS::Output,
        opening_hint: PCS::OpeningHint,
    ) -> Self {
        Self {
            slot,
            oracle,
            rows,
            commitment,
            opening_hint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedWitnessRequirement<N: WitnessNamespace> {
    pub slot: CommitmentSlot,
    pub requirement: ViewRequirement<N>,
    pub descriptor: OracleDescriptor<N>,
}

impl<N: WitnessNamespace> ResolvedWitnessRequirement<N> {
    pub const fn new(
        slot: CommitmentSlot,
        requirement: ViewRequirement<N>,
        descriptor: OracleDescriptor<N>,
    ) -> Self {
        Self {
            slot,
            requirement,
            descriptor,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamedWitnessChunk {
    pub index: usize,
    pub kind: jolt_witness::PolynomialChunkKind,
    pub rows: usize,
}

impl StreamedWitnessChunk {
    pub const fn new(index: usize, kind: jolt_witness::PolynomialChunkKind, rows: usize) -> Self {
        Self { index, kind, rows }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamedWitnessOutput {
    pub slot: CommitmentSlot,
    pub rows: usize,
    pub chunks: Vec<StreamedWitnessChunk>,
}

impl StreamedWitnessOutput {
    pub fn new(slot: CommitmentSlot, chunks: Vec<StreamedWitnessChunk>) -> Self {
        let rows = chunks.iter().map(|chunk| chunk.rows).sum();
        Self { slot, rows, chunks }
    }
}

#[derive(Clone)]
pub struct CommitmentResult<N: WitnessNamespace, PCS: CommitmentScheme> {
    pub resolved_witness: Vec<ResolvedWitnessRequirement<N>>,
    pub streamed_witness: Vec<StreamedWitnessOutput>,
    pub commitments: Vec<CommittedPolynomialOutput<N, PCS>>,
}

impl<N: WitnessNamespace, PCS: CommitmentScheme> CommitmentResult<N, PCS> {
    pub fn new(
        resolved_witness: Vec<ResolvedWitnessRequirement<N>>,
        streamed_witness: Vec<StreamedWitnessOutput>,
        commitments: Vec<CommittedPolynomialOutput<N, PCS>>,
    ) -> Self {
        Self {
            resolved_witness,
            streamed_witness,
            commitments,
        }
    }
}
